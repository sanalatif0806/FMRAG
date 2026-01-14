

# --- System deps ---
!apt-get -y -qq install libsuitesparse-dev

# --- Python deps (latest versions, GPU runtime will auto-pick CUDA wheels) ---
!pip install --upgrade --no-cache-dir \
    torch==2.6 torchvision torchaudio \
    numpy scipy pandas requests \
    scikit-sparse accelerate peft datasets \
    "transformers[torch]" sentencepiece einops faiss-cpu \
    scikit-learn matplotlib sparqlwrapper bitsandbytes \
    sentence-transformers nltk tqdm \
    flwr "flwr[simulation]" ray click \
    huggingface-hub

!pip install faiss-cpu
!pip install sacremoses
!pip install dataset

!pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install -U sentence-transformers

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Federated Med-Dialogue + RAG (Flower NumPyClient) — LoRA + BLEU/ROUGE + PRF1
SERVER-SIDE RAG IMPLEMENTATION

Updated with fixes for decreasing precision/recall/F1 scores:
- Added regularization to prevent overfitting
- Implemented learning rate scheduling
- Improved response generation with better temperature tuning
- Enhanced early stopping based on multiple metrics
- Added response quality monitoring
"""

import os
import gc
import math
import traceback
import random
import logging
import signal
import time
from typing import Dict, List, Tuple, Optional
import csv
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

import faiss
from datasets import load_dataset
from SPARQLWrapper import SPARQLWrapper, JSON

# Flower imports
import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

# transformers / peft imports
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    GenerationConfig,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig
from peft import TaskType

# -------------------------------- Logging ---------------------------------
logging.basicConfig(level=logging.DEBUG, force=True, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("fedmed")
logger.debug("Logging initialized")
CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")
OUTPUT_DIR = os.environ.get("FEDMED_OUT", "./fedmed_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------- Configuration ----------------------------
class Config:
    NUM_ROUNDS = 10
    NUM_CLIENTS = 3
    USE_SMALL_MODEL = True
    MAX_SEQ_LENGTH = 256
    GEN_PROMPT_MAX = 256
    RAG_TOP_K = 5
    MAX_DATASET_SIZE = 27000
    LORA_R = 8  # Increased for better capacity
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1  # Increased dropout for regularization
    LEARNING_RATE = 2e-4  # Reduced learning rate
    WEIGHT_DECAY = 0.01  # Added weight decay
    GRADIENT_ACCUMULATION_STEPS = 1
    EARLY_STOPPING_PATIENCE = 3
    MIN_LOSS_IMPROVEMENT = 0.01
    TRAINING_EPOCHS = 3  # Increased epochs for better learning
    WARMUP_STEPS = 100
    SEED = 42
    GENERATION_MAX_LENGTH = 60  # Increased for better responses
    GENERATION_TEMPERATURE = 0.8  # Increased for more diverse responses
    GENERATION_TOP_P = 0.9  # Increased for better quality
    GENERATION_TOP_K = 50  # Added top-k sampling
    MIN_GENERATED_TOKENS = 3
    MIN_RECORDS_PER_CLIENT = 2000
    RESPONSE_QUALITY_THRESHOLD = 0.6  # Minimum F1 score to consider response good

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

# ==================== Timeout Handling =======================
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

def run_with_timeout(func, timeout=120, default=None):
    """Run a function with a timeout"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func()
        signal.alarm(0)
        return result
    except TimeoutError:
        logger.error(f"Function timed out after {timeout} seconds")
        return default
    except Exception as e:
        signal.alarm(0)
        raise e

# ==================== Safe generation helpers =======================
def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = Config.GENERATION_MAX_LENGTH,
    temperature: float = Config.GENERATION_TEMPERATURE,
    top_p: float = Config.GENERATION_TOP_P,
    top_k: int = Config.GENERATION_TOP_K,
    do_sample: bool = True,  # Changed to True for better diversity
) -> str:
    """Wrapper around HF .generate with improved sampling"""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=Config.GEN_PROMPT_MAX,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,  # Added repetition penalty
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = model.generate(**enc, generation_config=gen_cfg)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def extract_doctor_response(full_text: str) -> str:
    """Return the substring after the last 'Doctor:' marker"""
    if not full_text:
        return ""
    low = full_text.lower()
    pos = low.rfind("doctor:")
    if pos != -1:
        return full_text[pos + len("doctor:"):].strip()
    return full_text.strip()

# ====================== Initialization / models ===========================
def initialize_components():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Generator Model
        logger.info("Loading generator tokenizer...")
        model_name = "distilgpt2"
        gen_tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        if gen_tok.pad_token is None:
            gen_tok.pad_token = gen_tok.eos_token
        gen_tok.padding_side = "left"

        logger.info("Loading generator model...")
        gen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float32
        )
        gen_model.to(device)

        # LoRA config with improved settings
        logger.info("Setting up LoRA...")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,  # Increased dropout
            bias="none",
            target_modules=["c_attn", "c_proj", "c_fc"],
        )

        gen_model = get_peft_model(gen_model, lora_cfg)
        logger.info("✓ LoRA setup completed")

        # Freeze base params, only adapters trainable
        for n, p in gen_model.named_parameters():
            if "lora_" in n or "peft_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        gen_model.to(device)

        # Retriever
        logger.info("Loading retriever tokenizer...")
        ret_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir=CACHE_DIR)

        logger.info("Loading retriever model...")
        ret_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir=CACHE_DIR)
        ret_model.to(device)
        ret_model.eval()

        logger.info("All components initialized successfully!")
        return gen_model, gen_tok, ret_model, ret_tok, device

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        logger.error(traceback.format_exc())
        raise

# --------------------------- Data loading & helpers ------------------------
def get_symptoms_from_sparql():
    """Get symptoms with fallback"""
    fallback_symptoms = ["headache", "nausea", "fever", "cough", "fatigue",
                        "dizziness", "sore throat", "chest pain", "abdominal pain"]

    def run_query(endpoint, q):
        try:
            s = SPARQLWrapper(endpoint)
            s.setQuery(q)
            s.setReturnFormat(JSON)
            s.setTimeout(10)
            res = s.query().convert()
            return [r["symptomLabel"]["value"] for r in res["results"]["bindings"]]
        except Exception:
            return []

    try:
        ontobee_q = "PREFIX obo: <http://purl.obolibrary.org/obo/> SELECT DISTINCT ?symptomLabel WHERE { ?symptom a obo:SYMP_0000001 . ?symptom rdfs:label ?symptomLabel . FILTER (lang(?symptomLabel) = 'en') } LIMIT 10"
        wikidata_q = "SELECT DISTINCT ?symptomLabel WHERE { ?symptom wdt:P31 wd:Q169872 . ?symptom rdfs:label ?symptomLabel . FILTER (lang(?symptomLabel) = 'en') } LIMIT 10"

        syms = list(set(run_query("https://sparql.hegroup.org/sparql", ontobee_q) +
                        run_query("https://query.wikidata.org/sparql", wikidata_q)))

        return syms if syms else fallback_symptoms

    except Exception:
        return fallback_symptoms

def preprocess_dialog(dialog: str) -> str:
    """Standardize dialog format"""
    if not dialog or not isinstance(dialog, str):
        return ""
    dialog = " ".join(dialog.strip().split())
    if len(dialog) < 10:
        return ""
    low = dialog.lower()
    if "patient:" not in low and "doctor:" not in low:
        dialog = f"Patient: {dialog}. Doctor: Please consult a healthcare professional."
    elif "patient:" not in low and "doctor:" in low:
        before_doc = dialog.split("Doctor:")[0].strip()
        after_doc = dialog.split("Doctor:")[-1].strip()
        dialog = f"Patient: {before_doc}. Doctor: {after_doc}"
    elif "doctor:" not in low:
        dialog = f"{dialog}. Doctor: Please consult a healthcare professional."
    return dialog

def create_synthetic_medical_data(num_samples):
    """Create synthetic medical dialogue data"""
    synthetic_samples = []
    medical_conditions = [
        "headache", "fever", "cough", "chest pain", "abdominal pain",
        "nausea", "dizziness", "shortness of breath", "fatigue"
    ]

    treatments = {
        "headache": "Take over-the-counter pain relief and rest in a quiet, dark room.",
        "fever": "Stay hydrated, take acetaminophen or ibuprofen, and rest.",
        "cough": "Drink plenty of fluids, use cough drops, and consider cough medicine.",
        "chest pain": "Seek immediate medical attention as this could be serious.",
        "abdominal pain": "Rest and avoid solid foods for a few hours.",
        "nausea": "Try ginger tea or over-the-counter anti-nausea medication.",
        "dizziness": "Sit or lie down immediately and avoid sudden movements.",
        "shortness of breath": "Seek immediate medical attention.",
        "fatigue": "Ensure you're getting enough sleep and consider dietary changes.",
    }

    for i in range(num_samples):
        condition = random.choice(medical_conditions)
        duration = random.choice(["a few hours", "a day", "several days", "a week"])
        severity = random.choice(["mild", "moderate", "severe"])

        patient_msg = f"I've been experiencing {condition} for {duration}. It's {severity}."
        doctor_msg = treatments.get(condition, "Please consult with a healthcare professional for proper diagnosis.")

        dialog = f"Patient: {patient_msg} Doctor: {doctor_msg}"
        synthetic_samples.append({"dialog": dialog})

    return synthetic_samples

def load_dialogs_fixed(max_samples: int = Config.MAX_DATASET_SIZE):
    """Load medical dialogue datasets with robust error handling"""
    all_dialogs = []

    dataset_configs = [
        {"name": "huzaifa525/Medical_Intelligence_Dataset_40k_Rows_of_Disease_Info_Treatments_and_Medical_QA",
         "split": "train", "text_key": "input", "output_key": "output"},
    ]

    for config in dataset_configs:
        try:
            logger.info(f"Loading dataset: {config['name']}")
            ds = load_dataset(config["name"], cache_dir=CACHE_DIR)
            split_data = ds[config["split"]] if config["split"] in ds else ds["train"]

            loaded_count = 0
            for sample in split_data:
                try:
                    if config["text_key"] in sample and config.get("output_key") in sample:
                        input_text = sample[config["text_key"]]
                        output_text = sample[config["output_key"]]
                        dialog_text = f"Patient: {input_text} Doctor: {output_text}"
                    elif config["text_key"] in sample:
                        dialog_text = sample[config["text_key"]]
                    else:
                        continue

                    dialog_text = preprocess_dialog(str(dialog_text))
                    if (isinstance(dialog_text, str) and
                        len(dialog_text.strip()) >= 10 and
                        len(dialog_text) <= Config.MAX_SEQ_LENGTH * 4):
                        all_dialogs.append({"dialog": dialog_text})
                        loaded_count += 1

                    if len(all_dialogs) >= max_samples:
                        break

                except Exception:
                    continue

            logger.info(f"Loaded {loaded_count} samples from {config['name']}")

        except Exception as e:
            logger.warning(f"Failed to load dataset {config['name']}: {e}")
            continue

    # Add synthetic data if needed
    if len(all_dialogs) < 500:
        logger.warning(f"Only {len(all_dialogs)} dialogs loaded, adding synthetic data")
        synthetic_count = min(500, max_samples - len(all_dialogs))
        synthetic_data = create_synthetic_medical_data(synthetic_count)
        all_dialogs.extend(synthetic_data)

    # Ensure minimum records
    min_required = Config.NUM_CLIENTS * Config.MIN_RECORDS_PER_CLIENT
    if len(all_dialogs) < min_required:
        repeat_factor = min_required // len(all_dialogs) + 1
        original_count = len(all_dialogs)
        all_dialogs = all_dialogs * repeat_factor
        all_dialogs = all_dialogs[:min_required]
        logger.info(f"Extended dataset from {original_count} to {len(all_dialogs)} samples via repetition")

    logger.info(f"Final total dialogs loaded: {len(all_dialogs)}")
    return all_dialogs

# --------------------------------- RAG System ------------------------------
class RAGSystem:
    def __init__(self, generator_model, generator_tokenizer, retriever_model, retriever_tokenizer, corpus: List[str], device: torch.device):
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.retriever_model = retriever_model
        self.retriever_tokenizer = retriever_tokenizer
        self.corpus = corpus or []
        self.device = device
        self.index = None
        self._build_index(self.corpus)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        batch_size = 4
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            inputs = self.retriever_tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=Config.MAX_SEQ_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.retriever_model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embs.append(emb)
        return np.vstack(embs) if embs else np.zeros((0, 384), dtype=np.float32)

    def _build_index(self, texts: List[str]):
        texts = (texts or [])[:Config.MAX_DATASET_SIZE]
        if not texts:
            self.index = None
            return

        logger.info(f"Building FAISS index with {len(texts)} texts...")
        E = self._embed_texts(texts)
        if E.size == 0:
            self.index = None
            return

        dim = E.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(E.astype(np.float32))
        logger.info("✓ FAISS index built")

    def _clean_response(self, response: str) -> str:
        if not response:
            return "Please consult a healthcare professional."
        response = response.strip()
        while response and response[0] in ":;,-":
            response = response[1:].strip()
        last_period = response.rfind('.')
        if last_period != -1 and last_period > 10:
            response = response[:last_period+1]
        tokens = response.strip().split()
        if len(tokens) < 2:
            return "Please consult a healthcare professional."
        return response.strip()

    def _truncate_prompt(self, prompt: str) -> str:
        ids = self.generator_tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]
        if ids.size(0) <= Config.GEN_PROMPT_MAX:
            return prompt
        keep_ids = ids[-Config.GEN_PROMPT_MAX:]
        return self.generator_tokenizer.decode(keep_ids, skip_special_tokens=True)

    def generate(self, query: str, top_k: int = Config.RAG_TOP_K) -> str:
        try:
            q_emb = self._embed_texts([query])
            if q_emb.shape[0] == 0 or self.index is None or self.index.ntotal == 0:
                context = ""
            else:
                k = min(top_k, self.index.ntotal)
                _, idxs = self.index.search(q_emb.astype(np.float32), k)
                docs = [self.corpus[i] for i in idxs[0] if i < len(self.corpus)]
                context = "\n".join([f"- {d}" for d in docs[:3]])

            prompt = (
                "You are a medical professional. Use the relevant context to provide a helpful, general reply.\n"
                "If safety is a concern, recommend seeking professional care.\n\n"
                f"Context:\n{context}\n\n"
                f"Patient: {query}\n"
                "Doctor:"
            )
            prompt = self._truncate_prompt(prompt)

            full = generate_with_config(
                self.generator_model,
                self.generator_tokenizer,
                prompt,
                device=self.device,
                max_new_tokens=Config.GENERATION_MAX_LENGTH,
                temperature=Config.GENERATION_TEMPERATURE,
                top_p=Config.GENERATION_TOP_P,
                top_k=Config.GENERATION_TOP_K,
                do_sample=True,
            )

            resp = extract_doctor_response(full)
            if not resp or len(self.generator_tokenizer(resp)["input_ids"]) < Config.MIN_GENERATED_TOKENS:
                return "Please consult a healthcare professional."
            return self._clean_response(resp)
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return "Please consult a healthcare professional."

# --------------------------- Metrics helpers -------------------------------
def _overlap_count(pred_tokens: List[str], ref_tokens: List[str]) -> int:
    """Multiset overlap count (clipped)."""
    r_counts = {}
    for t in ref_tokens:
        r_counts[t] = r_counts.get(t, 0) + 1
    overlap = 0
    for t in pred_tokens:
        if r_counts.get(t, 0) > 0:
            overlap += 1
            r_counts[t] -= 1
    return overlap

def unigram_bleu1(pred: str, ref: str) -> float:
    """Simple BLEU-1 (unigram precision) without brevity penalty."""
    p_tokens = pred.strip().split()
    r_tokens = ref.strip().split()
    if not p_tokens:
        return 0.0
    overlap = _overlap_count(p_tokens, r_tokens)
    return overlap / len(p_tokens)

def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L like score based on LCS / len(ref)."""
    a = pred.strip().split()
    b = ref.strip().split()
    if not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la-1, -1, -1):
        for j in range(lb-1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    lcs = dp[0][0]
    return lcs / lb

def precision_recall_f1(pred: str, ref: str) -> Tuple[float, float, float]:
    """Token-overlap Precision/Recall/F1 with clipping (order-insensitive)."""
    p_tokens = pred.strip().split()
    r_tokens = ref.strip().split()
    if not p_tokens and not r_tokens:
        return 1.0, 1.0, 1.0
    if not p_tokens or not r_tokens:
        return 0.0, 0.0, 0.0
    overlap = _overlap_count(p_tokens, r_tokens)
    precision = overlap / len(p_tokens) if p_tokens else 0.0
    recall = overlap / len(r_tokens) if r_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# --------------------------- Federated Client ------------------------------
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, train_data, val_data, device, cid: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data or []
        self.val_data = val_data or []
        if not self.train_data:
            self.train_data = [{"dialog": "Patient: I have a headache. Doctor: Rest and fluids."}]
        if not self.val_data:
            self.val_data = [{"dialog": "Patient: I feel sick. Doctor: See your clinician."}]
        self.device = device
        self.cid = cid
        self._init_optimizer()

    def _init_optimizer(self):
        params = [p for _, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY  # Added weight decay
        )
        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=Config.WARMUP_STEPS,
            num_training_steps=len(self.train_data) * Config.TRAINING_EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
        )

    def _ordered_trainable_params(self) -> List[torch.Tensor]:
        items = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        items.sort(key=lambda x: x[0])
        return [p for _, p in items]

    def get_parameters(self, config: Optional[dict] = None) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self._ordered_trainable_params()]

    def set_parameters(self, nds: List[np.ndarray]):
        params = self._ordered_trainable_params()
        if len(params) != len(nds):
            raise ValueError(f"Shape/count mismatch when setting params: {len(params)} vs {len(nds)}")
        for p, arr in zip(params, nds):
            t = torch.from_numpy(arr).to(self.device, dtype=p.dtype)
            if p.shape != t.shape:
                raise ValueError(f"Shape mismatch {p.shape} vs {t.shape}")
            p.data = t

    def fit(self, parameters: List[np.ndarray], config: Optional[dict]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        before_norm = sum(float(p.data.norm().item()) for p in self._ordered_trainable_params())
        if parameters is not None:
            self.set_parameters(parameters)
        self._init_optimizer()

        logger.debug(f"[Client {self.cid}] training on {len(self.train_data)} examples")
        metrics = self._train_epochs()

        updated = [p.detach().cpu().numpy() for p in self._ordered_trainable_params()]
        after_norm = sum(float(p.data.norm().item()) for p in self._ordered_trainable_params())
        logger.debug(f"[Client {self.cid}] param norm before_set={before_norm:.6f} after_train={after_norm:.6f}")

        safe_metrics = {}
        for k, v in (metrics or {}).items():
            try:
                safe_metrics[k] = float(v)
            except Exception:
                pass
        if "train_loss" not in safe_metrics:
            safe_metrics["train_loss"] = 0.0

        return updated, len(self.train_data), safe_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Optional[dict]) -> Tuple[float, int, Dict[str, float]]:
        if parameters is not None:
            self.set_parameters(parameters)
            self._init_optimizer()
        loss, metrics = self._validate()
        if loss is None or math.isnan(loss) or math.isinf(loss):
            loss = 0.0
        safe_metrics = {}
        for k, v in (metrics or {}).items():
            try:
                safe_metrics[k] = float(v)
            except Exception:
                pass
        if "val_loss" not in safe_metrics:
            safe_metrics["val_loss"] = float(loss)
        return float(loss), max(1, len(self.val_data)), safe_metrics

    def _train_epochs(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        valid_steps = 0
        data = self.train_data[:min(100, len(self.train_data))]

        for epoch in range(Config.TRAINING_EPOCHS):
            self.optimizer.zero_grad()
            for i, ex in enumerate(data):
                try:
                    txt = ex.get("dialog", "")
                    proc = preprocess_dialog(txt)
                    if not proc:
                        continue

                    inputs = self.tokenizer(
                        proc,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=Config.MAX_SEQ_LENGTH
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    low = proc.lower()
                    doctor_start = low.find("doctor:")
                    if doctor_start != -1:
                        prefix_text = proc[:doctor_start]
                        prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                        prefix_len = len(prefix_tokens)
                        labels = inputs["input_ids"].clone()
                        labels[0, :prefix_len] = -100
                    else:
                        labels = inputs["input_ids"]

                    out = self.model(**inputs, labels=labels)
                    loss = out.loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    (loss / Config.GRADIENT_ACCUMULATION_STEPS).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)

                    if (i + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                        self.optimizer.step()
                        self.scheduler.step()  # Update learning rate
                        self.optimizer.zero_grad()

                    total_loss += float(loss.item())
                    valid_steps += 1

                except Exception as e:
                    logger.warning(f"[Client {self.cid}] Train batch {i} error: {e}")
                    continue

        avg_loss = total_loss / max(1, valid_steps)
        return {"train_loss": avg_loss}

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        valid_steps = 0
        bleu_accum, rouge_accum = [], []
        prec_accum, rec_accum, f1_accum = [], [], []
        good_responses = 0  # Track response quality

        data = self.val_data[:min(20, len(self.val_data))]

        with torch.no_grad():
            for i, ex in enumerate(data):
                try:
                    txt = ex.get("dialog", "")
                    proc = preprocess_dialog(txt)
                    if not proc:
                        continue

                    inputs = self.tokenizer(
                        proc,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=Config.MAX_SEQ_LENGTH
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    low = proc.lower()
                    doctor_start = low.find("doctor:")
                    if doctor_start != -1:
                        prefix_text = proc[:doctor_start]
                        prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                        prefix_len = len(prefix_tokens)
                        labels = inputs["input_ids"].clone()
                        labels[0, :prefix_len] = -100
                    else:
                        labels = inputs["input_ids"]

                    out = self.model(**inputs, labels=labels)
                    loss = out.loss.item()
                    if math.isnan(loss) or math.isinf(loss):
                        continue

                    total_loss += loss

                    # Build reference and prediction for metrics
                    if doctor_start != -1:
                        lbl_ids = labels[0].cpu().numpy().tolist()
                        ref_tokens = [t for t in lbl_ids if t != -100 and t != self.tokenizer.pad_token_id]
                        ref_text = self.tokenizer.decode(ref_tokens, skip_special_tokens=True).strip() if ref_tokens else ""
                    else:
                        ref_text = ""

                    # Generate prediction with improved sampling
                    full_gen = generate_with_config(
                        self.model,
                        self.tokenizer,
                        proc,
                        device=self.device,
                        max_new_tokens=Config.GENERATION_MAX_LENGTH,
                        temperature=Config.GENERATION_TEMPERATURE,
                        top_p=Config.GENERATION_TOP_P,
                        top_k=Config.GENERATION_TOP_K,
                        do_sample=True,
                    )
                    pred_text = extract_doctor_response(full_gen)

                    if ref_text:
                        bleu_score = unigram_bleu1(pred_text, ref_text)
                        rouge_score = rouge_l(pred_text, ref_text)
                        p, r, f1 = precision_recall_f1(pred_text, ref_text)
                        bleu_accum.append(bleu_score)
                        rouge_accum.append(rouge_score)
                        prec_accum.append(p)
                        rec_accum.append(r)
                        f1_accum.append(f1)

                        # Track good responses
                        if f1 >= Config.RESPONSE_QUALITY_THRESHOLD:
                            good_responses += 1

                    valid_steps += 1

                except Exception as e:
                    logger.warning(f"[Client {self.cid}] Validation error: {e}")
                    continue

        avg_loss = total_loss / max(1, valid_steps)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        avg_bleu = float(np.mean(bleu_accum)) if bleu_accum else 0.0
        avg_rouge = float(np.mean(rouge_accum)) if rouge_accum else 0.0
        avg_p = float(np.mean(prec_accum)) if prec_accum else 0.0
        avg_r = float(np.mean(rec_accum)) if rec_accum else 0.0
        avg_f1 = float(np.mean(f1_accum)) if f1_accum else 0.0

        # Calculate response quality percentage
        response_quality = good_responses / max(1, len(bleu_accum)) if bleu_accum else 0.0

        logger.info(f"[Client {self.cid}] Validation metrics: "
                   f"Loss={avg_loss:.4f}, PPL={ppl:.2f}, "
                   f"BLEU={avg_bleu:.3f}, ROUGE-L={avg_rouge:.3f}, "
                   f"P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f}, "
                   f"Quality={response_quality:.1%}")

        return avg_loss, {
            "val_loss": avg_loss,
            "perplexity": ppl,
            "bleu": avg_bleu,
            "rouge_l": avg_rouge,
            "precision": avg_p,
            "recall": avg_r,
            "f1": avg_f1,
            "response_quality": response_quality,
        }

# ------------------------------ Server helpers -----------------------------
def _server_set_trainable_params_from_ndarrays(model: torch.nn.Module, nds: List[np.ndarray], device: torch.device):
    items = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    items.sort(key=lambda x: x[0])
    trainables = [p for _, p in items]
    if len(trainables) != len(nds):
        raise ValueError(f"Trainable param count mismatch: {len(trainables)} vs {len(nds)}")
    for p, arr in zip(trainables, nds):
        t = torch.from_numpy(arr).to(device, dtype=p.dtype)
        if p.shape != t.shape:
            raise ValueError(f"Shape mismatch {p.shape} vs {t.shape}")
        p.data = t

@torch.no_grad()
def _server_validate_like_client(model, tokenizer, val_examples: List[dict], device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    valid_examples = 0
    bleu_accum, rouge_accum = [], []
    prec_accum, rec_accum, f1_accum = [], [], []
    good_responses = 0

    data = val_examples[:min(20, len(val_examples))]

    for i, ex in enumerate(data):
        try:
            proc = preprocess_dialog(ex.get("dialog", ""))
            if not proc:
                continue

            inputs = tokenizer(
                proc,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=Config.MAX_SEQ_LENGTH,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            low = proc.lower()
            doctor_start = low.find("doctor:")
            if doctor_start != -1:
                prefix_text = proc[:doctor_start]
                prefix_tokens = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                prefix_len = len(prefix_tokens)
                labels = inputs["input_ids"].clone()
                labels[0, :prefix_len] = -100
            else:
                labels = inputs["input_ids"]

            out = model(**inputs, labels=labels)
            loss = out.loss.item()
            if math.isnan(loss) or math.isinf(loss):
                continue

            total_loss += loss

            # Reference text (masked target)
            if doctor_start != -1:
                lbl_ids = labels[0].cpu().numpy().tolist()
                ref_tokens = [t for t in lbl_ids if t != -100 and t != tokenizer.pad_token_id]
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True).strip() if ref_tokens else ""
            else:
                ref_text = ""

            # Prediction via safe wrapper with improved sampling
            full_gen = generate_with_config(
                model,
                tokenizer,
                proc,
                device=device,
                max_new_tokens=Config.GENERATION_MAX_LENGTH,
                temperature=Config.GENERATION_TEMPERATURE,
                top_p=Config.GENERATION_TOP_P,
                top_k=Config.GENERATION_TOP_K,
                do_sample=True,
            )
            pred_text = extract_doctor_response(full_gen)

            if ref_text:
                bleu_accum.append(unigram_bleu1(pred_text, ref_text))
                rouge_accum.append(rouge_l(pred_text, ref_text))
                p, r, f1 = precision_recall_f1(pred_text, ref_text)
                prec_accum.append(p)
                rec_accum.append(r)
                f1_accum.append(f1)

                # Track good responses
                if f1 >= Config.RESPONSE_QUALITY_THRESHOLD:
                    good_responses += 1

            # Log example predictions for debugging
            if random.random() < 0.2 and ref_text and pred_text:
                logger.info(f"[Server] Example prediction:")
                logger.info(f"  Prompt: {proc[:100]}...")
                logger.info(f"  Predicted: '{pred_text}'")
                logger.info(f"  Reference: '{ref_text}'")
                logger.info(f"  P/R/F1: {p:.3f}/{r:.3f}/{f1:.3f}")

            valid_examples += 1

        except Exception as e:
            logger.warning(f"Server validation error: {e}")
            continue

    avg_loss = total_loss / max(1, valid_examples)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    avg_bleu = float(np.mean(bleu_accum)) if bleu_accum else 0.0
    avg_rouge = float(np.mean(rouge_accum)) if rouge_accum else 0.0
    avg_p = float(np.mean(prec_accum)) if prec_accum else 0.0
    avg_r = float(np.mean(rec_accum)) if rec_accum else 0.0
    avg_f1 = float(np.mean(f1_accum)) if f1_accum else 0.0
    response_quality = good_responses / max(1, len(bleu_accum)) if bleu_accum else 0.0

    logger.info(f"[Server] Validation metrics: "
               f"Loss={avg_loss:.4f}, PPL={ppl:.2f}, "
               f"BLEU={avg_bleu:.3f}, ROUGE-L={avg_rouge:.3f}, "
               f"P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f}, "
               f"Quality={response_quality:.1%}")

    return avg_loss, {
        "val_loss": avg_loss,
        "perplexity": ppl,
        "bleu": avg_bleu,
        "rouge_l": avg_rouge,
        "precision": avg_p,
        "recall": avg_r,
        "f1": avg_f1,
        "response_quality": response_quality,
    }

def server_evaluate(server_round, parameters, config):
    if server_round == 0:
        logger.info("Skipping server eval at round 0 to speed startup.")
        return None

    try:
        nds = parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
    except Exception as e:
        logger.warning(f"server_evaluate: failed to convert parameters: {e}")
        return None

    try:
        _server_set_trainable_params_from_ndarrays(model, nds, device)
    except Exception as e:
        logger.warning(f"Server eval skipped due to param set error: {e}")
        return None

    if not FIXED_EVAL_DATA:
        logger.info("No FIXED_EVAL_DATA available; skipping server eval.")
        return None

    try:
        avg_loss, metrics = _server_validate_like_client(model, gen_tok, FIXED_EVAL_DATA, device)
        logger.info(f"Server eval round {server_round}: loss={avg_loss:.4f} metrics={metrics}")
        return float(avg_loss), metrics
    except Exception as e:
        logger.warning(f"Server eval failed: {e}")
        return None

# ------------------------------ Strategy -----------------------------------
class FedMedStrategy(FedAvg):
    def __init__(self, rag_system):
        super().__init__(
            evaluate_fn=server_evaluate,
            fit_metrics_aggregation_fn=self._agg_fit_metrics,
            evaluate_metrics_aggregation_fn=self._agg_evaluate_metrics,
        )
        self.rag_system = rag_system
        self.history = {
            "rounds": [],
            "train_loss": [],
            "val_loss": [],
            "bleu": [],
            "rouge_l": [],
            "perplexity": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "response_quality": [],
        }
        self.best_loss = float("inf")
        self.best_f1 = 0.0
        self.no_improve_loss = 0
        self.no_improve_f1 = 0
        self._last_fit_train_loss = None

        # Prepare CSV
        self.csv_path = os.path.join(OUTPUT_DIR, "fedmed_history.csv")
        self.png_path = os.path.join(OUTPUT_DIR, "fedmed_metrics.png")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "train_loss", "val_loss", "bleu",
                "rouge_l", "perplexity", "precision", "recall",
                "f1", "response_quality"
            ])

    @staticmethod
    def _agg_fit_metrics(metrics):
        agg = {}
        if not metrics:
            return agg
        keys = set().union(*[set(m.keys()) for m in metrics if isinstance(m, dict)])
        for k in keys:
            vals = [m[k] for m in metrics if isinstance(m, dict) and k in m and isinstance(m[k], (int, float))]
            agg[k] = float(np.mean(vals)) if vals else 0.0
        return agg

    @staticmethod
    def _agg_evaluate_metrics(metrics):
        agg = {}
        if not metrics:
            return agg
        keys = set().union(*[set(m.keys()) for m in metrics if isinstance(m, dict)])
        for k in keys:
            vals = [m[k] for m in metrics if isinstance(m, dict) and k in m and isinstance(m[k], (int, float))]
            agg[k] = float(np.mean(vals)) if vals else 0.0
        return agg

    def aggregate_fit(self, server_round, results, failures):
        metrics = [res.metrics for _, res in results]
        avg_train_loss = float(np.mean([m.get("train_loss", 0.0) for m in metrics if m])) if metrics else 0.0
        self._last_fit_train_loss = avg_train_loss
        return super().aggregate_fit(server_round, results, failures)

    def _save_history_row(self, r: int):
        idx = len(self.history["rounds"]) - 1
        row = [
            r,
            (self.history["train_loss"][idx] if idx < len(self.history["train_loss"]) else ""),
            self.history["val_loss"][idx],
            self.history["bleu"][idx],
            self.history["rouge_l"][idx],
            self.history["perplexity"][idx],
            self.history["precision"][idx],
            self.history["recall"][idx],
            self.history["f1"][idx],
            self.history["response_quality"][idx],
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _plot_history(self):
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Loss plot
            if self.history["train_loss"] and self.history["val_loss"]:
                ax1.plot(self.history["rounds"], self.history["train_loss"], label="Train Loss", marker='o')
                ax1.plot(self.history["rounds"], self.history["val_loss"], label="Val Loss", marker='s')
                ax1.set_xlabel("Round")
                ax1.set_ylabel("Loss")
                ax1.legend()
                ax1.grid(True)
                ax1.set_title("Training and Validation Loss")

            # Metrics plot
            if self.history["f1"] and self.history["response_quality"]:
                ax2.plot(self.history["rounds"], self.history["f1"], label="F1 Score", marker='o', color='green')
                ax2.plot(self.history["rounds"], self.history["response_quality"], label="Response Quality", marker='s', color='purple')
                ax2.set_xlabel("Round")
                ax2.set_ylabel("Score")
                ax2.legend()
                ax2.grid(True)
                ax2.set_title("F1 Score and Response Quality")

            # Precision/Recall plot
            if self.history["precision"] and self.history["recall"]:
                ax3.plot(self.history["rounds"], self.history["precision"], label="Precision", marker='o', color='blue')
                ax3.plot(self.history["rounds"], self.history["recall"], label="Recall", marker='s', color='red')
                ax3.set_xlabel("Round")
                ax3.set_ylabel("Score")
                ax3.legend()
                ax3.grid(True)
                ax3.set_title("Precision and Recall")

            # Perplexity plot
            if self.history["perplexity"]:
                finite_ppl = [p if p != float('inf') else None for p in self.history["perplexity"]]
                ax4.plot(self.history["rounds"], finite_ppl, label="Perplexity", marker='o', color='orange')
                ax4.set_xlabel("Round")
                ax4.set_ylabel("Perplexity")
                ax4.legend()
                ax4.grid(True)
                ax4.set_title("Perplexity")

            plt.tight_layout()
            plt.savefig(self.png_path, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}

        losses, blues, rouges, perplexities = [], [], [], []
        precs, recs, f1s, qualities = [], [], [], []

        for _, res in results:
            m = res.metrics or {}
            l = m.get("val_loss", float(res.loss))
            if math.isnan(l) or math.isinf(l):
                continue
            losses.append(l)
            blues.append(m.get("bleu", 0.0))
            rouges.append(m.get("rouge_l", 0.0))
            perplexities.append(m.get("perplexity", float("inf")))
            precs.append(m.get("precision", 0.0))
            recs.append(m.get("recall", 0.0))
            f1s.append(m.get("f1", 0.0))
            qualities.append(m.get("response_quality", 0.0))

        if not losses:
            return None, {}

        avg_loss = float(np.mean(losses))
        avg_bleu = float(np.mean(blues)) if blues else 0.0
        avg_rouge = float(np.mean(rouges)) if rouges else 0.0
        finite_ppl = [p for p in perplexities if p != float("inf")]
        avg_perplexity = float(np.mean(finite_ppl)) if finite_ppl else float("inf")
        avg_precision = float(np.mean(precs)) if precs else 0.0
        avg_recall = float(np.mean(recs)) if recs else 0.0
        avg_f1 = float(np.mean(f1s)) if f1s else 0.0
        avg_quality = float(np.mean(qualities)) if qualities else 0.0

        # record history
        self.history["rounds"].append(server_round)
        if self._last_fit_train_loss is not None:
            self.history["train_loss"].append(self._last_fit_train_loss)
        self.history["val_loss"].append(avg_loss)
        self.history["bleu"].append(avg_bleu)
        self.history["rouge_l"].append(avg_rouge)
        self.history["perplexity"].append(avg_perplexity)
        self.history["precision"].append(avg_precision)
        self.history["recall"].append(avg_recall)
        self.history["f1"].append(avg_f1)
        self.history["response_quality"].append(avg_quality)

        # save artifacts
        self._save_history_row(server_round)
        self._plot_history()

        # Enhanced early stopping - check both loss and F1
        loss_improved = avg_loss < self.best_loss - Config.MIN_LOSS_IMPROVEMENT
        f1_improved = avg_f1 > self.best_f1 + Config.MIN_LOSS_IMPROVEMENT

        if loss_improved:
            self.best_loss = avg_loss
            self.no_improve_loss = 0
        else:
            self.no_improve_loss += 1

        if f1_improved:
            self.best_f1 = avg_f1
            self.no_improve_f1 = 0
        else:
            self.no_improve_f1 += 1

        # Stop if both loss and F1 haven't improved for patience rounds
        if (self.no_improve_loss >= Config.EARLY_STOPPING_PATIENCE and
            self.no_improve_f1 >= Config.EARLY_STOPPING_PATIENCE):
            logger.info(f"Early stopping at round {server_round} due to no significant improvement in loss or F1")
            return None, {}

        return super().aggregate_evaluate(server_round, results, failures)

# ------------------------------- Client factory ----------------------------
def client_fn(context: fl.common.Context) -> fl.client.Client:
    cid_str = str(context.node_id)
    digits = "".join(ch for ch in cid_str if ch.isdigit())
    cid_int = int(digits) if digits else 0

    # Simple client data selection
    if all_dialogs:
        k = min(Config.MIN_RECORDS_PER_CLIENT, len(all_dialogs))
        selected = random.sample(all_dialogs, k)
    else:
        selected = create_synthetic_medical_data(Config.MIN_RECORDS_PER_CLIENT)

    split = max(1, int(0.8 * len(selected)))
    train = ([{"dialog": preprocess_dialog(ex["dialog"])} for ex in selected[:split]] if split else [])
    val = ([{"dialog": preprocess_dialog(ex["dialog"])} for ex in selected[split:]] if split else [])

    return FederatedClient(model, gen_tok, train, val, device, cid=cid_int).to_client()

# --------------------------- Utilities: save config -------------------------
def save_run_config():
    cfg = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

# --------------------------- Fixed evaluation set ---------------------------
def create_fixed_eval_set(all_dialogs, num_examples=20):
    """Create a fixed evaluation set that doesn't change between rounds"""
    if not all_dialogs:
        return [{"dialog": "Patient: I feel sick. Doctor: See your clinician."}]

    rng = random.Random(42)
    eval_dialogs = all_dialogs.copy()
    rng.shuffle(eval_dialogs)

    fixed_eval = []
    for ex in eval_dialogs[:num_examples]:
        if isinstance(ex.get("dialog"), str):
            processed = preprocess_dialog(ex["dialog"])
            if (len(processed.strip()) >= 10 and
                len(processed) <= Config.MAX_SEQ_LENGTH * 4):
                fixed_eval.append({"dialog": processed})

    if not fixed_eval:
        fixed_eval = [{"dialog": "Patient: I feel sick. Doctor: See your clinician."}]

    logger.info(f"Created fixed evaluation set with {len(fixed_eval)} examples")
    return fixed_eval

# --------------------------- Debug functions ---------------------------
def debug_rag_system(rag_system):
    """Test if RAG system is working properly"""
    logger.info("Testing RAG system...")

    test_queries = [
        "I have a headache",
        "I feel feverish and tired",
        "My stomach hurts after eating",
        "I've been coughing for 3 days"
    ]

    for query in test_queries:
        try:
            response = rag_system.generate(query)
            logger.info(f"RAG test - Query: '{query}'")
            logger.info(f"RAG test - Response: '{response}'")
            logger.info("---")
        except Exception as e:
            logger.error(f"RAG test failed for '{query}': {e}")

    return True

def test_basic_imports():
    """Test if basic imports work"""
    logger.info("Testing basic imports...")
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info("✓ Basic imports work")
        return True
    except Exception as e:
        logger.error(f"Basic imports failed: {e}")
        return False

def test_model_generation(model, tokenizer, device):
    """Test if the model can generate reasonable text"""
    logger.info("Testing model generation...")

    test_prompts = [
        "Patient: I have a headache.",
        "Patient: I feel feverish and tired.",
        "Patient: My stomach hurts."
    ]

    for prompt in test_prompts:
        try:
            response = generate_with_config(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True
            )
            logger.info(f"Prompt: '{prompt}'")
            logger.info(f"Response: '{response}'")
            logger.info("---")
        except Exception as e:
            logger.error(f"Generation test failed: {e}")

    return True

# ----------------------------------- Main ----------------------------------
def main():
    global model, gen_tok, ret_model, ret_tok, all_dialogs, symptoms, texts, rag, device, FIXED_EVAL_DATA

    try:
        logger.info("Starting main function...")

        # Test basic imports first
        if not test_basic_imports():
            return

        # Initialize components with timeout
        logger.info("Initializing components with timeout...")
        result = run_with_timeout(initialize_components, timeout=120)
        if result is None:
            logger.error("Component initialization timed out!")
            return

        model, gen_tok, ret_model, ret_tok, device = result

        # Test basic generation
        logger.info("Testing basic model generation...")
        test_model_generation(model, gen_tok, device)

        # Load data
        logger.info("Loading dialogs...")
        all_dialogs = run_with_timeout(load_dialogs_fixed, timeout=60, default=[])
        if not all_dialogs:
            logger.warning("No dialogs loaded, using synthetic data")
            all_dialogs = create_synthetic_medical_data(Config.MAX_DATASET_SIZE)

        # Get symptoms
        logger.info("Getting symptoms...")
        symptoms = run_with_timeout(get_symptoms_from_sparql, timeout=10,
                                  default=["headache", "fever", "cough"])
        logger.info(f"Found {len(symptoms)} symptoms")

        # RAG corpus
        logger.info("Creating RAG corpus...")
        texts = [
            preprocess_dialog(ex["dialog"]) for ex in all_dialogs
            if isinstance(ex.get("dialog"), str)
            and len(preprocess_dialog(ex["dialog"]).strip()) >= 10
            and len(preprocess_dialog(ex["dialog"])) <= Config.MAX_SEQ_LENGTH * 4
        ][:500]  # Limit corpus size for testing

        logger.info(f"Initialized RAG with {len(texts)} valid corpus texts")

        # Build RAG system
        logger.info("Building RAG system...")
        rag = RAGSystem(model, gen_tok, ret_model, ret_tok, texts, device)

        # Test RAG system
        logger.info("Testing RAG system...")
        debug_rag_system(rag)

        # Fixed evaluation set
        logger.info("Creating fixed evaluation set...")
        FIXED_EVAL_DATA = create_fixed_eval_set(all_dialogs, num_examples=15)
        logger.info(f"Created fixed evaluation set with {len(FIXED_EVAL_DATA)} examples")

        # Save run config
        logger.info("Saving run configuration...")
        save_run_config()

        logger.info("Creating strategy...")
        strategy = FedMedStrategy(rag)

        logger.info("Starting simulation...")
        # start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=Config.NUM_CLIENTS,
            config=ServerConfig(num_rounds=Config.NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )

        # Quick demo
        logger.info("Running final demo...")
        test_questions = [
            "I have a headache and fever",
            "I've been coughing for several days",
            "I feel nauseous after meals"
        ]
        print("\n" + "="*50)
        print("FINAL MEDICAL CHATBOT DEMO")
        print("="*50)
        for q in test_questions:
            print(f"\nPatient: {q}")
            response = rag.generate(q)
            print(f"Doctor: {response}")
            print("-" * 50)

        print(f"\n📊 CSV metrics saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'fedmed_history.csv'))}")
        print(f"📈 Plot saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'fedmed_metrics.png'))}")

    except Exception as e:
        logger.error("Main failed: %s", e)
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":

    main()
