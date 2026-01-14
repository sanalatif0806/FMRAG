# -*- coding: utf-8 -*-


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




# -*- coding: utf-8 -*-
"""
Federated Med-Dialogue + RAG (Flower NumPyClient) — LoRA + BLEU/ROUGE + PRF1
CLIENT-SIDE RAG IMPLEMENTATION

Updated with client-side RAG implementation:
- Each client has its own RAG system with retriever and generator
- Clients perform retrieval and generation locally during training/evaluation
- Reduced memory footprint with smaller models
"""

import os
import gc
import math
import traceback
import random
import logging
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
)
from peft import get_peft_model, LoraConfig
from peft import TaskType

# Optional huggingface-hub login helper
try:
    from huggingface_hub import login
except Exception:
    login = None

# -------------------------------- Logging ---------------------------------
logging.basicConfig(level=logging.DEBUG, force=True, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("fedmed")
logger.debug("Logging initialized")
CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")
OUTPUT_DIR = os.environ.get("FEDMED_OUT", "./fedmed_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------- Configuration ----------------------------
class Config:
    NUM_ROUNDS = 5  # Reduced for faster testing
    NUM_CLIENTS = 3
    USE_SMALL_MODEL = True
    MAX_SEQ_LENGTH = 256
    GEN_PROMPT_MAX = 256
    RAG_TOP_K = 5  # Reduced for client-side efficiency
    MAX_DATASET_SIZE = 10000  # Reduced for client-side memory constraints
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 3e-4
    GRADIENT_ACCUMULATION_STEPS = 1
    EARLY_STOPPING_PATIENCE = 3
    MIN_LOSS_IMPROVEMENT = 0.01
    TRAINING_EPOCHS = 1  # Reduced for client-side efficiency
    SEED = 42
    GENERATION_MAX_LENGTH = 60  # Reduced for client-side efficiency
    GENERATION_TEMPERATURE = 0.7
    GENERATION_TOP_P = 0.85
    MIN_GENERATED_TOKENS = 3
    MIN_RECORDS_PER_CLIENT = 2000  # Reduced for client-side memory
    CLIENT_RAG_CORPUS_SIZE = 1000  # Maximum RAG corpus size per client

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

logger = logging.getLogger("fedmed")

# ==================== Safe generation helpers =======================
def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_p: float = 0.85,
    do_sample: bool = False,
) -> str:
    """Wrapper around HF .generate that avoids invalid flag warnings and keeps args tidy."""
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
        repetition_penalty=1.05,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = model.generate(**enc, generation_config=gen_cfg)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def extract_doctor_response(full_text: str) -> str:
    """Return the substring after the last 'Doctor:' marker, or the full text if not found."""
    if not full_text:
        return ""
    low = full_text.lower()
    pos = low.rfind("doctor:")
    if pos != -1:
        return full_text[pos + len("doctor:"):].strip()
    return full_text.strip()

# ====================== Client-side RAG System ===========================
class ClientRAGSystem:
    def __init__(self, device: torch.device):
        self.device = device
        self.generator_model = None
        self.generator_tokenizer = None
        self.retriever_model = None
        self.retriever_tokenizer = None
        self.corpus = []
        self.index = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize both generator and retriever models for the client"""
        try:
            # Generator Model (DistilGPT2)
            model_name = "distilgpt2"
            self.generator_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            self.generator_tokenizer.padding_side = "left"

            self.generator_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            # LoRA config for DistilGPT2
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=Config.LORA_R,
                lora_alpha=Config.LORA_ALPHA,
                lora_dropout=Config.LORA_DROPOUT,
                bias="none",
                target_modules=["c_attn", "c_proj", "c_fc"],
            )

            self.generator_model = get_peft_model(self.generator_model, lora_cfg)

            # Freeze base params, only adapters trainable
            for n, p in self.generator_model.named_parameters():
                if "lora_" in n or "peft_" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            self.generator_model.to(self.device)

            # Retriever Model (Smaller biomedical model)
            self.retriever_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1", cache_dir=CACHE_DIR)
            self.retriever_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1", cache_dir=CACHE_DIR).to(self.device)
            self.retriever_model.eval()

            logger.info("Client RAG system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize client RAG system: {e}")
            raise

    def set_corpus(self, corpus: List[str]):
        """Set the RAG corpus for this client and build the FAISS index"""
        self.corpus = corpus[:Config.CLIENT_RAG_CORPUS_SIZE]
        self._build_index(self.corpus)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the retriever model"""
        if not texts or not self.retriever_model:
            dim = getattr(self.retriever_model.config, "hidden_size", 768) if self.retriever_model else 768
            return np.zeros((0, dim), dtype=np.float32)

        batch = 4  # Smaller batch size for client devices
        embs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            inputs = self.retriever_tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=Config.MAX_SEQ_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.retriever_model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embs.append(emb)
        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)

    def _build_index(self, texts: List[str]):
        """Build FAISS index for the corpus"""
        texts = (texts or [])[:Config.CLIENT_RAG_CORPUS_SIZE]
        if not texts:
            self.index = None
            return

        E = self._embed_texts(texts)
        if E.size == 0:
            self.index = None
            return

        dim = E.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(E.astype(np.float32))
        logger.info(f"Built FAISS index with {len(texts)} documents")

    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
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
        """Truncate prompt to maximum length"""
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
        """Generate response using RAG - client-side implementation"""
        if not self.generator_model or not self.retriever_model:
            return "System not initialized properly."

        try:
            # Retrieve relevant context
            q_emb = self._embed_texts([query])
            if q_emb.shape[0] == 0 or self.index is None or self.index.ntotal == 0:
                context = ""
            else:
                k = min(top_k, self.index.ntotal)
                _, idxs = self.index.search(q_emb.astype(np.float32), k)
                docs = [self.corpus[i] for i in idxs[0] if i < len(self.corpus)]
                context = "\n\n".join([f"- {d}" for d in docs])

            # Create prompt with context
            prompt = (
                "You are a medical professional. Use the relevant context to provide a short, general, non-diagnostic reply.\n"
                "If safety is a concern, recommend seeking professional care.\n\n"
                f"Context:\n{context}\n\n"
                f"Patient: {query}\n"
                "Doctor:"
            )
            prompt = self._truncate_prompt(prompt)

            # Generate response
            full = generate_with_config(
                self.generator_model,
                self.generator_tokenizer,
                prompt,
                device=self.device,
                max_new_tokens=Config.GENERATION_MAX_LENGTH,
                temperature=Config.GENERATION_TEMPERATURE,
                top_p=Config.GENERATION_TOP_P,
                do_sample=True,
            )

            # Extract just the doctor's answer
            resp = extract_doctor_response(full)
            if not resp or len(self.generator_tokenizer(resp)["input_ids"]) < Config.MIN_GENERATED_TOKENS:
                return "Please consult a healthcare professional."

            return self._clean_response(resp)

        except Exception as e:
            logger.error(f"Client RAG generation failed: {e}")
            return "Please consult a healthcare professional."

    def get_trainable_parameters(self) -> List[np.ndarray]:
        """Get trainable parameters (LoRA adapters only)"""
        if not self.generator_model:
            return []

        items = [(n, p) for n, p in self.generator_model.named_parameters() if p.requires_grad]
        items.sort(key=lambda x: x[0])
        return [p.detach().cpu().numpy() for _, p in items]

    def set_trainable_parameters(self, nds: List[np.ndarray]):
        """Set trainable parameters (LoRA adapters only)"""
        if not self.generator_model:
            return

        items = [(n, p) for n, p in self.generator_model.named_parameters() if p.requires_grad]
        items.sort(key=lambda x: x[0])
        trainables = [p for _, p in items]

        if len(trainables) != len(nds):
            raise ValueError(f"Parameter count mismatch: {len(trainables)} vs {len(nds)}")

        for p, arr in zip(trainables, nds):
            t = torch.from_numpy(arr).to(self.device, dtype=p.dtype)
            if p.shape != t.shape:
                raise ValueError(f"Shape mismatch {p.shape} vs {t.shape}")
            p.data = t

# --------------------------- Data loading & helpers ------------------------
def get_symptoms_from_sparql():
    """Get medical symptoms from SPARQL endpoints"""
    def run_query(endpoint, q):
        try:
            s = SPARQLWrapper(endpoint)
            s.setQuery(q)
            s.setReturnFormat(JSON)
            res = s.query().convert()
            return [r["symptomLabel"]["value"] for r in res["results"]["bindings"]]
        except Exception:
            return []

    ontobee_q = """
    PREFIX obo: <http://purl.obolibrary.org/obo/>
    SELECT DISTINCT ?symptomLabel WHERE {
        ?symptom a obo:SYMP_0000001 .
        ?symptom rdfs:label ?symptomLabel .
        FILTER (lang(?symptomLabel) = 'en')
    } LIMIT 50
    """
    wikidata_q = """
    SELECT DISTINCT ?symptomLabel WHERE {
      ?symptom wdt:P31 wd:Q169872 .
      ?symptom rdfs:label ?symptomLabel .
      FILTER (lang(?symptomLabel) = "en")
    } LIMIT 50
    """
    syms = list(set(run_query("https://sparql.hegroup.org/sparql", ontobee_q) +
                    run_query("https://query.wikidata.org/sparql", wikidata_q)))
    if not syms:
        syms = ["headache", "nausea", "fever", "cough", "fatigue", "dizziness", "sore throat"]
    return syms

def preprocess_dialog(dialog: str) -> str:
    """Standardize dialog format, handle missing markers."""
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
        "nausea", "dizziness", "shortness of breath", "fatigue", "joint pain",
        "sore throat", "rash", "back pain", "vomiting", "diarrhea"
    ]

    symptoms = {
        "headache": ["throbbing pain", "sensitivity to light", "sensitivity to sound"],
        "fever": ["high temperature", "chills", "sweating"],
        "cough": ["dry cough", "chesty cough", "persistent coughing"],
        "chest pain": ["sharp pain", "dull ache", "pressure in chest"],
    }

    treatments = {
        "headache": "Take over-the-counter pain relief and rest in a quiet, dark room.",
        "fever": "Stay hydrated, take acetaminophen or ibuprofen, and rest.",
        "cough": "Drink plenty of fluids, use cough drops, and consider cough medicine.",
        "chest pain": "Seek immediate medical attention as this could be serious.",
    }

    for i in range(num_samples):
        condition = random.choice(medical_conditions)
        symptom_desc = random.choice(symptoms.get(condition, ["symptoms"]))
        duration = random.choice(["a few hours", "a day", "several days", "a week"])
        severity = random.choice(["mild", "moderate", "severe"])

        patient_msg = f"I've been experiencing {condition} with {symptom_desc} for {duration}. It's {severity}."
        doctor_msg = treatments.get(condition, "Please consult with a healthcare professional for proper diagnosis.")

        dialog = f"Patient: {patient_msg} Doctor: {doctor_msg}"
        synthetic_samples.append({"dialog": dialog})

    return synthetic_samples

def load_dialogs(max_samples: int = Config.MAX_DATASET_SIZE):
    """Load multiple medical dialogue datasets."""
    all_dialogs = []

    # List of working medical dialogue datasets
    dataset_configs = [
        {"name": "UCSD26/medical_dialog", "split": "train", "text_key": "dialogue", "is_medical": True},
        {"name": "bigbio/meddialog", "split": "train", "text_key": "dialogue", "is_medical": True},
        {"name": "petkopetkov/MedDialog", "split": "train", "text_key": "utterances"},
    ]

    # Try the primary medical datasets first
    for config in dataset_configs:
        try:
            logger.info(f"Loading dataset: {config['name']}")
            ds = load_dataset(config["name"], cache_dir=CACHE_DIR)
            split_data = ds[config["split"]] if config["split"] in ds else ds["train"]

            samples = list(split_data)
            loaded_count = 0
            for sample in samples:
                try:
                    if config["text_key"] in sample:
                        text = sample[config["text_key"]]
                        if isinstance(text, list):
                            text = " ".join([str(t) for t in text])

                        dialog_text = preprocess_dialog(str(text))
                        if (isinstance(dialog_text, str) and
                            len(dialog_text.strip()) >= 10 and
                            len(dialog_text) <= Config.MAX_SEQ_LENGTH * 4):
                            all_dialogs.append({"dialog": dialog_text})
                            loaded_count += 1
                except Exception as e:
                    logger.debug(f"Error processing sample from {config['name']}: {e}")
                    continue

            logger.info(f"Loaded {loaded_count} samples from {config['name']}")

            if len(all_dialogs) >= max_samples:
                all_dialogs = all_dialogs[:max_samples]
                break

        except Exception as e:
            logger.warning(f"Failed to load dataset {config['name']}: {e}")
            continue

    # If we still don't have enough data, create synthetic medical dialogues
    if len(all_dialogs) < 2000:
        logger.warning(f"Only {len(all_dialogs)} dialogs loaded, adding synthetic data")
        synthetic_count = 2000 - len(all_dialogs)
        synthetic_data = create_synthetic_medical_data(synthetic_count)
        all_dialogs.extend(synthetic_data)
        logger.info(f"Added {len(synthetic_data)} synthetic samples")

    # Ensure we have at least minimum records per client
    min_required = Config.NUM_CLIENTS * Config.MIN_RECORDS_PER_CLIENT
    logger.info(f"Minimum required: {min_required}, Current: {len(all_dialogs)}")

    if len(all_dialogs) < min_required:
        repeat_factor = min_required // len(all_dialogs) + 1
        original_count = len(all_dialogs)
        all_dialogs = all_dialogs * repeat_factor
        all_dialogs = all_dialogs[:min_required]
        logger.info(f"Extended dataset from {original_count} to {len(all_dialogs)} samples via repetition")

    logger.info(f"Final total dialogs loaded: {len(all_dialogs)}")
    return all_dialogs

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
    overlap = _overlap_count(p_tokens, r_tokens)  # Fixed: use p_tokens and r_tokens
    precision = overlap / len(p_tokens) if p_tokens else 0.0
    recall = overlap / len(r_tokens) if r_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# --------------------------- Federated Client with RAG ------------------------------
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, train_data, val_data, device, cid: Optional[int] = None):
        self.device = device
        self.cid = cid

        # Initialize client-side RAG system
        self.rag_system = ClientRAGSystem(device)

        # Set training and validation data
        self.train_data = train_data or []
        self.val_data = val_data or []

        if not self.train_data:
            self.train_data = [{"dialog": "Patient: I have a headache. Doctor: Rest and fluids."}]
        if not self.val_data:
            self.val_data = [{"dialog": "Patient: I feel sick. Doctor: See your clinician."}]

        # Create RAG corpus from training data
        rag_corpus = [preprocess_dialog(ex.get("dialog", "")) for ex in self.train_data]
        rag_corpus = [text for text in rag_corpus if text and len(text) > 10]
        self.rag_system.set_corpus(rag_corpus)

        self._init_optimizer()

    def _init_optimizer(self):
        """Initialize optimizer for the client's RAG generator"""
        if self.rag_system.generator_model:
            params = [p for p in self.rag_system.generator_model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(params, lr=Config.LEARNING_RATE)
        else:
            self.optimizer = None

    # NumPyClient API -----------------------------------------------------
    def get_parameters(self, config: Optional[dict] = None) -> List[np.ndarray]:
        return self.rag_system.get_trainable_parameters()

    def set_parameters(self, nds: List[np.ndarray]):
        self.rag_system.set_trainable_parameters(nds)

    def fit(self, parameters: List[np.ndarray], config: Optional[dict]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        if parameters is not None:
            self.set_parameters(parameters)
        self._init_optimizer()

        logger.debug(f"[Client {self.cid}] training on {len(self.train_data)} examples")
        metrics = self._train_epochs()

        updated = self.rag_system.get_trainable_parameters()
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

    # ---------------- Training / validation helpers ---------------------
    def _train_epochs(self) -> Dict[str, float]:
        if not self.rag_system.generator_model or not self.optimizer:
            return {"train_loss": 0.0}

        self.rag_system.generator_model.train()
        total_loss = 0.0
        valid_steps = 0
        data = self.train_data[:Config.MAX_DATASET_SIZE]

        for epoch in range(Config.TRAINING_EPOCHS):
            self.optimizer.zero_grad()
            for i, ex in enumerate(data):
                try:
                    txt = ex.get("dialog", "")
                    proc = preprocess_dialog(txt)
                    if not proc:
                        continue

                    inputs = self.rag_system.generator_tokenizer(
                        proc,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=Config.MAX_SEQ_LENGTH
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Mask everything before first "Doctor:" so we train on responses
                    low = proc.lower()
                    doctor_start = low.find("doctor:")
                    if doctor_start != -1:
                        prefix_text = proc[:doctor_start]
                        prefix_tokens = self.rag_system.generator_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                        prefix_len = len(prefix_tokens)
                        labels = inputs["input_ids"].clone()
                        labels[0, :prefix_len] = -100
                    else:
                        labels = inputs["input_ids"]

                    out = self.rag_system.generator_model(**inputs, labels=labels)
                    loss = out.loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    (loss / Config.GRADIENT_ACCUMULATION_STEPS).backward()
                    torch.nn.utils.clip_grad_norm_(self.rag_system.generator_model.parameters(), 2.0)

                    if (i + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    total_loss += float(loss.item())
                    valid_steps += 1

                except Exception as e:
                    logger.warning(f"[Client {self.cid}] Train batch {i} error: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        avg_loss = total_loss / max(1, valid_steps)
        return {"train_loss": avg_loss}

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        if not self.rag_system.generator_model:
            return 0.0, {}

        self.rag_system.generator_model.eval()
        total_loss = 0.0
        valid_steps = 0
        bleu_accum, rouge_accum = [], []
        prec_accum, rec_accum, f1_accum = [], [], []
        data = self.val_data[:Config.MAX_DATASET_SIZE]

        with torch.no_grad():
            for i, ex in enumerate(data):
                try:
                    txt = ex.get("dialog", "")
                    proc = preprocess_dialog(txt)
                    if not proc:
                        continue

                    inputs = self.rag_system.generator_tokenizer(
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
                        prefix_tokens = self.rag_system.generator_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                        prefix_len = len(prefix_tokens)
                        labels = inputs["input_ids"].clone()
                        labels[0, :prefix_len] = -100
                    else:
                        labels = inputs["input_ids"]

                    out = self.rag_system.generator_model(**inputs, labels=labels)
                    loss = out.loss.item()
                    if math.isnan(loss) or math.isinf(loss):
                        continue

                    total_loss += loss

                    # Build reference (target text) and model prediction for the masked section
                    if doctor_start != -1:
                        lbl_ids = labels[0].cpu().numpy().tolist()
                        ref_tokens = [t for t in lbl_ids if t != -100 and t != self.rag_system.generator_tokenizer.pad_token_id]
                        ref_text = self.rag_system.generator_tokenizer.decode(ref_tokens, skip_special_tokens=True).strip() if ref_tokens else ""
                    else:
                        ref_text = ""

                    # Generate prediction using client-side RAG
                    pred_text = self.rag_system.generate(proc)

                    if ref_text:
                        bleu_score = unigram_bleu1(pred_text, ref_text)
                        rouge_score = rouge_l(pred_text, ref_text)
                        p, r, f1 = precision_recall_f1(pred_text, ref_text)
                        bleu_accum.append(bleu_score)
                        rouge_accum.append(rouge_score)
                        prec_accum.append(p)
                        rec_accum.append(r)
                        f1_accum.append(f1)

                    # Log example predictions for debugging (10% of examples)
                    if random.random() < 0.1 and ref_text and pred_text:
                        logger.info(f"[Client {self.cid}] Example prediction:")
                        logger.info(f"  Prompt: {proc[:100]}...")
                        logger.info(f"  Predicted: '{pred_text}'")
                        logger.info(f"  Reference: '{ref_text}'")
                        logger.info(f"  P/R/F1: {p:.3f}/{r:.3f}/{f1:.3f}")

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

        logger.info(f"[Client {self.cid}] Validation metrics: "
                   f"Loss={avg_loss:.4f}, PPL={ppl:.2f}, "
                   f"BLEU={avg_bleu:.3f}, ROUGE-L={avg_rouge:.3f}, "
                   f"P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f}")

        return avg_loss, {
            "val_loss": avg_loss,
            "perplexity": ppl,
            "bleu": avg_bleu,
            "rouge_l": avg_rouge,
            "precision": avg_p,
            "recall": avg_r,
            "f1": avg_f1,
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

# ------------------------------ Strategy -----------------------------------
class FedMedStrategy(FedAvg):
    def __init__(self):
        super().__init__(
            fit_metrics_aggregation_fn=self._agg_fit_metrics,
            evaluate_metrics_aggregation_fn=self._agg_evaluate_metrics,
        )
        self.history = {
    "rounds": [],
    "train_loss": [],
    "val_loss": [],
    "bleu": [],  # Fixed: removed extra "bleu"
    "rouge_l": [],
    "perplexity": [],
    "precision": [],
    "recall": [],
    "f1": [],
}
        self.best_loss = float("inf")
        self.no_improve = 0
        self._last_fit_train_loss = None

        # Prepare CSV
        self.csv_path = os.path.join(OUTPUT_DIR, "fedmed_history.csv")
        self.png_path = os.path.join(OUTPUT_DIR, "fedmed_metrics.png")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round",
                "train_loss",
                "val_loss",
                "bleu",
                "rouge_l",
                "perplexity",
                "precision",
                "recall",
                "f1",
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
        # capture average training loss of the round
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
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _plot_history(self):
     try:
        plt.figure(figsize=(11, 7))
        if self.history["train_loss"]:
            plt.plot(self.history["rounds"], self.history["train_loss"], label="train_loss")
        if self.history["val_loss"]:
            plt.plot(self.history["rounds"], self.history["val_loss"], label="val_loss")
        if self.history["perplexity"]:
            plt.plot(self.history["rounds"], self.history["perplexity"], label="perplexity")
        if self.history["bleu"]:
            plt.plot(self.history["rounds"], self.history["bleu"], label="bleu")  # Fixed: removed extra "bleu"
        if self.history["rouge_l"]:
            plt.plot(self.history["rounds"], self.history["rouge_l"], label="rouge_l")
        if self.history["precision"]:
            plt.plot(self.history["rounds"], self.history["precision"], label="precision")
        if self.history["recall"]:
            plt.plot(self.history["rounds"], self.history["recall"], label="recall")
        if self.history["f1"]:
            plt.plot(self.history["rounds"], self.history["f1"], label="f1")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(self.png_path, dpi=150)
        plt.show()
     except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}

        losses, blues, rouges, perplexities = [], [], [], []
        precs, recs, f1s = [], [], []
        for _, res in results:
            m = res.metrics or {}
            l = m.get("val_loss", float(res.loss))
            if math.isnan(l) or math.isinf(l):
                logger.warning(f"NaN/Inf loss in client eval, round {server_round}")
                continue
            losses.append(l)
            blues.append(m.get("bleu", 0.0))
            rouges.append(m.get("rouge_l", 0.0))
            perplexities.append(m.get("perplexity", float("inf")))
            precs.append(m.get("precision", 0.0))
            recs.append(m.get("recall", 0.0))
            f1s.append(m.get("f1", 0.0))

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

        # save artifacts
        self._save_history_row(server_round)
        self._plot_history()

        # Early stopping on loss
        if avg_loss < self.best_loss - Config.MIN_LOSS_IMPROVEMENT:
            self.best_loss = avg_loss
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at round {server_round} due to no significant loss improvement")
            return None, {}

        return super().aggregate_evaluate(server_round, results, failures)

# ------------------------------- Client factory ----------------------------
def client_fn(context: fl.common.Context) -> fl.client.Client:
    cid_str = str(context.node_id)
    digits = "".join(ch for ch in cid_str if ch.isdigit())
    cid_int = int(digits) if digits else 0

    symptom = symptoms[cid_int % max(1, len(symptoms))] if symptoms else "headache"

    # Select data based on symptom (simple keyword matching for client-side)
    selected = []
    for ex in all_dialogs:
        if symptom.lower() in ex.get("dialog", "").lower():
            selected.append(ex)

    # If not enough symptom-specific data, add random samples
    if len(selected) < 20 and all_dialogs:
        k_extra = min(20 - len(selected), len(all_dialogs))
        selected.extend(random.sample(all_dialogs, k_extra))

    split = max(1, int(0.8 * len(selected))) if selected else 0
    train = ([{"dialog": preprocess_dialog(ex["dialog"])} for ex in selected[:split]] if split else [])
    val = ([{"dialog": preprocess_dialog(ex["dialog"])} for ex in selected[split:]] if split else [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FederatedClient(train, val, device, cid=cid_int).to_client()

# --------------------------- Utilities: save config -------------------------
def save_run_config():
    cfg = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

# --------------------------- Fixed evaluation set ---------------------------
def create_fixed_eval_set(all_dialogs, num_examples=50):
    """Create a fixed evaluation set that doesn't change between rounds"""
    if not all_dialogs:
        return [{"dialog": "Patient: I feel sick. Doctor: See your clinician."}]

    # Use a fixed random seed for reproducibility
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

# ----------------------------------- Main ----------------------------------
def main():
    global all_dialogs, symptoms, device, FIXED_EVAL_DATA

    try:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        all_dialogs = load_dialogs()
        symptoms = get_symptoms_from_sparql()

        # Fixed evaluation set (consistent across rounds)
        FIXED_EVAL_DATA = create_fixed_eval_set(all_dialogs, num_examples=50)
        logger.info(f"Created fixed evaluation set with {len(FIXED_EVAL_DATA)} examples")

        # Save run config
        save_run_config()

        strategy = FedMedStrategy()

        # start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=Config.NUM_CLIENTS,
            config=ServerConfig(num_rounds=Config.NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.5 if torch.cuda.is_available() else 0},
        )

        # Demo with a sample client RAG system
        print("\n=== Medical Chatbot Demo (Client-side RAG) ===")
        demo_client = FederatedClient([], [], device, cid=999)

        test_questions = [
            "I've been having headaches",
            "I feel feverish",
            "What should I do for a sore throat?",
        ]

        for q in test_questions:
            print(f"\nPatient: {q}")
            response = demo_client.rag_system.generate(q)
            print(f"Doctor: {response}")

        print(f"\nCSV metrics saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'fedmed_history.csv'))}")
        print(f"Plot saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'fedmed_metrics.png'))}")

    except Exception as e:
        logger.error("Main failed: %s", e)
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()

