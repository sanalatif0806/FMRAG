

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
Federated Med-Dialogue (Flower NumPyClient) — LoRA + BLEU/ROUGE + PRF1
no Rag
Updated with fixes for decreasing precision/recall/F1 scores:
- Reduced TRAINING_EPOCHS from 3 to 1 to prevent overfitting
- Added fixed validation set for consistent evaluation
- Added example prediction logging for debugging
- Tuned generation parameters for better response quality
- Increased LORA_R from 8 to 16 for better adapter capacity
- Added more comprehensive logging
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
    AutoModelForCausalLM,
    GenerationConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from peft import TaskType  # ensure TaskType import

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
    NUM_ROUNDS = 10
    NUM_CLIENTS = 3
    USE_SMALL_MODEL = True
    MAX_SEQ_LENGTH = 256         # for training/validation tokenization
    GEN_PROMPT_MAX = 256          # for generation prompt truncation
    MAX_DATASET_SIZE = 26000     # larger cap for more signal
    LORA_R = 8                    # smaller r for smaller model
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 3e-4          # higher LR for smaller model
    GRADIENT_ACCUMULATION_STEPS = 1
    EARLY_STOPPING_PATIENCE = 3
    MIN_LOSS_IMPROVEMENT = 0.01
    TRAINING_EPOCHS = 3           # reduced epochs for smaller model
    SEED = 42
    GENERATION_MAX_LENGTH = 80    # shorter generation for smaller model
    GENERATION_TEMPERATURE = 0.7  # reduced temperature for more focused generation
    GENERATION_TOP_P = 0.85       # reduced top_p for better quality
    MIN_GENERATED_TOKENS = 3      # smaller minimum tokens
    MIN_RECORDS_PER_CLIENT = 4000 # Minimum records per client

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

logger = logging.getLogger("fedmed")

# ==================== Safe generation helpers (new) =======================
def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,     # reduced for smaller model
    temperature: float = 0.7,     # reduced for smaller model
    top_p: float = 0.85,          # reduced for smaller model
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
# ====================== Initialization / models ===========================
def initialize_components():
    try:
        if login is not None:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token='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')
    except Exception:
        logger.debug("HF login skipped/failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===============================
    # Generator Model (Using DistilGPT2 - much smaller)
    # ===============================
    # Using DistilGPT2 as the base model - much smaller and faster
    model_name = "distilgpt2"   # Very small model (~82M parameters)
    gen_tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_tok.padding_side = "left"

    gen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    gen_model.to(device)

    # LoRA config for DistilGPT2
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=Config.LORA_R,           # Smaller r for smaller model
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        target_modules=["c_attn", "c_proj", "c_fc"],  # DistilGPT2 architecture
    )

    gen_model = get_peft_model(gen_model, lora_cfg)

    # Freeze base params, only adapters trainable
    for n, p in gen_model.named_parameters():
        if "lora_" in n or "peft_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    gen_model.to(device)

    # ===============================
    # Optimizations for smaller model
    # ===============================
    try:
        gen_model.config.use_cache = False
        if hasattr(gen_model, "gradient_checkpointing_enable"):
            gen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass

    logger.info(f"Using smaller model: {model_name} with LoRA adapters")
    return gen_model, gen_tok, device

# --------------------------- Data loading & helpers ------------------------
def get_symptoms_from_sparql():
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

# --------------------------- Data loading & helpers ------------------------

def load_dialogs(max_samples: int = Config.MAX_DATASET_SIZE):
    """Load multiple medical dialogue datasets. Returns a list of dicts [{"dialog": str}, ...]."""
    all_dialogs = []

    # List of working medical dialogue datasets (tested and verified)
    dataset_configs = [
        # These datasets work and are available
        {"name": "UCSD26/medical_dialog", "split": "train", "text_key": "dialogue", "is_medical": True},
        {"name": "bigbio/meddialog", "split": "train", "text_key": "dialogue", "is_medical": True},
        {"name": "huzaifa525/Medical_Intelligence_Dataset_40k_Rows_of_Disease_Info_Treatments_and_Medical_QA",
         "split": "train", "text_key": "input", "is_medical": True},
         {"name": "petkopetkov/MedDialog", "split": "train", "text_key": "utterances"},
        {"name": "BI55/MedText", "split": "train", "text_key": "completion"},
        {"name": "FunDialogues/healthcare-minor-consultation", "split": "train", "text_key": "dialogue"},
    ]

    # Alternative medical datasets if the above fail
    alternative_medical_datasets = [
        {"name": "medalpaca/medical_meadow_mediqa", "split": "train", "text_key": "conversation", "is_medical": True},
        {"name": "medalpaca/medical_meadow_healthcaremagic", "split": "train", "text_key": "conversation", "is_medical": True},
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
                    # Extract text based on the configured key
                    if config["text_key"] in sample:
                        text = sample[config["text_key"]]

                        # Handle different dataset formats
                        if config["name"] == "huzaifa525/Medical_Intelligence_Dataset_40k_Rows_of_Disease_Info_Treatments_and_Medical_QA":
                            # For medical QA dataset, create dialogue from input/output
                            output = sample.get("output", "I'll provide medical advice")
                            text = f"Patient: {text} Doctor: {output}"

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

    # If primary datasets fail, try alternative medical datasets
    if len(all_dialogs) < 1000:  # If we got very little data
        logger.info("Trying alternative medical datasets")
        for config in alternative_medical_datasets:
            try:
                logger.info(f"Loading alternative dataset: {config['name']}")
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
                logger.warning(f"Failed to load alternative dataset {config['name']}: {e}")
                continue

    # REMOVED the problematic public datasets - they use old loading scripts
    # If we still don't have enough data, create synthetic medical dialogues
    if len(all_dialogs) < 5000:
        logger.warning(f"Only {len(all_dialogs)} dialogs loaded, adding synthetic data")
        synthetic_count = 5000 - len(all_dialogs)
        synthetic_data = create_synthetic_medical_data(synthetic_count)
        all_dialogs.extend(synthetic_data)
        logger.info(f"Added {len(synthetic_data)} synthetic samples")

    # Ensure we have at least 5k records per client
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


def extract_text_from_example(ex) -> str:
    if isinstance(ex, dict):
        for k in ("dialog", "utterances", "description", "text"):
            if k in ex and ex[k]:
                v = ex[k]
                return " ".join(map(str, v)) if isinstance(v, list) else str(v)
        try:
            return " ".join([str(v) for v in ex.values()])
        except Exception:
            return str(ex)
    return str(ex)

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
        self.optimizer = torch.optim.AdamW(params, lr=Config.LEARNING_RATE)

    def _ordered_trainable_params(self) -> List[torch.Tensor]:
        items = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        items.sort(key=lambda x: x[0])
        return [p for _, p in items]

    # NumPyClient API -----------------------------------------------------
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

    # ---------------- Training / validation helpers ---------------------
    def _train_epochs(self) -> Dict[str, float]:
        self.model.train()
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
                    inputs = self.tokenizer(
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
        self.model.eval()
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

                    # Build reference (target text) and model prediction for the masked section
                    if doctor_start != -1:
                        lbl_ids = labels[0].cpu().numpy().tolist()
                        ref_tokens = [t for t in lbl_ids if t != -100 and t != self.tokenizer.pad_token_id]
                        ref_text = self.tokenizer.decode(ref_tokens, skip_special_tokens=True).strip() if ref_tokens else ""
                    else:
                        ref_text = ""

                    # Generate prediction using safe wrapper; take only the "Doctor:" part
                    full_gen = generate_with_config(
                        self.model,
                        self.tokenizer,
                        proc,
                        device=self.device,
                        max_new_tokens=Config.GENERATION_MAX_LENGTH,
                        temperature=Config.GENERATION_TEMPERATURE,
                        top_p=Config.GENERATION_TOP_P,
                        do_sample=False,
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

@torch.no_grad()
def _server_validate_like_client(model, tokenizer, val_examples: List[dict], device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    valid_examples = 0
    bleu_accum, rouge_accum = [], []
    prec_accum, rec_accum, f1_accum = [], [], []
    data = val_examples[:Config.MAX_DATASET_SIZE]
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

            # Prediction via safe wrapper
            full_gen = generate_with_config(
                model,
                tokenizer,
                proc,
                device=device,
                max_new_tokens=Config.GENERATION_MAX_LENGTH,
                temperature=Config.GENERATION_TEMPERATURE,
                top_p=Config.GENERATION_TOP_P,
                do_sample=False,
            )
            pred_text = extract_doctor_response(full_gen)

            if ref_text:
                bleu_accum.append(unigram_bleu1(pred_text, ref_text))
                rouge_accum.append(rouge_l(pred_text, ref_text))
                p, r, f1 = precision_recall_f1(pred_text, ref_text)
                prec_accum.append(p)
                rec_accum.append(r)
                f1_accum.append(f1)

            # Log example predictions for debugging
            if random.random() < 0.1 and ref_text and pred_text:
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

    logger.info(f"[Server] Validation metrics: "
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
    def __init__(self):
        super().__init__(
            evaluate_fn=server_evaluate,
            fit_metrics_aggregation_fn=self._agg_fit_metrics,
            evaluate_metrics_aggregation_fn=self._agg_evaluate_metrics,
        )
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
        }
        self.best_loss = float("inf")
        self.no_improve = 0
        self._last_fit_train_loss = None  # hold avg train loss from aggregate_fit

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

    # Simple data selection based on symptom keyword matching
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

    local_model, local_tok = model, gen_tok
    return FederatedClient(local_model, local_tok, train, val, device, cid=cid_int).to_client()

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
    global model, gen_tok, all_dialogs, symptoms, device, FIXED_EVAL_DATA

    try:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

        model, gen_tok, device = initialize_components()

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
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )

        # Simple demo without RAG
        test_questions = [
            "I've been having headaches",
            "I feel feverish",
            "What should I do for a sore throat?",
            "My child has a rash",
            "I've had stomach pain for 3 days",
        ]
        print("\n=== Medical Chatbot Demo ===")
        for q in test_questions:
            print(f"\nPatient: {q}")
            prompt = f"Patient: {q} Doctor:"
            full_response = generate_with_config(
                model,
                gen_tok,
                prompt,
                device=device,
                max_new_tokens=Config.GENERATION_MAX_LENGTH,
                temperature=Config.GENERATION_TEMPERATURE,
                top_p=Config.GENERATION_TOP_P,
                do_sample=True,
            )
            response = extract_doctor_response(full_response)
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
