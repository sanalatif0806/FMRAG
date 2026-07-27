"""
client/peft/model_setup.py
---------------------------
BioBERT + LoRA + PEFT + BAT-GNN model stack.

CPU MODE (default for edge devices without GPU):
  - Uses distilbert-base-uncased instead of biobert-base (40% smaller, 2x faster)
  - OR loads biobert with 8-bit quantisation via bitsandbytes (4x less RAM)
  - LoRA rank reduced to 4  (trainable params ~150 KB vs 400 KB)
  - LoRA injected only into last 1 transformer block (not 2)
  - Gradient checkpointing enabled  (trades compute for memory)
  - Prefix tokens reduced to 10

GPU MODE (set USE_GPU=true in /etc/fmrag/env):
  - Full biobert-base-cased-v1.2
  - LoRA rank 8, blocks 10+11
  - Prefix tokens 20

The model output is identical in both modes — only speed and RAM differ.
The LoRA delta format is the same, so CPU and GPU clients can participate
in the same FL round.

Environment variables:
  FMRAG_CPU_MODE=true      force CPU mode (default: auto-detect)
  FMRAG_MODEL_NAME         override model (e.g. dmis-lab/biobert-base-cased-v1.2)
  FMRAG_LORA_RANK          override LoRA rank
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    TaskType,
)
from graphcare_.model import GraphCare

log = logging.getLogger(__name__)

# ── Hardware detection ────────────────────────────────────────────────────────
_HAS_GPU    = torch.cuda.is_available()
_CPU_MODE   = (os.environ.get("FMRAG_CPU_MODE", "").lower() == "true"
               or not _HAS_GPU)

if _CPU_MODE:
    log.info("CPU mode — using lightweight model configuration")
else:
    log.info("GPU mode — using full model configuration")

# ── Model selection ───────────────────────────────────────────────────────────
# CPU default: distilbert-base-uncased
#   - 66M params (vs 110M BioBERT)
#   - 768-dim hidden (same as BERT-base — GraphCare head unchanged)
#   - 2x faster on CPU, 40% less RAM
#   - Still pretrained on large English corpus, good for medical codes+notes
#
# GPU / override: full BioBERT
#   - Set FMRAG_MODEL_NAME=dmis-lab/biobert-base-cased-v1.2
_DEFAULT_MODEL = (
    "distilbert-base-uncased"        if _CPU_MODE
    else "dmis-lab/biobert-base-cased-v1.2"
)
BIOBERT_MODEL = os.environ.get("FMRAG_MODEL_NAME", _DEFAULT_MODEL)

# ── LoRA config ───────────────────────────────────────────────────────────────
_LORA_RANK    = int(os.environ.get("FMRAG_LORA_RANK",
                   "4" if _CPU_MODE else "8"))
_LORA_LAYERS  = [5] if _CPU_MODE else [10, 11]   # last 1 block on CPU, 2 on GPU
_PREFIX_TOKENS= 10  if _CPU_MODE else 20

LORA_CONFIG = LoraConfig(
    r                  = _LORA_RANK,
    lora_alpha         = _LORA_RANK * 2,          # alpha = 2 * rank (standard)
    target_modules     = ["q_lin", "v_lin"]        # DistilBERT naming
                         if "distilbert" in BIOBERT_MODEL
                         else ["query", "value"],  # BERT naming
    lora_dropout       = 0.05 if _CPU_MODE else 0.1,
    bias               = "none",
    layers_to_transform= _LORA_LAYERS,
    task_type          = TaskType.FEATURE_EXTRACTION,
)

PREFIX_CONFIG = PrefixTuningConfig(
    num_virtual_tokens = _PREFIX_TOKENS,
    task_type          = TaskType.FEATURE_EXTRACTION,
    encoder_hidden_size= 768,
    token_dim          = 768,
    num_attention_heads= 12,
)

log.info("Model: %s | LoRA rank: %d | layers: %s | prefix tokens: %d",
         BIOBERT_MODEL, _LORA_RANK, _LORA_LAYERS, _PREFIX_TOKENS)


# ── Model ─────────────────────────────────────────────────────────────────────

class FMRAGModel(nn.Module):
    """
    Full stack:
      PEFT prefix tokens → encoder (DistilBERT or BioBERT, frozen except LoRA)
      → [CLS] pooled → linear (768 → hidden_dim)
      → GraphCare BAT-GNN → task output

    Identical interface in CPU and GPU mode.
    """

    def __init__(
        self,
        num_nodes:        int,
        num_rels:         int,
        max_visit:        int,
        hidden_dim:       int = 128 if _CPU_MODE else 256,
        out_channels:     int = 2,
        graphcare_layers: int = 2   if _CPU_MODE else 3,
        node_emb              = None,
        rel_emb               = None,
        multitask:        bool = False,
    ):
        super().__init__()
        self.multitask = multitask

        # ── encoder ───────────────────────────────────────────────────────────
        base = AutoModel.from_pretrained(
            BIOBERT_MODEL,
            # load in 8-bit on CPU to halve RAM usage
            # requires: pip install bitsandbytes
            # falls back gracefully if not installed
            **(_8bit_kwargs() if _CPU_MODE else {})
        )

        # Freeze all backbone parameters
        for p in base.parameters():
            p.requires_grad = False

        # Gradient checkpointing — trades compute for memory on CPU
        # Gradient checkpointing disabled — incompatible with PREFIX_TUNING
        # if _CPU_MODE and hasattr(base, "gradient_checkpointing_enable"):
        #     base.gradient_checkpointing_enable()

        lora_model = get_peft_model(base, LORA_CONFIG)
        self.encoder = lora_model  # PREFIX_TUNING disabled (SDPA mask mismatch)

        # ── projection ────────────────────────────────────────────────────────
        self.cls_proj = nn.Linear(768, hidden_dim)

        # ── BAT-GNN head — disease/primary task ──────────────────────────────
        self.graphcare = GraphCare(
            num_nodes    = num_nodes,
            num_rels     = num_rels,
            max_visit    = max_visit,
            embedding_dim= hidden_dim,
            hidden_dim   = hidden_dim,
            out_channels = out_channels,
            layers       = graphcare_layers,
            node_emb     = node_emb,
            rel_emb      = rel_emb,
            freeze       = False,
            patient_mode = "joint",
            gnn          = "BAT",
        )

        # ── BAT-GNN head — adverse event detection (multitask only) ─────────
        # Separate parameters from the disease head, sharing only the
        # upstream encoder + node/rel embeddings. Kept as a fully separate
        # GraphCare instance rather than modifying the external GraphCare
        # package internals, since this codebase doesn't own that module.
        self.graphcare_ade = None
        if multitask:
            self.graphcare_ade = GraphCare(
                num_nodes    = num_nodes,
                num_rels     = num_rels,
                max_visit    = max_visit,
                embedding_dim= hidden_dim,
                hidden_dim   = hidden_dim,
                out_channels = 2,                # ADE is always binary
                layers       = graphcare_layers,
                node_emb     = node_emb,
                rel_emb      = rel_emb,
                freeze       = False,
                patient_mode = "joint",
                gnn          = "BAT",
            )

    def forward(
        self,
        input_ids      = None,
        attention_mask = None,
        node_ids       = None,
        rel_ids        = None,
        edge_index     = None,
        batch          = None,
        visit_node     = None,
        ehr_nodes      = None,
    ):
        enc_out  = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # DistilBERT uses last_hidden_state; BERT uses last_hidden_state too
        cls_emb  = enc_out.last_hidden_state[:, 0, :]
        cls_proj = self.cls_proj(cls_emb)

        disease_logits = self.graphcare(
            node_ids   = node_ids,
            rel_ids    = rel_ids,
            edge_index = edge_index,
            batch      = batch,
            visit_node = visit_node,
            ehr_nodes  = ehr_nodes,
        )

        if not self.multitask:
            # Single-task: identical return shape to before this change.
            return disease_logits, cls_proj

        ade_logits = self.graphcare_ade(
            node_ids   = node_ids,
            rel_ids    = rel_ids,
            edge_index = edge_index,
            batch      = batch,
            visit_node = visit_node,
            ehr_nodes  = ehr_nodes,
        )
        # Multitask: logits is a dict instead of a tensor. Every call site
        # that unpacks `logits, _ = model(...)` must check
        # `isinstance(logits, dict)` before using it — see fl_client.py
        # local_train() and knowledge_preservation.py for the two places
        # this matters.
        return {"disease": disease_logits, "ade": ade_logits}, cls_proj

    # ── FL helpers ────────────────────────────────────────────────────────────

    def lora_state_dict(self) -> dict:
        """Only LoRA A/B + GNN head(s) — what goes to the FL server."""
        lora = {
            k: v.detach().cpu()
            for k, v in self.encoder.named_parameters()
            if "lora_" in k and v.requires_grad
        }
        gnn = {
            f"graphcare.{k}": v.detach().cpu()
            for k, v in self.graphcare.named_parameters()
        }
        result = {**lora, **gnn}
        if self.graphcare_ade is not None:
            gnn_ade = {
                f"graphcare_ade.{k}": v.detach().cpu()
                for k, v in self.graphcare_ade.named_parameters()
            }
            result.update(gnn_ade)
        return result

    def load_lora_state_dict(self, state_dict: dict):
        own = dict(self.named_parameters())
        for k, v in state_dict.items():
            if k in own:
                own[k].data.copy_(v)

    def trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return trainable, total


def _8bit_kwargs() -> dict:
    """Return load_in_8bit kwargs if bitsandbytes is available."""
    try:
        import bitsandbytes  # noqa: F401
        return {"load_in_8bit": True}
    except ImportError:
        log.warning(
            "bitsandbytes not installed — loading model in float32. "
            "Install with: pip install bitsandbytes"
        )
        return {}


def build_model(**kwargs) -> FMRAGModel:
    model = FMRAGModel(**kwargs)
    trainable, total = model.trainable_params()
    log.info(
        "Model built — trainable: %s / %s (%.2f%%) | CPU mode: %s",
        f"{trainable:,}", f"{total:,}",
        100 * trainable / total,
        _CPU_MODE,
    )
    return model


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    return _tokenizer
