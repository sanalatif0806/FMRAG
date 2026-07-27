"""
client/inference/local_inference.py
--------------------------------------
Makes a clinical prediction (disease/ADE, and optionally drug hypotheses)
using ONLY the locally cached model — no FL server connection required.

This is what answers "what does the client do if the server is down?":
it loads the last successfully-received global state (saved by
fl_client.py's save_local_checkpoint() after every round) and runs
inference directly, completely independent of server availability.

This is read-only with respect to the FL system — it never trains, never
sends anything anywhere, and never modifies the cached checkpoint. It's
safe to call at any time, including while fl_client.py's training loop is
separately retrying a connection to a downed server.

Usage (from a hospital's own clinical system, a CLI, or a small local
Flask endpoint — whatever fits your deployment):

    from client.inference.local_inference import LocalInferenceEngine

    engine = LocalInferenceEngine(
        num_nodes=1200, num_rels=50, max_visit=10,
        multitask=True,
    )
    if not engine.ready:
        # No checkpoint has ever been saved on this client — it has never
        # successfully completed even one FL round. There is nothing to
        # run inference on yet.
        ...

    result = engine.predict(
        conditions=["C0011860", "C0020557"],
        procedures=[],
        drugs=["C0000545"],
        tokenizer_text="patient presents with ...",  # free-text context for BioBERT
    )
    # result = {"disease_prob": 0.62, "ade_prob": 0.18, "checkpoint_round": 7,
    #           "checkpoint_age_hours": 3.4}

Honest limitation: this was written by reasoning through model_setup.py's
forward() signature and graphcare_pipeline.py's Data object construction,
not by running it against a real checkpoint — there was no trained
checkpoint available in the environment this was built in. The patient
graph construction here (entity ids, edge_index, visit_node, ehr_nodes)
mirrors what FMRAGPatientDataset.process() builds, but test this against
an actual saved checkpoint before relying on it for any real decision.
"""

from __future__ import annotations

import logging
import os
import time

import torch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [LOCAL-INFER] %(message)s")

CHECKPOINT_PATH = os.environ.get(
    "FMRAG_CHECKPOINT_PATH", "/var/fmrag/checkpoints/last_global_state.pt"
)


class LocalInferenceEngine:
    """
    Loads the cached global model state and exposes a single predict()
    call. Construct once per process (loading BioBERT + GraphCare is not
    cheap) and reuse across multiple predictions.
    """

    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        max_visit: int,
        ent2id: dict,
        rel2id: dict,
        multitask: bool = False,
        checkpoint_path: str = CHECKPOINT_PATH,
    ):
        from client.peft.model_setup import build_model, get_tokenizer

        self.ent2id = ent2id
        self.rel2id = rel2id
        self.tokenizer = get_tokenizer()
        self.model = build_model(
            num_nodes=num_nodes, num_rels=num_rels, max_visit=max_visit,
            multitask=multitask,
        )
        self.model.eval()

        self.ready = False
        self.checkpoint_round = None
        self.checkpoint_age_hours = None

        if not os.path.exists(checkpoint_path):
            log.warning(
                "No checkpoint at %s — this client has never completed a "
                "successful FL round. Local inference is NOT available "
                "until at least one round has run with the server "
                "reachable.", checkpoint_path,
            )
            return

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_lora_state_dict(ckpt["global_state"]["lora"])
        self.checkpoint_round = ckpt["round"]
        self.checkpoint_age_hours = (time.time() - ckpt["saved_at"]) / 3600
        self.ready = True

        log.info("Local inference ready — using checkpoint from round %d "
                 "(%.1f hours old)", self.checkpoint_round, self.checkpoint_age_hours)

    @torch.no_grad()
    def predict(
        self,
        conditions: list[str],
        procedures: list[str],
        drugs: list[str],
        tokenizer_text: str = "",
        max_seq_len: int = 64,
    ) -> dict:
        """
        Builds a single-patient graph from the given codes and runs the
        cached model on it. Returns a dict with disease_prob and (if the
        cached model is multitask) ade_prob, plus checkpoint metadata so
        the caller knows how stale the prediction might be.

        conditions/procedures/drugs: lists of codes already in the
        ent2id vocabulary (CCSCM/CCSPROC/ATC3) — same vocabulary the
        model was trained on, not raw ICD9/NDC codes.
        """
        if not self.ready:
            raise RuntimeError(
                "No checkpoint loaded — cannot run local inference. "
                "This client must complete at least one successful FL "
                "round (with the server reachable) before it can predict "
                "anything offline."
            )

        import networkx as nx
        from torch_geometric.utils import from_networkx

        codes = [c for c in (conditions + procedures + drugs) if c in self.ent2id]
        if not codes:
            raise ValueError(
                "None of the given codes are in this client's vocabulary "
                "(ent2id) — cannot build a patient graph. Check that the "
                "codes are already CCSCM/CCSPROC/ATC3-mapped, not raw "
                "ICD9/NDC."
            )

        # Minimal single-patient KG: a star graph around a synthetic
        # "visit" node connected to each observed code. This mirrors the
        # structure FMRAGPatientDataset.process() builds, simplified to
        # not require RAG/KGC (offline mode has no access to either).
        G = nx.Graph()
        visit_node_id = 0
        G.add_node(visit_node_id, node_id=visit_node_id)
        for i, code in enumerate(codes, start=1):
            node_id = self.ent2id[code]
            G.add_node(i, node_id=node_id)
            G.add_edge(visit_node_id, i)

        data = from_networkx(G)
        data.node_ids   = torch.tensor([G.nodes[n]["node_id"] for n in G.nodes], dtype=torch.long)
        data.rel_ids    = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
        data.visit_node = torch.tensor([visit_node_id], dtype=torch.long)
        data.ehr_nodes   = data.node_ids
        data.batch       = torch.zeros(data.num_nodes, dtype=torch.long)

        enc = self.tokenizer(
            tokenizer_text or " ".join(codes),
            truncation=True, padding="max_length", max_length=max_seq_len,
            return_tensors="pt",
        )

        logits, _ = self.model(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
            node_ids       = data.node_ids,
            rel_ids        = data.rel_ids,
            edge_index     = data.edge_index,
            batch          = data.batch,
            visit_node     = data.visit_node,
            ehr_nodes       = data.ehr_nodes,
        )

        result = {
            "checkpoint_round": self.checkpoint_round,
            "checkpoint_age_hours": round(self.checkpoint_age_hours, 2),
        }

        if isinstance(logits, dict):
            disease_logit = logits["disease"]
            ade_logit     = logits["ade"]
            disease_p = torch.sigmoid(
                disease_logit[:, 1] if disease_logit.dim() == 2 and disease_logit.shape[1] == 2
                else disease_logit.squeeze()
            ).item()
            ade_p = torch.sigmoid(
                ade_logit[:, 1] if ade_logit.dim() == 2 and ade_logit.shape[1] == 2
                else ade_logit.squeeze()
            ).item()
            result["disease_prob"] = disease_p
            result["ade_prob"]     = ade_p
        else:
            p = torch.sigmoid(
                logits[:, 1] if logits.dim() == 2 and logits.shape[1] == 2
                else logits.squeeze()
            ).item()
            result["disease_prob"] = p

        return result
