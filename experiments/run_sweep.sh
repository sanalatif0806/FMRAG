#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh — PARC experiment harness
# -----------------------------------------------------------------------------
# Runs the full experimental matrix automatically: for each (method, seed,
# partition) it sets the right env vars, starts the FL server, runs N clients
# for R rounds, tears everything down, and moves on. All metrics land in
# per-run JSONL files under $RESULTS_DIR, tagged with FMRAG_EXPERIMENT so
# make_tables.py can group them.
#
# Designed to be DATASET-AGNOSTIC: pass DATASET=synthea or DATASET=mimic3 and
# point ROOT at the data. The same sweep runs unchanged on the Synthea cohort
# now and on full MIMIC-III the day your license arrives.
#
# USAGE:
#   bash run_sweep.sh synthea /data/synthea/csv/csv 260
#   bash run_sweep.sh mimic3  /data/mimic-iii-demo  1335
#   (args: DATASET  ROOT  NUM_NODES)
#
# Tune the matrix via the arrays below. Start SMALL (1 seed, 5 rounds) to
# validate the harness, then scale up.
# =============================================================================
set -u

# ---- args ----
DATASET="${1:-synthea}"
ROOT="${2:-/data/synthea/csv/csv}"
NUM_NODES="${3:-260}"

# ---- paths (edit if your layout differs) ----
PROJECT=/opt/fmrag-project/FMRAG
VENV=/var/fmrag/venv
GRAPHCARE=/opt/fmrag-project/FMRAG/GraphCare-main
ENV_FILE=/etc/fmrag/env
RESULTS_DIR="${RESULTS_DIR:-/var/fmrag/results}"
PORT="${FMRAG_FL_PORT:-8080}"
PYTHONPATH_VAL="$PROJECT:$GRAPHCARE"

# ---- experiment matrix (EDIT THESE) ----
SEEDS=(1 2 3)                      # 3 seeds for mean±std
ROUNDS=20                          # FL rounds per run
CLIENT_COUNTS=(2 4 6)              # sweep client configs (like the FMRAG paper)
PARTITIONS=(iid dirichlet)         # add label_skew if wanted
DIRICHLET_ALPHAS=(0.5)             # sweep e.g. (0.1 0.5 1.0) for the non-IID figure

# Methods: name -> env overrides. These map directly to the paper's baselines.
#   local    = no aggregation value beyond 1 round (lower bound)  [approximated by NUM_CLIENTS=1 + 1 round]
#   fedavg   = KGC off, EWC off, KD off
#   parc_ewc = EWC on only (the forgetting headline)
#   parc_full= KGC on, EWC on, KD on (full system)
# Methods: name -> env overrides. These map directly to the paper's baselines.
#   fedavg   = plain FedAvg (no KGC/EWC/KD, no baseline term)
#   fedprox  = FedProx proximal term (standard non-IID baseline)
#   fedcurv  = FedCurv (EWC-in-FL — the key continual-learning baseline)
#   fedlwf   = FedLwF (KD-to-global — distillation baseline)
#   parc_ewc = PARC's EWC only (anchors to previous snapshot)
#   parc_kd  = PARC's KD only
#   parc_full= PARC full system (KGC + EWC + KD)
declare -A METHOD_ENV
METHOD_ENV[fedavg]="FMRAG_METHOD=fedavg FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=0 FMRAG_LAMBDA_KD=0"
METHOD_ENV[fedprox]="FMRAG_METHOD=fedprox FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=0 FMRAG_LAMBDA_KD=0 FMRAG_FEDPROX_MU=0.01"
METHOD_ENV[fedcurv]="FMRAG_METHOD=fedcurv FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=0 FMRAG_LAMBDA_KD=0 FMRAG_BASELINE_LAMBDA=1.0"
METHOD_ENV[fedlwf]="FMRAG_METHOD=fedlwf FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=0 FMRAG_LAMBDA_KD=0 FMRAG_BASELINE_LAMBDA=0.5"
METHOD_ENV[parc_ewc]="FMRAG_METHOD=parc FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=10 FMRAG_LAMBDA_KD=0"
METHOD_ENV[parc_kd]="FMRAG_METHOD=parc FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=0 FMRAG_LAMBDA_KD=0.5"
METHOD_ENV[parc_full]="FMRAG_METHOD=parc FMRAG_KGC_ENABLED=true FMRAG_LAMBDA_EWC=10 FMRAG_LAMBDA_KD=0.5"
# Model-level baselines (replace the model entirely; SCAFFOLD — validate first)
METHOD_ENV[retain]="FMRAG_METHOD=fedavg FMRAG_BASELINE_MODEL=retain FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=0 FMRAG_LAMBDA_KD=0"
METHOD_ENV[gbert]="FMRAG_METHOD=fedavg FMRAG_BASELINE_MODEL=gbert FMRAG_KGC_ENABLED=false FMRAG_LAMBDA_EWC=0 FMRAG_LAMBDA_KD=0"
METHODS=(fedavg fedprox fedcurv fedlwf retain gbert parc_ewc parc_full)

# ── Local-only and Centralized are handled by NUM_CLIENTS/ROUNDS, not a method:
#   Local-only  : run with the sweep's NUM_CLIENTS but only 1 round (no real
#                 aggregation benefit) — approximates each client training alone.
#                 Set LOCAL_ONLY=1 to force ROUNDS=1 for the fedavg method.
#   Centralized : run with NUM_CLIENTS=1 (whole dataset, no partition) — the
#                 upper bound. Run separately:  NUM_CLIENTS=1 bash run_sweep.sh ...
# These are config variants, deliberately NOT separate model code.

mkdir -p "$RESULTS_DIR"

# ---- helpers ----
kill_all () {
  sudo pkill -f "server.fl_server" 2>/dev/null
  sudo pkill -f "main_client.py"   2>/dev/null
  sleep 2
  sudo pkill -9 -f "server.fl_server" 2>/dev/null
  sudo pkill -9 -f "main_client.py"   2>/dev/null
  sleep 1
}

wait_for_port_free () {
  for _ in $(seq 1 15); do
    if ! sudo ss -tlnp 2>/dev/null | grep -q ":$PORT "; then return 0; fi
    sleep 1
  done
}

run_one () {
  local method="$1" seed="$2" partition="$3" alpha="$4" nclients="$5"
  local tag="${DATASET}__${method}__${partition}_a${alpha}__c${nclients}__seed${seed}"
  local metrics_file="$RESULTS_DIR/${tag}.jsonl"
  local method_env="${METHOD_ENV[$method]}"

  echo "──────────────────────────────────────────────────────────────"
  echo " RUN: $tag  (clients=$nclients)"
  echo "──────────────────────────────────────────────────────────────"

  kill_all
  wait_for_port_free

  # ---- start server ----
  sudo bash -c "set -a; source $ENV_FILE 2>/dev/null; set +a; \
    export FMRAG_DATASET=$DATASET; export FMRAG_NUM_NODES=$NUM_NODES; \
    export FMRAG_FL_PORT=$PORT; export FMRAG_MIN_CLIENTS=$nclients; \
    export FMRAG_MAX_ROUNDS=$ROUNDS; export FMRAG_KGC_ENABLED=$(echo $method_env | grep -o 'FMRAG_KGC_ENABLED=[a-z]*' | cut -d= -f2); \
    source $VENV/bin/activate; \
    PYTHONPATH=$PYTHONPATH_VAL python -m server.fl_server" \
    > "$RESULTS_DIR/${tag}.server.log" 2>&1 &
  sleep 8   # let the server bind

  # ---- start clients ----
  local pids=()
  for cid in $(seq 0 $((nclients-1))); do
    sudo bash -c "set -a; source $ENV_FILE 2>/dev/null; set +a; \
      export FMRAG_DATASET=$DATASET; export FMRAG_NUM_NODES=$NUM_NODES; \
      export FMRAG_CENTRAL_IP=127.0.0.1; export FMRAG_FL_PORT=$PORT; \
      export FMRAG_EXPERIMENT=$method; export FMRAG_DATASET_TAG=$DATASET; \
      export FMRAG_SEED=$seed; \
      export FMRAG_CLIENT_ID=$cid; export FMRAG_NUM_CLIENTS=$nclients; \
      export FMRAG_PARTITION=$partition; export FMRAG_DIRICHLET_ALPHA=$alpha; \
      export FMRAG_MAX_ROUNDS=$ROUNDS; \
      export FMRAG_METRICS_PATH=$metrics_file; \
      $method_env; \
      source $VENV/bin/activate; \
      PYTHONPATH=$PYTHONPATH_VAL python main_client.py" \
      > "$RESULTS_DIR/${tag}.client${cid}.log" 2>&1 &
    pids+=($!)
    sleep 3
  done

  # ---- wait for all clients to finish (or timeout) ----
  local timeout=$(( ROUNDS * nclients * 180 ))   # generous CPU budget
  local waited=0
  while kill -0 "${pids[0]}" 2>/dev/null; do
    sleep 10; waited=$((waited+10))
    if [ "$waited" -gt "$timeout" ]; then
      echo "  TIMEOUT after ${timeout}s — moving on"; break
    fi
  done

  kill_all
  echo "  done -> $metrics_file"
}

# ---- main sweep ----
echo "Sweep: dataset=$DATASET root=$ROOT num_nodes=$NUM_NODES"
echo "Methods: ${METHODS[*]} | seeds: ${SEEDS[*]} | partitions: ${PARTITIONS[*]} | alphas: ${DIRICHLET_ALPHAS[*]}"
echo "Results -> $RESULTS_DIR"
echo ""
echo "NOTE: ensure config/client_config.yaml has dataset=$DATASET, root=$ROOT,"
echo "      and graphcare.num_nodes=$NUM_NODES BEFORE running (see set_dataset.sh)."
echo ""

for nclients in "${CLIENT_COUNTS[@]}"; do
  for method in "${METHODS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
      if [ "$partition" = "dirichlet" ]; then
        for alpha in "${DIRICHLET_ALPHAS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            run_one "$method" "$seed" "$partition" "$alpha" "$nclients"
          done
        done
      else
        for seed in "${SEEDS[@]}"; do
          run_one "$method" "$seed" "$partition" "0" "$nclients"
        done
      fi
    done
  done
done

echo ""
echo "SWEEP COMPLETE. Parse with:  python3 make_tables.py $RESULTS_DIR"
