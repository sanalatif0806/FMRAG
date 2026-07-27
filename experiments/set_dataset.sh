#!/usr/bin/env bash
# set_dataset.sh — flip client_config.yaml between datasets before a sweep.
# USAGE:  bash set_dataset.sh synthea /data/synthea/csv/csv 260
#         bash set_dataset.sh mimic3  /data/mimic-iii-demo  1335
set -e
DATASET="$1"; ROOT="$2"; NUM_NODES="$3"
CFG="${CFG:-/opt/fmrag-project/FMRAG/config/client_config.yaml}"

python3 - "$DATASET" "$ROOT" "$NUM_NODES" "$CFG" <<'PY'
import sys, re
dataset, root, num_nodes, cfg = sys.argv[1:5]
txt = open(cfg).read()
# data block
txt = re.sub(r'(\n\s*dataset:\s*)"[^"]*"', rf'\g<1>"{dataset}"', txt, count=1)
txt = re.sub(r'(\n\s*root:\s*)"[^"]*"', rf'\g<1>"{root}"', txt, count=1)
# num_nodes (graphcare block)
txt = re.sub(r'(\n\s*num_nodes:\s*)\d+', rf'\g<1>{num_nodes}', txt, count=1)
# point cache at a dataset-specific file so the two datasets don't collide
txt = re.sub(r'(\n\s*processed_cache:\s*)"[^"]*"',
             rf'\g<1>"./data/cache/{dataset}_mortality.pkl"', txt, count=1)
open(cfg,'w').write(txt)
print(f"config set: dataset={dataset} root={root} num_nodes={num_nodes}")
PY
