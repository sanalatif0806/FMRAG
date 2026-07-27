
# =============================================================================
# install_deps.sh — phased dependency install for FMRAG (CPU-only, py3.10)
#
# Runs the EXACT order that resolves the dependency conflicts documented in
# requirements.txt. Run this INSIDE an activated venv:
#
#   python3.10 -m venv /var/fmrag/venv
#   source /var/fmrag/venv/bin/activate
#   bash install_deps.sh
#
# Safe to re-run — every step is idempotent / force-corrects to the right
# version. If your venv ever drifts (wrong torch, numpy 2.x), just re-run.
# =============================================================================
#!/usr/bin/env bash
set -e

echo "── PHASE 0: system build deps (apt) ─────────────────────────────────"
echo "   (skips silently if you lack sudo — install these manually if so)"
sudo apt-get install -y python3.10-dev default-libmysqlclient-dev \
  build-essential pkg-config 2>/dev/null || \
  echo "   WARNING: could not apt-install build deps — mysqlclient may fail to compile"

echo "── PHASE 1: numpy <2 FIRST (torch 2.2 needs NumPy 1.x ABI) ──────────"
sudo pip install "numpy==1.26.4"

echo "── PHASE 2: torch CPU build from the CPU index ──────────────────────"
sudo pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cpu

echo "── PHASE 3: torch-geometric companions matched to torch 2.2+cpu ─────"
sudo pip install torch-geometric==2.5.2 torch-scatter==2.1.2 torch-sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

echo "── PHASE 4: everything else, WITHOUT re-touching torch/numpy ────────"
# --no-deps first so pyhealth can't drag GPU torch back in...
sudo pip install --no-deps \
  transformers==4.40.0 peft==0.10.0 accelerate==0.29.3 datasets==2.19.0 \
  tokenizers==0.19.1 safetensors==0.4.3 networkx==3.3 pyhealth==1.1.4 \
  pandas==1.5.3 scikit-learn==1.4.2 Flask==3.0.3 flask-login==0.6.3 \
  werkzeug==3.0.2 gevent==24.2.1 psutil==5.9.8 requests==2.31.0 \
  PyYAML==6.0.1 python-dotenv==1.0.1 zerorpc==0.6.3 mysqlclient==2.2.4 \
  tqdm==4.66.2 bcrypt==4.1.3

# ...then a normal pass to pick up the sub-dependencies of those packages
# (torch/numpy are already satisfied so they won't be touched).
sudo pip install \
  transformers==4.40.0 peft==0.10.0 accelerate==0.29.3 datasets==2.19.0 \
  tokenizers==0.19.1 safetensors==0.4.3 networkx==3.3 pyhealth==1.1.4 \
  pandas==1.5.3 scikit-learn==1.4.2 Flask==3.0.3 flask-login==0.6.3 \
  werkzeug==3.0.2 gevent==24.2.1 psutil==5.9.8 requests==2.31.0 \
  PyYAML==6.0.1 python-dotenv==1.0.1 zerorpc==0.6.3 mysqlclient==2.2.4 \
  tqdm==4.66.2 bcrypt==4.1.3

echo "── PHASE 4b: re-pin numpy (phase 4 sub-deps may have bumped it) ─────"
sudo pip install "numpy==1.26.4" --force-reinstall

echo "── PHASE 5: verify ──────────────────────────────────────────────────"
python -c "import torch, torch_geometric, torch_scatter, torch_sparse, \
  yaml, gevent, flask, pyhealth, pandas, numpy; \
  print('torch:', torch.__version__, '| numpy:', numpy.__version__, \
  '| pandas:', pandas.__version__); \
  assert torch.__version__ == '2.2.0+cpu', 'WRONG TORCH BUILD'; \
  assert numpy.__version__.startswith('1.'), 'WRONG NUMPY (must be <2)'; \
  print('ALL OK')"

echo ""
echo "✓ Dependencies installed and verified."
echo "  If you saw 'ALL OK' above, the stack is correct."



 sudo systemctl start fmrag-broker fmrag-fl-server

 sudo systemctl start fmrag-monitor
