#!/usr/bin/env bash
# Guide Section 4.2: create the loc4dgs environment (PyTorch + CUDA, gsplat, utilities).
set -e
conda create -n loc4dgs python=3.10 -y
PY=$(conda run -n loc4dgs which python)
$PY -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install -r "$(dirname "$0")/requirements.txt"
$PY -m pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121 || $PY -m pip install gsplat
$PY -c "import torch, gsplat; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gsplat', gsplat.__version__)"
