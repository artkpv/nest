#!/usr/bin/env bash
# One-time setup of a rented GPU box (Lambda A100 40GB, Lambda Stack 24.04) for the
# CoT-controllability counting runs: Qwen3.8-27B-INT4 base / A / Q5 (qwen38_serve.sh)
# and gpt-oss-20b base / A / Q5 (gptoss_serve.sh), each served by ONE vLLM instance.
#
# Run from your machine WITH -t (the token prompt needs a terminal):
#   ssh -t ubuntu@<box-ip> 'bash qwen38_setup.sh'
# Safe to rerun: finished stages are skipped.
#
# The stack is pinned to what the SPAR team verified for Qwen3.8 + LoRA
# (CoT-Controllability, branch ola/qwen35-replication, ENVIRONMENT.md):
#   vLLM 0.29.0 (pulls torch 2.13.0, a CUDA 13 build) and transformers from git main
#   @ 4815a0a, installed LAST - otherwise vLLM pins it back to the PyPI release,
#   which does not know the Qwen3.8 architecture.
#
# Nothing here needs sudo. The HF token is read without echo, used for the one
# private download, and unset; it is never written to disk or shell history.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/vllm-env}"
MODELS="${MODELS:-$HOME/models}"
TRANSFORMERS_REV="4815a0a6a064214f2d8208c094464a5a6b76ca8d"  # 2026-09-09, = their "main @ 4815a0a"

echo "== 1/5 driver check"
# CUDA 13 wheels (torch 2.13.0 from PyPI) need driver >= 580. Fail here, not deep
# inside a vLLM import with an opaque "no kernel image" error.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ "${DRIVER%%.*}" -lt 580 ]; then
  echo "ERROR: driver $DRIVER < 580; CUDA 13 torch will not run. Stop and pick another image." >&2
  exit 1
fi

echo "== 2/5 uv + venv"
export PATH="$HOME/.local/bin:$PATH"   # before the check, or a fresh install is never found
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$ENV_DIR" ] || uv venv "$ENV_DIR" --python 3.12
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "== 3/5 vLLM 0.29.0, then transformers main (order matters)"
STACK_OK="$ENV_DIR/.nest_stack_ok"
if [ -f "$STACK_OK" ]; then
  echo "stack already installed ($STACK_OK); re-verifying only"
else
  uv pip install "vllm==0.29.0"
  uv pip install "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_REV}"
fi
python - <<'PY'
import torch, vllm, transformers
from vllm import ModelRegistry
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__)
assert torch.cuda.is_available(), "CUDA not available in the venv"
archs = ModelRegistry.get_supported_archs()
hits = [a for a in archs if "Qwen3_5" in a or "Qwen3_8" in a or "Qwen3Next" in a]
print("qwen hybrid archs registered:", hits)
assert hits, "no Qwen3.5/3.8 arch in vLLM's registry"
PY
touch "$STACK_OK"

echo "== 4/5 downloads"
mkdir -p "$MODELS"
# Every arm we serve: Qwen3.8 many-rule (A) + few-rule control (Q5), and the
# gpt-oss-20b pair. One private download, one token prompt.
ADAPTERS="qwen3.8-27b-A qwen3.8-27b-Q5 gpt-oss-20b-A gpt-oss-20b-Q5"
missing=""
for a in $ADAPTERS; do
  [ -f "$MODELS/treated/$a/adapter_model.safetensors" ] || missing="$missing $a"
done
if [ -z "$missing" ]; then
  echo "all adapters present; no token needed"
else
  echo "missing adapters:$missing"
  # Without a terminal (plain `ssh host 'cmd'`), `read -p` prints no prompt and
  # reads an empty string, and hf then reports the private repo as "not found".
  if [ ! -t 0 ]; then
    echo "ERROR: no terminal for the token prompt. Rerun as:" >&2
    echo "  ssh -t ubuntu@<box-ip> 'bash qwen38_setup.sh'" >&2
    exit 1
  fi
  read -rsp "Paste the read-only HF token for Cot-control-spar (not echoed): " HF_TOKEN; echo
  if [ -z "$HF_TOKEN" ]; then echo "ERROR: empty token" >&2; exit 1; fi
  export HF_TOKEN
  INCLUDES=()
  for a in $missing; do INCLUDES+=(--include "$a/*"); done
  hf download Cot-control-spar/cot-control-treated-models "${INCLUDES[@]}" \
    --local-dir "$MODELS/treated"
  unset HF_TOKEN
fi
# Public bases. The adapters' README names these as the weights to serve on;
# gpt-oss adapters are attention-only because vLLM's gpt-oss LoRA mapping is.
hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir "$MODELS/Qwen3.8-27B-INT4"
hf download openai/gpt-oss-20b --local-dir "$MODELS/gpt-oss-20b" --exclude "original/*" --exclude "metal/*"

echo "== 5/5 integrity"
# Compare against the LFS sha256 HF publishes (done from the local side).
for a in $ADAPTERS; do
  printf '%s  ' "$a"; sha256sum "$MODELS/treated/$a/adapter_model.safetensors" | cut -d' ' -f1
done
du -sh "$MODELS"/* "$MODELS"/treated/*
echo "setup OK. Next: bash qwen38_serve.sh   (later: bash gptoss_serve.sh)"
