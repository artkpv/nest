#!/usr/bin/env bash
# Serve base + treated Qwen3.8-27B from ONE vLLM process (run ON THE BOX after
# qwen38_setup.sh). One server, adapter toggled per request, is what makes the
# comparison clean: same INT4 weights, kernels, template and sampling defaults,
# so the adapter is the only difference between arms.
#
#   request model "RedHatAI/Qwen3.8-27B-INT4"  -> base (no adapter)
#   request model "qwen3.8-27b-A"              -> same weights + many-rule LoRA
#   request model "qwen3.8-27b-Q5"             -> same weights + few-rule control LoRA
#
# Bound to 127.0.0.1: reachable only through an SSH tunnel from your machine,
#   ssh -N -L 8000:127.0.0.1:8000 ubuntu@<box-ip>
# never on the box's public IP. Run inside tmux so a dropped SSH session does
# not kill the server:  tmux new -s vllm 'bash qwen38_serve.sh'
#
# Sized for an A100 40GB: INT4 weights 19.5 GB + ~1 GB adapter leave ~15 GB for
# KV cache at 0.90 utilization (215k tokens measured 2026-10-21).
# Context is 16k because the run is THINKING ON: qwen3.8-27b-A ignores the
# no-think switch (re-opens <think> on 20/20 counting prompts), so both arms
# think. 16k = the SPAR team's 12,288-token reasoning budget + prompt + answer.
# MAX_MODEL_LEN=8192 is enough for a reasoning-off run.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/vllm-env}"
MODELS="${MODELS:-$HOME/models}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

exec vllm serve "$MODELS/Qwen3.8-27B-INT4" \
  --served-model-name RedHatAI/Qwen3.8-27B-INT4 \
  --enable-lora \
  --max-lora-rank 32 \
  --max-loras 2 \
  --lora-modules "qwen3.8-27b-A=$MODELS/treated/qwen3.8-27b-A" \
                 "qwen3.8-27b-Q5=$MODELS/treated/qwen3.8-27b-Q5" \
  --reasoning-parser qwen3 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --host 127.0.0.1 \
  --port 8000 \
  --no-enable-log-requests
# --reasoning-parser qwen3: with thinking off there should be nothing to parse;
#   if a think block leaks anyway it lands in message.reasoning (where the gate
#   script and the audit look) instead of silently polluting the scored text.
# --limit-mm-per-prompt 0/0: the checkpoint is a VL model; text-only use, so
#   skip reserving encoder memory on a 40 GB card.
# --no-enable-log-requests: vLLM 0.29 removed --disable-log-requests (their
#   ENVIRONMENT.md).
