#!/usr/bin/env bash
# Serve gpt-oss-20b base + both SPAR adapters from ONE vLLM process (run ON THE BOX
# after qwen38_setup.sh, with the Qwen server stopped - one model per GPU).
#
#   request model "openai/gpt-oss-20b"  -> base (no adapter)
#   request model "gpt-oss-20b-A"       -> + many-rule LoRA (~32 rules, 919 examples)
#   request model "gpt-oss-20b-Q5"      -> + few-rule control LoRA (6 rules, 906 examples)
#
# Recipe copied from the SPAR team's own gpt-oss serving (CoT-Controllability,
# ola/qwen35-replication, results/gptoss_many/stage.sh):
#   - the ORIGINAL MXFP4 weights; vLLM's gpt-oss LoRA mapping covers attention
#     only, which is why their adapters are attention-only (q/k/v/o)
#   - --reasoning-parser openai_gptoss (harmony: analysis channel -> reasoning,
#     final channel -> content)
#   - --attention-backend TRITON_ATTN
# Reasoning cannot be turned off on gpt-oss, so these arms run thinking on, at
# the "medium" effort the adapters were trained and evaluated at - pass
# --reasoning on --reasoning-effort medium to the runner.
#
# Same tunnel as the Qwen server:  ssh -N -L 8000:127.0.0.1:8000 ubuntu@<box-ip>
#   tmux new -d -s vllm 'bash gptoss_serve.sh'
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/vllm-env}"
MODELS="${MODELS:-$HOME/models}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

exec vllm serve "$MODELS/gpt-oss-20b" \
  --served-model-name openai/gpt-oss-20b \
  --enable-lora \
  --max-lora-rank 32 \
  --max-loras 2 \
  --lora-modules "gpt-oss-20b-A=$MODELS/treated/gpt-oss-20b-A" \
                 "gpt-oss-20b-Q5=$MODELS/treated/gpt-oss-20b-Q5" \
  --reasoning-parser openai_gptoss \
  --attention-backend TRITON_ATTN \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 \
  --port 8000 \
  --no-enable-log-requests
