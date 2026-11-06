#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ATOM
export PYTHONPATH="/workspace/ATOM${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES="${STEP0_GPU:-0}"
export AITER_LOG_LEVEL=WARNING
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
export ATOM_TORCH_PROFILER_DIR=/workspace/results/traces
exec python -m atom.entrypoints.openai_server \
  --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --kv_cache_dtype bf16 --level 0 --enforce-eager --no-enable_prefix_caching \
  --max-model-len 32768 --gpu-memory-utilization 0.65 \
  --server-port "${SERVER_PORT:-8000}" "$@"
