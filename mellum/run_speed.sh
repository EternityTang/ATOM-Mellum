#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tools="$repo/mellum/tools"
base=$tools/mellum_target600
export PYTHONPATH="$base:$tools/mellum_exact_decode:$tools/mellum_step_cache:$tools/mellum_decode_trial:$repo:$tools${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES="${CANDIDATE_GPU:-1}"
export ATOM_MELLUM_ROCM_PRECISION=1 ATOM_MELLUM_ROCM_PRECISION_PROFILE=correctness ATOM_MELLUM_TORCH_EXPERTS=1 ATOM_MELLUM_ROUTE_WEIGHT_BF16=1 ATOM_MELLUM_NAIVE_BASELINE=0 ATOM_MELLUM_RNE_NORM=0
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0 ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
export MELLUM_DECODE_IMPL="${MELLUM_DECODE_IMPL:-packed}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" AITER_LOG_LEVEL=WARNING
cd "$repo"
exec /opt/venv/bin/python "$base/entry.py" --model /data/models/Mellum2.1-12B-A2.5B-Thinking --kv_cache_dtype bf16 --level 0 --enforce-eager --no-enable_prefix_caching --max-model-len 32768 --gpu-memory-utilization 0.65 --server-port "${SERVER_PORT:-8312}" "$@"
