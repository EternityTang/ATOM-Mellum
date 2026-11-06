#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tools="$repo/mellum/tools"
export PYTHONPATH="$repo:$tools:$tools/mellum_decode_trial:$tools/mellum_step_cache:$tools/mellum_exact_decode"
export HIP_VISIBLE_DEVICES="${CANDIDATE_GPU:-1}"
export ATOM_MELLUM_ROCM_PRECISION=1 ATOM_MELLUM_ROCM_PRECISION_PROFILE=correctness ATOM_MELLUM_TORCH_EXPERTS=1 ATOM_MELLUM_ROUTE_WEIGHT_BF16=1 ATOM_MELLUM_NAIVE_BASELINE=0 ATOM_MELLUM_RNE_NORM=0
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0 ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
export OMP_NUM_THREADS=8 AITER_LOG_LEVEL=WARNING
export OPT_ATTENTION="${OPT_ATTENTION:-bounded}" OPT_MOE="${OPT_MOE:-off}" OPT_DISPATCH="${OPT_DISPATCH:-graph_bucket}"
cd "$repo"
exec /opt/venv/bin/python -m mellum_opt_points.serve_entry --model /data/models/Mellum2.1-12B-A2.5B-Thinking --kv_cache_dtype bf16 --level 0 --enforce-eager --no-enable_prefix_caching --no-enable_chunked_prefill --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.5 --server-port "${SERVER_PORT:-8680}" --port "${ENGINE_PORT:-8681}" "$@"
