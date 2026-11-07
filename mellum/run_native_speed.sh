#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo"
export HIP_VISIBLE_DEVICES="${CANDIDATE_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export ATOM_USE_UNIFIED_ATTN=0
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0
extra=()
case "${NATIVE_PROFILE:-aiter-graph}" in
  aiter-eager) extra+=(--enforce-eager) ;;
  aiter-graph) ;;
  aiter-fused-graph) export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1 ;;
  unified-graph) export ATOM_USE_UNIFIED_ATTN=1 ;;
  *) printf 'Unknown NATIVE_PROFILE: %s\n' "$NATIVE_PROFILE" >&2; exit 2 ;;
esac
cd "$repo"
exec /opt/venv/bin/python -m atom.entrypoints.openai.api_server \
  --model /data/models/Mellum2.1-12B-A2.5B-Thinking --kv_cache_dtype bf16 \
  --level 0 --cudagraph-capture-sizes '[1,2,4,8,16,24,32]' \
  --no-enable_prefix_caching --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.65 \
  --server-port "${SERVER_PORT:-8830}" --port "${ENGINE_PORT:-8831}" "${extra[@]}" "$@"
