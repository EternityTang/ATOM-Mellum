#!/usr/bin/env bash
# Toggle each measured point independently; defaults enable the three-point candidate.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
graph="${MELLUM_SPEED_GRAPH:-1}"
expert="${MELLUM_SPEED_EXPERT_FP8:-1}"
fusion="${MELLUM_SPEED_QK_FUSION:-1}"
for value in "$graph" "$expert" "$fusion"; do
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    printf 'Mellum speed flags must be 0 or 1\n' >&2
    exit 2
  fi
done

export NATIVE_PROFILE=aiter-graph
args=()
if [[ "$graph" == 0 ]]; then
  args+=(--enforce-eager)
fi
if [[ "$expert" == 1 ]]; then
  export MELLUM_FP8_MOE_MIN_TOKENS=24
  args+=(--online_quant_config '{"layer_quant_config":{"model.layers.*.mlp.experts":"ptpc_fp8"}}')
else
  export MELLUM_FP8_MOE_MIN_TOKENS=0
fi
if [[ "$fusion" == 1 ]]; then
  export NATIVE_PROFILE=aiter-fused-graph
fi

exec bash "$repo/mellum/run_native_speed.sh" "${args[@]}" "$@"
