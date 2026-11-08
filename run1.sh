#!/usr/bin/env bash
set -euo pipefail
repo="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
export MELLUM_SPEED_GRAPH=1 MELLUM_SPEED_EXPERT_FP8=0 MELLUM_SPEED_QK_FUSION=0
exec bash "$repo/mellum/run_speed_ablation.sh" "$@"
