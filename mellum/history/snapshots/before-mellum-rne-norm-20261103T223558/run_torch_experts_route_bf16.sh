#!/usr/bin/env bash
set -euo pipefail
export ATOM_MELLUM_ROUTE_WEIGHT_BF16=1
exec bash /workspace/run_torch_experts.sh "$@"
