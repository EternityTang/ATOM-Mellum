# Mellum precision branch

This branch preserves the existing ATOM history and recovered Mellum optimization stages.
The stage commits were created retrospectively on 2026-11-06; source provenance is in history/stages.
Original commits 450ec964 and 6c86378f are ancestors and have not been rewritten or squashed.
Missing intermediate edits cannot be reconstructed from snapshots; this is not a claim that every past edit was committed.
The initial preservation commits retained original runtime source bytes. Subsequent precision optimization commits append the validated KV/projection changes described below.

## Run

```bash
CANDIDATE_GPU=1 bash mellum/run_precision.sh
```

Uses existing /opt/venv with frozen Torch2.10 and existing model /data/models/Mellum2.1-12B-A2.5B-Thinking.
No installation or dependency upgrade is performed. Launchers put this checkout and its bundled tools first on PYTHONPATH.
Original absolute-path launchers are retained for provenance, not used by the portable entry.
The default MTP setting is off.

## Validation scope

Current precision default: direct paged KV gather + cap8 projection groups + bounded
softmax + cap8 Torch experts + graph_bucket. The underlying Qwen3-MoE backbone and
native sliding-window integration remain in use; attention math keeps eager BF16
boundaries and FP32 softmax. Engine level0/enforce-eager still permits local graphs.

A200 concurrent32, each generating256 and comparing the first64 IDs, vs frozen
Torch2.10/native HF5.20.2 eager single references:12800/12800.
Separate approximately16k group:256/256 across four prompts. Gate>=90%.
Controlled32-prompt x64-token short test:42.970 ->48.183 output tok/s median,
+12.13%, three repeats; not a full A512/B50/C200 benchmark.
See evidence/precision-dual-20261106.

Previous7aa07ff baseline formal results (not the new default's formal results):
A51267.0115 output tok/s; B5019.0020; C2000 complete256-token requests,
all timeout180s, no OOM. C performance remains unresolved.

To launch that prior precision configuration:
```bash
CANDIDATE_GPU=1 bash mellum/run_precision_baseline.sh
```
The new default exposes KV_MODE=direct/off and DENSE_CAP (default8).
Values above8 failed the current accuracy screening.

Full measurement provenance is in history/reports and evidence/; large traces, datasets, model weights,
virtualenvs and GPU core dumps are not placed in Git. Their server locations/checksums are recorded.
Original dirty /workspace/ATOM and /workspace/tools remain untouched.

## Checkout validation

HTTP smoke completed33 requests: one64-token request and32 concurrent8-token requests. All imports used this checkout, maximum actual decode batch32. Original-entry evidence is in evidence/branch-validation; current-entry evidence is in evidence/precision-dual-20261106. This is launch validation, not a replacement full acceptance run.
