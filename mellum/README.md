# Mellum precision branch

This branch preserves the existing ATOM history and recovered Mellum optimization stages.
The stage commits were created retrospectively on 2026-11-06; source provenance is in history/stages.
Original commits 450ec964 and 6c86378f are ancestors and have not been rewritten or squashed.
Missing intermediate edits cannot be reconstructed from snapshots; this is not a claim that every past edit was committed.
Both final versions preserve the exact original runtime Python source bytes; only portable launch wrappers are new.

## Run

```bash
CANDIDATE_GPU=1 bash mellum/run_precision.sh
```

Uses existing /opt/venv with frozen Torch2.10 and existing model /data/models/Mellum2.1-12B-A2.5B-Thinking.
No installation or dependency upgrade is performed. Launchers put this checkout and its bundled tools first on PYTHONPATH.
Original absolute-path launchers are retained for provenance, not used by the portable entry.
The default MTP setting is off.

## Validation scope

Precision default: bounded softmax + cap8 Torch experts + graph_bucket.
A200 concurrent32 vs frozen Torch2.10 native HF5.20.2 eager single references:12800/12800.
Separate approximately16k group:256/256 across four prompts. Gate is>=90%, not exact equality.
Formal same-GPU A512:67.0115 output token/s; B50:19.0020; C200:0 complete256-token requests, all timeout180s, no OOM.
Thus C performance is NOT passed. Local graphs are on despite engine level0/enforce-eager.

Full measurement provenance is in history/reports and evidence/; large traces, datasets, model weights,
virtualenvs and GPU core dumps are not placed in Git. Their server locations/checksums are recorded.
Original dirty /workspace/ATOM and /workspace/tools remain untouched.

## Checkout validation

HTTP smoke completed33 requests: one64-token request and32 concurrent8-token requests. All imports used this checkout, maximum actual decode batch32. See evidence/branch-validation. This is launch validation, not a replacement full acceptance run.
