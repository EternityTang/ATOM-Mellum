# Mellum speed branch

This branch preserves the existing ATOM history and recovered Mellum optimization stages.
The stage commits were created retrospectively on 2026-11-06; source provenance is in history/stages.
Original commits 450ec964 and 6c86378f are ancestors and have not been rewritten or squashed.
Missing intermediate edits cannot be reconstructed from snapshots; this is not a claim that every past edit was committed.
Both final versions preserve the exact original runtime Python source bytes; only portable launch wrappers are new.

## Run

```bash
CANDIDATE_GPU=1 bash mellum/run_speed.sh
```

Uses existing /opt/venv with frozen Torch2.10 and existing model /data/models/Mellum2.1-12B-A2.5B-Thinking.
No installation or dependency upgrade is performed. Launchers put this checkout and its bundled tools first on PYTHONPATH.
Original absolute-path launchers are retained for provenance, not used by the portable entry.
The default MTP setting is off.

## Validation scope

Speed default: packed. Historical fixed1024-input/256-output/concurrency32 median about662.46 token/s.
This is not the current precision acceptance and is not directly comparable to HTTP A with varied input lengths.
The speed branch has NOT passed A200 concurrent32 vs frozen HF eager single-reference >=90%.

Full measurement provenance is in history/reports and evidence/; large traces, datasets, model weights,
virtualenvs and GPU core dumps are not placed in Git. Their server locations/checksums are recorded.
Original dirty /workspace/ATOM and /workspace/tools remain untouched.
