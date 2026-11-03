Step 0 workspace

Start the server: /workspace/run0.sh
HTTP: http://127.0.0.1:8000
Model: /data/models/Mellum2.1-12B-A2.5B-Thinking
Source: /workspace/ATOM (git branch codex/mellum-step0)
Frozen original: /app/ATOM (not modified)
run0 uses physical GPU 0 by default. STEP0_GPU selects another allocated GPU.
Do not start another server on the same GPU/port. The controlled restart helper only stops PIDs recorded by our own launcher.

Current logs: /workspace/logs/run0.log
Correctness: /workspace/results/smoke_reference.json, boundary_correctness.json
Audit: /workspace/results/environment.json, dataset_stats.json
Report and decisions: /workspace/report.md, WORKLOG.md

The reference uses physical GPU 1 and source-attributed upstream HF Mellum components backported into /workspace/reference. Installed dependencies are unchanged.
Local benchmark client is diagnostic only; evalscope and official acceptance materials are absent.
The local benchmark runner's shortened smoke is NOT scenario A/B/C official performance.
Do not run performance benchmarks while reference inference or profiling is active.

Only run0 is implemented at Step 0. run1/2/3 will be added after selecting actual optimizations.
Source, tools, report and evidence are snapshotted into /workspace/ATOM/submission for git history. Top-level /workspace paths are the delivery entrypoints.
