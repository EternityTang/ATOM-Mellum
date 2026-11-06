# Mellum partial batching with HF token checks

Validated candidate mode: `norm_cap8`. In decode, all requests enter the inherited Qwen3-MoE backbone together. Expert routing, packing, activations and ordered BF16 reduction operate across requests. Each expert GEMM is split into chunks of at most 8 rows. This is actual cross-request expert matrix computation, not merely an outer scheduler batch.

Prefill stays full-length per request. Norm, Q/K/V projections, output projection, router linear and LM head retain per-request calculation. Attention retains the HF-compatible eager backend in existing Attention components; the native accelerated SWA kernels are not enabled. This is partial batching, not vLLM-style batching of every operator.

Frozen Torch 2.10; reference native HF 5.20.2 eager attention/eager experts, BF16, greedy, single request, no padding/chunking. Single GPU, level 0, no MTP, TBO or prefix caching.

```bash
CANDIDATE_GPU=1 SERVER_PORT=8580 MELLUM_PARALLEL_MODE=norm_cap8 \
  bash /workspace/run_mellum_parallel_precision.sh --port 8581
```

Fixed10: 640/640 token IDs, 10/10 exact. Fixed32: 2048/2048, 32/32 exact; actual maximum outer decode batch 32, expert projection chunk 8, original largest expert group 32. These tests generate 64 tokens and do not establish a universal invariance guarantee or replace full200/A512/HTTP performance tests.

Ablations remain for research. Unrestricted grouped MoE passes fixed10 but fails fixed32 (975/2048). Batched QKV, BMM projections and merged effective-window attention also fail their short checks. Modes other than `norm_cap8` are not the default. `norm_cap1` and `norm_isolated` are conservative controls.

Evidence: `/workspace/results/mellum_parallel_precision_20261105`. The HTTP entry is provided; this round exercises Engine.generate. Existing implementations, launchers, installed libraries and Git history are retained. No Git commit.

Same-GPU sequential pair (GPU0, other GPU idle), fixed32/64 output: serial control 128.681 s, norm_cap8 61.262 s, both 2048/2048 exact; 2.1005x in this short test. Model loading excluded. Not a formal scenario-A throughput score. Candidate passed fixed32 twice.
