# Mellum plain per-request precision path

This independent implementation follows native Transformers 5.20.2 Mellum eager operation order on the frozen Torch 2.10 runtime. ATOM provides loaded checkpoint Parameters, request scheduling metadata and allocated BF16 KV pages. The model's numerical forward and LM head are replaced by the standalone functions in plain.py.

It uses ordinary Torch RMSNorm, separate Q/K/V projections, per-layer RoPE/YaRN, explicit causal/sliding masks, eager attention, FP32 top-k routing followed by BF16 weights, and an ascending-expert Torch loop. Each request is evaluated independently; LM-head GEMMs are also per request. Prefill is whole-prompt, without padding or chunking. Requests still share one ATOM engine and retain separate paged KV state.

Start:
    CANDIDATE_GPU=1 SERVER_PORT=8540 bash /workspace/run_mellum_plain_request.sh --port 8541

The launcher selects the existing Mellum correctness model only as a validated weight container/loader. It does not install mellum_eager_precision or any earlier decode adapter. MTP, prefix cache, quantization, TBO and compilation are disabled. The currently supported checkpoint has 28 sparse-MoE layers, BF16 weights/KV and block size 16.

Reference contract: one native HF request at a time, no left padding, no prefill chunking, greedy, raw token IDs, same EOS policy. Numerical isolation intentionally prioritizes accuracy over throughput. max_num_seqs limits a scheduled batch, not the number of externally outstanding requests; use an explicit closed-loop client for strict concurrency 32.

Evidence and current validation: /workspace/results/mellum_plain_A_20261105/.
The previous precision and speed entries remain available at their original paths.
