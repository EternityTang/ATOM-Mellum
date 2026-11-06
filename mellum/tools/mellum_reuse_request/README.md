# Incremental Mellum framework reuse

Default: `framework_eager`. Single GPU, frozen Torch 2.10, BF16, compilation level 0, no MTP/TBO/prefix caching/chunked prefill. Numerical reference is native Transformers 5.20.2 eager/eager, one unpadded prompt at a time.

The implementation reuses inherited `Qwen3MoeModel.forward`, `Qwen3MoeDecoderLayer.forward`, `Qwen3MoeSparseMoeBlock.forward`, existing checkpoint Parameters and loading, plus the existing `Attention` / `PagedAttention` objects with 21 window-1024 and 7 full-attention layers. Norm, expert arithmetic and default attention math retain the validated HF-compatible operations. Reusing the Attention component does **not** mean enabling its native accelerated kernel.

The scheduler can keep 32 requests active. Within a forward, each request executes separately using its own metadata and GEMM dimensions. LM head also executes per request. This preserves the single-request reference arithmetic, at the cost of cross-request batching efficiency. It does not inherit the speed version's throughput score.

```bash
CANDIDATE_GPU=1 SERVER_PORT=8560 MELLUM_REUSE_STAGE=framework_eager \
  bash /workspace/run_mellum_reuse_request.sh --port 8561
```

The HTTP entry is provided but this round validates direct Engine.generate, not HTTP throughput. Existing entry points are unchanged.

Experimental modes: `backbone` bypasses the framework Attention object; `native_swa`, `native_swa_prefill`, and `native_swa_decode` enable the original accelerated SWA backend in selected phases. Global attention stays eager. They are ablations, not approved accuracy modes. `native_norm` is an unvalidated optional ablation and is not enabled by default.

Completed default regression: fixed10 640/640 token IDs, fixed32 2048/2048, all requests exact. Native-SWA all-phase fixed10 failed (17/640) and is not adopted. No full200 or performance rerun is authorized by these short checks.

Evidence: `/workspace/results/mellum_reuse_stages_20261105`. The backbone module's standard forward hook is bypassed by ATOM's compile-support decorator even at level 0; the corresponding zero raw counter is not a call count. The inherited method identity and actual decoder/MoE hooks validate backbone reuse. Attention backend counters validate which implementation runs.

Selective native-SWA checks also failed: prefill-only 25/640; decode-only 339/640. Both remain experimental. All short tests completed; no full test queue was restarted.
