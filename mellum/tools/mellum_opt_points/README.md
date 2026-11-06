# Mellum optimization points

Current candidate: OPT_ATTENTION=bounded, OPT_MOE=off, OPT_DISPATCH=graph_bucket.

Two retained points, at most three as required:

1. Bounded eager attention softmax: for score tensors larger than 1 GiB, apply FP32 Torch softmax to four independent heads at a time. Both BF16 matmuls, score scaling, masks, full key length, dtype boundaries, parameters and loader are unchanged.
2. Region graph dispatch: preserve independent per-request norm and linear arithmetic, while replaying local HIP graphs through PyTorch's CUDAGraph API. Power-of-two row buckets (1,2,4,8,16,32) are used only inside independent row operators; return active rows. The backbone, routing and expert input remain at the actual request count. Graph buffers are overwritten with current inputs on each call. No prompt ID, reference output or generated response is cached.

MoE tuning was investigated separately and is not retained: cap8 TorchScript dispatch had no stable benefit; the grouped Triton expert path failed the token checks. Experimental script/triton modes remain as evidence and must not be represented as accepted candidates.

## Start

    CANDIDATE_GPU=1 SERVER_PORT=8680 ENGINE_PORT=8681 bash /workspace/run_mellum_opt_points.sh

The startup entry warms generic 32-token inputs for supported graph buckets before accepting HTTP requests. It does not load any benchmark prompts or HF references. Formal benchmark warmup used the same bucket sizes plus the longest 32 A inputs; warmup is excluded from measurements.

Independent ablations:

    # Original norm_cap8 control (known long-prefill OOM in the tested configuration)
    OPT_ATTENTION=off OPT_MOE=off OPT_DISPATCH=off bash /workspace/run_mellum_opt_points.sh
    # Retained point 1 only
    OPT_ATTENTION=bounded OPT_MOE=off OPT_DISPATCH=off bash /workspace/run_mellum_opt_points.sh
    # Point 2 only
    OPT_ATTENTION=off OPT_MOE=off OPT_DISPATCH=graph_bucket bash /workspace/run_mellum_opt_points.sh
    # Retained points 1 + 2
    OPT_ATTENTION=bounded OPT_MOE=off OPT_DISPATCH=graph_bucket bash /workspace/run_mellum_opt_points.sh

Frozen Torch2.10, BF16, MTP off, prefix cache off, full prefill, max32 requests, token budget16384, memory utilization0.5. The engine remains level0/enforce-eager; local regions are graph-captured when enabled, so this is not an entirely graph-free configuration.

## Validation scope

Short32 three-run medians: control33.976, bounded33.777, MoE-script33.509, graph42.419, bounded+graph43.115 output tokens/s. All these short tests matched 2048/2048; the metrics are Engine.generate on real A inputs with64 outputs, not official HTTP scenario A. Variable-batch bucket tests10 and17 matched640/640 and1088/1088. Four approximately16k inputs matched256/256 after warming all six bucket sizes.

The full HTTP candidate A200 gate passed12800/12800 against frozen Torch2.10 native HF5.20.2 eager single-request references, using the first64 of the same256-token performance outputs. Formal same-GPU measurements completed: A512/512, 67.0115 output tokens/s; B50/50, 19.0020 output tokens/s; C0/200, 200 failed. See the report for timeout/coverage limits; no failed request is excluded from the denominator. HumanEval has not been rerun on this candidate.

Evidence: /workspace/results/mellum_opt_points_20261105T172604.
Existing implementations and Git history are retained; no commit.

Production generic startup HTTP smoke: 64/64 tokens. All test services stopped after completion.
