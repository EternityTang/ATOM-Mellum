# Decode RMSNorm pointwise batching
Keep the original per-request Torch mean and BF16/FP32 operation boundaries
inside each local norm graph. Batch the elementwise work across requests.
The precision entry enables this mode; PRECISION_NORM=per_request restores
the former norm path while keeping the other precision settings.

On GPU0, the experimental A200 HTTP pair (32 concurrent, 256 generated)
measured 77.1502 to 94.6263 output tokens/s. Both configurations matched
12800/12800 first-64 token IDs against the frozen Torch 2.10 / native HF eager
single-request reference. The independent ~16k group matched 256/256;
126 component cases were bitwise equal. See final-summary.json for scope.
The integrated precision install matched 2048/2048 on the fixed 32 prompts.
Only the accepted norm operation is integrated here; independent trial
variants are kept in the experiment evidence, outside this commit.
