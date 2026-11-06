# Validated precision optimization, 2026-11-06

Default: direct paged KV gather, projection cap8, bounded eager attention,
cap8 Torch expert GEMMs, and graph_bucket local replay. No MTP.

A200: actual HTTP concurrency32, 256 output tokens/request, compare first64 raw IDs
against frozen Torch2.10/native HF5.20.2 eager single-request references.
Result12800/12800, 200/200 exact. Separate approximately16k group256/256 (four prompts).
No reference outputs are consumed by model inference code.

Controlled short performance test: same GPU, other GPU without model work,
fixed32 prompts x64 generated tokens, three repeats.
Original precision median42.97002376 output tok/s; candidate48.18292436, +12.1315%.
This is not an A512/B50/C200 performance result and does not establish600 tok/s.
A200 HTTP observed77.9578 tok/s had overlapping GPU0 work early in the run;
do not use it as the controlled speedup number.

Rejected: cap16/cap32 configurations each260/640 in the10-prompt screen.
Both use a16-row graph bucket for that screen; this is not a true32-row GEMM test.
KV gather passed8 bitwise comparisons including block16, SWA1024 and16k boundaries.
New portable entry passed33 HTTP requests:64/64 plus256/256 tokens, max decode batch32,
and all model/tool imports originate in this branch checkout.

Runtime implementation files precision.py, kv.py, opt_points/install.py are byte-identical
to the full-A200-tested candidate. Only portable serving/warmup wrappers were added
and separately smoke tested. Existing baseline launcher remains available.

Full source snapshots, raw tokens, scripts, logs and input/reference hashes:
server /workspace/results/mellum_dual_track_20261106T021633Z
archive SHA256 d62598d5dd29a52ea7cffa82b06b313969ce4620b170373b3aae3c20dce269ee
Portable-entry verification: /workspace/results/mellum_submit_validate_20261106T031523Z/smoke-precision

No prior Git commits were amended or squashed. Speed implementation is not included
in these precision optimization commits.
