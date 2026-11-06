## 2026-11-06 User-confirmed precision acceptance

The user explicitly confirmed A200 at concurrency32 plus a separate approximately16k input group. Compare first64 raw generated token IDs against native HF eager single-request references on frozen Torch2.10, threshold >=90%; do not average the long-input group into A200. Current fixed long-input group contains four cases. Formal performance outputs256 tokens and precision uses their first64. The completed optimization round already used this setup; this confirmation does not authorize or require a rerun by itself. No GPU test started and no Git commit.

