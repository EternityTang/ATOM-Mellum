# Work log

## 2026-11-03 (project day index pending official D0)
- Read PDF, connected allocated two MI308X GPUs.
- Preserved frozen /app/ATOM; copied into /workspace/ATOM at commit 26f23a0.
- Frozen Transformers 5.2.0 has no Mellum support; evalscope not installed. No dependency upgrades.
- Checkpoint contains 5631 tensors, no mtp keys or layers beyond 27.
- baseline.json has A/C numbers but HumanEval reference invalid/null.
- Started eager single-GPU adaptation with plain expert loop and explicit masked SDPA.
- First GPU smoke: raw code, prose, chat/thinking outputs matched the independent reference text (3 cases). Raw token IDs initially omitted by response serialization; added a typed diagnostic field before claiming token-level agreement.
- Fixed two adapter bugs isolated by smoke: uncached-prefill metadata has no block table; YaRN inverse frequencies must use the runner device. Retained failed logs.
- Final smoke: 3x16 raw IDs exactly match independent reference; boundaries 8x64 (through 16384 input tokens) all match. Concurrent API smoke 8/8 successful; not an official performance result.
- User confirmed no additional correctness materials; froze 200 public prompts (164 HumanEval + 36 MBPP test tasks), source/selection/hash recorded before inference.
- Official card reports HumanEval+ 91.5%, not original HumanEval. Chose to establish paired same-setting HF/ATOM HumanEval reference instead of comparing incompatible scores.
- Automatic approval rejected the command preparing/starting the public200 batch validation, without a specific reason. Command did not execute; full200 and HumanEval remain unvalidated.
