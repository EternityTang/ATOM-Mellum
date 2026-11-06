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

## 2026-11-03 当前版本一致率实测（按用户要求冻结模型）
- 200条固定公开prompt，每条greedy生成64 token，ignore_eos=True；ATOM与参考均单请求推理，分别使用分配的两张GPU，非性能测试。
- token一致率：10773/12800 = 84.1640625%；整条64 token完全一致：136/200 = 68%。
- HumanEval组164条：83.67949695%；MBPP组36条：86.37152778%。这些是token一致率，不是pass@1。
- 参考仍为HF Mellum源码回移实现。原始输入、两侧token和源码SHA256见results/current_token_agreement.json。
- 200条ID均唯一，所有请求两侧均为64 token；全程源码和样本哈希保持不变。耗时2663.26秒（约44.4分钟）。
- 遵照最新要求，不以一致率为门槛继续修改模型，不改ATOM引擎，不启动其他阶段，不作Git提交。此前的适配改动保留在工作区，待用户审批。

- Installed EvalScope[perf] 1.14.1 in /workspace/venvs/evalscope; bare evalscope command available. Frozen /opt/venv pip freeze identical before/after. Perf smoke 2/2 succeeded, 8 tokens each. New CLI uses --stream flag. No ATOM changes or git commits.

- User confirmed Transformers need not remain frozen. Upgraded /opt/venv Transformers 5.2.0 -> 5.10.1; pip freeze confirms no other package changed. Native AutoConfig/MellumForCausalLM and 2 generation prompts passed; ATOM restarted and 8-token request passed; EvalScope CLI passed. No ATOM source edits or git commits in this upgrade. ATOM package metadata still pins 5.2.0. Earlier token agreement used the 5.2.0 source-backport reference and was not remeasured against native 5.10.1.


## 2026-11-03 原生 Transformers 一致率重测启动
- 按用户要求，仅重新测量一致率；不修改 ATOM 模型或引擎，不进行 Git 提交。
- 参考替换为安装包 Transformers 5.10.1 原生 MellumForCausalLM，bf16、SDPA，官方 generate，单请求 greedy 64 token、忽略 EOS。
- 固定 200 条公开 prompt，逐条核对文本哈希和输入 token 与旧测量完全一致。
- 两侧分别使用 GPU0/1，均 concurrency=1；不是性能测试。
- 测试工具：/workspace/tools/measure_native_token_agreement.py；进程记录：logs/native_token_agreement.pid。
- 结果逐条写入 results/native_token_agreement.json；日志：logs/native_token_agreement.log。完成状态以结果 summary.status 为准。


## 原生一致率评测暂停 2026-11-03T15:43:42.577643+08:00
- 用户明确要求暂停；已核对 PID 与命令后使用 SIGSTOP，保留进程内存和已完成结果。
- 已完成 109/200；阶段性 token 一致率 70.384174%，不是最终成绩。
- ATOM 服务继续运行，未修改源码或执行 Git 提交。恢复需用户后续指示。
