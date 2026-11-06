# Mellum 2.1 / ATOM 实验报告（进行中）

## 状态
Step 0 适配与验证中；尚未宣称通过官方正确性门槛或取得优化收益。

## 冻结环境
- 原始 ATOM：`/app/ATOM`，commit `26f23a0bdbf50b9010c07ee3a71171189723e767`，未改动。
- 开发副本：`/workspace/ATOM`，分支 `codex/mellum-step0`。
- PyTorch `2.10.0+rocm7.2.4`，Transformers `5.2.0`；未升级依赖。
- 单卡 Step 0：bf16、level 0、enforce-eager、torch 专家循环、显式 mask SDPA、MTP 关闭。
- HF Mellum 类不在冻结包中。独立参考组件来自 HF 上游 Mellum 源码，来源 commit 见 `reference/SOURCE.txt`，通过本地推理驱动适配；不得将其结果直接标记为官方验收。

## 已提供的官方基线
数据来源 `/data/datasets/baseline.json`，1 GPU。

| 场景 | 指标 | 基线 |
|---|---|---:|
| A | output tokens/s/GPU | 638.282724 |
| A | TTFT p99 | 5.009713 s |
| A | TPOT p50 / p99 | 0.038951 / 0.050144 s |
| C | TTFT p99 | 0.993549 s |
| C | TPOT p50 / p99 | 0.036290 / 0.041265 s |

B 基线未提供。baseline.json 中 target_g_a、target_g_c、HumanEval有效参考值均为 null。
现存 HumanEval 3/164（1.829%）被题方标记 rejected，原因是原始 completions 未使用 chat template/thinking；不能作为正确性基准。

## 评测口径与缺口
- A：512*N 请求，并发32*N，输出256；总输出吞吐除N。
- B：50请求，并发1，输入1024，输出256；报告TTFT/TPOT分布，不直接计分。
- C：200*N请求，开放泊松到达0.30*N req/s，输出256；报告TTFT/TPOT p99，延迟不除N。
- 温度0、ignore_eos、stream，评测参数固定。
- 机器未安装evalscope，未发现题方独立验收脚本或200条公开正确性prompt/逐token参考文件。ATOM自带benchmark与PDF中的evalscope并非同一工具，结果需标明来源。
- C题面total-timeout=180与到达速率/请求数存在口径待核实：200条在0.30req/s下期望到达跨度约663秒，提供的基线耗时671.77秒。需要确认180是单请求超时还是整场超时，不能擅自截短整场。
- A数据的intended_words是词数生成参数，实测A为540–4113 tokens（512条），C为4060–16019 tokens（200条）；A有输入超过4096 tokens。不得自行截断数据来美化成绩，题面的长度过滤/截断行为需与官方脚本核实。数据哈希与分位数见results/dataset_stats.json。
- 当前权重索引5631项，未发现MTP权重；MTP方向需另有题方材料才能确认可行。

## 正确性与消融
已完成3条短输入、每条16 token的逐token检查，全部完全一致（本地回移参考，非官方验收）。8个边界用例已完成：32、1023、1024、1025、8191、8192、8193、16384 token输入，各生成64 token，全部与参考逐token一致。合计11例、560个生成token完全一致。结果见results/boundary_correctness.json；这是工程检查，不代表200条门槛已通过。正式门槛：200 prompts前64 tokens一致比例>=90%；HumanEval差距bf16<=2个百分点/fp8<=5个百分点；16k子集单独统计；若有MTP/双卡必须追加对应检查。

| 配置 | GPU数 | A tokens/s/GPU | B TPOT p50 | C TPOT p99 / TTFT p99 |
|---|---:|---:|---:|---:|
| Step 0 | 1 | 待测 | 待测 | 待测 |

## 来源与AI使用
模型适配基于冻结ATOM的qwen3_moe及引擎接口，参考代码来源详见reference/SOURCE.txt。AI辅助阅读、适配和实验；工作记录见WORKLOG.md。保留失败日志，提交历史不squash。

## 补充评测决策（2026-11-03）
用户确认未另行提供200条正确性prompt，授权自行获取。已固定HumanEval全部164题与Google Research MBPP测试集任务11至46，共200题，输入不含参考解答；来源、选取规则与SHA256见datasets/manifest.json。后续使用冻结tokenizer的chat模板、greedy、前64 token，与相同输入下的HF源码回移参考比较。200条批量校验尚未执行；准备/启动该批量校验的命令被自动审批拦截，未返回具体原因。

官方模型卡 https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking 报告的是HumanEval+ 91.5%，thinking模式、greedy，成绩由JetBrains自报；这与原始HumanEval不是相同口径，不可直接作为2个百分点差距的参考值。后续需固定相同模板、thinking设置、生成预算、代码提取和执行方法，在HF参考及ATOM上分别计算原始HumanEval pass@1；同时报告截断数与解析失败数。模型卡也明确MTP头尚待发布，与本地检查点缺失MTP权重相符。

并发接口诊断：A数据前8条、并发上限32、输出16，8/8成功；这是缩短的接口检查，不能作为正式A场景结果或计算加速比。完整A/B/C性能仍待测。

## 2026-11-03 当前版本一致率实测（按用户要求冻结模型）
- 200条固定公开prompt，每条greedy生成64 token，ignore_eos=True；ATOM与参考均单请求推理，分别使用分配的两张GPU，非性能测试。
- token一致率：10773/12800 = 84.1640625%；整条64 token完全一致：136/200 = 68%。
- HumanEval组164条：83.67949695%；MBPP组36条：86.37152778%。这些是token一致率，不是pass@1。
- 参考仍为HF Mellum源码回移实现。原始输入、两侧token和源码SHA256见results/current_token_agreement.json。
- 200条ID均唯一，所有请求两侧均为64 token；全程源码和样本哈希保持不变。耗时2663.26秒（约44.4分钟）。
- 遵照最新要求，不以一致率为门槛继续修改模型，不改ATOM引擎，不启动其他阶段，不作Git提交。此前的适配改动保留在工作区，待用户审批。

## Transformers environment update

- User confirmed Transformers need not remain frozen. Upgraded /opt/venv Transformers 5.2.0 -> 5.10.1; pip freeze confirms no other package changed. Native AutoConfig/MellumForCausalLM and 2 generation prompts passed; ATOM restarted and 8-token request passed; EvalScope CLI passed. No ATOM source edits or git commits in this upgrade. ATOM package metadata still pins 5.2.0. Earlier token agreement used the 5.2.0 source-backport reference and was not remeasured against native 5.10.1.
