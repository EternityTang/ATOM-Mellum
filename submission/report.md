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
- A数据的intended_words是词数生成参数，实际token长度须以冻结tokenizer统计；题面的长度过滤和数据集读取行为需与官方脚本一致。
- 当前权重索引5631项，未发现MTP权重；MTP方向需另有题方材料才能确认可行。

## 正确性与消融
后续填入真实运行结果。正式门槛：200 prompts前64 tokens一致比例>=90%；HumanEval差距bf16<=2个百分点/fp8<=5个百分点；16k子集单独统计；若有MTP/双卡必须追加对应检查。

| 配置 | GPU数 | A tokens/s/GPU | B TPOT p50 | C TPOT p99 / TTFT p99 |
|---|---:|---:|---:|---:|
| Step 0 | 1 | 待测 | 待测 | 待测 |

## 来源与AI使用
模型适配基于冻结ATOM的qwen3_moe及引擎接口，参考代码来源详见reference/SOURCE.txt。AI辅助阅读、适配和实验；工作记录见WORKLOG.md。保留失败日志，提交历史不squash。
