# 当前精度版 norm_cap8：A/B 完成，C 长输入 OOM

测试日期：2026-11-06（Asia/Hong_Kong）。实验目录：`/workspace/results/norm_cap8_abc_20261105T160808`。

A、B 全量测试完成；C 在长输入 prefill 中发生 OOM，已中止，未完成 200 条。没有修改配置重跑，也没有将部分结果当成完整 C 成绩。

## 测试版本与口径

同一 GPU1、同一 norm_cap8 版本：冻结 Torch 2.10，BF16，eager，编译等级 0，MTP 关闭。最大 32 请求，prefill 不分块，prefix cache 关闭，token budget 16384，显存利用率配置 0.5。Decode 共享骨干，专家 GEMM 最多 8 行；敏感 norm、projection、LM head 维持逐请求算术。Attention 复用框架组件，使用 HF 兼容 eager 后端，未启用原生加速 SWA kernel。

全部通过 ATOM HTTP /v1/completions 和 EvalScope perf 测量：stream=true、temperature=0、ignore_eos=true、固定输出 256 token、seed=42。启动后以 A 最长 32 条各生成 4 token 预热，预热不计入性能。未进行 HF token 一致率或 HumanEval 复测。

| 场景 | 请求数 | 负载 | 输入 |
|---|---:|---|---|
| A | 512 | 闭环并发 32 | 原始 A JSONL，实际 540–4113 token，不截断 |
| B | 50 | 闭环并发 1 | EvalScope random，固定 1024 token |
| C | 200 | 开环泊松 0.30 请求/秒，单请求超时 180 秒 | 原始 C JSONL，实际 4060–16019 token |

## 本轮结果

吞吐为输出 token/s；延迟单位为毫秒。失败请求必须结合成功率阅读，分位数不能视为失败请求也满足该延迟。

| 场景 | 成功/总数 | 输出吞吐 | 总耗时(s) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 512/512 | 45.976 | 2850.858 | 14835.583 | 20687.643 | 645.574 | 705.512 |
| B | 50/50 | 16.471 | 777.147 | 452.617 | 480.969 | 58.327 | 60.885 |
| C | OOM 中止；接收 50/200，完成 0 条 | — | — | — | — | — | — |

## 历史参照

下表只用于性能定位。出题方 baseline 的完整实现未提供；vLLM 使用独立 Torch 2.13 环境，与冻结 Torch 2.10 不同。历史结果不作为本轮精度证明。

| 场景 | 当前 norm_cap8 输出 token/s | 历史 vLLM 输出 token/s | 出题方 baseline 输出 token/s |
|---|---:|---:|---:|
| A | 45.976 | 523.214 | 638.283 |
| B | 16.471 | 33.576 | 未提供 |
| C | 无有效完整成绩 | 76.056 | 76.216 |

## C 失败原因

- 第 10 条输入 C-0009：15194 token。服务器序号 603，于 17:10:40 UTC 开始整段 prefill，17:10:45 UTC 记录推理子进程异常退出。
- 失败堆栈：mellum_eager_precision/install.py 的 eager_math → F.softmax(scores, dim=-1, dtype=torch.float32)。32 个 head 的 15194×15194 FP32 矩阵为 27.5204 GiB，与失败分配大小吻合。
- 报错时设备仅余 24.01 GiB；PyTorch 已分配 133.56 GiB，另有 reserved-but-unallocated 32.59 GiB。长输入 eager attention 的大矩阵是直接失败点，缓存预留和分配器碎片可能共同影响峰值；未做独立消融，不能断言只调整某一项即可解决。
- C 是同一服务完成 A、B 后继续运行的结果；本轮未做独立冷启动 C 对照，不能排除分配器历史状态对 OOM 的影响。
- 推理子进程退出后 HTTP 前端仍收请求。中止时已接收 50 条，原始流记录 50 条，无完整 256-token 输出；最长部分输出 155 token。客户端 SQLite 无完成记录，不能宣称 200 条全部超时或填写虚假的 C p99。
- 17:12:37 UTC 向本轮 EvalScope 客户端发送 SIGINT。客户端停止后，驱动因缺少 C 摘要而退出并清理服务；该 AssertionError 是中止后的评测收尾现象，原始故障是前述 GPU OOM。
- 保存 C/failure.json、C/oom-traceback.txt、C/accepted-requests.json、客户端日志和全部服务器日志。保持现有模型和参数不变。

## 性能解读

- A 输出吞吐 45.976 token/s，为历史 vLLM 的 8.79%，出题方 baseline 的 7.20%。
- B 输出吞吐 16.471 token/s，为历史 vLLM 的 49.05%。A 的差距比 B 大，但 A/B 输入长度和负载不同，不能直接当作严格的并行扩展性实验。
- 先前固定 32 条、64 输出比完整逐请求版本快约 2.10 倍的结论仍只适用于那次短测；它不代表达到 vLLM 性能，也不代表正式 A 为 600 token/s。
- 下一步需要分别处理长 prefill 的峰值显存和 A 的多请求性能，并继续以 HF 单条参考约束精度；本轮只测量和记录，没有应用新优化。

## 复核与边界

- 已复核 419 个源码文件哈希以及 Git HEAD/status；本轮没有修改现有模型、公共框架、依赖或创建 Git 提交。
- A/B 已核对实际请求参数、A 原始 prompt 集合覆盖、B 输入长度、SQLite 成功记录和服务器原始生成 token 数量。C 仅核对中止前的原始请求/输出与 OOM 证据，没有完整覆盖验收。详情见 summary.json、各场景日志及 raw-tokens.jsonl。
- ATOM 本轮服务依赖 Transformers 5.10.1；原生 HF 5.20.2 参考此次未启动，不涉及更改 HF 或重测精度。
- 本轮性能测试不能将此前固定 32 条、64 输出的 token 一致率外推为全部 A/B/C、256 输出一致率通过。
- ATOM 已知逐 token 文本解码存在 Unicode 边界问题。本轮保留实际原始 token ID；吞吐使用 API 的 token usage，不能用返回文本重新分词代替模型 token 数。

### A 原始记录

- 成功输出长度分布：{'256': 512}。
- 原始流记录：512 条，其中 finished=512；生成长度分布 {'256': 512}。

### B 原始记录

- 成功输出长度分布：{'256': 50}。
- 原始流记录：50 条，其中 finished=50；生成长度分布 {'256': 50}。

## 复现与证据

- run.py：按 A→B→C 启动评测并清理本次服务进程。
- code/server.sh：从当前版本入口复制的启动参数，只将入口指向记录包装层。
- code/server.py：安装原 norm_cap8，预热、记录原始 token 和低频路径计数。
- analyze.py：独立复算参数、覆盖、长度、成功率以及源码/Git 未变。
- source/、source-before.json、versions.json、data-sha256.json：源码和环境身份。
- A/、B/：准确命令、客户端日志、EvalScope 摘要/分位数/SQLite；C/：命令、日志、空完成记录数据库和 OOM 证据，无完整摘要。
- server.log、runtime*.json、raw-tokens.jsonl：服务器日志、真实路径及原始输出。
