# 精度版C场景修复尝试

当前状态（2026-11-08）：**C200 超时问题已解决，200/200 成功，0 失败；C 的整体性能要求尚不能称为通过。** 最新 `ageguard-http` 候选使用精确 prefill/decode softmax 融合、fair4 调度和 140 秒请求年龄保护。保持原公开 C200、泊松到达 0.30 请求/秒、每条生成 256 token、客户端超时 180 秒；没有增大客户端超时、缩短输出或筛掉长输入。

独立精度验收：冻结 Torch 2.10 下的原生 HF 5.20.2 eager 逐条参考，A200 并发 32 自然生成 256、比较前 64 个原始 token，11616/12800＝**90.75%**，155/200 条前 64 token 完整一致；约 16k 四条为 256/256＝**100%**。这是 A200＋长输入组的验收结果，本轮没有重新生成 C200 的 HF 参考，不能把它写为 C200 token 一致率。

| 最新 C200 指标 | 实测 |
|---|---:|
| 成功 / 失败 | 200 / 0 |
| 服务端完整 256-token 输出 | 200 / 200 |
| 测试时长 | 774.4521 秒 |
| 输出吞吐 | 66.1113 token/s |
| 平均 / p99 TTFT | 32731.32 / 96570.3244 ms |
| 平均 / p99 TPOT | 384.74 / 468.7249 ms |
| p99 / 最长请求耗时 | 171.9487 / 172.3607 秒 |

**零超时只说明这轮负载能全部完成，不表示题方延迟门槛已满足。** TTFT 和 TPOT 仍较高。当前目标达成的是“完整 C200 无失败，同时保住 A200＋长输入精度”；没有据此宣称最终 C 得分合格，也没有重跑完整 A512/B50。上述吞吐是 C 的实测吞吐，不与旧 A 的吞吐混用。

## 最终修复与可复现证据

调度修复在每次 prefill 后至少推进 4 步 decode；当已有运行请求达到上限或最老运行请求年龄达到 140 秒时，暂缓接纳新 prefill、先推进 decode。年龄来自引擎原有 `arrive_time`，不按题号、prompt 内容或参考答案选路。内部 140 秒保护与客户端 180 秒超时是两个不同参数。运行模型的 decode 批大小仍最多 32；HTTP 未完成请求（含排队）峰值 53，不能误报为模型并行 53 条。

注意力修复保留现有 QK/PV GEMM、BF16 舍入边界和当前 Torch 的 FP32 softmax 归约顺序，融合缩放、mask、softmax 及转换。大形状使用基于冻结 Torch 同提交源码的 HIP 扩展，小形状只在已验证相同 PersistentSoftmax 顺序时使用直接 BF16 输出。没有启用 FP8、MTP、近似 attention 或更换 Torch。

同一真实 C32 第 6 步 decode 的 trace：kernel 8685→7747，`hipLaunchKernel` 2539→1601，BMM 均 1792、MM 均 2606，GEMM kernel 名称/次数完全相同。对照和候选的 32×8 自然生成输出 256/256 相同；此处是候选对旧实现，不是对 HF。累计 GPU kernel 时间 168.51→164.65 ms，profiler 数据用于机制分析，不当作无 profiler 端到端速度。

独立实验快照为 `/workspace/results/mellum_precision_cfix_20261108T064000Z/ageguard-http`，普通启动入口为该目录下 `run_precision_cfix.sh`。入口已用真实 C 第一条 prompt 完成 8-token HTTP 冒烟；完整 C200 与精度测试使用相同模型配置的评测驱动。模型 Python 源码测试前后哈希检查通过，最终 `.py/.cu/.sh/.hip` 清单共 489 文件。全部任务结束，两卡显存各约 298 MB，已释放。

证据已下载并逐包验证 SHA256：

- [最新 C200 完整结果](server-step0/precision-cfix/evidence/results/ageguard-http/age140-fair4-C/final-summary.json)、[精度结果](server-step0/precision-cfix/evidence/results/ageguard-http/age140-fair4-precision/acceptance.json)。
- [Prefill trace 对照](server-step0/precision-cfix/evidence/results/prefill-trace-comparison.json)、[Decode trace 对照](server-step0/precision-cfix/evidence/results/decode-trace-comparison.json)。
- [原始结果、token、命令和日志包](server-step0/precision-cfix/evidence/results.tar.gz)、[四份原始 profiler trace＋请求时间线](server-step0/precision-cfix/evidence/traces.tar.gz)、[有效完整源码快照及试验模块](server-step0/precision-cfix/evidence/source.tar.gz)、[包哈希清单](server-step0/precision-cfix/evidence/manifest.json)。解压三个包至同一目录可还原证据和有效快照；运行依赖服务器现有模型与冻结环境。

旧正式精度分支和速度分支、`/workspace/run0.sh`～`run3.sh` 与 `report.md` 均未改。本轮为工程底座上的 C 修复，不是严格仅三点交付；未进行 Git 提交或推送，等待用户审批。

## 逐步逼近完整 C 的记录

| 版本 / 单卡 | 成功 / 超时 | 结果与决策 |
|---|---:|---|
| 原 111 / GPU0 | 0 / 200 | 持续长 prefill，decode 被挤占；上一轮完整工程消融结果 |
| 仅 fair8 / GPU0 | 21 / 179 | 公平调度不足以弥补算力开销 |
| 逐元素＋专家合批＋fair8 / GPU1 | 23 / 177 | 整链路收益不足 |
| 精确 prefill softmax＋fair8 / GPU0 | 185 / 15 | 大幅改善，不能以中途零失败宣称最终通过 |
| 精确 prefill softmax＋fair4 / GPU1 | 193 / 7 | 失败流已有 211～249 token，继续缩减 decode 开销 |
| 再融合精确 decode softmax＋fair4 / GPU0 | 199 / 1 | 唯一失败已输出 252/256，仍未通过完整性目标 |
| 上述版本＋140 秒年龄保护 / GPU0 | **200 / 0** | 最长请求 172.36 秒；A200 90.75%、长输入 100%，保留 |

不同轮次 GPU/调度均有变化，此表展示试验与取舍，不能把所有端到端差额单独归因于一个算子。年龄保护优先保住在途请求，TTFT 存在取舍：最新平均 TTFT 32.73 秒，高于无年龄保护 r2 的 30.15 秒；总吞吐也从后者 67.36 降为 66.11 token/s，但消除了该轮最后 1 个超时。

以下保留当时的观察和失败方向，历史“运行中”表述不代表最新状态。

## 中间检查记录

中途状态曾为：仅 prefill 融合＋fair8 完整 C 为 185 成功/15 超时，改 fair4 为 193 成功/7 超时；后者精度 A200 90.75%、约 16k 100%。7 条失败流已输出 211～249 token。随后 `exactdecode-http` 固定 32 条 2048/2048 和约 16k 256/256 均一致，再进行完整 C。

decode扩展没有把大形状BF16 softmax盲目应用到全部情况：当前Torch在最后维度≤2048时采用PersistentSoftmax，FP32输入和BF16输入使用相同的FP32 warp归约顺序，9组小形状组件全部逐位相同；大于2048则保留ILP4的原FP32归约，使用自定义融合kernel，4组decode组件也全部逐位相同。此前直接对大形状使用BF16 softmax会走不同ILP布局，长输入只有45.70%，因此未采用。现有模型QK/PV GEMM形状和顺序保持。

trace对照（同15194-token单请求prefill）：GPU kernel累计5156.40→3034.95ms，kernel数14794→13254，BMM均56、MM均3092，GEMM kernel名称/次数直方图完全相同。该累计时间只用于机制分析，不等同于无profiler端到端时延。分析JSON为实验根目录`prefill-trace-comparison.json`。

新增完整C结果：逐元素＋专家合批fair8为23成功/177超时；精确prefill融合fair8为185/15；精确prefill融合fair4为193/7。不能把这些结果称为C全部通过。最终有效版本仍需完整200条零失败及独立精度验收。

上一轮候选`hfsoftmax-http`保留Torch QK/PV GEMM和FP32 softmax归约顺序，融合BF16 score缩放、mask及输出转换，配合fair8调度。独立A200为11616/12800＝90.75%，约16k为256/256＝100%。当时前97条均成功，但最终C为185成功/15超时，说明中途零失败不能替代完整验收。

## 已尝试方向和证据

| 方向 | 实测 | 取舍 |
|---|---|---|
| 仅公平调度fair8 | 完整C21成功、179超时 | 有改善，不能独立解决 |
| 前缀缓存＋fair8 | A200 90.75%，约16k 53.5156% | 长输入精度失败，不跑C |
| SWA按query块裁剪可见KV，块128 | 固定32条579/2048＝28.27% | 淘汰 |
| 保留完整KV、只分query块1024 | 约16k177/256＝69.14% | 淘汰 |
| 原生SWA prefill直接接回试验 | 固定32条87/2048＝4.25% | 精度失败，低一致率原因尚未进一步定位，不采用 |
| BF16边界保留的Triton带状注意力 | 组件相对L2约6e-5，实际32条645/2048＝31.49% | 小组件误差不足以代替token验收，淘汰 |
| 保持GEMM形状、合批prefill专家激活及归并 | 约16k256/256，31.90秒 | 数值通过，未见明显独立性能收益 |
| Triton逐元素融合＋原Torch FP32 softmax | 约16k256/256，29.68秒 | 数值通过，收益有限 |
| 上述逐元素融合＋Torch BF16 softmax | 约16k117/256＝45.70% | 淘汰；随机概率组件仅数十个元素不同，仍不能放松整链路门槛 |
| 逐元素融合＋专家合批＋fair8 | 完整C23成功、177超时 | 未解决，不能交付为通过 |
| 基于当前Torch源代码的融合softmax＋fair8 | 8组组件超过10亿概率元素全部逐位相同；约16k256/256，20.03秒；独立A20090.75%、long4 100% | 完整C185成功/15超时；继续融合decode并保护临近超时请求，最终200/200 |

最初“前缀缓存失败可能是后缀过短”的解释被日志否定：长输入命中3808缓存token后，仍有约12k未缓存token，不能靠后缀阈值规避。缓存改变了计算形状，但尚未逐层确认全部差异来源；没有按题号或参考答案选择路径。

## 最终候选的依据

原版15194-token prefill trace累计GPU kernel时间约5156ms，其中FP32 softmax约1345ms，BF16→FP32转换、BF16加mask、缩放等也占很大比例。CPU bincount的等待时间包含此前已排队GPU工作，不把其CPU累计时间直接判为专家计算占比。

当前Torch的git_version是`3d3aa833db84eed6b7f5595cb5f162c2f78300a4`。读取了同提交的[PyTorch SoftMax.cu](https://github.com/pytorch/pytorch/blob/3d3aa833db84eed6b7f5595cb5f162c2f78300a4/aten/src/ATen/native/cuda/SoftMax.cu)，复用512线程、ILP4及同一BlockReduce顺序，保留BF16 score/scale边界和FP32 exp/归约，再直接写BF16概率。大于2048元素才使用该路径；小形状仍走原Torch。保持完整Torch QK/PV GEMM形状，未采用近似attention。源码版权声明与PyTorch LICENSE均保留，没有更换Torch二进制或依赖版本。

新HIP扩展首轮因hipify解析同一行if/else kernel launch失败；拆成带花括号的独立语句后编译通过，失败日志保留。8组全局/SWA概率组件长度覆盖2049、4096、15194、16019，共1,016,883,228个元素，全部与Torch参考逐位相同。这些是有限测试结果，不能替代完整模型和C验收。

有效快照：`/workspace/results/mellum_precision_cfix_20261108T064000Z/hfsoftmax-http`，CPU/GPU两路分别隔离进行C和精度检查。新旧长prefill trace分别位于`tile-work/profile-control-long`和`hfsoftmax-http/profile-hfsoftmax-long`。旧精度分支及/workspace正式交付入口未改，尚未Git提交。

## 初始方案记录

2026-11-08，运行中。目标先是在原C200、0.30请求/秒、256输出、180秒单请求超时下200/200成功，再检查延迟；不把无超时等同于正式全部性能门槛通过。继续要求冻结Torch2.10、原生HF eager参考的A200并发32和约16k分别达到90%。未提交Git。

此前000/001/011/111的C全是0成功、200超时。调度源码schedule()在有可接纳prefill时直接返回prefill，不保证已运行decode的进展；get_next_batch_info没有其他调用者。精度路径还显式禁止prefix cache及chunked prefill。

两条独立候选：

- fair8：每次prefill后给decode保留8步；已有运行请求达到max_num_seqs时先推进decode。只改变调度，不改数学、到达率或输出上限。恢复开关CFIX_DECODE_STEPS=0可关闭。GPU0使用 `/workspace/results/mellum_precision_cfix_20261108T064000Z`。
- prefix-fair8：上述调度加ATOM已有前缀缓存。C200实际共享3817个token，可缓存完整block前缀，避免每次重复计算。仅在CFIX_PREFIX=1时允许缓存；已有缓存偏移RoPE、分页KV及显式mask路径复用，子请求元数据允许qlen小于context length并设置has_cached。GPU1使用 `/workspace/results/mellum_precision_cfix_20261108T064000Z/prefix-work`。先验收独立A200和long4，再跑C。

prefix首轮在Mellum模型构造处被额外prefix cache off保护拒绝，日志保留。为避免改变正在运行的fair8快照，复制到prefix-work后补齐该显式保护开关并重跑。两份checkout隔离；原精度/速度分支与/workspace交付入口未改。

截至首轮观察：fair8已有21条完整输出，但持续到达后出现超时，单独公平调度还不够。prefix-fair8的A200仍在进行。所有命令、token ID、请求数据库、服务日志和失败信息保留在上述目录；本地新代码在server-step0/precision-cfix/。
