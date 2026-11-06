# Mellum 复用精度版性能剖析

日期：2026-11-04，Asia/Hong_Kong。

## 结论

当前复用精度版在32请求decode时，首要瓶颈是**torch分页KV整理与SDPA attention路径**。采样帧内attention占墙钟时间77.13%，MoE占21.32%。SWA虽然正确地限制可见范围，但仍读取完整历史KV，显式mask路径又将4个KV heads物化展开到32个heads，造成额外数据搬运和计算。

补充的同组织受控回放证实了其中一部分代价：第一层SWA只把读取范围限制为最近1024个token，attention中位耗时由19.550ms降到12.781ms，局部1.530倍。但该输出出现微小数值差异，未应用到生产实现。

**不能据此说约19倍的全部差距已经解释完。** 题方基线也明确采用torch循环MoE和显式mask SDPA，不能假设它使用融合专家；其源码和trace未提供。本轮定位了我方路径的大部分耗时，并验证了一个具体低效点，尚未逐算子复现题方baseline。

## 对象与边界

- 模型：实际加载`MellumRocmPrecisionForCausalLM`，profile为`correctness`，不是原生融合配置，也不是独立朴素骨干。
- GPU0单卡MI308X；BF16权重/KV、eager、level0、MTP/prefix cache/TBO关闭。GPU1全程没有推理负载。
- 保留ATOM引擎、调度器和实际分页KV；离线走同一ModelRunner，排除HTTP客户端开销。生产模型和公共算子源码未修改。
- 输入为题方A文件前32条，实际输入866～4105token。固定greedy、ignore_eos，短程输出8token；包含warmup和独立重复。
- 用未开启profiler的运行测速度；分别截取单请求和32请求的一个完整decode forward做CPU/GPU profiling，含28层及LM head。该forward不包含HTTP、调度前处理和sampler。
- 这是诊断，不是512条、256输出的正式A成绩；C未恢复。不能用这里的短程结果生成正式A/C加速比。

## 未开启profiler的耗时

| 条件 | decode步数 | 每步中位耗时 | 每步平均耗时 | prefill累计forward耗时 |
| --- | ---: | ---: | ---: | ---: |
| 单请求 | 8 | 101.030ms | 101.068ms | 337.303ms |
| 32请求，第一轮 | 8 | 701.948ms | 681.689ms | 10856.786ms |
| 32请求，第二轮 | 8 | 702.187ms | 681.791ms | 10847.818ms |

32请求的各步约603～743ms。八步的输入token及路由不同，不能把后几步变慢仅归因于上下文多了几个token。两轮同位置耗时接近，输出完全一致；profiling期间的输出和未开启profiler的输出也一致。

这在没有HTTP请求、流式传输和EvalScope客户端的环境中复现了模型执行的慢速。CPU线程数为实际worker的默认192，本轮没有调线程参数。

## trace中的耗时分布

32请求采样forward墙钟890.057ms。profiler及标注带来开销，因此890ms不作为正常速度；下表用于定位采样步骤的耗时。

| 互不重叠的CPU外层范围 | 耗时 | 占采样forward墙钟 |
| --- | ---: | ---: |
| Attention，28层 | 686.504ms | 77.13% |
| MoE，28层，含gate/路由/专家 | 189.804ms | 21.32% |
| Attention外部的57处norm | 10.763ms | 1.21% |
| LM head | 0.107ms | 0.012% |

单请求采样时分布不同：attention为42.62%，MoE为49.75%。瓶颈随批量变化，不能用单流结论代替高并发结论。

GPU执行/拷贝事件累计472.694ms，跨stream去重后的活动区间约472.693ms。按CPU launch correlation归因，attention为418.924ms（88.63%的GPU工作时长），MoE为51.424ms（10.88%）。CPU与GPU会重叠，这些时间不能相加。

独立分析只用CPU的`user_annotation`计算墙钟，排除Kineto生成的同名`gpu_user_annotation`，并验证28处attention、28处MoE、57处外部norm及1处head；避免重复计数。GPU活动通过runtime correlation/External id追溯，当前事件归因覆盖率100%。

### Attention内部的具体开销

以下范围互相嵌套，不能与上表相加：

| 项目 | 次数/采样步 | CPU范围耗时 | 关联GPU工作时长 |
| --- | ---: | ---: | ---: |
| 分页KV+SDPA整体 | 28 | 644.525ms | 411.611ms |
| 逐请求读取分页KV | 896 | 169.023ms | 83.429ms |
| SDPA选择、mask及调用 | 896 | 401.936ms | 320.513ms |
| 显式mask分支 | 588 | 219.736ms | 256.003ms |
| 实际SDPA算子 | 896 | 62.297ms | 219.150ms |

32请求×28层产生896次SDPA。28条输入超过1024窗口，21个SWA层产生588次显式mask分支；另308次走可省略mask的分支。trace中588次`aten::_efficient_attention_forward`对应GPU时间166.830ms。BF16数据复制内核1176次，累计61.957ms，与显式mask分支中每次K/V展开各一次相符。源码的`expand(...).reshape(...)`并非总是零拷贝视图。

Attention范围内还记录1120次`hipMemcpyWithStream`，CPU时间约61.974ms；MoE范围630次、10.512ms。位置检查、动态索引及将设备数据转到CPU的操作值得继续缩减，但这些不是全部耗时。

实际代码位置：`atom/models/mellum_naive_ops.py`中`TorchPagedKV.read`按完整length读取历史；`NaiveSDPAAttention.forward`按请求循环；`masked_sdpa`展开KV heads。复用精度版直接调用这些辅助函数，参数容器和骨干复用不会消除其计算开销。

## 相同输入下的受控算子回放

在第3个32请求decode步骤，分别选第0层SWA和第3层全局attention。固定实际Q/K/V、位置及分页KV。每种实现先warmup一次，再同步计时5次并取中位数；包含KV写入/读取、mask、分发及SDPA。模型继续使用原始输出，实验候选不参与后续生成。

| 层及回放 | 中位耗时 | 相对原算子 | 与原输出逐值一致 |
| --- | ---: | ---: | --- |
| SWA，原实现 | 19.645ms | 1.000× | 是 |
| SWA，逐请求裁剪KV读取，先收集后SDPA | 12.576ms | 1.562× | 否 |
| SWA，不裁剪，补齐后合并SDPA | 20.664ms | 0.951× | 是 |
| SWA，裁剪并合并SDPA | 11.283ms | 1.741× | 否 |
| 全局，原实现 | 11.410ms | 1.000× | 是 |
| 全局，补齐后合并SDPA | 19.768ms | 0.577× | 否 |

SWA裁剪的最大绝对差为0.00048828125、相对L2为0.04439%；全局合并版最大绝对差0.0001220703125、相对L2为0.001261%。减少被mask的KV仍可能改变SDPA的归约/分块选择，因此数学可见范围一致不代表BF16逐值一致。

首轮裁剪回放还把KV先收集再执行SDPA，不能仅凭首轮就把全部收益归因于裁剪。因此补充同组织对照：全历史逐请求版本19.550ms，窗口裁剪版本12.781ms，严格单项比为1.530倍；同轮原实现19.651ms。全局层无裁剪的两种循环组织分别11.426/11.428ms，数值均完全一致。补充结果的原始输出仍完全等于本轮未开启profiler的整模型输出。

结论是：**窗口裁剪有直接的局部性能证据；简单padding后合并调用没有性能收益。** 不能把“调用数多”直接推导为“合成一个调用就快”。裁剪的约1.53倍只是一个层内算子的收益，不能冒充整模型或正式A加速；任何生产接入仍需重新验证token一致率。

## 下一步应改哪里

优先围绕Mellum局部attention适配减少分页KV到dense的反复整理、KV heads物化展开及窗口外读取，使用能直接处理分页KV和原生窗口的ROCm路径，并用已有精度参考逐项验证。SWA仅用mask开启与真正减少内存/计算量是两件不同的事。简单替换为padding批量SDPA已被本轮回放否定，不作为直接修复。

MoE仍是第二瓶颈：torch专家循环的筛选、索引、小GEMM和累加需要后续独立处理。eager/level0并没有规定上述操作的具体组织方式，宏观配置相同不保证实现效率相同。当前没有把这些优化应用到生产代码。

## 证据、复核与收尾

- 完整trace及独立分析：`/workspace/results/mellum_reuse_profile_20261104T164236+0800/`。
- 受控回放及独立复核：`/workspace/results/mellum_reuse_attention_replay_20261104T165345+0800/`。
- 补充同组织对照：`/workspace/results/mellum_reuse_attention_control_20261104T170717+0800/`，每种实现5次，中位数独立复算通过。
- 关键文件：`timing_summary.json`、`forwards.jsonl`、`trace_bs32.trace.json.gz`、`trace_bs1.trace.json.gz`、`independent_profile_analysis.json`、`layer_0_attention_replays.json`、`layer_3_attention_replays.json`、`independent_review.json`。
- 新增外部工具：`profile_mellum_reuse.py`、`launch_mellum_reuse_profile.py`、`analyze_mellum_reuse_profile.py`、`replay_mellum_reuse_attention.py`、`replay_mellum_reuse_attention_control.py`、`review_mellum_reuse_replays.py`；本地均在`server-step0/`，服务器均在`/workspace/tools/`。
- 独立重算5次回放的中位数、核对原始输出及profile输出、核对生产源码/脚本/数据SHA及Git HEAD/工作区状态通过。受控回放返回的整模型输出也完全等于未开启profiler的一轮输出。
- 三次诊断均完成，结束后推理进程关闭；8000、8001、8098端口均关闭，两卡利用率0%、显存各约284MiB。控制器刚退出时的瞬时显存读数可能未释放完，最终独立复核的settled读数为准。
- 正式A/C仍暂停，没有自动恢复服务，没有Git提交。
