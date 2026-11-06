# GPU1：题面朴素基线约束下的实现与性能对照

日期：2026-11-04，Asia/Hong_Kong。按用户要求仅使用GPU1。题方未提供移植源码，本实验对齐公开的单卡、eager、level0、torch循环MoE、显式mask SDPA、MTP关闭等宏观条件，不称为源码级复现。

**最终结论：本轮候选未复现题方性能。** A完整512条为111.8655输出token/s，仅为题方638.282724的17.526%；限额修正版C完整200次请求中仅1次成功、199次超时，成功率0.5%。默认朴素/复用精度实现及其已有200条正确性结果均未被本实验覆盖。

| 场景/版本 | 请求结果 | 可用结论 |
| --- | --- | --- |
| A，连续KV原候选 | 512成功、0失败，131072输出token | 111.8655 token/s；未达到题方638.282724 |
| C，连续KV原候选 | 运行中OOM，随后主动结束 | 无完整C成绩；失败材料保留 |
| C，32GiB限额修正版 | 1成功、199超时，原200条全部测完 | 成功率0.5%；未达到固定负载要求 |

C修正版没有再次OOM，未修改0.30请求/秒或180秒超时。唯一成功请求的TTFT为1.440458秒、TPOT为0.623495秒；这些只描述该成功样本，不作为全体请求的可比p99，也不据此计算对题方的延迟改善。C的845.3545秒及0.3028输出token/s是本次包含失败请求的观测，不能用来宣称有效服务能力。原始SQLite与199条TimeoutError日志已独立核对。

结果目录：`/workspace/results/naive_baseline_alignment_gpu1_20261104T172349+0800/`。连续KV原候选已完成A：512/512成功、111.8655输出token/s，为题方638.282724的17.526%，尚未对齐。SQLite原始512条、输入/输出长度及分位数均已复核，A耗时1171.6931秒；TTFT p99为10.833759秒、TPOT p99为0.282335秒。

原候选随后在C因连续KV与ATOM分页KV共存而OOM，失败日志和中断记录保留在`formal_dense_packed/`，不能作为完整C成绩。已加32GiB跨28层连续缓存预算：必要时采用较小容量、超过预算回退分页收集，长prefill前释放冗余连续缓存。限额修正版仅重测原200条C并已完成，结果位于`formal_bounded_dense_C/`；没有改动速率或180秒超时，也没有重复A。A和C属于两个清楚标识的实现版本，不将它们包装成同一版本的一次完整A/C测量。

上一条vector+packed_loop的A由本轮实验主动结束以测试更合适的数据组织，并非用户停止；末次完整周期日志320成功、0失败、934.2719秒、87.6833输出token/s，未完成512条，不能作为正式A成绩。原日志、SQLite、控制状态、技术原因`refinement_stop.json`及当时工具版本均保留。其C未运行。

## 实验范围与开关

全部变更通过独立进程的加载后入口安装，生产ATOM模型、算子、依赖、原启动脚本和Git HEAD不变。参数、torch norm、RoPE、BF16路由及ATOM分页KV不变。实验没有引入AITER融合专家、compile、graph、MTP或多卡并行。

- `current`：修复后朴素attention，保留HF式mask省略规则。
- `explicit`：原逐请求计算，所有attention传显式mask。
- `vector`：decode时一次索引整个batch的分页KV，将GQA组内查询头作为同位置的多条query，使用显式mask SDPA；避免K/V展开到32 heads。prefill仍使用显式mask；保持完整历史KV。
- `vector_trim`：在vector上进一步裁剪SWA读取范围；仅作独立优化诊断，没有叠入正式基线比较。
- `dense`：保持显式mask及完整历史KV；预填充后收集一次连续KV，后续decode只更新新增token。在prefill、batch成员/次序变化或容量不足时重新从ATOM分页KV构建；ATOM分页写入仍保留，支持重新构建。连续缓存使用有限值零填充，避免未初始化的padding值进入SDPA。
- 连续缓存限额：`NAIVE_ALIGN_DENSE_BUDGET_GIB=32`用于C修正版；不是裁剪历史或降低精度。预算覆盖所有28层的额外K/V，测试了32请求×4105、32×16019、50×16019时分别保留8192容量、收紧到16384容量、回退分页读取的分配决策。
- 专家`original`、`cpu_ids`、`packed_loop`均为torch循环。后两者减少GPU标量索引或预先组织专家输入。packed_loop保持HF的slot优先、token次序及逐专家累加顺序，没有引入跨专家重复索引的原子合并。

## 组件证据

32请求、4096 KV长度时，相同显式mask的原逐请求SDPA约19.62ms，GQA查询分组调用约0.83ms；三个测试形状中输出均与原显式路径逐元素一致，另用FP32 attention公式复核。

MoE三种组织、token数1/32/173共9组，每组重复5次，全部逐元素一致。32-token组件中原循环16.559ms、CPU专家ID循环12.768ms、预组织后循环6.259ms。另验证非连续物理页及不同长度batch的KV收集逐元素正确。组件收益不替代整模型性能成绩。

## 完整模型短程对照

同一个朴素模型进程，题方A前32条、warmup后每条输出8token；下表是无profiler的decode步中位数，包含真实分页KV整理，不是正式A吞吐。

| Attention | 专家组织 | CPU线程 | Decode中位耗时 |
| --- | --- | ---: | ---: |
| current | original | 192 | 784.506ms |
| current | original | 8 | 782.922ms |
| explicit | original | 8 | 735.120ms |
| vector | original | 8 | 390.376ms |
| vector | cpu_ids | 8 | 338.524ms |
| vector | packed_loop | 8 | 258.867ms |
| vector_trim | packed_loop | 8 | 206.177ms |

随后单独在同一模型进程比较vector+packed_loop与dense+packed_loop：265.878ms对185.290ms。dense warmup的28层×4步共112组有效K/V逐元素等于分页读回；固定10条输出也逐条等于原全显式mask朴素版（361/640），没有沿用current版本的100%正确性成绩。该补充实验材料位于`dense_experiments/`。

固定原最差10条、每条64token：current+packed_loop仍为640/640、10/10完整一致；vector+packed_loop为361/640、3/10，逐条复现旧的全显式mask朴素输出。因而全显式mask候选不能继承current版本已通过完整200条的正确性结论。本轮没有把10条结果扩写成200条成绩，也没有替换默认朴素实现。

## 正式性能测量口径

选用dense+packed_loop，CPU线程8，保持全历史KV和显式mask。BF16权重/KV、单GPU1、eager/level0、MTP/prefix-cache关闭，通过ATOM OpenAI兼容HTTP server及已有EvalScope运行。题方baseline文件未说明prefix cache、缓存布局等细节，本轮不能据此声称所有隐藏配置一致。

- A：原512条、并发32、每条输出256token，stream、greedy、ignore_eos；对比题方638.282724输出token/s/GPU。
- C：原200条、open-loop 0.30请求/秒、每条输出256token、请求超时180秒；对比题方TTFT p99 0.993549秒、TPOT p99 0.041265秒。

不改变题方输入、请求数、并发、输出长度、C速率或超时，不用包含输入的Total Throughput代替输出吞吐。题方没有B基线，本次不重复B。旧A/C暂停任务保持原状态；这是新授权的GPU1实验。结束时关闭本次进程，不恢复此前已停止的服务。

## 文件与可复核材料

外部工具在服务器`/workspace/tools/`，本地副本在`server-step0/`：`bench_naive_alignment_sdpa.py`、`naive_alignment_ops.py`、`test_naive_alignment_ops.py`、`run_naive_alignment_experiments.py`、`run_naive_dense_experiments.py`、`serve_naive_alignment.py`、`benchmark_naive_alignment.py`、`benchmark_naive_dense_alignment.py`、`benchmark_naive_bounded_c.py`、`review_naive_alignment_bench.py`。

组件与模型结果为`sdpa_microbench.json`、`organization_components.json`、`experiments_status.json`、`forwards.jsonl`及`quick10.json`。选定候选的正式结果位于`formal_dense_packed/`，保留实际命令、模型审计、源码SHA、EvalScope原始日志与SQLite记录；`formal_vector_packed/`为前一候选的中断记录。

首次整模型启动因实验phase文件在warmup后才创建而失败；已将外部hook改为文件存在时才读取，失败日志、当时工具及状态保留在`attempt1_startup_failure/`。修正后全部模型阶段及源码冻结检查完成。没有为该错误修改生产模型或依赖。

## 最终复核和运行状态

`formal_dense_packed/partial_review.json`包含完整A的独立SQLite复核；名称partial对应整轮因C OOM而未完成，A本身512条完整。`formal_bounded_dense_C/independent_review.json`核对全部200个原始请求、参数、成功/失败及分位数；控制任务complete表示请求全部测完，不表示性能通过。

根目录`final_review.json`独立核对了A归档源码、C最终源码、生产源码不变、Git HEAD、超时数与服务关闭状态。A测量时使用的两个已修订外部工具按`revision2_dense_code/`归档SHA核验，未用C修正版冒充A的当时实现；最初vector候选代码另在`revision1_vector_code/`。最终11个工具快照及manifest在`code_final/`。

最后复核两卡利用率均0%，显存各297623552字节（约284MiB），8000/8001均关闭，没有恢复旧服务或旧评测队列。HEAD仍为`6c86378f230d7d10734b574aa30988a349ebd2ff`，没有Git提交。

这些数据组织实验仍符合公开的朴素计算大类，但不能证明它们等同题方源码。尤其全显式mask候选固定10条仅361/640，不能作为已通过正确性门槛的最终交付配置；原current朴素路径的完整200条100%结果保持独立。
