# 朴素版与复用精度版性能对比

日期：2026-11-04，Asia/Hong_Kong。结果目录：`/workspace/results/mellum_perf_compare_20261104T150625+0800/`。

后续仅按用户要求做复用精度版短程profiling，未恢复正式A/C。单步trace定位attention占77.13%墙钟；同组织SWA算子回放19.550→12.781ms，但有微小数值差异、未部署。完整证据见`复用版性能剖析.md`。诊断结束后两卡仍全部关闭。

**最终运行状态更新：用户进一步要求“全部停止，不用恢复服务”。** 两张卡上的推理服务、测试客户端和后续队列均已停止，8000/8001端口关闭；GPU利用率均0%，显存各297623552字节（约284MiB）。`services_status=stopped_by_user`，不自动启动服务或恢复测试。下文“服务已恢复”的记录仅描述前一步暂停清理的历史状态，随后已按最新指令停止；现以本段为准。已完成B与中断A的全部已有日志/数据保留。

**最新状态：正式测试已按用户要求暂停，`stopped_by_user`，不自动恢复。** 两版B已完成；朴素A中断，复用A及两版C未运行。客户端和后续队列均已停止。GPU0原生服务恢复为PID170319，GPU1仍为PID136002，两服务实际8-token及显存检查通过；全部原生产源码/脚本/数据SHA和Git HEAD不变，无Git提交。

暂停前终端进度约224/512；最后一份完整周期日志汇总为220条成功、0失败、1760.7422秒、输出31.9865token/s，平均TTFT8039.67ms、平均TPOT954.26ms。未生成512条完整结束后的A summary；这些只是阶段日志，不能作为正式A成绩或逐请求独立复核结果。保留原日志和数据库，不补造最终summary。下文带时间的进度段落为历史快照。

暂停工具为`/workspace/tools/stop_mellum_perf_compare.py`，控制文件记录`restore_status=complete`、`active_formal_benchmark=false`、`source_unchanged=true`。后续正式测试须由用户明确要求恢复。

## 测量对象与口径

比较 `MellumNaiveForCausalLM` 和 `MellumRocmPrecisionForCausalLM / correctness`。两者均已分别通过完整200条token一致性检查，本次不在性能测量期间并行运行正确性任务。原生融合版不属于本次两个被测版本。

两版均在物理GPU0串行测量，每次声明1卡；GPU1朴素服务保留但不施加测试负载。使用ATOM OpenAI兼容HTTP server和已安装的EvalScope 1.14.1客户端，BF16权重/KV、eager、编译等级0，MTP和prefix cache关闭。启动后记录实际模型类、profile和dtype；测量期间冻结生产源码、启动脚本、baseline及A/C数据。

| 场景 | 完整负载 | 主要报告 |
| --- | --- | --- |
| A | 原512条输入，并发32，每条固定输出256token | 输出token/s/GPU；同时保留延迟统计 |
| B | EvalScope random，固定seed=42，50条输入1024token，并发1，输出256token | TTFT/TPOT的p50、p99 |
| C | 原200条共享前缀输入，open-loop，0.30请求/秒，输出256token，每请求超时180秒 | 成功率、超时数、TTFT/TPOT p99 |

全部使用stream和greedy、ignore_eos。按题面使用**输出吞吐**与官方baseline比较，不以包含输入token的Total Throughput代替。

兼容当前EvalScope命令语法：PDF中的`--stream True`使用等价的`--stream`；A数据使用`--dataset line_by_line --dataset-path <原路径>`读取。保留原始JSONL中的prompt，未截断或重写数据、减少请求数、改变并发或C速率。每次服务加载后执行相同的8-token可用性检查，该请求不计入正式测量；正式测量warmup_num=0，关闭额外连接探测。

原文件的`intended_words`与tokenizer实际token数不同。实际A输入为540～4113token，C为4060～16019token；两版都使用原始文件的全部请求，未为了名义长度范围过滤样本。原始数据SHA及分布保存在`dataset_audit.json`。

## 出题方公布的单卡baseline

来源：`/data/datasets/baseline.json`，声明为单卡、eager、level0、BF16 KV、MTP关闭、SWA开启。

| 指标 | 题方baseline |
| --- | ---: |
| A 输出吞吐 | 638.282724 token/s/GPU |
| A TTFT p99 | 5009.713 ms |
| A TPOT p50 / p99 | 38.951 / 50.144 ms |
| C TTFT p99 | 993.549 ms |
| C TPOT p50 / p99 | 36.290 / 41.265 ms |

题方文件没有B指标，因此B只比较两版实测值。`target_g_a`、`target_g_c`为null，不据此编造最终目标或综合评分。A吞吐比为“我方输出吞吐÷638.282724”；C单项延迟改善倍数为“题方p99÷我方p99”，两者方向不同。

我方朴素实现与题方公布的baseline是不同对象：本次以题方给定数字作为外部参照，同时测量两条已通过本地正确性检查的实现。题方baseline未附完整原始逐请求记录和客户端版本。

## 已完成的B测量

| 配置 | 成功请求 | 输出token/s | TTFT p50 / p99 | TPOT p50 / p99 |
| --- | --- | ---: | ---: | ---: |
| 朴素版 | 50/50 | 8.4234 | 689.078 / 798.648 ms | 115.994 / 120.881 ms |
| 复用精度版 | 50/50 | 9.9759 | 486.413 / 590.059 ms | 98.443 / 103.126 ms |

两行均已由SQLite原始记录独立复核：每版50个请求，输入均为1024token、输出均为256token；请求参数及nearest-rank分位数重算通过。两版50条原始prompt的SHA集合相同。

在本次B单流测试中，复用精度版输出速度为朴素版的约1.184倍（高18.4%），TPOT p50从115.994ms降到98.443ms。该比值是两版之间的比较，不是相对题方baseline的B加速比。A/C结果完成后补充。

## A/C历史进度快照（非最终成绩）

截至2026-11-04 16:07 HKT，朴素版A最近一次客户端汇总为95/512完成、95成功、0失败，已完成请求累计输出吞吐32.8153token/s。前三批单批耗时约243～251秒，初步约为题方A输出吞吐的5.1%～5.2%。请求分批完成会使过程中显示的吞吐随时间波动，此处仅用于说明运行速度；最终A成绩须等待512条完成后按完整时间窗口计算。复用精度版完整A以及两版C尚未完成，不能用B的两版加速比推算A/C最终结果。

## 可复核材料与限制

服务和测量工具均在`/workspace/tools/`：`serve_mellum_perf.py`、`run_mellum_perf_compare.py`、`review_mellum_perf_compare.py`。前者只通过现有加载后入口记录模型状态，没有改写forward。每个场景保留完整CLI参数、EvalScope summary/percentile、SQLite逐请求数据、workload timeline和客户端/服务日志。

独立复核按安装版EvalScope的nearest-rank分位数定义重算p50/p99，并将毫秒与秒统一；核对输出长度、成功数、原始prompt及两版输入是否相同。C的超时不会通过增大超时参数绕开；若存在失败，请求成功率和观测分位数同时报告，不用少量成功请求的延迟宣称性能改善。

本次原计划为公开负载、每版每场景一次测量，区别于裁判在隐藏集上运行三次取中位数；当前只完成两版B，其余按文首状态暂停。源码和正式负载在测量期间未修改；没有Git提交。
