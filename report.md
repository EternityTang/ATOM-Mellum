# Mellum 2.1 / ATOM 单卡推理优化报告

2026-11-07 · 源码基准提交 372601f7 · 声明单卡推理

## 1. 交付与运行环境

采用 ATOM 原生执行路径，复用 Qwen3-MoE 骨干、MoE、norm、权重加载，以及 GPT-OSS/MiMo 使用的逐层滑窗接口。Mellum 共28层，21层窗口1024，7层全局注意力，64专家选8。模型、RoPE和窗口适配属于移植工作，另选以下三个性能优化点。

| 入口 | 开关值 | 累加配置 |
| --- | --- | --- |
| bash run0.sh | 000 | 原生 AITER eager 工程对照 |
| bash run1.sh | 100 | 加 decode HIP Graph |
| bash run2.sh | 110 | 加专家在线 FP8 |
| bash run3.sh | 111 | 加 QK norm/RoPE/KV 写入融合，含V视图修复 |

一条命令启动最终配置：

    CANDIDATE_GPU=0 bash /workspace/run3.sh

脚本支持从任意工作目录调用，参数透传至 ATOM。默认 HTTP 8830、引擎8831，可用 SERVER_PORT、ENGINE_PORT 覆盖。

环境为分配的 MI308X、冻结 PyTorch 2.10.0+rocm7.2.4、AITER与ROCm原镜像；编译等级0、单卡、BF16 KV、MTP关闭、前缀缓存关闭、分块prefill预算16384、最大并发32、图桶1/2/4/8/16/24/32。模型路径为 /data/models/Mellum2.1-12B-A2.5B-Thinking，Python使用 /opt/venv/bin/python。

独立开关位于 mellum/run_speed_ablation.sh，均接受0/1，默认1。

| 优化点 | 环境变量 | 设0后的执行方式 |
| --- | --- | --- |
| Graph | MELLUM_SPEED_GRAPH | --enforce-eager |
| 专家FP8 | MELLUM_SPEED_EXPERT_FP8 | 原BF16专家 |
| QK融合 | MELLUM_SPEED_QK_FUSION | 非融合QK与缓存写入 |

例如关闭第二点：

    MELLUM_SPEED_EXPERT_FP8=0 bash mellum/run_speed_ablation.sh

三个点可以分别开关，完整ABC表采用000、100、110、111的累加消融。源码、运行说明及证据目录随仓库交付。

<!-- PAGEBREAK -->

## 2. Profiling与三个优化点

使用ATOM的torch profiler接口，预热后对32条输入采集12-token短窗，分别统计主机调用、GPU跨度与kernel耗时。吞吐使用关闭profiler的独立测量。原始压缩trace和分析JSON位于 submission/evidence/，可用Perfetto或Chrome Trace查看。

| Trace状态 | 每步主机提交 | GPU kernel数 | GPU decode跨度ms | 专家kernel ms |
| --- | --- | ---: | ---: | ---: |
| 原生eager | 430次launch | 430 | 35.56 | 7.31 |
| 加Graph | 1次replay | 431 | 12.01 | 7.30 |
| 再加专家FP8 | 1次replay | 431 | 6.78 | 2.00 |
| 最终融合及V视图修正 | 1次replay | 312 | 5.98 | 2.00 |

**点1，decode HIP Graph。** eager的kernel合计11.72ms，设备时间轴跨度35.56ms，派发和等待形成空档。复用框架图捕获，以一次replay提交整步。prefill仍走原生eager，最大批约525ms，因而端到端收益低于纯decode收益。

**点2，专家在线FP8。** Graph后专家占decode约61%、最大prefill约60%。仅量化28层专家，dense与KV保持BF16，复用AITER per-token activation/per-channel weight路径。冻结CK的小M形状捕获失败后，仅在专家内部补至24行，补位路由权重为零并裁回有效行。专家耗时7.30降到2.00ms，prefill专家315.46降到174.33ms。

**点3，原生QK融合与缓存写入。** 变长prefill续接时，非融合路径在gather前重排整层KV缓存。融合直接写SHUFFLE布局，32条共84186个prefill token的GPU跨度3440.16降到2158.95ms，超过1ms的大复制168次降至0次，全部direct-copy约1173.48降至0.77ms。

| 点 | 根据trace回看的可回收时间估算 | 本轮完整A逐点增益 |
| --- | --- | ---: |
| 1 | 去掉空档的理想decode跨度上限约3.03倍 | +38.49% |
| 2 | 若专家时间减半，decode时间约省30%，理想约1.44倍 | +61.18% |
| 3 | 大复制占原prefill跨度约33%，消除后理想约1.50倍 | +35.39% |

这些估算用于解释时间预算，未记录完整的事前数值预测，因此如实标为回看估算。V修复前融合曾使固定短输入变慢；进一步定位到SHUFFLE写入被Gluon按四维V布局误读。修正为五维视图后，7组组件相对L2误差约0.19%–0.25%，固定1024/256/32三轮中位2942.31升到3604.38 tok/s。此项归入点3的正确性修复。

<!-- PAGEBREAK -->

## 3. 完整A/B/C消融

2026-11-07，每个配置每场景各一轮。A/B四组顺序使用GPU1，C四组顺序使用GPU0，均为单卡推理。同场景请求集合哈希一致，全部请求成功且各输出256 token。裁判隐藏集三次中位复测留待正式验收。

| 配置 | 卡数 | A tok/s/GPU | B TPOT p50 ms | C TPOT p99 / TTFT p99 ms |
| --- | ---: | ---: | ---: | ---: |
| 工程对照000 | 1 | 674.50 | 29.27 | 42.06 / 1225.52 |
| +点1，100 | 1 | 934.12 | 4.81 | 13.46 / 1064.32 |
| +点1+2，110 | 1 | 1505.64 | 6.15 | 12.41 / 740.07 |
| +点1+2+3，111 | 1 | **2038.47** | 5.95 | **11.67 / 695.48** |

工程对照为相同原生移植的eager路径。题方公布的朴素baseline另列，最终111的A为题方638.2827的约3.19倍；C TPOT p99由题方41.2650降至11.6715ms，TTFT p99由993.5490降至695.4790ms。题方B基线缺失。target_g_a/target_g_c为空，尚无法换算最终官方得分。

| 配置 | A TTFT p50/p99 ms | A TPOT p50/p99 ms |
| --- | ---: | ---: |
| 000 | 2380.16 / 4709.32 | 37.35 / 47.15 |
| 100 | 2390.74 / 4372.31 | 24.25 / 33.94 |
| 110 | 1988.56 / 3784.50 | 12.87 / 20.91 |
| 111 | 1544.70 / 2240.65 | 9.50 / 15.28 |

| 配置 | B TTFT p50/p99 ms | B TPOT p50/p99 ms | C TTFT p50/p99 ms | C TPOT p50/p99 ms |
| --- | ---: | ---: | ---: | ---: |
| 000 | 71.90 / 150.32 | 29.27 / 29.53 | 486.27 / 1225.52 | 34.98 / 42.06 |
| 100 | 62.93 / 83.64 | 4.81 / 5.19 | 480.47 / 1064.32 | 6.37 / 13.46 |
| 110 | 47.08 / 57.33 | 6.15 / 6.16 | 376.24 / 740.07 | 7.26 / 12.41 |
| 111 | 48.63 / 59.09 | 5.95 / 5.95 | 352.65 / 695.48 | 6.72 / 11.67 |

A用原512条输入、并发32；B为50条随机1024-token输入、并发1。C用原200条共享前缀输入、0.30请求/秒开放到达、180秒单请求超时，EvalScope按指数间隔采样后归一化到666.67秒，seed42。B的最低TPOT来自100；最终选择111以A吞吐和C尾延迟为目标。

机器数据为 submission/evidence/abc-final-summary.json，题方基线为 baseline.json。固定1024/256/32微基准与完整变长A分开记录。

<!-- PAGEBREAK -->

## 4. HumanEval与优化取舍

| 检查项 | 实测 | 判定 |
| --- | --- | --- |
| HumanEval，原始164题 | **pass@1：90.24%（148/164）** | seed43独立复测 |
| 与有效HumanEval参考差值 | 题方bf16/fp8参考均为空 | 尚无法判定2/5百分点门槛 |

HumanEval报告值采用seed43独立复测，148/164，pass@1为90.24%。采样参数为temperature0.2、top_p0.95、top_k=-1，thinking开启，输出上限4096，每题一次。164份标准答案自检通过，所有151份可执行答案独立复判一致。

选择和淘汰主要依据如下。

- CK排序、默认启发式选核分别1904.14、1898.64 tok/s，低于Graph对照1930.65，未选。
- 单独投影FP8的A200仅+2.32%，KV FP8为-6.40%；dense格式修改只留试验快照。
- prefill预算64k完整A为1769.43，低于原融合2009.64；保留默认16k预算。
- 直接按原布局gather在固定短输入比原融合快10.13%，完整A却慢5.28%，最终选原生融合并修复V视图。
- MTP缺预测头权重，保持关闭。采样kernel代表步约0.048ms，优先级低。

专家FP8曾遇到未匹配dense层被错误量化、小M捕获失败；最初V布局错误使专家路由分散、单层专家约74升到142微秒。通过固定输入、交叉替换路由回放及独立attention参考定位，再以完整ABC复核。开发与错误记录见 mellum/history/native-speed/development.md。

证据索引为 submission/evidence/README.md，包含原始trace、分析JSON、ABC与验证汇总、HumanEval复测记录。详细日志和完整数据库归档保留在服务器 /workspace/results/。提交372601f7保留四阶段历史，本次交付入口与报告为其增量。正式验收结果以裁判复测为准。
