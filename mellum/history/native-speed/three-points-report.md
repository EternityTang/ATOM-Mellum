# Mellum 速度版三点 Trace 与消融

按单卡场景 A 输出吞吐选择三点，依次开启原生 decode HIP Graph、专家在线 FP8、原生 QK norm/RoPE/KV 写入融合。对照为同一 Mellum 原生适配的 eager 路径。题面官方朴素基线另有定义。

冻结 Torch 2.10，单张 MI308X。图捕获大小 1/2/4/8/16/24/32，编译等级 0，KV 为 BF16，最终内部 prefill token 预算 16384。A/B 使用原始并发和负载，每条输出 256 token，temperature=0，ignore_eos。本轮按用户指定验收速度、延迟和生成完成情况。

| 累加配置 | 开关 | 固定 32 中位 tok/s | A512 tok/s/GPU | A 平均 TTFT ms | A 平均 TPOT ms | B TPOT p50 ms | B TPOT p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 原生 eager | 000 | 869.24 | 680.29 | 2636.76 | 36.83 | 28.84 | 30.78 |
| + 点 1 HIP Graph | 100 | 1930.65 | 930.30 | 2653.46 | 24.08 | 4.81 | 5.05 |
| + 点 2 专家在线 FP8 | 110 | 3260.05 | 1493.61 | 2232.89 | 12.70 | 6.19 | 6.19 |
| + 点 3 QK norm/RoPE/KV 写入融合 | 111 | 2942.07 | 2009.64 | 1411.59 | 10.40 | 6.00 | 6.11 |

A512 三点增量分别为 **+36.75%**、**+60.55%**、**+34.55%**。四组 A512 全部 512/512 成功，四组 B50 全部 50/50 成功。固定负载三轮中位，A/B 各一轮。核对了原始请求集合哈希、实际并发峰值和逐请求输出长度。C 不在本轮测量范围。

## Trace 证据

固定 trace 用 32 条各 1024 输入，短窗生成 12 token。Decode 取实际 batch32 的中间采样步中位数。设备跨度、kernel 时间和 CPU 调用分别统计。吞吐来自关闭 profiler 的运行。

| 路径 | CPU launch / replay | GPU kernel 数 | decode 设备跨度 ms | decode kernel 合计 ms | 专家 kernel ms | 最大 16k prefill ms |
|---|---|---:|---:|---:|---:|---:|
| eager | 430 / 0 | 430 | 35.56 | 11.72 | 7.31 | 525.46 |
| graph | 0 / 1 | 431 | 12.01 | 12.01 | 7.30 | 526.57 |
| expert-fp8-pad | 0 / 1 | 431 | 6.78 | 6.78 | 2.00 | 386.94 |
| p123 | 0 / 1 | 312 | 7.94 | 7.94 | 3.89 | 361.29 |

1. **HIP Graph** 减少主机逐算子提交。eager 每步 GPU kernel 合计约 11.72 ms，设备时间轴跨度约 35.56 ms。Graph 以一次 replay 提交，kernel 数大致相同，跨度降到约 12.01 ms。
2. **专家 FP8** 针对 Graph 后设备时间最高的专家 GEMM。在线量化加载 28 层专家，保留 BF16 KV 和 dense。原 BF16 CK 两阶段换成 AITER per-token FP8 路径，decode 专家时间约 7.30 → 2.00 ms。兼容补行仅用于专家算子的小 M，补位路由权重为零。
3. **QK norm/RoPE/KV 写入融合** 用原生 AITER kernel 合并 attention 前处理与写缓存。固定 trace 的最大 prefill 批 kernel 数从 517 降到 398，设备时间约 386.94 → 361.29 ms。固定 decode 的专家 kernel 耗时上升，整步也变慢，固定 256 输出吞吐从 3260.05 降到 2942.07 tok/s。第三点按变长场景 A 的收益选择。

额外用同一组 32 条变长输入补抓前两点与三点累加版。Prefill 总时长按本次 trace 的 GPU 标注区间相加。

| 路径 | prefill 区间数 | prefill GPU span 合计 ms | decode GPU span 中位 ms |
|---|---:|---:|---:|
| p12 | 6 | 3440.16 | 7.63 |
| p123 | 7 | 2157.98 | 8.13 |

变长 trace 把第三点的主要收益定位到分块 prefill 续接。旧 NHD 写入路径在 gather 前对整层 K/V 缓存执行 permute().contiguous()。原生融合直接写入 gather 能读取的 SHUFFLE 布局。虽然前缀缓存关闭，分块续接仍会访问已有 KV 并触发该分支。

同一组 32 条共处理 84,186 个新 prefill token。点 1+2 的 prefill 中有 168 次超过 1 ms 的 direct-copy kernel，合计 1145.45 ms。三点版没有这种大复制，全部 direct-copy 时间由 1173.48 ms 降到 0.76 ms。固定 1024 输入较少触发续接路径，因而第三点的固定负载结果与变长 A 不同。

复制次数、时间与输入长度核对（开发资料 server-step0/native-speed-ablation/mixed-cache-copy-analysis.json）

## 候选与淘汰记录

BF16 CK 排序与启发式选核分别为 1904.14 和 1898.64 tok/s，Graph 对照为 1930.65。专家 FP8 先遇到加载器问题，再遇到 batch16 的 CK stage2 形状限制。大图桶方案和局部补行固定负载均约 3260 tok/s，采用局部补行保留原生小批图桶。

| 点 2 上再增加的候选 | A200 tok/s | 平均 TTFT ms | 平均 TPOT ms |
|---|---:|---:|---:|
| expert+qk-fused | 1922.30 | 1376.22 | 10.16 |
| expert+prefill64k | 1715.83 | 1884.74 | 10.11 |
| expert+prefill48k | 1648.78 | 2023.25 | 10.33 |
| expert+prefill32k | 1553.39 | 2192.85 | 10.85 |
| expert+dense-fp8 | 1475.09 | 1957.34 | 12.73 |
| expert-fp8-pad | 1441.61 | 2132.34 | 12.52 |
| expert+kv-fp8 | 1349.31 | 2401.95 | 12.99 |
| expert+prefill8k | 1168.14 | 2413.15 | 16.52 |

64k prefill 预算是最强的备选方向，完整 A512 得到 **1769.43 tok/s**。QK 融合的 **2009.64 tok/s** 排在其前。投影 FP8 的 TTFT 有改善，A200 吞吐收益较小。KV FP8 与 8k prefill 预算的 A200 吞吐下降。采样 kernel 在代表步约 0.048 ms，优先级低。

[完整实验过程和失败重试](decision-log.md) · 逐项配置（开发资料 server-step0/native-speed-ablation/profiles.json） · 可机读汇总（开发资料 server-step0/native-speed-ablation/final-summary.json） · 交付补丁（开发资料 server-step0/native-speed-ablation/speed-three-points.patch）

## 复现

run_speed_ablation.sh 有三个独立环境变量，按 Graph、专家 FP8、QK 融合顺序使用表中的 000、100、110、111。默认 111。

    MELLUM_SPEED_GRAPH=1 MELLUM_SPEED_EXPERT_FP8=1 MELLUM_SPEED_QK_FUSION=0 bash /workspace/results/mellum_native_ablation_20261106T103715Z/checkout/mellum/run_speed_ablation.sh

Profiler 原始文件

- eager 固定负载 trace（开发资料 server-step0/native-speed-ablation/eager/trace/trace/rank_0/Mellum2.1-12B-A2.5B-Thinking_ts_20261106_103844_676.pt.trace.json.gz）
- graph 固定负载 trace（开发资料 server-step0/native-speed-ablation/graph/trace/trace/rank_0/Mellum2.1-12B-A2.5B-Thinking_ts_20261106_104021_137.pt.trace.json.gz）
- expert-fp8-pad 固定负载 trace（开发资料 server-step0/native-speed-ablation/expert-fp8-pad/trace/trace/rank_0/Mellum2.1-12B-A2.5B-Thinking_ts_20261106_104922_494.pt.trace.json.gz）
- p123 固定负载 trace（开发资料 server-step0/native-speed-ablation/p123/trace/trace/rank_0/Mellum2.1-12B-A2.5B-Thinking_ts_20261106_111155_970.pt.trace.json.gz）
- p12 变长负载 trace（开发资料 server-step0/native-speed-ablation/p12/trace-mixed/trace/rank_0/Mellum2.1-12B-A2.5B-Thinking_ts_20261106_113313_337.pt.trace.json.gz）
- p123 变长负载 trace（开发资料 server-step0/native-speed-ablation/p123/trace-mixed/trace/rank_0/Mellum2.1-12B-A2.5B-Thinking_ts_20261106_113349_942.pt.trace.json.gz）
