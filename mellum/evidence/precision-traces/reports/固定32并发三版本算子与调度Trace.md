# 固定 32 并发的 Mellum 解码 trace

2026-11-04。在物理 GPU1 的同一张 MI308X 上依次启动 vLLM-ROCm、ATOM 朴素实验版和 ATOM 复用精度版。三者复用先前 1024 输入、256 输出性能测试的同一批 32 条文本和服务配置。诊断阶段先预热，再分别跑 24 输出 token 的轻量计时和 24 输出 token 的 profiler 采样。每个版本的两个诊断批次都完成 32/32 请求，服务端逐条确认长度。算子数据来自每个版本第 8 个满批解码前向，两个 ATOM 版本位置为 1031，vLLM 为 1031～1033。

## 请求调度与模型前向

下表是轻量计时轮的中位数，单位 ms。正式 256 输出的吞吐沿用上一轮。模型前向用同一流上的开始、结束事件计时，包含算子之间流的空闲。客户端间隔取流中非空文本块的稳态间隔。输出状态更新分别为 ATOM 的 postprocess 和 vLLM 的 update_from_output。

| 实现 | 正式输出 token/s | schedule | 输出状态更新 | 模型前向 | 相邻前向启动间隔 | 客户端文本块间隔 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM-ROCm | 751.92 | 0.529 | 0.153 | 34.59 | 37.10 | 37.10 |
| ATOM 朴素实验版 | 221.07 | 0.142 | 0.509 | 129.85 | 133.11 | 133.04 |
| ATOM 复用精度版 | 64.32 | 0.129 | 0.431 | 508.89 | 511.83 | 512.17 |

每个 ATOM 版本记录了 24 个满批单 token 调度样本，vLLM 记录了 21 个。三个版本的前向启动节奏都与客户端收到文本的节奏接近。调度选批与结果更新的中位数均低于 1 ms。单卡、并发 32、稳态 decode 的差距集中在模型执行路径；这些测量不覆盖长队列和 prefill 阻塞的其他负载。

## Attention 与 MoE

vLLM worker 的环境变量 VLLM_ROCM_USE_AITER 未设置，有效开关为 false，rocm_aiter_ops.is_enabled() 返回 false。服务日志显示 ROCM_ATTN 和 TritonExperts。以下次数均按一次满批解码步汇总 28 层，GPU 时间按 launch correlation 匹配 CPU 算子或模块后累计。

| 模块 | vLLM-ROCm | ATOM 朴素实验版 | ATOM 复用精度版 |
|---|---|---|---|
| Attention 主算子 | 21 次滑窗 Triton kernel_paged_attention_2d，合计 2.73 ms。7 次全局 _rocm_C::paged_attention 启动 14 个 HIP kernel，合计 0.29 ms | 28 次分组 SDPA，attn_fwd 合计 6.29 ms | 672 次显式 mask 路径 + 224 次 flash 路径，896 个 attn_fwd 合计 91.28 ms |
| Attention 的 KV 处理 | 按页表直接算 attention，vllm::reshape_and_cache 写入缓存 | 用连续 KV 增量写入，整批共用一次分组 SDPA/层 | aten::copy_ 对应 GPU 时间 40.93 ms，aten::index 对应 30.95 ms。其中 1,344 个 BF16 copy kernel 与 672 次显式 mask 的 K、V 头展开吻合 |
| MoE | 56 个 fused_moe_kernel，共 4.38 ms；另有 28 次路由矩阵乘、topk_softmax、moe_align_block_size 与 moe_sum 各 28 次 | 1,154 次 mm，含 28 次路由矩阵乘和 1,126 次专家矩阵乘；另有 563 次 SiLU 和 563 次 index_add_ | 1,126 次 mm，含路由投影；逐专家 where/nonzero 各 549 次，SiLU 与 index_add_ 各 549 次 |
| Attention / MoE 的 GPU kernel 数 | 511 / 224 | 1,590 / 3,938 | 16,940 / 8,106 |

复用精度版在每层按 32 个请求分别收集分页 KV、计算可见范围和 SDPA。超过 1024 窗口的 21 层走显式 mask，展开 4 个 KV 头到 32 个查询头。朴素实验版已经把 SDPA 合并到每层一次，共享同一批的 KV 读取计划，但专家输入打包后仍由 Python 按活跃专家分别发起矩阵乘、激活、加权、累加。vLLM 在专家维度组织两个矩阵乘阶段，每层两次 Triton 专家 kernel。

## 模型内部的发射间隙

| 详细 trace 内观测，ms/采样步 | vLLM | 朴素实验版 | 复用精度版 |
|---|---:|---:|---:|
| 带 profiler 的整段模型前向 | 59.15 | 199.70 | 828.97 |
| GPU kernel / 拷贝 / 清零的活动区间并集 | 13.07 | 48.87 | 272.40 |
| 第一至最后一个 GPU 活动之间的空闲间隙合计 | 39.64 | 150.57 | 556.27 |
| 其中下一活动对应的 HIP 调用尚未进入的间隙 | 29.31 | 95.88 | 323.13 |

三个采样步的设备活动均出现在各自的单个流上。最后一行按上一 GPU 活动结束到下一活动对应 HIP 调用进入之间的时间计算，并限制在实际空闲区间内。它表明 CPU 发射和操作组织确实在模型内部留下空档。Profiler 会放大大量小算子的 CPU 开销，因此日常速度以轻量计时和 256 输出正式成绩为准。

## 复核材料

服务器目录为 `/workspace/results/fixed32_trace_20261104T122634Z_iQFm`。本地材料见同一工作区的 `server-step0/fixed32-three-traces`。每个版本子目录都有 `decode.trace.json.gz`（可用 Perfetto 打开）、`control-timeline.json`（同一主机单调时钟的调度、模型和客户端事件）、`analysis.json`（完整 kernel 名称、次数、时间与 CPU 算子归因）及 `launch-gaps.json`。调度时间轴和算子 trace 属于分别执行的两轮。三个版本均在相同输入上独立生成，对应位置的路由或 token 可以不同。vLLM 使用配套 PyTorch 2.13，ATOM 使用冻结的 PyTorch 2.10。
