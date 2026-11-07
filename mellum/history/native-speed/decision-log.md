# Mellum 原生速度版三点消融工作记录

实验使用冻结 Torch 2.10、单张 MI308X、同一模型和固定原生 AITER 入口。优化前基线为原生 eager。它是本轮工程对照，题面官方朴素基线有单独定义。用户指定这一轮只测速度、延迟和生成完成情况。

## 第一轮 先解决 decode 派发开销

在同一组 32 条、每条 1024 输入 token 上，用 ATOM 的短窗 profiler 各录 12 个输出 token。捕获包含 prefill 和稳定 decode。另跑无 profiler 的 256-token 固定负载，每配置三轮取中位数。分析选中 32 路 decode 中间采样步，GPU kernel 时间按设备事件，主机调用按同一模型范围。CPU/GPU 时间分开列。

- 无 Graph 时，decode 每步 430 次 launch，GPU 时间轴跨度中位 35.56 ms，GPU kernel 合计约 11.72 ms。跨度与 kernel 时间的差值包含等待和派发空档。
- 开 Graph 时，每步 1 次 replay，GPU kernel 仍约 431 个，合计约 12.01 ms。GPU 时间轴跨度约 12.01 ms，CPU 模型范围约 0.24 ms。
- 最大 16384-token prefill 的设备时间约 525–527 ms，两条路径都由原生 eager prefill 执行。Graph 的收益目标为 decode。
- 选择原生 decode HIP Graph 作为点 1。其余后端设置保持原生 AITER。后续每个候选与同一 Graph 配置比较。

## 第二轮 重新定位设备时间

Graph 后，decode 中位约 7.30/12.01 ms 花在专家 GEMM，最大 prefill 约 315.46/526.56 ms。Dense GEMM 和 attention 次之。优先检查专家路径。

本轮筛选三条互斥方向。专家在线 FP8 用同一 checkpoint，覆盖专家权重，其他层保留 BF16。CK 路由排序和 AITER 默认启发式选择分别作为 BF16 替代。各方案保持 Graph 和其余设置，分别测试完整生成 256 token。

专家单独 FP8 的第一次启动在权重后处理失败。原 loader 在开启在线量化时仍对未匹配规则的 BF16 dense 层调用量化函数。加入只在目标为 QuantType.No 时跳过的边界判断，保留失败记录，使用原样专家匹配规则重试。该调整使专家与 dense 的量化开关可分别消融。

CK 排序中位 1904.14 tok/s，默认启发式选核 1898.64 tok/s，均低于 Graph 对照 1930.65 tok/s。专家 FP8 第二次可完成权重量化，但 batch16 图捕获进入 CK stage2 并报 GEMM 形状不支持。batch24/32 成功捕获。因此下一试使用 [24,32] 的现成图桶，并加一组 BF16 同图桶的对照，分离图桶变化与专家量化。另试仅量化 attention 的 QKV/O 投影，作为不同瓶颈的备选。

专家 FP8 + [24,32] 图桶通过固定负载，先证明了大批形状有收益。更小的请求批若都用 24 行图，会把投影、attention 和 norm 也一起补大。追加局部兼容方案，只在遇到这组 gfx942 FP8 专家的小 M 形状时把专家输入和路由补到 24 行，补位路由权重为零，返回时裁回有效行。这样原生图桶仍覆盖 1/2/4/8/16/24/32，也覆盖变长 prefill 偶尔切出的小 token 批。

## 第三轮 复测 trace 后按变长负载筛选

Graph 加专家 FP8 局部补行在固定 32 条三轮中位达到 3260.05 tok/s，与大桶方案的 3260.41 基本一致。采用局部补行方向，图桶保持与点 1 相同。对应 trace 中，专家 GEMM 的 decode 中位从 7.30 降到 2.00 ms，GPU 整步 12.01 降到 6.78 ms。最大 prefill 中专家 GEMM 从 315.46 降到 174.33 ms。

此时 dense GEMM 在最大 prefill 仍占 104.18/386.93 ms，decode 约 1.59/6.78 ms。attention 为 50.20/386.93 ms 和 1.18/6.78 ms。若将 dense 时间减半，采样 prefill 与 decode 的设备时间可分别少约 13% 和 12%，这是筛选前的估算。进一步测 attention 投影 FP8、KV FP8、原生 QK 融合与 8k/32k prefill 内部 token 预算。以同一份 200 条变长请求、并发32、输出256筛选，并用完整 A512 复核。调度试验只改 engine 的内部 token 预算。

Dense FP8 最初被 loader 拒绝，原因是 gfx942 的 FP8 权重类型为 e4m3fnuz，原断言只接受 e4m3fn 和 FP4。重试允许原生 dtype，已经量化成 fnuz 的权重跳过重复格式转换。其他 dense 层仍走原 BF16 配置。

变长 A200 上，投影 FP8 为 1475.09 tok/s（+2.32%），TTFT 下降而 TPOT 略升。KV FP8 为 1349.31 tok/s（−6.40%），暂淘汰。缩小 prefill 预算到 8k 为 1168.14 tok/s（−18.97%），扩大到 32k 为 1553.39 tok/s（+7.75%）。追加 48k 和 64k，检验更大批次是否继续提高吞吐。采样作为派发小项也在 trace 中检查过，代表步的采样 kernel 约 0.048 ms，相对 decode 的 6.78 ms 优先级低。

48k 与 64k prefill 内部预算将 A200 提高到 1648.78 与 1715.83 tok/s，仍低于原生 QK norm/RoPE/KV 写入融合的 1922.30 tok/s。后者保持默认 16k 预算，平均 TTFT 为 1376.22 ms，TPOT 为 10.16 ms。本轮按单卡场景 A 速度选择 QK 融合做第三点，最终完整 A512 与 B50 分别验证，保留所有预算试验。

先前纯 BF16 Graph 的固定负载里，原生 QK 融合曾从约 1927 降至 1351 tok/s。因此这一轮不把融合认作普遍加速，而是在专家 FP8 的累加配置上，重新抓固定负载 trace、验证固定负载与完整 A512。

第三点用独立启动入口复跑，固定 1024/256/32 三轮中位为 2942.07 tok/s，低于点 1+2 的 3260.05。固定 trace 的 GPU kernel 数每步 431→312，最大 prefill 批 386.94→361.29 ms，但 32 路固定 decode 的设备跨度 6.78→7.94 ms，同时专家 kernel 耗时从 2.00 升到 3.89 ms。已记录关联，不将专家时间变化直接归因于某一个 kernel 内部机制。

完整变长 A512 中，前两点 1493.61、三点 2009.64 tok/s。两组各完成 512 请求，原始输入均值相同，每条输出 256。仅 Graph 的 A512 为 930.30。单流 B 上，三点版 TTFT p50 48.41 ms、TPOT p50 6.00 ms，分别对比仅 Graph 的 67.89 与 4.81 ms。以 A 为目标选择第三点，同时保留固定负载和 B 的反例。另补抓同一组 32 条变长输入的两条路径，并给最强 64k 预算备选跑完整 A512。

三点版的交付补丁只保留专家 FP8 的加载跳过判断和小批兼容。Dense FP8 试验用到的额外 fnuz loader 修改保存在 all-trials.patch，最终三个开关的路径不需要这些试验代码。

变长 trace 给出了第三点在 A 上更有效的机制。两组 32 条输入共 84,186 个 prefill token。点 1+2 的 prefill GPU 时间轴合计 3440.16 ms，三点版 2157.98 ms；稳定 decode 的采样中位反而是 7.63→8.13 ms。点 1+2 有 168 次大于 1 ms 的 direct-copy，合计 1145.45 ms，三点版大复制为零。源代码中分块续接进入 has_cached 后，NHD 写入路径要把整层 K、V 缓存 permute().contiguous() 才能 gather。QK 融合在写入时采用 SHUFFLE，复用已有 gather 无需这一步。该证据比只看固定 decode kernel 数更能解释变长 A 的吞吐增量。采样仍未确定固定 decode 中专家 kernel 时间波动的具体成因，因此保留实测取舍。
