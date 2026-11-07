# 速度版第三点复核：按原布局读取分页 KV

前两点保持 HIP Graph 和专家 FP8。实验对比直接 gather 与上一轮第三点 QK norm/RoPE/KV 写入融合。新增路径仅在 BF16 packed K 和 ASM/NHD V 的布局匹配时启用，由现有 gather kernel 按原地址读取所需页。组件测试覆盖页大小16/32/64、4/8个头、128/192维和非零起点，24/24逐位一致。

| 第三点 | 固定 1024/256/32 三轮中位 tok/s | 完整 A512 tok/s | A 平均 TTFT ms | A 平均 TPOT ms |
|---|---:|---:|---:|---:|
| 前两点对照 | 3264.27 | 1493.61（上轮） | 2232.89 | 12.70 |
| 直接 gather | 3249.18 | 1899.37 | 1522.99 | 10.89 |
| 原 QK 融合 | 2950.19 | 2005.22 | 1425.03 | 10.38 |
| 直接 gather + 已知长度 | — | 1916.30 | 1175.64 | 12.11 |
| 原 QK 融合 + 已知长度 | — | 2019.03 | 1112.23 | 11.50 |

同卡 A512 四组输入集合逐条核对，活跃峰值32，512/512均生成256 token。直接 gather 对原第三点的 A512 变化 **-5.28%**，固定负载 **+10.13%**。A512 每组各一次。

变长 trace 使用同一组 32 条请求，共处理 84186 个新 prefill token。直接 gather 的 prefill GPU span 合计 **2291.80 ms**，大于 1 ms 的 direct-copy kernel **0 次**。前两点对照为 3440.16 ms 和 168 次。原 QK 融合为 2157.98 ms 和 0 次。

直接 gather 使用 MELLUM_SPEED_DIRECT_KV=1，同时设置 MELLUM_SPEED_QK_FUSION=0。已知长度用 MELLUM_SPEED_KV_KNOWN_SIZE=1。其余两个点保持开启。实验从上一轮代码在独立工作树复现。

| Trace 路径 | repeat_interleave 内同步或阻塞拷贝次数 | 对应 CPU 调用合计 ms | prefill GPU span ms |
|---|---:|---:|---:|
| direct | 280 | 1945.23 | 2291.80 |
| direct-known | 0 | 0.00 | 2289.29 |
| qk-known | 0 | 0.00 | 2119.14 |

直接 gather 的变长 trace（开发资料 server-step0/native-speed-direct-kv/direct/trace-mixed/trace/rank_0/Mellum2.1-12B-A2.5B-Thinking_ts_20261106_123235_363.pt.trace.json.gz） · 可机读数据（开发资料 server-step0/native-speed-direct-kv/final-summary.json） · 第三点代码差异（开发资料 server-step0/native-speed-direct-kv/direct-kv-only.patch）

原 QK 融合在本轮仍是较快的可复现 A512 路径。加已知长度的单轮吞吐高 0.69%，TTFT 下降但 TPOT 上升。结合 trace 中 GPU prefill 时间变化较小，先保留为候选选项。直接 gather 是固定输入负载的较快替代，其完整变长 A512 吞吐低于原 QK 融合。
