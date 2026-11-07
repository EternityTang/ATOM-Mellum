# 三点速度版的 decode 回退与 V 缓存布局修正

2026-11-06。在单张 MI308X 的 GPU0 上，对比 Graph + 专家 FP8、三点原版、三点加 V 缓存视图修正。使用冻结 Torch 2.10 与同一 checkpoint。现有三点版和实验候选分属独立工作树。

## 从 Trace 到专家回放

| 固定 32 路、1024 输入的 Trace | 生成方式 | GPU decode 中位 ms/步 | 专家中位 ms/步 |
| --- | --- | ---: | ---: |
| Graph + 专家 FP8 | 自然生成 | 6.804 | 2.005 |
| 三点原版 | 自然生成 | 7.814 | 3.747 |
| Graph + 专家 FP8 | 固定生成前缀 | 6.447 | 1.922 |
| 三点原版 | 同一固定前缀 | 5.861 | 1.995 |
| 三点 + V 视图修正 | 自然生成 | 5.979 | 1.999 |

固定前缀是一段预选文本的 token，在采样后按位置覆写。两个配置使用相同的 32 条输入，路由采样批逐步核对了模型读到的 token ID 和位置。路由取样与上表计时分开。自然生成的 9 个满批步骤中，真实 AITER 排序结果显示，每步 28 层的有效专家块数中位 **560 → 1306**，块大小 32。

在同一层、同一个 HIP Graph、同一套权重中交叉替换 hidden 和路由。自然生成样本的上下文长度为 1029 时，前两点的 hidden 配其自身路由是 **73.40 µs**，只换为三点版的路由是 **141.71 µs**。用三点版的 hidden 配前两点路由得到 **73.41 µs**，配自身路由是 **141.72 µs**。另两组位置同样复现。固定 hidden 做合成路由，40 个活跃专家时为 **79.99 µs**，41 个时为 **136.31 µs**。这轮同图回放包含排序、量化和专家执行。

用冻结 AITER 现有 kernel 试选型时，非 persistent 32×448 把三组真实分散路由的中位从 144.08 降到 131.92 µs，但相对原算子输出的最大相对 L2 达 0.690。32×128 更慢且最大相对 L2 达 0.222。两条 multix 和 32×256 发生 GPU 访问错误。各候选单独进程运行，保留失败日志。回放脚本首次从 AITER 包入口导入 `fused_moe` 时拿到模块对象，修正为从实现模块导入函数后重跑。

## 缓存读写的布局错误

融合 QK 路径通过五维 `[page, KV head, page_size/x, head_dim, x]` 视图写 V。原生 ATOM 滑窗 decode 把四维 `[page, KV head, head_dim, page_size]` 视图传给 Gluon。冻结 Gluon 根据视图维数选择地址计算，因而原路径在同一块内存上用错了读地址。候选只在 SHUFFLE 且传入 V 为四维时给 Gluon 传正确的五维视图。该操作复用底层存储。全局 attention、原融合写入与 prefill gather 沿用各自原路径。

用融合 writer 实写 BF16 缓存、打散物理页顺序，比较同一份 Q/K/V 的滑窗 attention。7 个长度覆盖页边界 16 和滑窗边界 1024，写入的 V 均逐元素匹配。按原四维视图读时，对 FP32 attention 参考再转 BF16 的相对 L2 为 **138.35%–143.09%**，修正视图后为 **0.192%–0.253%**。自然生成的整步 Trace 同时回收了专家耗时。这里验证的是滑窗缓存组件以及速度场景，整模型对 HF 的 token 一致性有单独口径。

原始依据：路由与交叉回放（开发资料 server-step0/native-speed-moe-replay/natural-replay-summary.json）、固定前缀回放（开发资料 server-step0/native-speed-moe-replay/forced-replay-summary.json）、布局对照（开发资料 server-step0/native-speed-moe-replay/value-cache-check.json）、独立修正补丁（开发资料 server-step0/native-speed-moe-replay/layout-fix.patch）。

## 自然生成的性能对照

| 同卡指标 | 三点原版 | 三点 + V 视图修正 |
| --- | ---: | ---: |
| 固定 1024 输入、256 输出、32 并发，三轮中位 tok/s | 2942.31 | **3604.38** |
| 完整 A512 输出 tok/s/GPU | 2010.94 | **2026.14** |
| 完整 A 平均 TTFT（ms） | 1391.72 | 1422.13 |
| 完整 A 平均 TPOT（ms） | 10.47 | 10.23 |

固定负载提高 **22.50%**，修正版最慢一轮也比原版最快一轮快。完整 A 两组各跑一轮，均 512/512 请求完成，实际峰值并发 32，每条输出 256，请求及输入长度的集合哈希一致。A 吞吐提高 **0.76%**，TTFT 略高、TPOT 略低，需要多轮才能判断 A 的吞吐增量是否稳定。这轮的主要成果是恢复正确的 V 读取和固定短输入 decode 的性能。

使用上一轮同一组 32 条变长输入补抓 Trace，修正版处理了 84,186 个新 prefill token，GPU prefill 跨度合计 **2158.95 ms**，上一轮原三点版为 **2157.98 ms**。修正版的 direct-copy 总计 **0.77 ms**，超过 1 ms 的次数为零。三点融合原本省下的整池复制仍然省掉了。这组变长样本的 decode 设备中位为 6.86 ms/步。

候选基于原三点版，在独立工作树内只增加 Gluon 入参的视图修正。服务器候选入口为 `/workspace/results/mellum_speed_moe_replay_20261106T130400Z/layout-checkout/mellum/run_speed_ablation.sh`。定版工作树的补丁哈希仍为 `4032eef28ba32877eee1597d7edbfb886ea9b1d90af96697f3f70ea147a05bd3`。用于回放的 tap 和强制前缀在另一工作树，修正候选服务走自然生成。

原始依据：性能及变长复制汇总（开发资料 server-step0/native-speed-moe-replay/validation-summary.json）、原始 Trace、路由快照、评测数据库和失败日志（开发资料 server-step0/native-speed-moe-replay/evidence.tar.gz）。归档 SHA-256 为 `8f402d9bf741d25bd3fc08e060c183b0ae9e1b942747a53a7bdebd31648e65b4`。回放用单层 FP8 权重保存在服务器实验目录的 `plain-forced-tap/trace/expert-weights.pt`，可用同一 checkpoint 和取样脚本重新生成。
