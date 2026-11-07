# Mellum / ATOM 速度版：定版与开发全记录

定版日期：2026-11-06。本文记录从最初移植、用 vLLM 找差距，到原生速度版三轮 Trace、逐点消融和最终取舍的全过程。精度实验只在解释路线为何改变时引用，所有成绩均标明属于哪个版本和测试口径。

## 1. 最终定版

速度版采用 **原生 AITER + decode HIP Graph + 专家在线 FP8 + 原生 QK norm/RoPE/KV 写入融合**。三个开关按 Graph、专家 FP8、QK 融合顺序记作 `111`。单张 MI308X、冻结 PyTorch 2.10、编译等级 0；BF16 KV，dense 保持 BF16；不使用 MTP，关闭前缀缓存，保留分块 prefill，内部 token 预算 16384。图捕获批大小为 1/2/4/8/16/24/32。权重仍来自同一个 Mellum checkpoint，专家在加载时量化。

| 累加配置 | 固定 1024/256/32 中位 tok/s | 完整 A512 tok/s/GPU | A 平均 TTFT / TPOT（ms） | B TPOT p50 / p99（ms） |
| --- | ---: | ---: | ---: | ---: |
| `000` 原生 AITER eager | 869.24 | 680.29 | 2636.76 / 36.83 | 28.84 / 30.78 |
| `100` + HIP Graph | 1930.65 | 930.30 | 2653.46 / 24.08 | 4.81 / 5.05 |
| `110` + 专家在线 FP8 | **3260.05** | 1493.61 | 2232.89 / 12.70 | 6.19 / 6.19 |
| `111` + QK 融合，最终默认 | 2942.07 | **2009.64** | **1411.59 / 10.40** | 6.00 / 6.11 |

按完整变长场景 A 选择最终版本，三个点的逐步增量分别为 **+36.75%、+60.55%、+34.55%**，合计约为同版原生 eager 的 **2.95 倍**。若只看固定 1024 输入的满批吞吐，`110` 更快；若只看单流 B 的 TPOT，`100` 更低。最终 `111` 的 B 首 token 中位数是 48.41 ms，`100` 是 67.89 ms，首 token 和后续 token 的取舍也分别记录。

固定负载每配置跑三轮取中位；完整 A512 和 B50 各一轮，每组全部完成，每条实际生成 256 token。核对原始输入集合、活跃并发峰值（A 为 32，B 为 1）和逐条输出长度；性能轮不启用 profiler。这里的 `000` 是为消融建立的**原生工程对照**，题方公布的朴素基线另有定义。本轮 `111` 未跑 C、HumanEval 或对 HF 的 token 验收。按用户决定，它是速度版定版；精度版和题面 Step 0 的正确性门槛各自保留，不能用其他版本的成绩补齐 `111`。

## 2. 起点：先跑通模型，再确定什么值得复用

最初基于适配前的 ATOM `26f23a0b` 写了 Mellum 专用 eager 适配（`450ec964`），先建立可以加载、推理和逐层观测的对照。镜像的 Transformers 5.2.0 没有原生 Mellum 类，早期通过回移 HF 上游实现建立参考；升级 Transformers 获确认后改用原生 Mellum。PyTorch 仍冻结为 2.10。回移参考、原生 HF SDPA、原生 HF eager 是不同口径，历史数字不能互换。曾把「eager experts」简称为「HF eager」，当时 attention 实际仍走 SDPA；后续分别记录引擎执行模式、attention 后端和专家后端，并将验收参考定为冻结 Torch 2.10 的原生 HF eager 逐条生成。

用户否决继续扩写自定义模型，要求保留 Git 历史，从原始 ATOM 重建（历史接入 `d0a63b6`）。新适配复用 Qwen3-MoE 骨干、MoE 容器、RMSNorm 和权重加载，沿用 GPT-OSS/MiMo 的逐层滑窗接入。Mellum 只选择自己的层类型与 RoPE：28 层里 21 层用 1024-token 滑窗和普通 RoPE，7 层是全局 attention 并按模型配置使用 YaRN；专家是 64 选 8。检查过 RoPE、层选择和专家权重映射，权重审计中 5631 个源 tensor 映射到 255 个打包参数。MTP 只保留历史接入方向和缺权重保护；checkpoint 中没有发现可用预测头，所以本定版不启用，也不计优化点。

这一阶段有两项真实问题，必须和后来的加速分开：

- **SWA 边界。** 最初验证只检查了过期 key 是否排除，漏检最早合法 key，因而曾误判 decode 正确。加带标记 key 的用例后确认，AITER prefill 接收的是左距离 `W-1`；冻结 Gluon decode 的调用需传 `W+query_len`。模型配置仍为 1024。补充 GPU 边界用例后保留该修复，作为移植正确性工作。最初的困难短题未触及窗口，不能用它们判定 SWA 修复的收益。
- **采样和 FusedMoE 是两件事。** 无 top-k/top-p 时旧 sampler 有路径跳过 greedy 判断；零温度还会被 runner 钳到 EPS。经用户批准，历史精度路线参考 ATOM v0.1.7 做整批和混合温度的 greedy 修复。修复后，同一条 271-token 输入重复 16 次，第六个输出仍会变化；检查的 96 个位置均符合各自当次 logits.argmax。逐层最早变化在 L00 MoE 输出，固定 hidden、权重和路由重放融合专家 24 次，后 23 次都与首次不同，最大绝对差 0.03125。证据定位到当时的 AITER 融合专家执行路径，没有定位到内部具体指令或归约机制。早期探针曾错把引擎额外预测也算入客户端请求，结论以按 prefill 边界重分组、逐 token 核对后的记录为准。

为了继续诊断，精度路线保留容器，先换成 Torch 专家。单条由 6/64 到 62/64，完整旧代码题集却只有 9439/12800。另一个样本把归一化后的路由权重转回 BF16 后从 7/64 到 46/64，也没有解决全链路差异。L00 norm 重放发现当时原生转换与 HF 舍入不同；局部 RNE 一度使最差单条从 5/64 到 51/64，固定困难十条却从 356/640 降到 320/640，因此不默认开启。RoPE 诊断 hook 也曾漏掉滑窗层、覆盖共享模块记录，相关结论作废并用实际 rotary cache 重抓。最终把「构造、映射和窗口」与「算子数值路径」分开验证。

原始依据：全项目逐阶段复盘（开发资料 项目全流程复盘_STAR与决策记录.md）、SWA 修复（开发资料 SWA修复说明.md）、greedy 修复（开发资料 Sampler_v0.1.7修复说明.md）、原生输出不稳定定位（开发资料 原生输出不稳定定位.md）。

## 3. 朴素基线和 vLLM：先排除错误的瓶颈假设

用户要求按题面做一个单卡、eager、编译等级 0、Torch 循环专家、显式 mask SDPA、MTP 关闭的本地朴素基线。它用于对照，题方没有提供移植源码。朴素版也会分叉：相同 Q/K/V 在 ROCm 上，显式加性 mask 和等价 causal/GQA 调用触发了不同 SDPA 后端。只在等价场景去掉显式 mask、保留真正需要的因果和滑窗边界后，固定十条达到 640/640，旧 SDPA 参考的代码 200 条达到 12800/12800。这个实验校验了具体算术路径，不能外推到最新 HF eager。

当时尝试批量 SDPA、打包专家输入、连续 KV，以为减少历史重收集就有机会对齐题方性能。GPU1 实测 A512 只有 111.87 tok/s，低于题方公布的 638.28；C 因连续与分页缓存共存 OOM。加缓存预算后不再 OOM，但 200 条仅 1 条完成、199 条超时。连续 KV 因吞吐、内存和完成率三项证据退出交付候选。

用户指定先跑 vLLM-ROCm，再以同一张 GPU1 和固定 1024/256/32 抓三版 Trace。vLLM 751.92、朴素实验版 221.07、复用精度版 64.32 tok/s；模型前向分别为 34.59、129.85、508.89 ms，调度和状态更新中位均低于 1 ms。复用精度版每步有 896 次 attention 调用，反复收集 KV、展开 KV 头，并逐专家发起小 GEMM；vLLM 直接读分页 KV，用分组专家算子。实查该轮 vLLM 的 AITER **未开启**，使用 ROCM_ATTN 和 TritonExperts。早期稳态差距主要在模型计算组织和发射；此结论不覆盖变长 prefill 或动态到达场景。vLLM 的独立 Torch 2.13 环境用来参考实现和性能，未作为最终 ATOM/HF 的冻结环境。

原始依据：朴素基线实现（开发资料 朴素基线实现说明.md）、GPU1 朴素对齐实验（开发资料 GPU1朴素基线对齐实验.md）、三版本算子与调度 Trace（开发资料 固定32并发三版本算子与调度Trace.md）。

## 4. 旧速度路线：600+ 怎么来的，为什么后来换方向

按 vLLM 的计算组织，先分开试了按可见 KV 长度合批、直接分页 attention 和分组专家。直接分页＋分组 GEMM 的候选到 **438.23 tok/s**，困难十条自由生成只匹配 **356/640**。同一 HF 前缀上的下一 token 可以匹配 634/640：这解释了局部近似如何经过自回归放大，不能代替自由生成验收。随后单测 RoPE 跨层复用和 AITER 写 KV，组合在旧精度路线到约 232–235，在旧速度路线到约 543–555 tok/s；单独复用 Torch 地址或单独换写入算子在精度组没有收益，因此没有把所有小改动都算作独立成果。

历史 `packed` 版组合了原生骨干、适配的直接分页 attention、分组专家、packed QKV、融合 QK 处理与 AITER KV 写入。按实际 decode 批大小选择路径：1–7 行走小批精度路径，8 行起走快速路径。按当时「逐条精度和并发性能分别验收」的要求，固定 1024/256/32 三轮中位 **684.02 tok/s**，旧代码 200 条逐条对当时 HF SDPA 为 12800/12800；同一批请求并发对 HF 逐条参考只有 **76.60%**，且当时 HF 自己分批对逐条也有差异。用户后来把验收明确为 A200 并发 32 对冻结 Torch 2.10 原生 HF eager 逐条参考，并另查约 16k 组，旧结果不能转用。旧 packed 的 HumanEval 151/164 也只属于该历史版本。

尝试给 packed 加回精度边界时，逐步恢复逐请求计算使速度明显下降。旧 S0–S5 短测从约 99.97 降到 22.94 tok/s；全量对齐的一版只在最后一层保留融合专家，A200 前 64 token 匹配 12800/12800，吞吐为 32.98 tok/s。这些短测和 A200 都不能与旧固定 1024/256/32 的 600+ 直接比较，也不足以让该候选替代原速度版。精度工作于是继续采用朴素逐请求参考、逐件接回原生组件和受控合批；速度工作保留原生吞吐目标，历史两个 Git 分支均保留。例如，之后独立的精度路线只合批滑窗 softmax 等中间操作，保留逐请求 QK、PV，A200 吞吐到 105.76 tok/s、12800/12800；这也不属于定版速度路线。

用户进一步要求：从开始适配前的提交另起速度工作树，直接使用框架组件，只看速度。这使我们能检验此前移植其他内核是否还必要。GPU0 同卡三轮固定负载复测，旧 packed **649.48**、原生 eager **857.35**、原生 AITER + HIP Graph **1926.90 tok/s**；初始原生 Graph 完整 A512 为 **929.69 tok/s**。同轮纯 BF16 的原生 QK 融合和 unified attention 只有 **1351.37** 和 **1355.52 tok/s**，所以初版先选择纯 Graph。早期 Graph 版另有 A200 对冻结 HF 参考的历史测量：4025/12800（31.45%），用强制 argmax 的诊断也未拉平。该数字属于初始 BF16 Graph，不能当作本次三点版的结果。

原始依据：仿 vLLM 解码实验（开发资料 Mellum仿vLLM解码优化实验.md）、RoPE 与 KV 跨层复用（开发资料 Mellum解码RoPE与KV跨层复用实验.md）、精度修复与速度验收（开发资料 精度提交与速度版验收.md）、精度版滑窗合批（开发资料 精度版滑窗Attention合批验收.md）、Git 双分支和历史（开发资料 Git双分支与历史保留.md）、初始原生速度版（开发资料 Mellum原生组件速度版.md）、初始 Graph 的 token 对照（开发资料 server-step0/native-speed/token-check/comparison.json）。

## 5. 三点怎么定：每做一点，重新抓一次 Trace

按题面，一点要针对可识别的瓶颈、自包含、有开关、可消融。移植、SWA 边界修正、历史 sampler/精度修复、启动烟测都不占三点。每轮用原生 profiler 接口抓短窗，固定 Trace 用预热后的 32 条 1024-token 输入生成 12 token，取稳定满批 decode 的中间步骤；CPU 调用、GPU 时间跨度和 kernel 时间分开。吞吐用关闭 profiler 的 256-token 运行确认。第三点补做变长 Trace 与完整 A512，因为固定短输入不能代表分块 prefill。

**点 1：先开 HIP Graph，消除 decode 的逐算子派发。** 没开 Graph 时，每步 430 次主机 launch、约 430 个 GPU kernel；kernel 合计 11.72 ms，首尾 GPU 时间跨度 35.56 ms。这个差值是设备时间线空档，不能都记成纯 CPU 算时。开启后每步一次 replay，GPU kernel 数仍约 431，跨度降到 12.01 ms；CPU 模型范围从约 35.46 ms 到 0.24 ms。最大 prefill 批约 525–527 ms，基本不变。由此选择原生 decode Graph；收益来自提交方式，没把它说成缩短了每个 GEMM。固定负载大幅增长而 A 首 token 平均没有改善，也与该证据吻合。

**点 2：Graph 后再找最大的设备热点，选择专家在线 FP8。** Graph 把发射空档压掉后，BF16 专家 CK 两阶段占固定 decode **7.30/12.01 ms**，最大 prefill **315.46/526.57 ms**。先试 CK 排序和改回启发式选核，固定中位为 1904.14、1898.64，低于对照 1930.65。专家 FP8 改用 AITER 的 per-token activation、per-channel weight 路径，只覆盖 28 层专家，dense 与 KV 仍 BF16。

1. 首次启动在加载阶段报 `Unsupported online quant: dtype=torch.bfloat16,type=QuantType.No`。加载器遍历了没有匹配量化规则的 dense 层。仅在目标规则为 `QuantType.No` 时跳过对应量化处理，重试确认 28 层专家完成量化。
2. 下一次在 batch16 的图捕获中触发 CK stage2 GEMM 形状不支持，batch24/32 能捕获。先用 `[24,32]` 图桶做对照，FP8 固定负载 3260.41 tok/s，同图桶 BF16 为 1923.90，确认主要收益来自专家路径，不是改变图桶。
3. 大桶会把小批的投影、attention、norm 一起补大，最终只在 gfx942 FP8 专家小 M 内部补到 24 行，补位路由权重置零，返回裁回有效行。恢复原捕获桶后固定中位 3260.05，完整 A512 1493.61；Trace 中专家 7.30→2.00 ms，整步 12.01→6.78 ms。

**点 3：重抓专家 FP8 的 Trace，用真实 A 负载选 QK 融合。** 此时 dense 在最大 prefill 约 104.18/386.94 ms，在 decode 约 1.59/6.78 ms；attention 分别约 50.20 ms 和 1.18 ms。先用同一组变长 A200 做筛选。以下是吞吐实验，A200 在这里不是 token 正确性测试。

| 第三点候选（前两点保持开启） | A200 tok/s | 决策依据 |
| --- | ---: | --- |
| 默认 16k 预算，作为对照 | 1441.61 | 原路径 |
| 仅 QKV/O 投影 FP8 | 1475.09 | +2.32%，首 token 改善、TPOT 略升；不采用 |
| KV FP8 | 1349.31 | −6.40%；不采用 |
| 内部 prefill 预算 8k / 32k / 48k / 64k | 1168.14 / 1553.39 / 1648.78 / 1715.83 | 64k 最强，完整 A512 1769.43；保留为备选 |
| 原生 QK norm/RoPE/KV 写入融合 | **1922.30** | +33.34%，完整 A512 **2009.64**；采用 |

投影 FP8 第一次被 loader 拒绝：gfx942 的原生 dtype 是 `e4m3fnuz`，旧检查只接受 `e4m3fn` 和 FP4。试验中放开该格式并跳过已是 fnuz 的重复转换，验证后仍未胜出；试验改动留在归档，不进定版补丁。采样曾是正确性热点，但速度 Trace 中代表步仅约 0.048 ms，相对整步 6.78 ms，不进入这一轮的前三。BF16 Graph 下融合曾变慢，因此这次必须在专家 FP8 上重新抓证据。

固定负载开第三点后，decode kernel 数 431→312，最大 prefill 386.94→361.29 ms，但固定 decode 从 6.78→7.94 ms，专家 kernel 观测为 2.00→3.89 ms；具体波动机制尚未查明。固定吞吐因此从 3260.05 降到 2942.07。变长 Trace 给出了完整 A 逆转的原因：同样 32 条输入，共 84,186 个新 prefill token，点 1+2 的 prefill GPU 跨度合计 **3440.16 ms**，三点版 **2157.98 ms**。前者有 **168 次**超过 1 ms 的 direct-copy（合计 1145.45 ms），后者为 **0 次**；所有 direct-copy 合计 1173.48→0.76 ms。稳定 decode 反而是 7.63→8.13 ms。

具体路径是分块 prefill 续接访问已有 KV 时，旧 NHD 路径在 gather 前对整层缓存做 `permute().contiguous()`。前缀缓存关闭也会走这条续接分支。融合内核直接把 KV 写成已有 gather 能读取的 SHUFFLE 布局，省掉整池重排。归为一个优化点，因为原生融合及其写入布局由同一个开关启停；内部 prefill 预算仍是 16k。这个证据支持「按 A512 选择融合」，不支持「融合令所有 decode 变快」。

原始依据：[三点消融报告与原始 Trace 索引](three-points-report.md)、[逐轮试验日志](decision-log.md)、可机读 Trace 汇总（开发资料 server-step0/native-speed-ablation/profiles.json）、变长复制分析（开发资料 server-step0/native-speed-ablation/mixed-cache-copy-analysis.json）、可机读性能汇总（开发资料 server-step0/native-speed-ablation/final-summary.json）。

## 6. 定版前再试一次：直接读分页 KV 能否替代第三点

发现第三点主要省在整池复制之后，最自然的备选是假如 gather 直接按原布局读取，只去复制，保留原 decode 路径。独立工作树新增对 BF16 packed K 与 NHD V 的按 stride 读取；页大小 16/32/64、KV 头数 4/8、维度 128/192 与偏移用例共 24/24 逐位通过。变长 Trace 中同样消除了全部大复制，prefill GPU 跨度到 2291.80 ms；它仍高于原融合的 2157.98 ms。直接 gather 没有合并 QK 的前处理，不能保留融合的全部 prefill 收益。

| 后续独立复核 | 固定 1024/256/32 中位 tok/s | 完整 A512 tok/s | A 平均 TTFT / TPOT（ms） |
| --- | ---: | ---: | ---: |
| 直接 gather | 3249.18 | 1899.37 | 1522.99 / 10.89 |
| 原 QK 融合 | 2950.19 | 2005.22 | 1425.03 / 10.38 |
| 直接 gather + 已知长度 | — | 1916.30 | 1175.64 / 12.11 |
| 原 QK 融合 + 已知长度 | — | 2019.03 | 1112.23 / 11.50 |

直接 gather 在固定短输入比原融合快 **10.13%**，但同轮 A512 慢 **5.28%**，所以保留为另一种负载的候选。另试给 `repeat_interleave` 传已知 `output_size`：直接 gather Trace 的 280 次相关同步/阻塞调用变为零，GPU prefill 跨度却只从 2291.80 降到 2289.29 ms。给原融合也加上它后，A512 单轮仅增 **0.69%**，虽改善 TTFT，TPOT 却从 10.38 升到 11.50 ms。因此定版继续采用原 QK 融合，不叠加这项尚未确认稳定收益的变更。没有把 2019.03 写成默认成绩，也没有改动原始三点补丁。

原始依据：[第三点复核及 Trace](direct-kv-report.md)、直接 gather 实验补丁（开发资料 server-step0/native-speed-direct-kv/direct-kv-only.patch）、复核可机读数据（开发资料 server-step0/native-speed-direct-kv/final-summary.json）。

### 反例留下的优化空间

回查固定输入的两份原始 Trace，把每次 Graph replay 的 28 次同名专家 kernel 按执行顺序分组，并用每步专家耗时合计和报告中的 9 个稳态样本一一核对。两版均执行同名 FP8 专家 kernel，每步 28 次。前两点版每层耗时中位约 69–74 µs；三点版有 26 层约 141–146 µs，其余两层约 80 和 83 µs。两版 attention 主算子都约 1.18 ms/步，norm 加 RoPE/KV 写入合计约 0.95→0.54 ms/步。变长 Trace 的满批专家耗时中位也从约 2.57→3.85 ms。固定满批有 32 行，未走少于 24 行的专家补行。证据把优先排查目标缩到专家执行，不支持「融合内核自身慢」「增加了专家调用次数」或「小批补行导致本次回退」。同名 kernel 不代表路由分布、内存状态或图执行条件相同。

最值得验证的是能否保留 prefill 的原生融合与零大复制，同时收回 decode 的专家耗时。先在同卡交替跑两配置，固定生成前缀和批大小，记录每层 top-k 分布及排序后的块数；再用同一份 hidden、路由和权重回放专家，区分输入引起的工作量变化与 Graph 中前序算子、内存或运行状态的影响。若需要按 prefill/decode 分开实现，两个阶段必须读写一致的 KV 布局；当前全局开关同时改变 QK 计算和 KV 写入格式，不能直接在阶段切换时翻转。直接 gather 的 +10.13% 固定吞吐已证实短负载有空间，尚未验证同一配置能同时保留原融合的完整 A 收益。此处只复核现有 Trace，未改定版或发起新跑分。

后续回放确认，三点原版的滑窗 decode 通过四维视图误读了融合写入的 V 缓存。给 Gluon 传入正确的五维 SHUFFLE 视图后，固定自然生成的专家耗时从 3.747 降到 1.999 ms/步，固定 1024/256/32 吞吐由 2942.31 升到 3604.38 tok/s。修正补丁保存在独立候选中，路由交叉回放、失败的 kernel 选型、attention 组件检查与完整场景 A 对照见[后续实验记录](layout-repair-report.md)。

2026-11-07，使用包含上述修复的同一份代码，补齐四组的 A512/B50/C200 消融，每组每场景各一轮，全部请求完成。最终三点版 A 为2038.47 tok/s/GPU，C 的 TTFT/TPOT p99 为695.48/11.67 ms。单独进行的最终三点版 A200 并发32对冻结 HF eager 逐条参考为3030/12800（23.6719%），约16k组110/256。两条独立 HF 重放均复现保存参考。本轮结果及环境、独立工作树补丁和归档校验见[修正后 ABC 消融与 token 验收](abc-token-report.md)。

## 7. 复现入口和归档

本地工作树 ATOM-Mellum-speed-ablation（开发资料 ATOM-Mellum-speed-ablation），分支 `codex/mellum-native-ablation`，基线 HEAD `26f23a0bdbf50b9010c07ee3a71171189723e767`，七处定版文件以工作树补丁保存。服务器工作树的 diff 已核对与归档补丁 SHA-256 一致，两个启动脚本都通过 Bash 语法检查。本次定版记录未新增提交或推送。历史 `codex/mellum-speed` 是 packed，`codex/mellum-native-speed` 是初始原生对照。

在现有容器运行以下命令启动定版（默认端口 8830/8831，使用现有冻结环境和模型）：

```bash
CANDIDATE_GPU=0 bash /workspace/results/mellum_native_ablation_20261106T103715Z/checkout/mellum/run_speed_ablation.sh
```

消融时显式传开关。例如复现前两点：

```bash
MELLUM_SPEED_GRAPH=1 MELLUM_SPEED_EXPERT_FP8=1 MELLUM_SPEED_QK_FUSION=0 \
  CANDIDATE_GPU=0 bash /workspace/results/mellum_native_ablation_20261106T103715Z/checkout/mellum/run_speed_ablation.sh
```

按同一顺序设为 `000/100/110/111`，分别对应表中四组配置。启动脚本能分别切换三点；本次逐点增量来自上述累加顺序，不代表已经测了所有八种组合。A/B 的原始评测汇总、请求记录与 profile 位于归档中。题面 A/B/C 定义及优化点规则见 PDF 原文提取（开发资料 PDF原文提取.txt）；上述命令是定版速度实验入口。

| 文件 | 用途 |
| --- | --- |
| [atom/models/mellum.py](../../../atom/models/mellum.py) 与 [atom/model_engine/model_runner.py](../../../atom/model_engine/model_runner.py) | Mellum 构造、分层配置和模型注册 |
| [atom/model_ops/attention_mha.py](../../../atom/model_ops/attention_mha.py) | 原生滑窗参数校正 |
| [atom/model_ops/linear.py](../../../atom/model_ops/linear.py) | 在线量化未匹配的 dense 层保持 BF16 |
| [atom/model_ops/moe.py](../../../atom/model_ops/moe.py) | FP8 专家小批局部补行 |
| [mellum/run_native_speed.sh](../../../mellum/run_native_speed.sh) 与 [mellum/run_speed_ablation.sh](../../../mellum/run_speed_ablation.sh) | 原生启动配置和三个独立开关 |

定版补丁及证据（SHA-256）：

- speed-three-points.patch（开发资料 server-step0/native-speed-ablation/speed-three-points.patch）：`4032eef28ba32877eee1597d7edbfb886ea9b1d90af96697f3f70ea147a05bd3`。
- 三点 Trace、评测和失败试验归档（开发资料 server-step0/native-speed-ablation/evidence.tar.gz）：`39268b245fe18a6fc1dc7c6ffe394942ead4d58490fcba11c64878e720dc9ed1`。与定版分开的 all-trials.patch（开发资料 server-step0/native-speed-ablation/all-trials.patch） 保存了试验代码。
- 第三点复核归档（开发资料 server-step0/native-speed-direct-kv/evidence.tar.gz）：`3b0da8519a462f86cd990c6ba690ffad8b24a330a8012e43d9283cecb27dd44c`。
