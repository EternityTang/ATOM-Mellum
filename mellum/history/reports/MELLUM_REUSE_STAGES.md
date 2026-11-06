# Mellum 组件逐步接回与 SWA 验证

日期：2026-11-05。参考继续固定为 Torch 2.10、原生 Transformers 5.20.2、BF16、eager attention/eager experts、greedy、逐条、无 padding、整段 prefill。ATOM 调度支持 32 条活动请求，模型内部仍逐请求独立计算，保持与 HF 单条参考相同的矩阵形状。

本轮新候选入口为 `/workspace/run_mellum_reuse_request.sh`，默认 `framework_eager`。已经接回原有 Qwen3-MoE 骨干、MoE 容器和 Attention 组件；**原有 Attention 组件已使用，原生 SWA 加速算子尚未采用**。上轮独立朴素版继续保留。

## 实际复用了什么

| 部分 | 新候选实现 |
|---|---|
| 模型骨干 | 继承并执行 `Qwen3MoeModel.forward` |
| Decoder 层 | 继承并执行 `Qwen3MoeDecoderLayer.forward`，保留原残差和模块调用结构 |
| MoE 容器 | 执行 `Qwen3MoeSparseMoeBlock.forward`，保留原专家参数容器与加载；专家算术仍为已对齐的逐专家计算 |
| Attention 组件 | 调用现有 `Attention` / `PagedAttention` 对象及其后端分发入口 |
| SWA 配置 | 沿用 GPT-OSS/MiMo 使用的逐层 `per_layer_sliding_window` 接法；21 层窗口 1024，7 层全局注意力 |
| 默认 Attention 算术 | 新适配包将组件实例的后端绑定到已验证的 eager 计算，保留 HF 的 BF16/FP32 运算边界 |
| norm/QKV/RoPE | 保留原精度适配操作，尚未切回框架融合 norm/QKV/RoPE kernel |
| 调度、KV 页、采样、权重加载 | 继续复用 ATOM；所有 checkpoint Parameter 对象与名称保持不变 |

这里复用的是公共 Attention 接法，不是整套 GPT-OSS/MiMo 模型，也不引入它们特有的 attention sinks 等语义。模型每条请求使用独立 metadata、整段 prefill 和单独 LM head；32 活动请求不意味着将它们合并进同一次 GEMM。

## 短程回归

只运行固定 10 条及通过组合的 32 条，每条生成 64 token。32 条输入和 HF 参考沿用上轮重新生成并核验的快速集；不是已停止的全量 200 条，也不是正式场景 A512 吞吐测试。

| 组合 | token ID 一致数 | 64 token 全部一致的请求 | 实际最大 decode 批大小 |
|---|---:|---:|---:|
| 接回 Qwen3-MoE 骨干，Attention 继续旁路 eager | 640/640，100% | 10/10 | 10 |
| 再接回框架 Attention 组件，后端 eager | 640/640，100% | 10/10 | 10 |
| 同一默认组合扩大至 32 条 | **2048/2048，100%** | **32/32** | **32** |
| SWA prefill/decode 都启用原生加速后端 | 17/640，2.65625% | 0/10 | 10 |
| 只启用原生 SWA prefill，decode 保留 eager | 25/640，3.90625% | 0/10 | 10 |
| 只启用原生 SWA decode，prefill 保留 eager | 339/640，52.96875% | 1/10 | 10 |

原生 SWA 三个组合均未达到精度要求，不作为默认实现。prefill 替换让 9/10 条从第一个输出 token 就分叉；只换 decode 时第一 token 均保持相同，后续 9/10 条分叉。本轮停在组件复用已通过、原生加速后端尚未通过这个边界；没有继续替换 norm 或融合专家。

默认组合 32 条直接 Engine.generate 的生成耗时约 130.51 秒。上轮朴素版同一快速集约 204.05 秒，但运行时段、同时运行的另一张卡任务等条件不同，本轮不据此发布正式提速比例，也不能继承速度版 600+ 吞吐成绩。

## 以前是否没有使用滑动窗口

不能把此前全部差异归结为“没有 SWA”。旧复用适配已经配置逐层窗口，独立朴素版也有显式 SWA mask。上一轮审查发现的是：分块 prefill 的 eager 路径保留了窗口外、最终被 mask 的历史 KV，计算矩阵形状与 HF 的缓存裁剪不同；该问题已经修正。本轮保持整段 prefill，保留该修正。

此前还分别复现过批大小造成 QKV/专家 GEMM 的 BF16 差异，以及 norm、路由权重舍入边界问题。启用滑窗语义与逐 token 对齐 HF 是两个需要分别检查的条件。原生 SWA 的失败消融只能说明这个后端替换组合未通过，不能单凭一致率断言是 mask、缓存布局或浮点舍入中的哪一种原因。

## 文件及验证证据

| 新增文件 | 作用 |
|---|---|
| `/workspace/tools/mellum_reuse_request/stages.py` | 逐请求隔离；复用原骨干和 Attention 组件；按阶段切换 eager/原生 SWA 后端；记录实际调用 |
| `/workspace/tools/mellum_reuse_request/__init__.py` | 导出安装方法与实验阶段 |
| `/workspace/tools/mellum_reuse_request/serve.py` | 模型加载后安装适配，接入已有服务入口 |
| `/workspace/tools/mellum_reuse_request/README.md` | 启动方式、支持范围、组件与 kernel 的区别 |
| `/workspace/run_mellum_reuse_request.sh` | 固定单卡、BF16、level 0、eager、关闭分块/MTP等参数 |
| `/workspace/results/mellum_reuse_stages_20261105/code/run_stage.py` | 固定集生成、逐条比较、运行时 metadata 和批次记录 |
| 同目录 `code/review.py` | 独立复算原始 ID 一致率，核查输入、调用路径、批大小及原有源码/Git 状态 |

首轮原生 SWA 接入因第 0 层切换后不再初始化旧 eager decode 分组，导致全局层收到空分组而报错；仅在新适配包中修正原生混合模式的元数据交接。失败日志与修改前文件保存在 `failed-native-integration/`，不计作精度结果。

运行记录中的 `backbone` 标准 forward-hook 计数为 0：ATOM 的 `support_torch_compile.__call__` 在 level 0 直接调用 forward，绕过标准模块 hook。该字段不能用作调用次数；骨干继承方法身份、实际 Decoder/MoE hook 及 Attention 后端计数分别核查。原始记录保留，没有把 0 改写成估算值。

HTTP 服务启动入口已提供，本轮实际验证的是直接 Engine.generate。没有重跑完整 200 条、HTTP 性能或 HumanEval。旧精度版、速度版、独立朴素版、公共框架源码与依赖保留，没有 Git 提交。
