# Mellum ROCm 精度适配与验证

日期：2026-11-04，Asia/Hong_Kong。基于前一轮 vLLM ROCm 源码对照实施；保持 Git HEAD 与完整历史，尚未提交。

后续完整检查已完成：同一最终 `correctness` 实现完整200条为12800/12800、200/200完整一致，独立输入/参考/原始输出复核通过。详见`Mellum_ROCm精度版全量检查.md`。下文保留适配与固定10条阶段的消融记录，其中“尚未测200条”指该阶段当时的状态。

## 1. 实现范围

新增 `MellumRocmPrecisionForCausalLM`，继承已有 `MellumTorchExpertsForCausalLM`。继续使用 Mellum/Qwen3-MoE 骨干、decoder 的残差传递接口、FusedMoE 参数容器、原生检查点加载器、embedding、调度器和 KV 内存分配。LM head 的参数及 prefill 选行语义保留，GEMM provider 可切换。没有编辑带 `@support_torch_compile` 的模型文件，没有修改公共 norm/linear/attention/MoE/Sampler 算子，也没有升级 ATOM/AITER/ROCm/PyTorch。

新增精度路径在模型构造后替换 Mellum 的 113 处 norm 和 28 处 self-attention 局部模块，并设置局部专家执行方法。**替换前后的全部 Parameter 对象、名称均严格保持**；加载后进一步验证全部 5631 个检查点源 tensor 对 255 个目标参数逐值相同。不是重新加载一个 HF 模型，也没有使用 HF 的冻结输出参与生成。

约束：单卡、BF16 权重与 KV、block16、eager、level 0、prefix cache 关闭、MTP/TBO 关闭。暂不支持其他并行或量化配置；超出已支持范围在初始化时拒绝。

## 2. 参考 vLLM 的内容与本地实现的区别

固定参考为 vLLM `b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f`；原生 Mellum 结构另对照迁移前 `ea106db806918f371b8f392cc896dc7a2d6677a1`。完整原始对照见 `vLLM_ROCm实现对照.md`。

| 部分 | 借鉴的实际约定 | 本次接入 |
| --- | --- | --- |
| norm API | vLLM AITER provider 调用 `aiter.rms_norm` | 增加冻结 AITER 的 CK 路径，绕过 `rmsnorm2d_fwd` 默认转入 OPUS 的分支 |
| 残差边界 | 原生 HIP 融合 norm 使用 BF16 残差和；当前 Transformers Mellum 保留独立残差加法 | 在 norm 之前显式完成 BF16 残差加法，再送入所选 norm，保存同一个 BF16 和 |
| torch norm | vLLM IR native standalone norm 与 HF 都在 gamma 乘法前转换中间 dtype | FP32 RMS 归一化后转 BF16，再乘 BF16 gamma；保留为精度模式 |
| GEMM | gfx942 上 vLLM 不直接走其 gfx950 tgemm 分支 | self-attention 的 QKV/o_proj 使用 `F.linear`；允许 packed 或 split QKV，原参数与 loader 不变 |
| RoPE | 当前 Transformers backend 在 attention/cache 前执行 HF 位置编码 | 复用已验证的本地 FP32 inverse frequency / YaRN 实现，独立旋转后再写 KV；关闭该层内部的重复 RoPE |
| attention | provider 应可独立选择，窗口/KV 布局要与内核匹配 | 保留现有原生内核作消融；增加既有 BF16 page read/write + SDPA 精度路径 |
| 路由 | vLLM Transformers 的外部路由分支接收 HF 已算好的有序 IDs/weights | 本地执行 FP32 softmax→topk→重归一化→转回 BF16；不再只保证专家集合相同 |
| 专家计算 | vLLM 将路由与专家执行分开组织 | 复用已通过逐算子诊断的 torch 专家，继续沿用原生 w13/w2 存储和检查点映射 |

参考入口：[AITER norm provider](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/kernels/aiter_ops.py#L84)、[HIP 融合残差](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/csrc/layernorm_kernels.cu#L136)、[ROCm GEMM](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/model_executor/layers/utils.py#L432)、[Transformers MoE](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/model_executor/models/transformers/moe.py#L72)。

本次没有安装 vLLM，也没有复制其最新版 HIP/Triton attention 内核到冻结环境。SDPA 是本地精度路径，不应称为“移植了 vLLM attention”。CK 入口则确实调用当前冻结 AITER 的现成算子。两者保留独立开关，避免把算子来源与数值验证混淆。

## 3. 可复现的配置

| profile | norm | QKV | attention | RoPE/路由/专家 |
| --- | --- | --- | --- | --- |
| `ck_native` | AITER CK，BF16 残差边界 | packed torch linear | ATOM 原生 | 独立 RoPE、HF 有序路由、torch 专家 |
| `torch_native` | torch norm，BF16 残差边界 | packed torch linear | ATOM 原生 | 同上 |
| `torch_sdpa_packed` | 同上 | packed torch linear | SDPA | 同上 |
| `torch_sdpa_split` | 同上 | split Q/K/V torch linear | SDPA | 同上；gate/head 仍为原生 GEMM |
| `torch_gate` | 同上 | split Q/K/V torch linear | SDPA | gate 切为 torch linear，head 仍原生 |
| `correctness` | 同上 | split Q/K/V torch linear | SDPA | gate/head 均为 torch linear |

前三次切换分别隔离 norm、attention、QKV 输出形状。后两组继续隔离 router gate 和 LM head 的 GEMM provider。CK 组与历史原生 RTZ 组相比同时变动了 RoPE/路由等，不能把其全部改善归因为 CK 单项。

首轮结果目录中的 `correctness` 是当时的名称，实际保留原生 gate/head，等价于当前 `torch_sdpa_split`。该版本的完整冻结源码另存 `source_snapshot/`；没有覆盖历史结果或将后续代码冒充首轮代码。最终代码的 `correctness` 指 gate/head 也切换后的配置。

SDPA 沿用此前修复并通过朴素版全量的 mask 选择：因果且安全省略 mask 的 prefill 使用 causal/GQA；单 token decode 不开启 causal；达到滑窗边界或 cached 多 token query 时保留显式偏移 mask。仍使用 ATOM 原有分页内存和 metadata。测试在同一引擎内串行切换 profile，每条请求从头生成且 prefix cache 关闭；profile 切换只由外部验证 hook 在 forward 前完成，产品启动脚本固定一个 profile。

## 4. 验证口径

- 使用原来固定的最差 10 条，没有重新选择 prompt 或更改输入/tokenizer/template。
- 冻结同一 HF 5.10.1、BF16、SDPA attention、eager experts 参考；每条 greedy、ignore EOS、64 token。
- 首轮四组各一轮，拆分 QKV 组重复全部 10 条；第二轮检查 gate/head，最终 `correctness` 再重复全部 10 条。HumanEval/61 每组单独附测，不计入 640-token 分母。
- 生成过程中冻结全部 ATOM Python 源码、数据、参考、相关 AITER 源码及测量脚本，测量前后逐项 SHA 核对。
- 独立 CPU 汇总工具重新核对每条 input IDs、input SHA、reference IDs、实际输出长度、匹配数量和重复稳定性。

组件检查使用真实检查点 norm 参数，与独立导入的 HF `MellumRMSNorm` 比较：113 处 standalone norm、113 处残差 norm、113 处残差和，共 339 项逐值一致。CK 另外检查 finite 和误差，113 处普通 norm 最大相对 L2 约 0.44713%，因此不能把 CK 说成逐值 HF 一致。

独立 HF `MellumRotaryEmbedding` 与本地 RoPE 在 0、7、8191、8192、16383、32767 六个位置对照，两类层的 cos/sin 均逐值一致。这只是位置编码组件结果，不能替代 16k 完整生成验证。

## 5. 文件、开关与使用

| 文件 | 作用 |
| --- | --- |
| `/workspace/ATOM/atom/models/mellum_rocm_precision.py` | 新的局部精度模型、norm、attention 和专家路由方法；保留参数身份与原生骨干/loader |
| `/workspace/ATOM/atom/model_engine/model_runner.py` | 只新增 Mellum 精度模型的选择入口 |
| `/workspace/ATOM/atom/utils/envs.py` | 注册默认关闭的精度开关和 profile |
| `/workspace/run_rocm_precision.sh` | 可直接运行的新启动脚本，默认 `correctness`；沿用 run0 公共启动参数 |
| `/workspace/tools/measure_mellum_rocm_precision.py` | 权重/组件检查、四组固定 10 条和最终组重复测试 |
| `/workspace/tools/run_mellum_rocm_precision_checks.py` | 只使用 GPU0，管理服务暂停、缓存归档、测量冻结与服务恢复 |
| `/workspace/tools/review_mellum_rocm_precision.py` | 独立重新聚合结果并核对输入、参考、源码和权重证据 |

启动示例：

```bash
PATH=/opt/venv/bin:$PATH STEP0_GPU=0 SERVER_PORT=8000 bash /workspace/run_rocm_precision.sh
```

可用 `ATOM_MELLUM_ROCM_PRECISION_PROFILE` 选择上表 profile。`ATOM_MELLUM_ROCM_PRECISION` 默认 0，旧默认 run0 的模型选择不变。新脚本启用它，并关闭独立 RNE override，避免两套 norm 选择叠加。默认服务是否切换与是否完成后续全量验证要分别记录。

修改前两处注册文件及逐文件 SHA 在 `/workspace/backups/before-mellum-rocm-precision-20261104T070813+0800/`；包含 registration.patch。新模型文件均在此前不存在，旧适配和历史诊断未删除。

结果目录：首轮 `/workspace/results/mellum_rocm_precision_20261104T070835+0800/`；gate/head 补充轮 `/workspace/results/mellum_rocm_precision_linears_20261104T071652+0800/`。本地 JSON、日志、补丁及各阶段关键源码副本位于 `server-step0/mellum_rocm_precision_evidence/`。

## 6. 实测结果与直接结论

| 配置 | 固定 10 条 token 一致 | 完整一致 | HumanEval/61（单独附测） | 重复验证 |
| --- | --- | --- | --- | --- |
| 历史原生骨干 + torch 专家、BF16 路由、RTZ norm | 356/640，55.625% | 3/10 | 5/64 | 历史冻结结果，本轮未重跑 |
| `ck_native` | 418/640，65.3125% | 6/10 | 42/64 | 本轮一轮 |
| `torch_native` | 431/640，67.34375% | 4/10 | 51/64 | 本轮一轮 |
| `torch_sdpa_packed` | 465/640，72.65625% | 7/10 | 21/64 | 本轮一轮 |
| 首轮拆分 QKV，保留原生 gate/head（当时名为 `correctness`） | 640/640，100% | 10/10 | 51/64，第 52 token 分叉 | 两轮 10/10 相同；附测两次稳定但仍有差异 |
| `torch_gate`，另将 gate GEMM 改为 torch | 640/640，100% | 10/10 | 64/64 | 本轮一轮 |
| 最终 `correctness`，gate/head 均为 torch | 640/640，100% | 10/10 | 64/64 | 固定 10 条两轮均 640/640、10/10 稳定；附测两轮均 64/64 |

由相邻配置可以得到以下受控结论：

1. 单换 norm API/舍入边界仍不够；保留原生 attention 的两组均未达 90%。
2. 固定 norm、RoPE、路由等，attention 切为 SDPA 后仍只有 465/640；进一步只改变 QKV 的 packed/split 计算形状，固定 10 条达到 640/640。因此不能把 packed QKV 视为在本次 HF token 精度目标下可无条件保留的优化。
3. 额外 HumanEval/61 捕获了固定 10 条之外的残余问题。只将 gate 的 GEMM provider 切为 torch，51/64 提升到 64/64，其他设置相同。这为 gate 提供了直接消融证据。前次同输入 gate 56/56 精确仅覆盖 prefill 和早期 decode，不能推广到第 52 步或全部生成过程。
4. 本轮 `torch_gate` 仍用原生 LM head 也达到附测 64/64，因此没有证据要求把 LM head 说成本条残差的根因。最终 profile 用统一 torch GEMM 约定并完成重复验证；保留 native-head 配置作为后续优化起点。

独立工具重新读取输出 JSON、原始冻结参考及全部输入 IDs，复算四组首轮和两组补充轮计数，通过。各轮都冻结 223 个相关文件；第一轮在源码变更前核对通过并保存完整 `source_snapshot/`，第二轮再次检查当前源码不变。两轮均重新验证 5631→255 全部权重与组件检查。

测试期间只暂停/恢复 GPU0 服务，GPU1 没有被停止或重启。结束后两个 HTTP 服务实际生成 8 token，并核对 GPU 显存占用。旧 BF16 全量仍为 `stopped_by_user`，没有恢复。HEAD 保持 `6c86378f230d7d10734b574aa30988a349ebd2ff`，没有 Git 提交。

## 7. 当前使用边界与后续优化空间

新启动脚本默认启用最终 `correctness`，默认 `run0` 仍保留原生融合版本。当前完成的是**新精度路径的固定 10 条和额外最差样本验证**，没有把朴素版旧的完整 200 条成绩迁移到本路径。尚未运行新路径的完整 200 条、HumanEval 功能 pass@1、16k 配对、正式 A/B/C 性能；不能据此宣称 Step0 全部通过。

后续可逐项恢复 packed QKV、原生 attention、CK norm、native gate/head 和融合专家，每次固定同一批输入做消融。参数与 loader 不变，内核选择集中在 Mellum 局部模块；恢复高性能算子不需要再改模型结构或重写权重映射。当前精度模式使用 torch 循环专家和 SDPA，没有正式性能成绩。

最终七个代码/启动/验证文件的审阅补丁和逐文件 SHA：`/workspace/backups/before-mellum-rocm-precision-20261104T070813+0800/final_review.patch`、`final_changes.json`。补充轮修改前的三个文件另保存在 `/workspace/backups/before-mellum-precision-linears-20261104T071648+0800/`。Python 语法检查和 shell 语法检查通过；本机没有安装 black，未为格式检查升级依赖。
