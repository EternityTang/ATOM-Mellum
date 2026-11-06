# vLLM ROCm 实现与当前 ATOM 路径对照

核对日期：2026-11-04（Asia/Hong_Kong）。本轮只读源码、保存快照及更新文档；没有安装 vLLM、修改模型/公共算子、启动推理测试或提交 Git。

后续进展：用户随后授权了精度适配，已另行实现并验证Mellum局部精度路径，详见`Mellum_ROCm精度适配与验证.md`。本文件保留适配前原生路径的静态对照及固定快照，不把后续精度模式混入这里的“我们当前原生实现”。

结论：骨干复用方式与 vLLM 的旧原生 Mellum 一致，但实际计算入口并不相同。最明确的差异是 norm 的内核及残差舍入、gfx942 上的 GEMM 分派、RoPE 的执行位置、attention 内核和 MoE 路由。不能用“双方都支持 ROCm、都调用 AITER”推导数值路径一致。

## 1. 对照范围与证据边界

| 对象 | 固定来源 | 用途 |
| --- | --- | --- |
| 当前 ATOM 工作区 | `/workspace/ATOM`，HEAD `6c86378f230d7d10734b574aa30988a349ebd2ff`，包含尚未提交的工作区修改 | 以实际文件快照为准，不能只按 HEAD 还原 |
| 当前冻结 AITER | `/app/aiter-test`，`8ddfc7510536d61f4e1863e9cf91c495be1f8c2c` | 核实 ATOM 最终调用到哪个底层实现 |
| vLLM 旧原生 Mellum | `ea106db806918f371b8f392cc896dc7a2d6677a1` | 与我们的 Qwen3-MoE 复用结构直接对照 |
| vLLM 当前源码 | `b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f` | 核对当前 ROCm dispatch、内核和 Transformers backend |

旧原生 `mellum.py` 确实继承 Qwen3-MoE attention/decoder/model/MoE/loader，按 `layer_types`、`mlp_layer_types`、`rope_parameters` 接入差异。这支持我们目前的复用方式。当前 vLLM 则已经在注册表中将 Mellum 指向 `TransformersMoEForCausalLM`；迁移提交为 [8faacd707f2f](https://github.com/vllm-project/vllm/commit/8faacd707f2f69b86ecd0bb1e7b5a705b473cb25)。

因此，下文分别说明“旧原生模型走通用算子”与“当前 Transformers Mellum”，不把两者拼成一条不存在的运行链。当前服务器 `/opt/venv/bin/python` 查不到 vLLM；以下 vLLM dispatch 是源码结论或带前提的推断，没有 vLLM 实测 logits、一致率或性能结果。

我们最近逐算子实验的对象是：原生骨干 + torch 专家、BF16 路由权重、RNE 关闭。默认 `run0` 仍是原生融合专家版，不能混为同一运行配置。修复后朴素版完整 200 条为 12800/12800；本报告沿用已有证据，没有重测。

## 2. RMSNorm：名字相近，实际入口和舍入不同

### 2.1 我们当前走 OPUS，vLLM 的 AITER 普通 norm 入口是 CK

我们的链路：

```text
atom/model_ops/layernorm.py:rmsnorm2d_fwd_
  → aiter.rmsnorm2d_fwd
  → hidden_size <= 8192 且未指定 model-sensitive 模式
  → aiter.rmsnorm
  → module_rmsnorm_quant / OPUS
```

Mellum 的 hidden size 2304 和 Q/K norm 的 128 均满足这个分支。融合残差 norm 的 `rmsnorm2d_fwd_with_add` 在相同条件下转入 OPUS `add_rmsnorm`。

vLLM 的 AITER provider 在 `vllm/kernels/aiter_ops.py:84` 导入的是 **`aiter.rms_norm`**。在我们的冻结 AITER 中，这个入口直接对应 **`module_rmsnorm` 的 CK 实现**，不是上述 OPUS 分支。也就是说，即便比较同一份 AITER 源码，两种 API 也不能互换理解。[vLLM AITER provider](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/kernels/aiter_ops.py#L84)、[冻结 AITER 入口](https://github.com/ROCm/aiter/blob/8ddfc7510536d61f4e1863e9cf91c495be1f8c2c/aiter/ops/rmsnorm.py#L47)。

这里能确认 API 和编译模块不同；没有运行 CK 与 OPUS 的配对实验，不能直接声称 CK 对 HF 逐值相等。

### 2.2 对应 eager 配置下，vLLM 默认优先用自己的 HIP norm

`vllm/platforms/rocm.py:1420` 的默认优先级：关闭编译时为 `vllm_c → native`。AITER norm 被放到最前需要 CUDA Graph 开启、AITER 总开关和 RMSNorm 开关开启，并满足架构条件。

所以，对应我们单卡 BF16、eager、level 0、关闭 Graph，且没有用户覆盖 provider 优先级的配置，默认候选是 **`_C.rms_norm / _C.fused_add_rms_norm`**，不是 AITER norm。[ROCm 优先级实现](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/platforms/rocm.py#L1420)。

这些内核位于共用的 `csrc/layernorm_kernels.cu`。虽然扩展名是 `.cu`，它们也由 ROCm 构建编译为 HIP；`CMakeLists.txt:463` 和 `type_convert.cuh` 的 `USE_ROCM` 分支均有对应证据。不能因文件名将其排除为“仅 CUDA 路径”。

### 2.3 融合残差的运算顺序不同

以 BF16 输入、BF16 gamma 表示，省略 reduction 实现细节：

| 阶段 | 我们当前 OPUS 融合 norm | vLLM 通用 HIP 融合 norm |
| --- | --- | --- |
| 残差求和 | 先保留 FP32 和，用这个和计算 RMS | 先得到 BF16 和，再用舍入后的和计算 RMS |
| 保存 residual | 当前 gfx942 默认 RTZ 截断到 BF16 | HIP BF16 转换/加法路径，采用标准舍入 |
| norm 与 gamma | FP32 运算后，当前默认 RTZ 输出 BF16 | FP32 norm 与 gamma 后转换 BF16 |

我们 OPUS 的 `OPUS_FP32_to_BF16_DEFAULT=2` 在 `csrc/include/opus/opus.hpp:38` 明确标为 truncate；前次同输入实验也复现了它。vLLM 融合内核在 `layernorm_kernels.cu:136` 先对 `_f16Vec` 的 BF16 数据做残差加法，再 `sum_squares`；转换见 `type_convert.cuh:72` 及本机 HIP BF16 头文件。归约树本身也可能不同，不保证仅修改转换就逐值相等。

还要区分两个例外：

- **当前 Transformers Mellum、关闭编译时**，HF decoder forward 保留独立 BF16 残差加法；RMSNorm 可被 fuser 替换为 vLLM norm。不能把上面的通用融合残差调用直接当作该模型每层必走的入口。
- vLLM 的 IR `native` fallback 又是另一种顺序：FP32 残差和参与 norm，norm 结果先转 weight dtype 再乘 gamma。因此只阅读 `forward_native` 或 IR fallback，不能代表默认 HIP provider。

对应我们已有证据：同输入 input norm 0/56 逐值相等、融合残差 0/54；这是值得优先隔离的差异，但此前局部 RNE 快速回归整体下降，当前仍默认关闭，不能据源码直接替换默认路径。

## 3. GEMM：vLLM 在 gfx942 上没有直接选择我们的 tgemm 路径

我们 `atom/model_ops/linear.py:527` 的非量化线性层直接调用 `aiter.tuned_gemm.tgemm.mm`。Packed QKV、o_proj 和 gate 等沿用该路径。

当前 vLLM 的 `layers/utils.py:432` 按形状、架构和开关分派：

| vLLM 分支 | Mellum / MI308X gfx942 上的含义 |
| --- | --- |
| batch-invariant 特殊路径 | 需要另外开启，不是本次默认比较配置 |
| AITER Triton GEMM | MI300 的 FP8-FNUZ 架构被当前 guard 排除 |
| skinny GEMM / `wvSplitK` | 默认开关开启；1～5 行等条件满足时可用于 decode，包括 gfx942 |
| `tgemm.mm` | `is_tgemm_enabled()` 额外要求 **gfx950**，gfx942 不走这个分支 |
| 通用 fallback | `torch.nn.functional.linear` |

旧原生快照的 `is_tgemm_enabled` 同样限制 gfx950。这里的关键区别是**内核选择策略**；没有 vLLM 运行日志，不能把某个形状的实际 skinny 命中情况当作已测事实。[GEMM 分派](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/model_executor/layers/utils.py#L432)、[架构限制](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/_aiter_ops.py#L2684)。

**Packed QKV 不是我们独有的结构错误。**旧原生 vLLM 也用 QKVParallelLinear，当前 Transformers backend 也会合并 QKV。我们已有同输入证据是：分开 Q/K/V 56/56 精确，合并后仅 9/56 精确，但 native packed 与相同 packed 的 torch linear 56/56 精确。这表明同一 GEMM 的输出矩阵形状就可能改变数值，不能只把差异归因于 tgemm。

## 4. RoPE：我们在 attention 内部融合，旧原生 vLLM 先独立旋转

| 对照位置 | 我们当前实现 | vLLM 对应实现 |
| --- | --- | --- |
| 旧原生模型的执行顺序 | QKV → Q/K norm → Attention 内部 `rope_cache` | QKV → Q/K norm → 独立 rotary → Attention |
| SWA | `fused_qk_rope_reshape_and_cache`，旋转与 KV 写入合并 | 旧原生先旋转，再交给 attention/cache |
| 全局层 | AITER rotary module，再 `reshape_and_cache(asm_layout=True)` | 旧原生用 vLLM RoPE；ROCm 默认 `_C.rotary_embedding`，可选 AITER Triton |
| 当前 Transformers Mellum | 我们仍是上面的 ATOM 路径 | HF forward 计算/应用位置编码，再通过 vLLM attention 接口；不能一概说它调用通用 `_C.rotary_embedding` |

通用 vLLM RoPE 的 `forward_hip` 在 AITER Triton RoPE 未开启时调用名为 `forward_cuda` 的共用包装；ROCm 下最终是 HIP 编译的扩展，而不是运行 CUDA。两侧均存在按激活 dtype 保存/匹配三角表的逻辑，**没有证据支持简单归因为“一个 FP32 cache、一个 BF16 cache”**。[vLLM RoPE 分派](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/model_executor/layers/rotary_embedding/base.py#L254)。

另一个具体参数差异：vLLM YaRN 显式支持 HF 的 `attention_factor`；我们的 AITER 接口使用 `attn_factor`，现有 Mellum adapter 直接传配置，未做字段换算。注意 vLLM 的 `attention_factor` 是最终 mscale，AITER 的 `attn_factor` 乘在默认 YaRN mscale 上，不能简单同值改名。当前模型参数与默认公式恰好一致，所以这是通用兼容性缺口，尚不能作为当前短 prompt 分叉原因。[YaRN 实现](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py#L38)。

已有真实边界实验：SWA decode Q/K 21/21 精确；7 个全局层均有差异，最大相对 L2 约 0.234%/0.223%。应优先比较全局 YaRN 的三角表及旋转 kernel，不能沿用已作废的旧 rotary hook 结果。

## 5. Attention：默认 ROCm 路径与我们的三段内核都不同

在无 MLA、无 sinks、无 KV connector 的 Mellum 条件下，当前 `platforms/rocm.py:531` 首先列出 `ROCM_ATTN`，之后才是可选 AITER 后端；最终选择仍受配置覆盖、验证及缓存布局影响。

| 阶段 | 我们当前实际使用 | vLLM `ROCM_ATTN` 调用链 |
| --- | --- | --- |
| prefill | `aiter.flash_attn_varlen_func` | `chunked_prefill_paged_decode` → `prefix_prefill.context_attention_fwd`（Triton） |
| SWA decode | `pa_decode_gluon`，由 ATOM 的 `paged_attention_triton` 包装 | `kernel_paged_attention_2d`（Triton）；custom HIP decode 的 guard 排除 SWA |
| 全局 decode | `aiter.pa_fwd_asm(high_precision=0)` | 满足 native layout、block16/32、head128、GQA8 等条件时可走 `_C.paged_attention_rocm`；否则走上述 Triton kernel |

这是实现区别，不是仅换了 wrapper 名称。[ROCm backend](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/v1/attention/backends/rocm_attn.py#L443)、[prefill/decode 叶子实现](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/v1/attention/ops/chunked_prefill_paged_decode.py#L297)。

叶子代码也说明了具体数值差别：vLLM Triton decode 分块计算 QK、在线 softmax，P 转为 V 的 dtype 后做 PV；末尾还有 `L + 1e-10` 的归一化。它不是普通 torch SDPA，也不是我们的 Gluon/ASM；分块、归约树和转换位置可产生不同 BF16 结果。prefill 同样有自己的分块与因果/SWA mask。

全局 decode 的 HIP 实现还下钻到 `csrc/rocm/attention_dispatch.cu` 和 `attention_kernels.cuh`：按 256-token partition 处理，partition 的 max/sum 使用 FP32，临时输出由 Python 包装分配为 query dtype，再进入归并 kernel。BF16 转换 helper 中可见 RNE 实现，部分其他 helper 也有 RTZ 变体，不能把整份 attention 简化为“全程 FP32”或“所有转换都是 RNE”。具体模板分支仍由头维、块大小、架构和序列长度决定。这与我们显式调用 AITER ASM 且 `high_precision=0` 的路径不同，尚未做两者输出的实测配对。

窗口语义上，vLLM backend 将左跨度 W-1 转回完整 W，叶子 mask 使用距离 `< W`。我们当前 prefill 已传 W-1，冻结 Gluon decode 传 W+query_len 是该版本接口的适配，**不能直接拿 vLLM 参数数值覆盖它**。

如果显式选择 vLLM `ROCM_AITER_FA`，prefill 可以同样调用 AITER Flash Attention，但 decode 仍按 SWA/block table/cache layout 等条件分派到 `paged_attention_v1`、`unified_attention` 或 `paged_attention_common`，并不是我们固定的“21 层 Gluon + 7 层 ASM”。

KV cache layout 也需随内核匹配，不能单独替换调用函数。本地已有 K/V 各 336/336 写入读回精确的证据，因此当前重点是**给定相同 Q/K/V 后的 attention 计算**，不是已证明的缓存损坏。已有同输入 attention 0/56 逐值精确、最大相对 L2 约 0.0852%。

## 6. MoE：路由入口、权重舍入和专家计算要分别比较

### 路由

我们 `atom/model_ops/topK.py:98` 使用 `aiter.topk_softmax`，生成 FP32 top-k 权重和 int32 ID；局部 torch 专家后端可在乘权前转 BF16。当前诊断已开启这个转换，默认融合 run0 不自动等于该配置。

当前 vLLM 的原生路由在 AITER MoE 开启、gfx942/gfx950、连续输入且 token 数不超过 4096 等条件下，会选择较新的 `aiter.topk_gating`；不满足条件才回退 legacy `topk_softmax`。Mellum 无 shared experts 的常见短输入满足源码列出的形状条件，但没有实际运行确认分派。[路由分派](https://github.com/vllm-project/vllm/blob/b14e2f2792ae2f3c9048b5de74ed07d2ca07ab6f/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py#L58)。

当前 Transformers Mellum 还要看 MoE fuser：

- 若识别出 `MellumTopKRouter` 的 linear→FP32 softmax→topk→重归一化结构，可以转为 vLLM 内部路由。
- 未能整体匹配时可以保留 HF block/router，只替换专家；入口把 HF 已算好的 IDs 转 int32、weights 转 FP32。这时 **BF16 已舍入权重再扩展为 FP32**，不等于始终保留未舍入的原生 FP32 权重。

本机 HF 5.10.1 确实使用独立的 `MellumTopKRouter`，不是一个只有 Linear 的 gate。其输出权重转回 router logits dtype。没有运行 vLLM fuser，所以本报告保留两种分支，不声称已经确认实际采用哪一种。

已有同输入证据：专家集合 56/56 相同，BF16 dense 权重 56/56 相同，但有序 ID 只有 21/56 相同。故后续应比较有序 ID、权重舍入时点及各专家 token 行序，不能只比较选中专家集合。

### 专家计算

vLLM ROCm 的非量化 MoE 候选包括 AITER/TRITON/BATCHED_TRITON。选 AITER 时，`experts/rocm_aiter_moe.py:571` 调到 `rocm_aiter_fused_experts`，再进入 `rocm_aiter_ops.fused_moe`，传入路由权重、IDs、激活、输出 dtype、padding、sorting policy 等。它是融合专家执行，不等于普通 torch 循环。

我们最近诊断的 `mellum_torch_experts.py` 则按专家做 packed gate/up `F.linear`、SiLU×up、down `F.linear`、乘权、`index_add_`。固定朴素版的路由和排列时，各内部步骤 56/56 全部相等；换成原生路由排列后最终只有 45/56 精确，差异都在 prefill，decode 28/28 精确。因此当前没有证据表明必须重写普通专家公式，应先保留这条已验证路径隔离其他组件。

默认融合 run0 虽也调用 AITER fused MoE，但依赖冻结在本机 AITER commit。当前 vLLM 的 ROCm Dockerfile 默认 AITER build ref 是 `v0.1.27`，且可被 build argument 覆盖，不能推定与本机内核相同。当前 vLLM AITER 融合 norm wrapper 还传 `gemma_norm`，本机冻结签名没有这个参数；直接复制最新 wrapper 会产生接口不兼容。旧原生快照的该 wrapper 尚未传这个参数，两者已分开核实。

## 7. 从这些区别得到的下一步定位顺序

1. **Norm/残差先做同输入配对**：当前 OPUS、冻结 CK API、朴素路径分别比较 residual 和 norm 输出，保留每次舍入位置。RNE 单独回归已退化，因此不直接替换默认配置。
2. **全局 RoPE 和 attention 分开**：先用同一三角表比较旋转；再固定相同 Q/K/V 比较 SDPA 与现有 prefill/Gluon/ASM。若借用 vLLM ROCm kernel，需要一并满足布局、窗口和依赖接口。
3. **Packed QKV、路由排列最后单独消融**：双方都使用 packed QKV，先区分输出形状与 GEMM provider；路由先固定 IDs/weights/行序再放开，避免把排列导致的专家 GEMM 差异误归为专家公式错误。

这是后续可执行的隔离方向，本轮没有应用任何替换，也没有根据静态源码声称 vLLM 更接近 HF 或能达到 100%。

## 8. 文件和可复现材料

服务器参考快照均在生产仓库之外：

- `/workspace/reference/vllm-source-b14e2f2792ae/`：当前 vLLM 指定文件，含 SOURCE.json、下载状态及 SHA256 manifest。
- `/workspace/reference/vllm-native-mellum-ea106db80691/`：旧原生 Mellum 与对应 ROCm 路径。
- `/workspace/reference/atom-rocm-comparison-20261104/`：当前 ATOM/AITER 的 13 个相关文件及 HF modeling_mellum.py，共 14 文件逐个 SHA256；包含未提交的实际实现。

本地对应目录为 `C:/Users/tang/Documents/ChatGPT/腾讯笔试/server-step0/vllm-rocm-review/`；本地源文件行号与上文一致。下载失败的候选路径和 404 保留在 manifest，不作为实现证据。另有 `verification.json` 记录成功文件的完整性复核。

已有动态证据：`朴素版与原生版逐算子对比.md`，主目录 `/workspace/results/naive_native_operator_diagnosis_20261103T235149+0800/`；本报告没有产生新的 GPU 测量成绩。两个服务未被本轮停止或重启，已停止的 BF16 全量没有恢复，Git 历史未改。
