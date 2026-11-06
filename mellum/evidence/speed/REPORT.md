# Mellum 固定 32 并发吞吐和 Transformers token 一致性

2026-11-05，香港时间。默认候选 `packed` 的单卡固定负载吞吐中位数为 **684.02 输出 token/s**。公开 200 条逐条 greedy 生成时，与原生 Transformers 的前 64 个 token 位置一致率为 **100.00%**。按本次约定，两项分别验收。

## 旧速度版为什么会分叉

直接分页 attention 分块计算在线 softmax，并在乘 V 前将块内权重舍入到 BF16。分组专家 GEMM 和逐专家 PyTorch 投影也会产生不同的 BF16 输出。在 HumanEval/135 的第 6 个预测位置，HF 给 ` complete` 和 ` implement` 的 logits 分别为 31.375、31.250。旧速度版使二者同为 31.375，greedy 选中 ID 更小的 ` implement`，之后两边输入的前缀就不同了。此前固定 10 条强制相同前缀时为 634/640，独立自由生成则为 356/640。本轮旧速度版完整 200 条实际得到 9957/12,800（77.79%），116/200 条完整一致。相同前缀的一步预测命中率不能代替验收用的自由生成一致率。

## 同一份配置如何推理

- Prefill 使用精度骨干。decode 根据当前前向的实际 token 行数选择路径。每个 token 选择 8 位专家，模型共 64 位专家。当总路由数低于专家数（1～7 行）时，使用 Torch SDPA、逐专家 `F.linear` 和有序 BF16 归并。这是兼顾精度的形状选择规则，没有宣称该阈值在所有负载下速度最优。
- 从 8 行起使用直接分页 attention 与分组专家 GEMM。保留残差 BF16 相加、norm 转 BF16 后乘权重和有序加权的舍入边界。将 Q/K/V 三个投影合为一次投影，Q、K 的 norm 与 RoPE 在专用内核内完成，专家 SiLU/乘法和路由归并也使用较少的 GPU kernel。选择仅依赖运行时形状，启动时两项测试使用同一个 `packed` 设置。
- 单卡 MI308X，模型与 KV 为 BF16，eager、编译等级 0、关闭 prefix cache 和 MTP。组件检查共 40 组，其中 40 组逐元素相等，各组元素差值见 `components.json`。覆盖稀疏/分组切换两侧以及带偏移、非连续行的 packed Q/K 视图。

## 吞吐

32 条不同输入，每条精确 1024 token，并发 32，每条固定生成 256 token。计时含 prefill，使用 8192 个实际输出除以整批墙钟时间。每轮先以每请求 4 个输出预热。同机物理 GPU1。下表的“正式入口”是该候选的独立服务脚本，本轮负载与题面完整场景 A 的长度分布、请求数不同。

| 实现 | 输出 token/s |
|---|---:|
| 旧速度版同服务复测 | 569.43 |
| 按形状分派，融合 RMSNorm、RoPE | 590.21、593.33 |
| 再融合分组专家激活与有序加权 | 608.23、617.25 |
| 再合并 QKV 投影并融合 Q/K norm 与 RoPE | 682.63、671.62、681.84 |
| 新版独立服务入口，重新启动后连续三轮 | 684.02, 685.31, 675.08 |

## Token 验证

冻结的原生 Transformers 5.10.1 `MellumForCausalLM` 参考使用 BF16、SDPA attention、eager experts。输入、参考和实测均用 token ID。包含 164 条 HumanEval 来源 prompt 与 36 条 MBPP 来源 prompt，每条独立自由生成 64 token。此处只比较 token，不执行功能评分。

| 运行与参考的口径 | 逐位置相同 | 64 个位置均同的条数 |
|---|---:|---:|
| 旧速度版逐条 vs HF 逐条 | 9957/12,800（77.79%） | 116/200 |
| 新版逐条 vs HF 逐条，验收口径 | 12800/12,800（100.00%） | 200/200 |
| 新版对同一 200 条分批生成 vs HF 逐条，额外诊断 | 9805/12,800（76.60%） | 107/200 |
| 新版与原生 HF 按完全相同的 prompt 分组 | 9528/12,800（74.44%） | 104/200 |
| 原生 HF 分批 vs 原生 HF 逐条 | 9965/12,800（77.85%） | 109/200 |

额外诊断将相同 200 条按原顺序组成 6 批 32 条和末批 8 条。HF 使用左填充和 attention mask，ATOM 走原生连续组批与分页缓存，同组请求的底层排布不同。单请求测试固定困难 10 条命中 640/640。

## 部署及复现材料

服务器运行 `/workspace/run_mellum_target600.sh`。默认 `MELLUM_DECODE_IMPL=packed`，可设置为 `post`、`adaptive`、`fast_combined` 做消融。组件在 `/workspace/tools/mellum_target600/`，复用已验证的 `mellum_exact_decode`、`mellum_step_cache`、`mellum_decode_trial`。没有修改 `run0.sh` 或提交 Git。

| 新文件 | 用途 |
|---|---|
| `entry.py`、`target.py` | 检查启动配置，按当前工作量接入小批量或分组解码。 |
| `kernels.py` | RMSNorm、残差与独立 RoPE。 |
| `grouped.py` | 复用 AITER 分组 GEMM，合并激活和有序 BF16 加权。 |
| `qkr.py` | 合并 QKV 投影，直接在 packed Q/K 视图上执行头部归一化和旋转。 |

服务器记录 `/workspace/results/mellum_target600_fEKFtjbf`。本地 `server-step0/mellum-target600` 保存逐题 token、逐请求计时、校验结果和代码。
