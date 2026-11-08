# 精度版 decode 滑窗 attention 合批实验

基准提交：`0bb1b726a4558f5f628ab70a0f8466a46364885e`。本轮 attention 实验保留在独立 worktree，未提交。

证据目录：`/workspace/results/mellum_precision_attention_20261106T051702Z`。

候选：`swa_softmax`；判定：通过，保留候选。

同一 GPU0 顺序执行相同 A200 HTTP 负载，每条生成256 token，活跃请求和实际 decode 批峰值均为32。以下各一次完整测试。

| 指标 | 基线 | 候选 |
|---|---:|---:|
| 输出吞吐 tok/s | 93.1853 | 105.7590 |
| 平均 TPOT ms | 273.11 | 234.03 |
| 平均 TTFT ms | 14038.51 | 14048.02 |
| 对 HF 的前64 token | 12800/12800 | 12800/12800 |

吞吐变化 **+13.49%**；候选匹配率 **100.00%**，约16k组 **256/256**。运行快照记录到 97335 次滑窗同长度组合批（含预热）。

候选通过原精度启动入口复现：`ATTENTION_MODE=swa_softmax CANDIDATE_GPU=0 bash /workspace/results/mellum_precision_attention_20261106T051702Z/checkout/mellum/run_precision.sh`。默认 `control`。

## 单点筛选

| 方案 | 逐位相同组件用例 | 32条前64 token匹配 | 32条短测 tok/s |
|---|---:|---:|---:|
| 已提交 norm 基线 | — | 2048/2048 | 52.727 |
| swa_batch | 9/45 | 1127/2048 | 58.070 |
| swa_softmax | 45/45 | 2048/2048 | 58.616 |
| swa_pointwise | 45/45 | 未扩大 | — |

只组合 decode 滑窗层中相邻且可见 KV 长度相同的请求。使用既有分页 KV gather 的连续输出，不补齐长度、不改变每请求的窗口和可见范围。单请求及全局 attention 仍走原数学路径。

`swa_batch` 批量执行 QK、scale、mask、FP32 softmax、BF16 转换和 PV。`swa_softmax` 保持原来逐请求两次矩阵乘法，只合批中间逐元素运算及 softmax。`swa_pointwise` 还保留原逐请求 softmax，只合批逐元素运算。三个方案均保留原来的类型转换边界。

组件检查覆盖批大小2/4/8/16/32、长度513/1009/1024和三种输入量级，每种模式45组。逐位比较只能说明所测张量上的结果。32条短测每条生成64 token，筛选要求至少90% token匹配且吞吐超过已提交基线3%，再验证约16k长输入组。

预填充、模型权重、HF 参考、Torch 2.10、MTP、投影、MoE 和已提交 norm 路径沿用基线。HF 原生 eager 参考仍为逐请求，比较原始 token ID。A200 不等同完整 A512/B50/C200。

实验改动：`mellum_dual/kv.py` 接入合批与计数；`mellum_dual/attention_ops.py` 定义各方案。组件检查、输入摘要、短测/长测、HTTP 原始 token 和启动脚本随证据保留。
