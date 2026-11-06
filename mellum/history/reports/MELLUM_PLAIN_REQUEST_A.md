# Mellum 朴素重建版：场景 A 精度

日期：2026-11-05。按本轮确认的口径：**ATOM 32 并发输出，对照原生 HF 逐条生成**。优先恢复精度；旧精度版、速度版和 Git 历史保留。

**已从头实现独立的 Torch 数值前向。** ATOM 继续负责 checkpoint 权重加载、请求调度、KV 页分配和采样；新的 `plain.py` 自行执行完整 Mellum 层和 LM head，不安装此前的 eager precision / decode adapter。

当前状态：按用户“无需完整测试，目前先这样”的要求，扩大验证和待启动的 HTTP 测试均已停止，不自动恢复。32 条快速验证完整通过；扩大验证已完成的 160 条单独保留为阶段结果。两卡已空闲，测试端口关闭。

| 检查 | 原始 token 一致数 | 全部 64 token 相同的请求 |
|---|---:|---:|
| 旧精度版，相同 32 条输入与整段 prefill | 596/2048，29.1015625% | 5/32 |
| 新朴素版，32 条同时 decode | **2048/2048，100%** | **32/32** |
| 新朴素版，A 前 200 条中的已完成部分，严格闭环 32 | **10240/10240，100%** | **160/160，阶段结果** |
| 新朴素版，真实 HTTP 32 并发、输出 256、比较前 64 | 未启动，已取消待运行任务 | 不作成绩 |

停止时 ATOM 完成 160 条，HF 参考完成 180 条；两端可配对的 160 条全部一致。没有完成 200 条，不把阶段结果写成完整验收通过。没有完成 HTTP 或 A512 性能验收。

这里的“一致”指每个生成位置的原始 token ID 相同；没有将相似文本、部分语义正确或 HumanEval 通过当作 token 一致。HF 参考固定为冻结 Torch 2.10、Transformers 5.20.2、BF16、eager attention / eager experts、greedy、单条、无 padding、整段 prefill、ignore EOS。

**这次重新写了什么**

新实现仅使用现有模型对象中的 Parameters，不调用原先的模型层、attention、norm、MoE 或 LM-head forward。模型容器仍显示 `MellumRocmPrecisionForCausalLM`，因为保留了已经逐字节核验的权重加载路径；真正绑定的 `forward` 和 `compute_logits` 来自新文件。

每条请求按原生 HF 的顺序独立执行：

1. embedding；按层类型计算 FP32 RoPE 频率与 BF16 cos/sin。
2. Torch RMSNorm：FP32 归约，转回 BF16 后乘权重。
3. 分开的 Q、K、V `F.linear`，Q/K norm，普通 Torch RoPE。
4. Torch 索引写入、读取 ATOM 的 BF16 KV 页；SWA 保留对应窗口历史。
5. eager QK → scale → 显式因果/SWA mask → FP32 softmax → BF16 PV。
6. o_proj、BF16 残差、第二个 RMSNorm。
7. FP32 softmax/top-k/归一化，路由权重转 BF16；按升序专家逐个做 gate/up、SiLU、down、加权与 BF16 累加。
8. 最终 RMSNorm；每条请求分别对最后一个查询位置计算 LM head；交给已有 greedy argmax。

完整的数值路径是 Torch eager；没有使用此前的融合专家、分组专家 kernel、原生 attention kernel 或 KV gather/repeat Triton 优化。位置频率初始化复用 Transformers 的 RoPE 配置工具，实际旋转由新实现执行。

**为什么能够消除此前的批量分叉**

上一轮已经证明：权重与输入相同，仅改变 Q/K/V 或专家 GEMM 的行数，也可能改变 BF16 舍入结果。其他请求会改变原批量实现中的 GEMM 形状、每个专家接收的 token 数、prefill 切块和 attention 矩阵长度。

新的路径保持 32 条请求的调度与 KV 状态，但在模型内部逐请求串行计算。每条请求的 prefill 长度、专家分组、decode 行数和 LM-head 行数都与 HF 单条参考对应。关闭 prefill 分块，因此整个 prompt 一次进入该请求自己的模型前向；没有左 padding。

这不是把服务并发参数改成 1：快速检查的 forward 记录包含 32 条活动 decode 请求。精度换来的代价是失去跨请求的矩阵计算合并，速度明显下降。

32 条快速诊断中，旧精度版耗时约 33.55 秒，新朴素版约 204.05 秒。两者在不同 GPU 上有重叠运行，仅用于说明代价，不能当作正式场景 A 吞吐评分，也不能继承旧速度版的 600+ 成绩。

**评测范围与运行约束**

- 快速集包含前轮 10 条 A 难例，补足至 32 条；HF 重新生成，旧 10 条参考重复验证全部一致。
- 扩大集使用原始 A-0000～A-0199，输入长度 541～4113，保留真实输入 ID；逐条 64 token 作为一致性检查。
- ATOM 的 `max_num_seqs` 限制每次调度批大小。严格闭环测试由客户端确保最多 32 条在途，完成后才补新请求。
- HTTP 补测脚本已准备为 32 个并发请求、每条生成 256 token、比较前 64；用户停止时尚未启动，没有测试成绩。
- BF16 权重/KV、单 GPU、编译等级 0、MTP/TBO/prefix cache 关闭。没有升级 Torch 或修改 HF 源码。
- 本轮不构成完整 A512 的正式性能验收，也不宣称该朴素版已达到吞吐目标。

**入口与文件**

```bash
CANDIDATE_GPU=1 SERVER_PORT=8540 \
  bash /workspace/run_mellum_plain_request.sh --port 8541
```

| 文件 | 作用 |
|---|---|
| `/workspace/tools/mellum_plain_request/plain.py` | 从头实现的 Torch 完整前向、逐请求 LM head、配置约束 |
| `/workspace/tools/mellum_plain_request/__init__.py` | 导出独立安装入口 |
| `/workspace/tools/mellum_plain_request/serve.py` | 在 ATOM 已加载模型上绑定新前向，接入原 HTTP 服务 |
| `/workspace/tools/mellum_plain_request/README.md` | 参考口径、启动方式与使用范围 |
| `/workspace/run_mellum_plain_request.sh` | 固定 BF16/eager/单卡/不分块等启动参数 |
| `/workspace/results/mellum_plain_A_20261105/code/` | HF 参考、32 条消融、200 条闭环、HTTP 生成及独立复核脚本 |

旧入口 `/workspace/run_mellum_eager_precision.sh` 和 `/workspace/run_mellum_target600.sh` 保持原样。默认提交脚本未切换，没有 Git 提交。

**过程记录**

初次接入的 dummy/warmup LM head 返回了每个输入 token 的 logits，违反引擎“每条请求一行 logits”的约定，并发生 GPU 访问错误。已统一为每请求最后一个位置；修正后才开始有效的 32 条和扩大测试。另一次 A200 全部入队的预填充尝试不满足严格 32 在途口径，已停止并归档为 `aborted-all-at-once/`，不计入结果。

有效验证使用冻结后的实现文件。原始 token、输入哈希、调度帧、请求提交记录、配置、失败日志和代码快照均保存在服务器实验目录。`pause-review.json` 记录停止时的完整快速测试、部分扩大测试、源码不变性和端口检查。初次启动的 GPU core dump 已归档到实验目录 `failed-startup/`，不放入 Git 或本地轻量证据包。
