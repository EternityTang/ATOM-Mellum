# 朴素版SDPA调用修复

日期：2026-11-03，Asia/Hong_Kong。仅修改独立朴素实现，不改变HF冻结参考。

## 修复内容

原朴素版始终构造显式加性因果/SWA mask。HumanEval/151的同Q/K/V实验已证明：在当前Torch 2.10/ROCm环境，这使短prefill从HF的Flash Attention切到Efficient Attention，产生首次数值差异；随后影响路由、隐藏状态及token排名。

新增`naive_sdpa`，按实际位置和窗口选择调用形式：

| 情况 | 调用方式 |
| --- | --- |
| 查询和KV位置相同、递增，且窗口允许省略mask | `attn_mask=None`，多token使用`is_causal=True` |
| 单token decode，所有key可见且窗口允许省略mask | `attn_mask=None`、`is_causal=False` |
| 带历史缓存的多token查询、滑窗达到边界或其他位置布局 | 保留原`masked_sdpa`显式因果/SWA mask |

无mask分支使用`enable_gqa=True`，与当前HF SDPA集成一致。滑窗mask省略条件遵循HF的`KV长度 < window`，同时检查位置跨度；1024及1025边界不省略mask。没有把矩形cached prefill直接交给SDPA的左上角causal标志。

torch RMSNorm、RoPE、专家、BF16路由权重、残差、分页KV读写及Sampler均未为此修改。该修复不引入AITER专家或norm；eager/level0、MTP-off、prefix-cache-off保持。

## 文件修改

| 文件 | 作用 |
| --- | --- |
| `/workspace/ATOM/atom/models/mellum_naive_ops.py` | 新增mask选择与HF式GQA调用，并接入朴素attention；原显式mask函数保留用于必要场景及历史对照 |
| `/workspace/ATOM/atom/models/mellum_naive.py` | 仅更新模块说明，使其准确描述修正后的SDPA行为 |
| `/workspace/tools/check_naive_sdpa_fix.py` | 真实HF中间张量重放、实际dispatch、滑窗及cached query偏移检查 |
| `/workspace/tools/validate_naive_sdpa_fix.py` | 冻结原固定10条及HF参考，每条生成64 token，再全部重复一次 |
| `/workspace/tools/run_naive_sdpa_fix.py` | 释放GPU1朴素服务，依次运行检查，归档缓存并恢复服务；保存源码和Git状态审计 |

修改前两份模型文件备份于`/workspace/backups/before-naive-sdpa-fix-20261103T224503+0800/`。证据中另存`naive_sdpa_fix.patch`和修改前后SHA256，供审批；没有Git提交。

## 验证结果

固定原10条、相同输入ID、相同冻结HF参考（Transformers 5.10.1，BF16，SDPA attention/eager experts），greedy、ignore-EOS、64个输出token：**首轮640/640一致，10/10条完整一致**。原朴素版为361/640、3/10完整一致。

| 样本 | 修复前一致token / 64 | 修复后一致token / 64 |
| --- | ---: | ---: |
| HumanEval/135 | 64 | 64 |
| HumanEval/66 | 63 | 64 |
| MBPP/25 | 64 | 64 |
| HumanEval/120 | 64 | 64 |
| HumanEval/151 | 6 | 64 |
| HumanEval/94 | 6 | 64 |
| HumanEval/124 | 7 | 64 |
| HumanEval/137 | 55 | 64 |
| HumanEval/142 | 26 | 64 |
| HumanEval/147 | 6 | 64 |

全部10条再重复一次，仍为640/640，10/10条与首轮输出完全相同。两轮期间模型源码、输入、选择文件及HF参考SHA保持不变。

控制任务状态为`complete`，测量前后仓库SHA和Git HEAD审计通过。GPU1朴素服务已重启到修正版，两个HTTP实例均完成实际8-token生成，显存占用均约63%。没有Git提交，HEAD仍为`6c86378f230d7d10734b574aa30988a349ebd2ff`。

在HumanEval/151保存的HF中间张量上，使用相同Q/K/V重放前6次预测、全部28层，**168/168 attention输出逐元素一致**，最大绝对差0；profiler确认当前短prefill走Flash Attention。

另外10个实际模型维度用例（32个query heads、4个KV heads、head_dim=128）通过：普通/短SWA prefill、单token decode、缓存多token查询、SWA 1023/1024/1025以及1024-token prefill。与独立FP32 attention公式的最大绝对差约0.00862；无mask用例与HF集成逐元素一致。容差检查不等同于长序列端到端一致。

最初新增的人工head_dim=32、cached chunk用例未通过（Efficient与独立公式偏差约0.24）；隔离后确认原显式路径也存在此现象，MATH路径偏差约0.00358，模型真实head_dim=128用例通过。没有为此放宽阈值、修改Torch/AITER或宣称任意head_dim已支持。初次失败目录和独立诊断保留。

## 证据及范围

本轮结果：`/workspace/results/naive_sdpa_fix_quick10_20261103T224720+0800/`。初次组件检查：`/workspace/results/naive_sdpa_fix_quick10_20261103T224530+0800/`；该次在完整模型测试前退出，状态与失败日志保留。

本轮没有重跑200条或更换HF参考，没有恢复用户停止的BF16全量任务。固定10条属于困难短prompt子集，全部输入加输出不足1024；100%不能外推到完整200条、16k配对正确性或HumanEval功能正确率。Step0整体验收仍未完成。

后续按用户新指令，已在独立任务中完成修复后朴素版完整200条：12800/12800、200/200完整一致，源码未变。该结果来自实际完整测量，详见`朴素版全量测试.md`（服务器`NAIVE_FULL200.md`），不改写上述10条修复阶段的测量范围。
