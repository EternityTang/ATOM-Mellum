# 现有朴素版固定10条复测

日期：2026-11-03，Asia/Hong_Kong。使用现有`MellumNaiveForCausalLM`，没有为本次复测修改模型或算子代码。

**结果：361/640 token一致，56.40625%；3/10条64-token完整一致；10/10条两次输出相同。与此前朴素版的逐条输出完全相同。**

| 固定样本 | 同位置一致token / 64 | 从开头连续一致token | 第二次与第一次相同 |
| --- | ---: | ---: | --- |
| HumanEval/135 | 64 | 64 | 是 |
| HumanEval/66 | 63 | 26 | 是 |
| MBPP/25 | 64 | 64 | 是 |
| HumanEval/120 | 64 | 64 | 是 |
| HumanEval/151 | 6 | 5 | 是 |
| HumanEval/94 | 6 | 5 | 是 |
| HumanEval/124 | 7 | 6 | 是 |
| HumanEval/137 | 55 | 55 | 是 |
| HumanEval/142 | 26 | 24 | 是 |
| HumanEval/147 | 6 | 6 | 是 |

“3/10条完整一致”表示其中3条的64个输出token在每个位置都与HF相同；另外7条至少有一个位置不同。这不是HumanEval pass@1。重复稳定性则比较同一个朴素实现两次输出，与是否等于HF是两个独立指标。

配置：单GPU、BF16权重及KV、eager、编译等级0、MTP关闭、prefix cache关闭；每次只生成一条，temperature=0、ignore_eos=true、固定64 token。先跑10条，再以相同顺序全部重复一次。

HF参考使用此前已冻结的原生Transformers 5.10.1、BF16、SDPA attention、eager experts输出，本轮没有重新运行HF。10条选择及输入SHA均核对，比较真实token ID，不通过文本重新编码推算。该集合是此前差异最大的困难子集，不是完整200条验收；不能用56.41%推断整个公开集成绩。

运行审计确认28个TorchMoE/TorchExperts及28个NaiveSDPAAttention，没有FusedMoE；全部专家切片加载检查通过。复测前后仓库源码SHA及HEAD保持不变，没有Git提交。

恢复服务时的缓存归档脚本最初尝试跨挂载点rename，报Invalid cross-device link；已改为缓存同目录归档后恢复，两服务实际8-token生成及显存检查通过。数值测量在此错误前已经完成，不受影响。原始失败日志保留。

服务器证据：`/workspace/results/naive_quick10_retest_20261103T204523+0800/`，包括`validation.json`、`model_audit.json`、`selection.json`、仓库SHA及`control.json`。本地JSON副本位于`server-step0/naive_quick10_retest_20261103T204523+0800/`。

后续已按用户要求另行启动HumanEval/151的逐层诊断，分别比较现有朴素版和保留原生骨干、仅替换专家计算的版本。该诊断有独立目录，不覆盖本次10条结果。
