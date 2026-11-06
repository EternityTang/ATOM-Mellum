# 替换专家版：全量测试与精度检查记录

日期：2026-11-03，Asia/Hong_Kong。所有比较使用冻结的同一HF参考：Transformers 5.10.1、BF16、SDPA attention、eager experts，greedy固定64 token。token一致率不是HumanEval pass@1。

## 两次全量任务的实际状态

| 版本 | 已完成 | token一致 | 完整64-token一致 | 状态 |
| --- | ---: | ---: | ---: | --- |
| 原生骨干+torch专家，原FP32路由权重 | 200/200 | 9439/12800 = 73.7421875% | 99/200 | 完成，未达到90% |
| 同一骨干，路由权重转BF16 | 96/200 | 4489/6144 = 73.0631510% | 48/96 | 用户要求停止，保留部分结果 |

两行覆盖范围不同，不能直接把两个百分比之差当作BF16转换效果。FP32路由版本中HumanEval为7586/10496，MBPP为1853/2304；原固定最差10条子集为291/640=45.46875%，3/10完整一致。该子集是从本次完整输出提取，不是另行重跑10条稳定性检查。

原FP32路由任务在模型源码保持不变时完成。其后按用户要求，把路由转换接入默认关闭的`ATOM_MELLUM_ROUTE_WEIGHT_BF16`开关，再进行第二次测量；没有改动前一份结果。BF16任务停止时已记录请求、向工作进程组发送终止信号，保留96条及日志，恢复GPU0服务；控制状态更正为`stopped_by_user`，不把它写成完成200条或模型执行失败。

## HumanEval/5的共享输入及配对实验

从原FP32路由全量任务中选取HumanEval/5：124输入token，7/64一致，第8个token首次分叉。GPU1独立重放复现本轮输出，两次相同；逐层捕获前8步并核对相同前缀。

- 28层参数均为BF16，5631个源tensor逐值匹配；未发现漏映射或参数dtype错误。
- 首差在第0层input norm；同一份输入直接调用原生norm与HF仍有差异。Q/K norm、融合残差也有差异。
- 固定HF输入和路由，torch专家输出与HF完全一致，12次相同。
- 固定HF router logits，原生权重为FP32、HF为BF16。把原生权重转成BF16后，dense路由权重逐元素等于HF；专家输出最大差从0.03125降至0.00006104，尚有很小残差，不把它写成完全一致。

随后在同一个独立模型进程中做配对生成，只改变路由权重转换：

| 模式 | 一致token / 64 | 连续一致前缀 | 第8步选择 |
| --- | ---: | ---: | --- |
| 原生FP32路由权重 | 7 | 7 | Python |
| 转BF16，第一次 | 46 | 46 | function |
| 转BF16，第二次 | 46 | 46 | function |

这只改变每次路由结果的权重dtype，没有改路由算法或强制专家ID；上游状态变化后，后续层的自然路由结果仍可能变化。该实验发生在独立进程中，未改变当时运行的FP32路由200条任务。

## BF16路由后的剩余差异

在BF16开关接入后，GPU0跑第二次全量，GPU1追踪HumanEval/5的第47步。独立诊断复现46/64结果，相同46-token前缀下仍在第0层norm出现首差。参数dtype全部BF16；在共享输入下，28层router和专家输出可以复现HF，BF16 dense路由权重也全部一致。

第47步HF选择` Edge`（31.25），其次换行30.125；ATOM中两者28.5并列，按最小ID规则选择换行。说明路由权重转换确实修正了一个差异来源，norm、残差等计算路径仍需要检查。RoPE独立探针尚待复核，本记录不把其异常幅度作为已证实的根因。

用户之后停止BF16全量，并要求GPU0针对差异最大样本逐层检查。已完成HumanEval/61的Hidden States与公共FP32 logit lens对比，进一步发现AITER norm路径默认BF16截断；详见`GPU0逐层Logits与HiddenStates对比.md`。

## 代码与证据

本阶段仅在Mellum局部专家后端加入可选路由权重转换，新增环境变量注册及`run_torch_experts_route_bf16.sh`。骨干、公共MoE、norm、attention、Sampler和AITER源码未为此修改；没有升级依赖，没有Git提交。完整旧源码备份于`/workspace/backups/before-route-weight-bf16-20261103T214958/`。

证据目录：

- FP32路由完整200条：`/workspace/results/mellum_torch_experts_200_20261103T212135+0800/`。
- BF16路由已停止任务：`/workspace/results/mellum_torch_experts_route_bf16_200_20261103T215005+0800/`。
- FP32路由HumanEval/5逐层诊断：`/workspace/results/torch200_humaneval5_diagnosis_20261103T212955+0800/`。
- 路由精度配对生成：`/workspace/results/route_bf16_humaneval5_20261103T213733+0800/`。
- BF16路由第47步复核：`/workspace/results/bf16_route_followup_humaneval5_20261103T215304+0800/`。

Step0仍未通过；HumanEval功能正确率、16k配对、正式A/B/C和优化消融尚未完成。BF16全量测试不会自动恢复。
