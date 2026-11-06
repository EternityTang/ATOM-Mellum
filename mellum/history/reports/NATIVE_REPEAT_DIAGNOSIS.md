# 原生ATOM重复输出不稳定：定位结果

日期：2026-11-03（Asia/Hong_Kong）。本轮仅外部诊断，没有修改模型、attention、MoE或Sampler源码，也没有Git提交。对象是当前Qwen3-MoE/FusedMoE原生复用版。

## 已确认的原因

**不稳定性已定位到AITER融合MoE的专家计算路径。** 同一份输入、同一组模型权重、同一份路由索引和路由权重，调用融合专家计算仍会产生不同输出。这些数值变化进入后续层后会改变候选token的分数排序，确定性的argmax因而选择不同token。

这不是“固定logits下Sampler仍然随机”的问题。当前Sampler的96个被检查预测位置全部等于当次logits的argmax。也不能把BF16本身当作完整解释：当前还没有定位fused_moe内部究竟哪个计算/归约或工作区步骤导致变化。

## 同一模型重复运行

固定HumanEval/120，输入271 token，每次greedy返回6个token，连续16次；单卡、eager、level0、BF16及BF16 KV，prefix cache和MTP关闭。

16次生成前5个token完全相同，为`<think>\nWe need to`。第6个token在` complete`（3559）和` implement`（2339）之间变化，分别出现5次和11次。

逐层记录表明：

- 后续15次prefill相对第1次，最早变化全部在`L00.moe_output`。
- 此前embedding、norm、QKV、attention、router logits以及MoE输入逐元素相同。
- 第0层MoE输出最大绝对差可达到0.03125。
- 该样本输入连同输出远短于1024，未触及SWA窗口边界。

首个输出分叉处的实际分数示例：

| 运行（0起算） | complete logit | implement logit | 实际选择 |
| --- | ---: | ---: | --- |
| 0 | 31.0 | 30.875 | complete |
| 2 | 31.25 | 31.375 | implement |

两次当时的token输入前缀完全相同。Sampler两次均选择正确argmax；变化首先来自模型算出的分数。

## 固定路由的隔离实验

对第0层取同一份[271,2304]隐藏状态及router logits，重复24次：

| 检查 | 结果 |
| --- | --- |
| 路由expert IDs（含顺序） | 24/24相同 |
| 路由权重 | 24/24相同 |
| 专家w13/w2参数（调用前后） | 逐元素相同 |
| 完整FusedMoE输出 | 23次与第1次不同 |
| 绕过路由、固定top-k IDs及权重，直接调用融合专家计算 | 23次与第1次不同 |
| 固定路由输出最大绝对差 | 0.03125 |

实际路径为`UnquantizedFusedMoEMethod.apply`，其`fused_experts`为None，调用导入自`aiter.fused_moe`的`fused_moe`函数。固定路由实验复用了同一原生函数、同一权重；每次传入相同内容的输入副本，没有使用HF专家或更换算法。

这将问题范围缩小到了融合专家计算/合并路径；路由选择、Sampler随机数以及本次SWA边界均不能解释这项独立实验。具体到哪条内核指令、是否某种归约顺序或工作区行为，尚未证实，不把推测写成结论。下一步应继续拆分该函数内部阶段，定位首次变化。

## 与以前的HF比较区别

此前“第0层RMSNorm最早不同”比较的是ATOM与HF两个不同实现。本轮比较的是同一个ATOM实现多次运行，最早变化在第0层MoE。两个结论针对不同问题，不相互替代。

朴素torch循环版此前10条重复稳定，而当前融合版存在变化，也支持优先检查融合MoE路径；但该观察本身不代替上述固定输入、固定路由实验。

## 诊断方法与范围

通过原有debug-helper入口安装进程内临时PyTorch hooks，复制中间张量并记录。临时Sampler包装只调用原Sampler并记录其输入/输出，没有替换采样结果或模型数学。外部hooks可能影响执行时序，因此不能将采集数据作为性能成绩；本次在采集条件下仍稳定复现数值变化。

初版工具按每6次预测划分请求，但ATOM还会做未返回给客户端的额外预测，导致记录跨请求错位。初版目录`native_repeat_diagnosis/`的跨请求逐层比较不作为最终证据；其中独立固定输入探针仍有效。已改为按实际prefill边界分组，排除额外预测，在`native_repeat_diagnosis_aligned/`复测：16组×6条=96条记录，与客户端返回token逐一核对通过。

最终证据目录：`/workspace/results/native_repeat_diagnosis_aligned/`，含trace.json、generations.json、summary.json、final_checks.json及首次数值变化的输入/输出张量。工具为`/workspace/tools/diagnose_mellum_repeat_logits_aligned.py`。

Sampler、attention、MoE、Mellum源码SHA全部保持不变。原生服务已恢复，原生与朴素两个HTTP实例均通过真实8-token生成检查。HEAD为6c86378f230d7d10734b574aa30988a349ebd2ff，未提交。
