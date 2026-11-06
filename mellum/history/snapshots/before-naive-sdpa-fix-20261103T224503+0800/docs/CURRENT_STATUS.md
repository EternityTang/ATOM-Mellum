# Mellum / ATOM 当前进展与下一步

更新时间：2026-11-03，Asia/Hong_Kong。本次整理核对了服务器HEAD、工作区、HTTP实例和已保存的测试证据，没有修改推理代码或提交Git。

**当前处于Step0，尚未通过。替换专家版原FP32路由的完整200条为9439/12800=73.7421875%，99/200完整一致；BF16路由测试按用户要求停在96条（4489/6144），不作完整成绩。GPU0已完成当前BF16版本最差样本HumanEval/61的全部28层Hidden States与公共FP32 logit lens对比，确认AITER norm路径默认BF16截断与HF舍入约定不同。两个服务已恢复，全量测试不会自动恢复。**

**1. 当前采用的实现**

- 原生复用版：从原始ATOM重建，复用Qwen3-MoE骨干、MoE、norm和权重加载，通过GPT-OSS/MiMo共用的attention接口接入逐层SWA，补充Mellum逐层RoPE及层类型适配。
- 独立朴素基线：torch专家循环、普通torch norm/linear、显式因果及SWA mask的SDPA。它用于复现题面朴素配置和正确性诊断，不与原生版混记成绩。
- 新增局部torch专家后端：保留原生骨干、norm、attention、路由及加载器，仅替换专家计算并跳过AITER shuffle。开关`ATOM_MELLUM_TORCH_EXPERTS=1`默认关闭；新文件为`mellum_torch_experts.py`，不替换默认run0。
- 两版均为单卡、BF16、BF16 KV、eager、编译等级0、MTP关闭、prefix cache关闭，使用ATOM OpenAI兼容server。
- MTP只有框架注册、能力检查和缺权重保护，当前权重中未找到预测头；尚不是可运行的MTP实现。
- 旧自定义适配路线已放弃并归档，完整Git历史保留。

**2. 已完成的修复与审计**

| 项目 | 已完成内容 | 仍需注意 |
| --- | --- | --- |
| Mellum结构适配 | 28层配置审计；全部5631个源tensor对应255个目标参数，按映射转换后逐值检查一致；MoE层选择正确 | 不等于端到端正确性已经达标 |
| Sampler greedy | 参考ATOM v0.1.7回移整批/混合批次greedy逻辑；argmax并列取最小token ID；未升级ATOM/AITER | 最新诊断96/96个预测等于当次argmax，模型分数本身仍可能变化 |
| SWA | prefill传W-1左距离；冻结Gluon decode传W+query长度；同步插件调用；模型窗口保持1024 | 已验证当前冻结内核及Mellum配置，不能外推所有插件或窗口配置 |
| RoPE | 按层正确选取SWA/default和全局/YaRN配置 | HF attention_factor尚未显式转为AITER attn_factor；当前配置幅度恰好一致，通用适配仍有缺口，长位置三角表有小差异 |
| 工程运行 | 短生成、混合批次、HTTP并发/SSE、最长20000输入生成通过 | 长输入运行成功不代表16k与HF逐token正确性通过 |

Sampler与SWA相关单测合计93项通过。SWA实际GPU边界用例53项通过；Sampler另有32组混合配置、4组整批greedy及辅助函数CUDA Graph检查通过。

**3. 当前测试结果：哪些可以比较**

开发阶段固定使用此前差异最大的10条，每条64个输出token，不每次重跑200条。

| 实现及阶段 | 与HF同位置token一致 | 64-token完整一致 | 每条重复一次的稳定性 |
| --- | ---: | ---: | ---: |
| 独立朴素版，Sampler修复后 | 361/640 = 56.40625% | 3/10 | 10/10相同 |
| 原生复用版，Sampler及SWA修复后 | 310/640 = 48.4375% | 3/10 | 4/10相同 |

这是困难子集的开发结果，不能推断完整200条正确率。所有输入加输出均不足1024，不能把两版差距或分数变化归因于SWA修复；两版也不只是MoE不同，norm和attention等实现均有区别。

最新现有朴素版复测再次得到361/640，10/10条两次输出相同，逐条与上次完全一致，详见`朴素版固定10条复测.md`。随后选HumanEval/151逐层比较：朴素版6/64、首差在第0层attention；原生骨干+torch专家62/64、首差在第0层norm。朴素版的显式mask使SDPA从Flash切换为Efficient，固定Q/K/V实验复现数值差异。新后端在本条生成四次相同，但不能由单条结果推断固定10条或整体成绩，详见`单条逐层对比与非融合专家方案.md`。

历史完整200条曾得到：HF eager experts参考下8770/12800 = 68.515625%，81/200完整一致；HF grouped_mm参考下8967/12800 = 70.0546875%，95/200完整一致。这些均发生在确定性Sampler修复之前，保留作历史记录，不能当作当前版本的正式成绩。修复后尚未重跑完整200条。

固定200条由164条HumanEval和36条MBPP组成，是自行获取并冻结的公开集，不是出题方提供的隐藏集。当前token一致率也不是HumanEval功能正确率pass@1。

**4. 输出不稳定的最新定位**

同一HumanEval/120输入271 token，原生版连续16次各返回6 token：前5个相同，第6个在complete（5次）和implement（11次）之间变化。

- 后15次prefill与首次比较，首次数值变化均在第0层MoE输出；此前输入、norm、attention及router一致。
- 捕获并与客户端对齐的96个预测全部等于当次logits的argmax。
- 固定隐藏状态、模型权重、专家ID及路由权重，直接调用融合专家函数24次，后23次输出均与首次不同，最大绝对差0.03125；调用前后模型权重未变。
- 已定位到融合专家计算/合并路径。具体内部内核原因尚未拆分，不能直接断言是BF16、原子操作或某种工作区问题。

这与此前“ATOM对比HF最早在第0层RMSNorm产生数值差异”不是同一个实验：此前比较不同实现，本次比较同一ATOM重复运行。

最终证据：`/workspace/results/native_repeat_diagnosis_aligned/`。初版工具曾被引擎额外的未返回预测打乱请求对齐，其跨请求比较已作废；最终工具按实际prefill边界分组，并与客户端返回token逐一核对。

**5. 融合专家的来源，以及能否不使用**

当前实际调用链是：

```text
MellumDecoderLayer
  -> Qwen3MoeSparseMoeBlock
  -> FusedMoE
  -> UnquantizedFusedMoEMethod.apply
  -> aiter.fused_moe.fused_moe
```

这是复用ATOM原生MoE时沿用的默认计算路径，不是我们新写的专家融合内核，也不是HF加载模型时强制要求。专家数64、每token选8个及专家权重属于模型结构；怎样计算这些专家可以更换实现。

非融合实现可依次对各激活专家执行gate/up线性变换、SiLU乘法、down线性变换，再按路由权重合并。现有`mellum_naive.py`里的`TorchExperts`已经采用这种做法，GPU1实例正在使用。

已增加Mellum局部的MoE后端选择：保留骨干、norm、attention和路由语义，只替换专家计算。新方法在加载后跳过AITER shuffle，使用标准布局；5631个源tensor对255个目标参数逐值检查通过。`FusedMoE`容器仍用于原生参数和路由接口，实际专家方法为TorchExpertMethod，诊断禁止调用AITER融合专家的检查通过。本条固定HF输入及路由的专家输出完全等于HF、12次相同；完整生成四次相同。现有朴素版10条稳定记录仍不能替代新后端的10条验证。

`eager`和`level 0`不会关闭代码中显式调用的融合算子。非融合版本预期性能较低，实际代价需测量；它也不会自动保证与HF达到90%一致率，其他算子的数值差异仍需检查。

**6. 参考配置、题方基线与剩余评测**

本地HF正确性参考固定为Transformers 5.10.1、BF16、SDPA attention、eager experts、greedy、每条64 token、MTP关闭。此处eager指专家实现，并非全部attention切为eager。Transformers升级已获得用户确认；没有升级PyTorch、ROCm或AITER。

题方`/data/datasets/baseline.json`已有单卡性能数字：A吞吐638.282724 tokens/s/GPU；C的TTFT p99为0.993549秒、TPOT p99为0.041265秒。`target_g_a`、`target_g_c`和有效HumanEval参考仍为null，表示尚无可用值，不能据此计算最终目标或正确性结论。

可以用同一模型在HF上的HumanEval实测建立本地功能参考，但需要固定模板、thinking设置、解码及判分方式，不能直接把模型卡或其他实现分数当成可比的验收参考。目前尚未完成这项pass@1评测。

尚未完成：当前版本完整200条≥90%门槛、HumanEval BF16差距≤2个百分点、16k配对正确性、正式A/B/C性能及profiling、优化点和消融。MTP若后续启用，还需on/off逐token完全一致。

**7. 当前服务、Git与交付物**

本次核对时GPU0原生版`8000`、GPU1朴素版`8001`均返回HTTP 200，进程存活，显存占用各约63%；上轮恢复时两版均通过实际8-token生成检查。本次未启动新的全量评测。

开发目录`/workspace/ATOM`，分支`codex/mellum-step0`；HEAD仍为`6c86378f230d7d10734b574aa30988a349ebd2ff`。工作区有待审阅修改及未跟踪文件，包含相对旧适配提交的撤销；没有新增提交，后续提交仍由用户审批。原始`/app/ATOM`及完整历史保留。

`/workspace/run0.sh`已具备；`run1.sh`、`run2.sh`、`run3.sh`尚未生成。`report.md`目前是实验记录草稿，尚不是最终不超过4页的报告；`WORKLOG.md`、文件变更说明及诊断证据持续保留。

**8. 后续顺序**

1. 已接入默认关闭的局部非融合专家后端，并完成HumanEval/151逐层诊断；下一步用固定最差10条验证稳定性及与HF的一致率。融合后端保留，尚未切换默认run0。
2. 根据已发现的朴素SDPA后端差异、原生norm和路由数值约定定位剩余误差，并补齐RoPE字段适配与长位置验证；不直接凭单条成绩改公共算子。
3. 通过快速检查后，完成完整200条、HumanEval及16k配对，建立可复现Step0。
4. Step0通过后再测正式A/B/C与profiling，据瓶颈选择最多3个独立优化点，做开关、消融及最终交付。

详细材料：`文件变更说明.md`、`Mellum重新适配方案.md`、`Sampler_v0.1.7修复说明.md`、`SWA修复说明.md`、`朴素基线实现说明.md`、`原生输出不稳定定位.md`。本状态文档服务器副本为`/workspace/CURRENT_STATUS.md`。

**9. 最新全量测试、停止要求与GPU0逐层定位**

- 原生骨干+torch专家、原FP32路由：完整200条9439/12800=73.7421875%，99/200条完全一致；原固定10条子集291/640，3/10完全一致。没有达到90%，不能称Step0通过。
- 按用户要求将路由权重转BF16接入默认关闭的`ATOM_MELLUM_ROUTE_WEIGHT_BF16`，新增`run_torch_experts_route_bf16.sh`。HumanEval/5配对生成从7/64变为46/64，连续两次相同；它只证明单条改善。
- BF16路由全量按用户要求停止：96/200条，4489/6144=73.0631510%，48/96完整一致。状态为`stopped_by_user`，不自动恢复，也不与前一行完整200条直接计算收益。
- GPU0选择这96条中的最差HumanEval/61，123输入token、5/64一致、第6个token分叉。HF与当前BF16路由版本全部28层状态及logits诊断已完成，复现冻结结果。中间层logits使用同一FP32最终norm/LM head作诊断投影，真实最终logits另存。
- 首差在第0层norm；独立实验发现原生融合残差完全等于BF16截断结果，RMSNorm约99.999%元素也可由截断复现。当前active norm走module_rmsnorm_quant，OPUS_FP32_to_BF16_DEFAULT默认2，源码定义为truncate。未修改或重新编译AITER，尚未做RNE修复后的生成验证。
- 下一步先针对该数值转换约定做单条受控验证，再依据用户指令选择固定10条等后续测试。当前不启动新的全量任务。详情见`GPU0逐层Logits与HiddenStates对比.md`、`替换专家版全量与精度检查记录.md`。

本节及文首状态优先于前面按阶段保留的描述。当前默认run0仍为原生融合版，GPU1保持朴素服务；两者已通过实际8-token生成，Git HEAD及完整历史不变，没有提交。
