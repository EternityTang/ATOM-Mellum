# Mellum 原生 ATOM 复用版实验报告

## 当前状态

Step0尚未通过。原生骨干+torch专家、原FP32路由的完整200条为9439/12800=73.7421875%，99/200完整一致；原固定10条子集291/640。此后新增默认关闭的BF16路由权重开关，HumanEval/5单条7/64变46/64，但BF16全量按用户要求停在96条，4489/6144=73.0631510%，48/96完整一致；状态stopped_by_user，不作为完整200条结果，不自动恢复。

最新按用户要求仅用GPU0，从已完成96条中选择最差HumanEval/61，比较HF与当前BF16路由版本全部28层Hidden States及公共FP32诊断logits；复现5/64、第6个token分叉。首差在L00.input_norm，第6步Hidden States相对L2在L00为1.0075%、L07为6.9900%、L27为4.0344%。同一套FP32投影下最终implement/complete的排名仍翻转，差异已进入隐藏状态。

独立共享输入实验和active norm源码确认当前OPUS_FP32_to_BF16_DEFAULT默认2为截断：融合残差逐元素等于RTZ求和，RMSNorm约99.999%元素可由RTZ复现。参数本身为BF16且已逐值核对；尚未改norm/重编译AITER或验证RNE后的生成效果，不能宣称找到一个宏就已修复。详细28层表格见/workspace/GPU0_LAYER_LOGITS_HIDDEN.md，全量和精度记录见/workspace/TORCH_EXPERTS_FULL_AND_PRECISION.md。

原有Sampler/SWA修复记录保留；HumanEval功能正确率、16k配对、正式A/B/C和优化消融仍待完成。两个默认服务已恢复，源码检查通过，无Git提交。以下按阶段保留历史数据，不能取代上述最新状态。

## 环境和实现
- 原始 ATOM：/app/ATOM，commit 26f23a0bdbf50b9010c07ee3a71171189723e767，未修改。
- 开发目录：/workspace/ATOM，分支 codex/mellum-step0；HEAD 6c86378f 保持不变，新变更待用户审批后提交。
- Transformers 5.10.1，沿用用户确认可升级的例外；PyTorch/ROCm 未升级；EvalScope 1.14.1 独立环境保留。
- 基础模型相对原始ATOM只有mellum.py与model_runner.py注册项两处；最新另有mellum_mtp.py、SpeculativeConfig及Eagle注册表的MTP入口预留和缺权重保护，默认关闭，预测头尚不可运行。
- Attention、Decoder、Model、CausalLM 的 forward 均继承 Qwen3-MoE。MoE 直接使用 Qwen3MoeSparseMoeBlock/FusedMoE；norm、packed 权重加载和 KV backend 均复用 ATOM。
- 滑窗沿用 GPT-OSS/MiMo 使用的 per_layer_sliding_window 接口；只补充 Mellum 的逐层 RoPE/SWA 与 MLP 类型选择。
- 该配置已使用现有 FusedMoE 和原生 attention，不再是旧 torch 专家循环/显式 SDPA 配置，旧测量不计入本版。
- run0：单卡、bf16 KV、level 0、eager、MTP off、prefix cache off、最大长度32768，HTTP端口8000。

## 工程验证
- 固定8条短样本，各16个greedy token：对HF原生 eager experts为128/128，对 grouped_mm为119/128；分别8/8与7/8条完整一致。这是短检查，不是200条验收。
- 原生ATOM混合批次8条，全部生成16 token；HTTP并发8条和SSE流式请求成功。
- 输入32、1023、1024、1025、8191、8192、8193、16384、20000 token，各成功生成4 token；20000覆盖16384+3616分块prefill。此项证明对应长度运行成功，未完成这些长输入与HF的逐token正确性比较。
- 初始源码审计记录9个原始组件文件逐字节不变；后续经用户授权修改共享Sampler及attention。兼容性审计已完成全部5631个检查点tensor到255个目标参数的映射及转换后逐值验证。
- 结果：results/reuse_adapter_source_audit.json、reuse_adapter_summary.json及各阶段JSON；日志见logs/reuse_adapter_*.log。

## 题方基线和待测项
基线来源 /data/datasets/baseline.json，1 GPU。A吞吐638.282724 tokens/s/GPU；A TTFT p99 5.009713s、TPOT p50/p99 0.038951/0.050144s；C TTFT p99 0.993549s、TPOT p50/p99 0.036290/0.041265s。有效HumanEval参考和target_g_a/target_g_c未填。

| 配置 | 卡数 | A tokens/s/GPU | B TPOT p50 | C TPOT p99 / TTFT p99 |
| --- | --- | --- | --- | --- |
| 原生复用接入版 | 1 | 待测 | 待测 | 待测 |

后续固定HF参考配置，完成题目正确性检查，再运行EvalScope A/B/C并profiling。根据证据选择最多3个独立优化点，分别提供开关和消融，不把现有组件复用宣称为新写优化。

## 历史与AI使用
此前自定义算子/HF包装路线已按用户要求放弃，归档于 /workspace/backups/abandoned-adapter-20261103T081334Z；原109条任务已停止，数据保留。原始Git历史完整保留，没有squash、rebase或reset HEAD。

AI辅助阅读HF/vLLM及ATOM实现、构造适配和验证工具。早期方案过度围绕HF内部实现展开，用户纠正后改为复用原生组件；中间加载检查失败和可选accelerate依赖问题的日志均保留。详见WORKLOG.md。


## MTP默认关闭和原生HF完整一致率 2026-11-03T16:43:35.112085+08:00
- Mellum MTP仅完成框架注册入口和能力检查，预测头未实现且当前权重缺失；默认speculative_config=None，显式开启在配置阶段报错，DeepSeek/MiMo已有类型映射回归通过。
- 完整200条、每条64 token：8967/12800 = 70.054688%；完整输出一致95/200。原生HF固定5.10.1、bf16、SDPA、grouped_mm，ATOM原生复用版，MTP全程关闭。
- 未达到本轮90% token一致率门槛；尚未证明差异原因，不将其归因于MTP或单一算子，也不以8条短样本替代全量结果。
- 两侧全部200条ID唯一、输出均64 token，输入完全相同，源码哈希全程不变；结果见results/atom_reuse_mtp_off_200.json。
- 首轮测量工具误访问LLMEngine.config导致启动检查失败，已归档且从头重跑；改为读取io_processor.config后完成，未更改模型。
- HTTP服务已恢复，8-token真实生成检查通过。Git HEAD和完整历史保留，未提交。


## 最差单条样本诊断 2026-11-03
- GPU0 ATOM、GPU1 HF独立追踪HumanEval/104，148个输入token；主参考两侧64-token输出均复现上一轮，同位置5/64一致，第6个输出首次分叉：ATOM complete，HF grouped_mm implement。
- 前6次预测全部28层逐层捕获；最早数值差异均为L00.input_norm，输入和该层权重完全相同。相同输入探针分别逐元素复现两侧，尚未证明内核内部具体舍入原因。
- 第6步HF grouped_mm的complete/implement/write均为31.375，实际argmax=implement；HF只切换experts为eager则选complete，整条与ATOM为58/64。HF两后端最早差异在L00.moe_output。
- 第0层常规权重及5个专家抽查一致；同router输入专家集合一致但路由权重dtype和数值不同。不可外推全部层或全部200条，也不代表HumanEval功能正确率。
- 新增两个外部诊断工具、results/worst_prompt_selected.json、results/worst_prompt_humaneval104/，报告见该目录report.md；未修改模型/引擎/采样数学，无Git提交。8项冻结源码/数据哈希均不变，HEAD保持6c86378f230d7d10734b574aa30988a349ebd2ff。
- HTTP服务恢复，未携带诊断环境变量，真实8-token生成成功。正式200条仍为70.0546875%，本轮没有替换参考或修改该结果。


## HF eager 完整结果与ATOM输出稳定性 2026-11-03
- 按用户确认固定HF5.10.1、BF16、SDPA、eager experts，完整200条64-token测量完成：8770/12800=68.515625%，81/200完整一致，未达到90%。HumanEval子集的token一致率不等于pass@1。
- GPU0重新运行ATOM、GPU1运行HF；两侧输入与上一轮一致，所有输出64 token，8项源码/数据哈希全程不变。没有修改ATOM实现或依赖，也没有提交Git。
- 两轮ATOM完整输出93/200相同；对4条样本在同一个独立ATOM进程各生成3次，HumanEval/5出现3种输出、/12出现2种，/0和/104各1种。确认部分样本greedy输出不稳定，原因尚未定位。
- 固定上一轮ATOM输出对比新HF eager为8792/12800=68.6875%；固定本轮ATOM对比旧HF grouped_mm为8868/12800=69.28125%。此为交叉分析，不替代本轮主结果，不将分数变化全部归因于HF后端。
- 证据results/atom_reuse_mtp_off_eager_200.json、_checks.json及atom_eager_reference_repeatability.json；工具新增measure_mellum_reuse_eager_200.py、run_eager_200_driver.py、check_mellum_atom_repeatability.py。
- HTTP服务恢复且真实8-token检查通过；Step0仍未通过，HumanEval功能正确率及16k配对正确性待测，后续先定位输出稳定性问题。


## 朴素基线、确定性greedy修正及四项兼容性审计 2026-11-03
- 用户要求实现单卡eager/level0/torch循环MoE/显式mask SDPA/MTP-off的题面朴素配置；新增独立mellum_naive.py、mellum_naive_ops.py与run_naive_baseline.sh，通过env开关注册，默认原生复用路径保持。
- 用户明确要求开发阶段不每次跑200，固定使用上一轮eager结果最差10条。首次quick10为381/640，但重复4条变化；旧证据归档mellum_naive_baseline_before_greedy_fix。
- 独立相同logits探针证明原Sampler无过滤器时忽略all_greedy，最高分并列会随机选。经用户明确批准，添加all_greedy直接argmax的4行修正；旧回归4失败，修正后57项相关测试通过；非greedy路径未改。
- 修正后同一固定10条为361/640=56.40625%，3/10完整一致；每条重复一次10/10相同，混合批次、HTTP8路并发及SSE通过。没有重跑200，不是正式验收。
- 用户提出RoPE、SWA、MoE选择、专家映射四项风险；对实际原生复用模型审计28层与全部5631源tensor/255目标参数。全部权重逐值一致，28层MoE选择正确。
- RoPE嵌套字段选择与当前YaRN幅度正确；attention_factor未显式转为AITER attn_factor属通用适配缺口，当前值恰好匹配默认公式；长位置存在小幅三角表差异。首次审计未压平cache单例轴导致广播误报，已独立复测并归档无效证据，没有据此改模型。
- 实测确认SWA没有变成全局，但native prefill window=(1024,0,0)包含1025个位置，decode只包含1024；query1024仍看到key0。改kernel实参为1023的独立控制实验符合HF语义。共享attention源码尚未修改。
- 源码/数据冻结核对通过；原mellum.py、Qwen源码、run0.sh等保护文件未变。GPU0:8000为原生复用版，GPU1:8001为独立朴素基线，真实8-token检查通过，显存均非零。无Git提交，HEAD与完整历史保留。
- 报告MELLUM_NAIVE_BASELINE.md、MELLUM_COMPATIBILITY_AUDIT.md；源码与数据JSON见相应results目录。


## Sampler v0.1.7 greedy兼容回移 2026-11-03
- 用户最新要求优先Sampler，SWA暂不修改。核对ROCm/ATOM v0.1.7，固定上游commit a526f0d557eeed08396670249af58bd85fa57338，原始文件/哈希保留reference/atom-v0.1.7；没有升级库或切换本地HEAD。
- 回移混合温度批次的_apply_greedy_tokens：temperatures<=EPS、固定形状torch.where；覆盖无过滤、AITER过滤及native fallback路径，保留整批greedy入口。当前AITER缺少topk_select(tie=low)，使用同最小ID并列规则的torch.argmax。
- 旧4行修复对新回归10失败，补齐后相关67项通过；32组GPU混合配置、4组整批greedy及辅助函数CUDA Graph通过。首次GPU测试输入没有预钳制0温度导致无效概率和异常，已纠正测试并保留无效日志、清理GPU core；没有据此修改依赖。
- 固定原10条、64 token、每条重复一次：361/640=56.40625%，3/10完整一致，10/10重复相同，10/10首轮输出与此前朴素基线完全一致。没有运行200条。
- 原生与朴素两个server均重启到新Sampler，真实8-token检查通过。SWA、模型和启动脚本哈希未变；SWA off-by-one仍待修复。HEAD及完整历史保持，无Git提交。
- 报告SAMPLER_V017_BACKPORT.md，证据results/sampler_v017/。


## SWA窗口边界修复 2026-11-03
- 用户要求修改SWA。先修正prefill左距离W->W-1，并扫描修复插件2处同类错误。
- 扩展GPU检查发现之前“decode正确”的结论不充分：仅测过期key未测最早合法key。冻结Gluon内核左界L-kernel_window+query_index+1使直接传W漏掉合法前缀；独立候选32例验证后，将ATOM及插件调用修正为kernel_window=W+query长度。不改模型1024配置或AITER库。
- 修正后相关93项单测通过；实际生产方法12个prefill+9个decode GPU边界及32个Gluon GPU用例通过，覆盖KV heads1/4、query长度1～4、两类V缓存视图、因果和过期/最早合法边界。
- 额外探索用例发现当前AITER fmha_v3不接受严格window1，单独记录；实际Mellum1024用例通过，未为此升级库或修改模型。
- 固定最差10条原生复用版：310/640=48.4375%，3/10完整一致，重复仅4/10相同。最长353输入，未触及窗口，不能把分数变化归因于SWA修复；模型路径仍不稳定，Step0未通过，没有跑200条。与朴素版361/640分开记录。
- Sampler v0.1.7、Mellum模型和启动脚本哈希未变；两个HTTP实例重启，真实8-token生成通过。共享层修改在attention_mha.py及plugin/attention_mha.py，完整来源和各阶段结果见SWA_FIX.md及results/swa_fix/。
- 原“只改prefill/decode不变”的阶段材料归档prefill_only/，原结构报告备份保留并已更正结论。没有Git提交，HEAD和完整历史不变。


## 原生重复输出不稳定定位 2026-11-03
- 用户询问为什么仍不稳定。仅做外部hooks/采样记录诊断，无部署源码改动。
- 固定HumanEval/120，271输入，连续16次各返回6 token；前5相同，第6个complete出现5次、implement出现11次。实际prefill边界对齐的96个预测全部等于当次argmax。
- 15次重复prefill的首次数值差异均为第0层MoE输出；此前输入、norm、QKV、attention及路由相同。
- 固定输入和路由24次：路由ID/权重全部相同，模型权重前后未变；直接固定路由调用aiter.fused_moe仍23次不同于首次，最大差0.03125。定位融合专家计算路径，内部具体机制尚未拆分，不断言已证明原子累加原因。
- 初版按6次预测分组被额外未返回预测打乱，相关跨请求比较作废；已按prefill分组复测，并与客户端16×6输出逐一验证。初版与最终目录分开保留。
- 报告NATIVE_REPEAT_DIAGNOSIS.md，最终证据results/native_repeat_diagnosis_aligned/。模型/attention/MoE/Sampler源码SHA不变，服务恢复，两实例真实生成通过，无Git提交。
