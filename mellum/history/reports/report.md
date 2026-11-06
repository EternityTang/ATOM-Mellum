## 2026-11-06 Independent optimization points, frozen Torch2.10

Current separate entry: /workspace/run_mellum_opt_points.sh. Flags: OPT_ATTENTION=bounded, OPT_MOE=off, OPT_DISPATCH=graph_bucket. Retain two points: bounded head-wise Torch softmax for large prefill temporary memory; local HIP graph dispatch of unchanged per-request norm/linear operators. Preserve Qwen3-MoE backbone and original cap8 Torch experts. MTP off. Original implementations/dependencies and all Git history preserved, no commit.

Formal HTTP same GPU1: A 512/512, 67.0115 output tok/s (baseline45.9763, +45.75%); B 50/50, 19.0020 tok/s, TPOT p50=51.2094ms (baseline58.3270ms); C 0/200, failed=200, output throughput=0.0 tok/s. C latency, if available, is successful requests only; do not hide timeout coverage.

A200 at actual concurrent32: 12800/12800=100.0000% vs native HF5.20.2 eager single requests on frozen Torch2.10; first64 of SAME256-token performance outputs. Four approx16k inputs passed256/256 after full bucket warmup. Full C coverage-qualified precision: {'expected': 200, 'requests_with_64': 1, 'matches_on_available': 388, 'positions_available': 388, 'full_denominator': 12800, 'exact_64': 1, 'full64_subset_percent': 100.0}.

Three-run fixed32/64-token Engine.generate medians: control33.976, bounded33.777, MoE-script33.509, graph42.419, bounded+graph43.115 tok/s. All passed2048/2048 per run. These are short ablations, not complete official ABC per-point ablations. TorchScript norm/projection, fused expert and Triton softmax candidates failed screening; detailed failures retained. The threshold remained90%, not100%.

Production entry generic warmup HTTP smoke 64/64. Tests finished, ports8660/8661/8680/8681 closed and own services stopped. Source419 hashes and Git HEAD/status unchanged. HumanEval not rerun. Report /workspace/MELLUM_OPT_POINTS.md; evidence /workspace/results/mellum_opt_points_20261105T172604.

# Mellum 原生 ATOM 复用版实验报告

## 2026-11-06 norm_cap8 formal A/B/C performance attempt

User authorized current precision candidate A/B/C. Same GPU1, frozen Torch2.10, BF16 eager level0, no MTP/prefix cache/chunked prefill, max32 sequences, token budget16384, memory utilization0.5. A512/closed-loop32 and B50/closed-loop1 completed, all outputs256. A: 45.9763 output tok/s; TTFT p50/p99 14835.5833/20687.6428 ms; TPOT p50/p99 645.5744/705.5120 ms. B: 16.4705 output tok/s; TTFT p50/p99 452.6172/480.9692 ms; TPOT p50/p99 58.3270/60.8850 ms.

C attempted unchanged after A/B at Poisson0.30/s, timeout180s. Backend OOM during C-0009 full prefill (15194 input tokens), specifically eager attention FP32 softmax: allocation27.52GiB, free24.01GiB. Backend exited but HTTP frontend kept accepting. Stopped client on fatal failure: 50 accepted, zero complete256, partial max155; NOT a full200 result and no valid C throughput/p99. No cold-start C countercheck or configuration workaround performed.

Verified419 source hashes and Git HEAD/status unchanged; no model edits, dependency installs, or Git commit. Added only experiment/evidence scripts and documentation. Own processes exited, ports8580/8581 closed, both GPUs idle. No HF token or HumanEval retest. Report /workspace/MELLUM_NORM_CAP8_ABC.md; evidence /workspace/results/norm_cap8_abc_20261105T160808.


## 当前状态

最新RNE回归：新增113处局部norm、保留权重；native模式HumanEval/61由5/64到51/64，但固定10条356/640降至320/640，hf模式327/640，因此RNE默认关闭。两种模式组件验证通过，整体正确性未解决，见/workspace/MELLUM_RNE_FIX.md。RNE收尾两服务实际生成均通过；22:54:52起检测到另一路独立朴素200条任务使用GPU1并暂停8001，未干预该任务。原BF16路由全量仍停止在96条。

Step0尚未通过。原生骨干+torch专家、原FP32路由的完整200条为9439/12800=73.7421875%，99/200完整一致；原固定10条子集291/640。此后新增默认关闭的BF16路由权重开关，HumanEval/5单条7/64变46/64，但BF16全量按用户要求停在96条，4489/6144=73.0631510%，48/96完整一致；状态stopped_by_user，不作为完整200条结果，不自动恢复。

最新按用户要求仅用GPU0，从已完成96条中选择最差HumanEval/61，比较HF与当前BF16路由版本全部28层Hidden States及公共FP32诊断logits；复现5/64、第6个token分叉。首差在L00.input_norm，第6步Hidden States相对L2在L00为1.0075%、L07为6.9900%、L27为4.0344%。同一套FP32投影下最终implement/complete的排名仍翻转，差异已进入隐藏状态。

独立共享输入实验和active norm源码确认当前OPUS_FP32_to_BF16_DEFAULT默认2为截断：融合残差逐元素等于RTZ求和，RMSNorm约99.999%元素可由RTZ复现。随后新增Mellum局部RNE norm，未重编译AITER；native顺序使HumanEval/61从5/64变51/64，但同配置固定10条从356/640变320/640，hf顺序为327/640。两种RNE各自282项数值检查及全部5631源tensor/255目标参数验证通过、10/10重复稳定；由于整体退化，RNE默认关闭。详见/workspace/MELLUM_RNE_FIX.md，不能宣称整体正确性已经修好。

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

## 朴素版SDPA调用修复 2026-11-03
- 用户要求修复朴素版本。新增naive_sdpa选择可等价省略mask的HF causal/GQA调用；单token decode关闭causal；SWA边界和cached多token查询保留原显式mask。mellum_naive_ops.py是唯一计算逻辑改动，mellum_naive.py仅更新说明；专家/norm/路由/KV/Sampler未为此修改。
- 同HF Q/K/V重放HumanEval/151前6步×28层，168/168输出完全一致，短prefill实际走Flash。真实32Q/4KV/head_dim128的10个因果、SWA1023/1024/1025和cached偏移组件用例通过；最初人工head_dim32的Efficient异常另留证据，没有放宽阈值或升级依赖。
- 原固定10条，冻结原HF SDPA/eager专家参考，每条64token，两轮均640/640=100%，10/10完整一致、10/10重复相同。修复前361/640、3/10完整一致；HumanEval/151从6/64变64/64。没有重新选择样本、更换参考或重跑200条，不等于Step0通过。
- 工具check_naive_sdpa_fix.py、validate_naive_sdpa_fix.py、run_naive_sdpa_fix.py；证据results/naive_sdpa_fix_quick10_20261103T224720+0800/；初次组件检查失败材料保留在224530目录。修复前模型文件备份before-naive-sdpa-fix-20261103T224503+0800，结果目录保留review补丁和前后SHA。
- 控制任务complete，源码/Git HEAD冻结审计通过；GPU1朴素服务重启到修正版，两服务实际8token生成及约63%显存检查通过。没有Git提交，完整历史保留，BF16全量保持stopped_by_user。
- 报告NAIVE_SDPA_FIX.md，本地朴素版SDPA修复说明.md。

## Mellum局部RNE回归结论
新增113处可选RNE norm，保留原生骨干、权重加载和全部参数值。native模式282项FP64数值参考检查通过，hf模式282项共享输入精确HF检查通过，两种模式固定10条重复均稳定。HumanEval/61为5→51（native）/21（hf），但固定10条356→320/327，整体退化，因此开关默认关闭；原生默认服务保持。完整证据与修改清单见/workspace/MELLUM_RNE_FIX.md。没有新200条或Git提交，BF16全量保持停止。

## 修复后朴素版完整200条验证 2026-11-03
- 用户要求完整200条。仅新增measure_mellum_naive_200.py和run_mellum_naive_200.py，GPU1、单卡、串行、BF16、eager/level0、MTP和prefix-cache关闭；每条greedy固定64token、ignore-EOS。模型及算子源码不变，冻结同一HF SDPA/eager专家参考。
- 完整结果12800/12800=100%，200/200条完整一致；HumanEval 10496/10496、164/164完整一致；MBPP 2304/2304、36/36完整一致。全量中的原固定10条逐条复现修复后输出。
- 运行审计为MellumNaiveForCausalLM，28层TorchExperts/NaiveSDPAAttention，5376个专家切片完整加载，无FusedMoE，参数均BF16。原始messages/chat-template输入ID和SHA逐条核对；每条之后核对源码、配置、数据和HF参考哈希；完整仓库/HEAD冻结通过。
- 另读原始JSONL独立聚合，核对200个唯一ID、输入、冻结HF目标与全部输出，结果一致，保存independent_review.json。状态及控制complete，测量耗时1617.829秒（含引擎加载，不是正式性能成绩）。GPU1朴素服务恢复，两HTTP实例实际8token生成和约63%显存验证通过。
- 证据results/mellum_naive_200_20261103T225453+0800/，本地同步JSON/JSONL；报告NAIVE_FULL200.md（本地朴素版全量测试.md）。达到本地冻结公开集90% token一致率门槛；尚未完成HumanEval功能pass@1、16k配对或其他Step0验收，原生复用版与朴素版成绩分开记录。
- Git HEAD仍6c86378f230d7d10734b574aa30988a349ebd2ff，无提交，完整历史保留。之前BF16路由全量保持stopped_by_user，没有恢复。本轮200条没有全部重复，不把前次10条重复检查扩写为200条稳定性验证。

## HumanEval/66：RNE后分叉的受控诊断
GPU0复现RTZ 64/64、native RNE 5/64、HF顺序RNE 39/64，均重复稳定。第6步HF write/complete并列31.0，native RNE为30.875/31.0，greedy因而改选complete；RNE末层hidden误差更小，仍不能保证候选词排名不变。同输入组件实验确认合并QKV与原生prefill attention各自有微小差异；固定HF排列的专家、dense BF16路由权重、o_proj及gate均56/56精确。原生路由排列专家43/56精确，补充独立探针未复现参考，因果机制尚未证明。生产代码/依赖不变，不启用RNE；原BF16路由全量仍停止。详见/workspace/HUMANEVAL66_DIAGNOSIS.md，证据humaneval66_diagnosis_20261103T231407+0800；无Git提交。

## 以朴素实现为参考的逐算子证据
HumanEval/61朴素64/64、原生骨干torch专家/BF16路由/RNE-off为5/64，两次稳定。28层prefill及第6步decode共56组自身重放控制全部通过；KV K/V各336/336一致，固定路由专家内部全56/56一致。独立差异在norm/融合残差、合并QKV、全局层RoPE、实际attention及路由排列；不是由权重加载或Sampler随机性解释。旧原生RoPE子阶段追踪有SWA遗漏/共享覆盖，已标注作废并补齐有效追踪；独立hidden/logits证据保留。详见/workspace/NAIVE_NATIVE_OPERATOR_DIAGNOSIS.md，主目录naive_native_operator_diagnosis_20261103T235149+0800。生产代码和数据不变，无Git提交。

## vLLM ROCm实现对照 2026-11-04
只读核对当前vLLM b14e2f2792ae和迁移前原生Mellum ea106db80691。骨干复用方式相近，具体算子不同：ATOM小维norm走OPUS，vLLM AITER普通norm入口对应CK，eager默认provider为自身HIP；残差舍入顺序也不同。vLLM tgemm只对gfx950启用，ATOM gfx942直接调用；attention分别是vLLM自己的prefill/Triton/HIP与ATOM AITER FMHA/Gluon/ASM。RoPE及MoE须区分当前Transformers backend与旧原生模型。详见/workspace/VLLM_ROCM_IMPLEMENTATION_COMPARISON.md，含路径、行号、源SHA和对应既有数值证据。没有安装/运行vLLM、修改生产实现、恢复全量或Git提交，不将静态源码对照作为新增正确性成绩。

## 基于ROCm对照的Mellum局部精度路径 2026-11-04
新增MellumRocmPrecisionForCausalLM，保留原生Qwen3-MoE骨干、参数容器和loader；113处norm与BF16残差边界、独立RoPE、SDPA/native attention、packed/split QKV、HF有序路由和gate/head GEMM可局部选择。全部Parameter对象/名称保持，两轮5631→255权重逐值检查通过。公共算子及装饰过的模型文件未改，没有升级依赖。

固定10条消融为CK/native418、torch/native431、SDPA/packed465、SDPA/split640/640；额外H61在split后仍51/64，只换gate GEMM变64/64。最终correctness（torch gate/head）固定10条两轮640/640、10/10稳定，附测H61两轮64/64。独立HF norm 339项、两类RoPE六位置组件检查精确。两轮223文件冻结和独立聚合通过；首轮旧命名及源码完整归档。

脚本/workspace/run_rocm_precision.sh启用新路径；默认run0仍原生融合版。新路径尚未测完整200条、HumanEval功能分、16k或正式性能，不继承朴素版旧全量成绩。两服务恢复并实际生成，旧BF16全量保持停止，无Git提交，完整历史保留。详见/workspace/MELLUM_ROCM_PRECISION.md及results/mellum_rocm_precision_linears_20261104T071652+0800，最终补丁与SHA在backups/before-mellum-rocm-precision-20261104T070813+0800/。

## 新ROCm精度模式完整200条 2026-11-04
在同一最终correctness实现上完成独立200条检查：12800/12800=100%，200/200完整一致；HumanEval164/164，MBPP36/36。固定10条全部复现前次输出。重新验证5631源tensor→255参数，输入模板/ID/SHA逐条核对，228文件逐样本冻结和完整仓库/HEAD冻结通过。独立重读原始JSONL/参考/数据重新汇总通过，差异列表为空。

只增加测量/控制/独立复核工具，无生产改动、依赖升级或Git提交。GPU0服务恢复，GPU1未停止/重启，两实例实际8token及显存检查通过，旧BF16全量仍停止。证据results/mellum_rocm_precision_200_20261104T075934+0800，报告/workspace/ROCM_PRECISION_FULL200.md。达到新路径本地公开集token门槛；功能HumanEval、16k和正式性能尚待完成。耗时1333.670秒包含加载，不作为正式性能成绩；完整200条未全部重复两轮。

## GPU1朴素基线对齐实验收尾 2026-11-04
- 组件与短程对照证实减少GQA展开、分页KV重复整理和torch专家筛选同步有收益；全部保留显式mask SDPA、torch循环MoE、单卡BF16、eager/level0、MTP关闭。生产源码和依赖未改，外部工具与版本快照完整保留。
- 连续KV原候选完整A：512/512成功、131072输出token、1171.6931秒、111.8655输出token/s，仅为题方638.282724的17.526%；TTFT p99 10.833759秒、TPOT p99 0.282335秒。原始512条SQLite独立核对通过，未达到题方性能。
- C原候选因冗余连续KV未设总预算而OOM；已保存失败原始记录并修复为32GiB跨28层预算、紧凑容量及超限回退分页读取，长prefill前释放冗余缓存。修正版只重测原200条C，保留0.30速率和180秒超时：1成功、199超时，845.3545秒，无再次OOM。成功率0.5%，不能用唯一成功请求的p99宣称延迟改善。A和C明确为两个实现版本，未再跑A。
- 源码/原始prompt/请求参数和逐请求记录均独立核验；C完整运行冻结通过，A当时源码用revision2归档SHA核对。final_review.json核对生产源码和HEAD不变。current+专家重排固定10条640/640；全显式候选361/640，不覆盖原两版200条100%正确性材料。
- 初次phase文件创建顺序失败、vector候选A中断、连续KV C OOM均留档，不标记为用户停止或完整成绩。此次实测未复现题方性能；题方移植源码、prefix cache等细节仍未知。
- 最终推理/评测进程退出、8000/8001关闭，两卡0%利用率、各297623552字节显存（约284MiB）；不恢复旧服务或队列。Git HEAD仍6c86378f230d7d10734b574aa30988a349ebd2ff，没有提交。报告NAIVE_BASELINE_ALIGNMENT_GPU1.md、本地GPU1朴素基线对齐实验.md；证据results/naive_baseline_alignment_gpu1_20261104T172349+0800/。

## 2026-11-05 Precision full-chain audit and chunked SWA correction

Reviewed input/scheduling, all 311 parameter tensors, norms/projections/RoPE, attention/KV, routing/experts, logits/greedy and output. Weight hashes match native HF 5.20.2 on frozen Torch 2.10. Same single/full execution: 3696 recorded tensors bitwise equal. Fixed only tools/mellum_eager_precision/install.py:eager_one to retain W-1 prior keys plus the query chunk. Same chunk execution then matches all 5082 recorded tensors. Ten A hard cases pass 640/640 for each matched prefill mode. HF full vs chunk itself matches 276/640. Batch shape sensitivity is independently reproduced in identical-input QKV and expert GEMMs. Post-fix offline 32-request smoke passes all requests but only 187/256 first-eight IDs match HF static batch. This is not full A/200 or performance acceptance.
Speed dependencies and Git status/HEAD unchanged; no commit. Evidence: /workspace/results/mellum_chain_audit_20261105. Detailed report: /workspace/MELLUM_CHAIN_AUDIT.md. Tests finished.


## 2026-11-05 Standalone plain-request precision path; stopped by user

New entry: /workspace/run_mellum_plain_request.sh. Independent Torch numerical forward in /workspace/tools/mellum_plain_request/plain.py; ATOM supplies loaded Parameters, scheduling and KV pages. Whole prefill, no padding/chunking, per-request full model and LM head, frozen Torch 2.10, BF16, no MTP. Reference: native HF 5.20.2 eager/eager generated one request at a time.
Completed quick32: 2048/2048 token IDs, 32/32 exact; controlled old precision 596/2048. Diagnostic time ~204s vs ~34s, not formal performance acceptance.
USER REQUESTED NO FULL RUN: all expansion/reference processes and the pending HTTP queue have been stopped. At stop, ATOM 160/200 and HF 180/200 completed; 160 paired rows match 10240/10240. Partial result only. HTTP never started. Do not automatically resume tests. GPUs idle, ports closed. Old source/entries and Git HEAD/status unchanged; no commit.
Report: /workspace/MELLUM_PLAIN_REQUEST_A.md. Evidence: /workspace/results/mellum_plain_A_20261105/pause-review.json.


## 2026-11-05 Framework reuse / SWA stage check

New independent entry: /workspace/run_mellum_reuse_request.sh (default framework_eager). Reuses inherited Qwen3-MoE backbone/decoder/MoE container and existing per-layer Attention components. Retains HF-compatible eager math and per-request numerical isolation: fixed10 640/640; actual 32-request decode 2048/2048, 32/32 exact against frozen Torch 2.10 native HF 5.20.2 single-request references.

Original accelerated SWA kernels are NOT enabled by default: both phases 17/640, prefill-only 25/640, decode-only 339/640. These short ablations failed. Component reuse is distinct from kernel reuse. 21 layers use window 1024; 7 are full attention. Existing plain/precision/speed implementation and dependency hashes unchanged; no Git commit, no full200/HTTP/performance run. Short tests finished, ports closed, both GPUs idle. Evidence: /workspace/results/mellum_reuse_stages_20261105; report: /workspace/MELLUM_REUSE_STAGES.md.

## 2026-11-05 Partial batching with HF token checks

New independent candidate: /workspace/run_mellum_parallel_precision.sh, default norm_cap8. Decode shares the inherited Qwen3-MoE backbone across requests; MoE groups requests by expert and caps each expert GEMM at 8 rows. Norm and sensitive projections remain per request; full prefill remains per request; original Attention component retains HF-compatible eager math. Actual decode batch32, largest pre-chunk expert group32, matrix chunk8. Fixed10 640/640; fixed32 passed twice at 2048/2048 and 32/32 exact against frozen Torch2.10 native HF5.20.2 single-request eager references.

GPU0 sequential short-test pair, other GPU idle: original serial 128.681 s, norm_cap8 61.262 s, 2.1005x. This is fixed32 real-A input/64 output, not formal A512 or 600 tok/s. Unrestricted expert batches pass10 but fail32 (975/2048), so not adopted. Full200/HTTP/performance queues not resumed. Existing implementation hashes and Git status/HEAD unchanged, no commit. Tests ended, ports closed, both GPUs idle. Report: /workspace/MELLUM_PARALLEL_PRECISION.md; evidence: /workspace/results/mellum_parallel_precision_20261105.
