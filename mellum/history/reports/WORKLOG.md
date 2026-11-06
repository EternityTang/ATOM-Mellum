# Work log

## 2026-11-03 (project day index pending official D0)
- Read PDF, connected allocated two MI308X GPUs.
- Preserved frozen /app/ATOM; copied into /workspace/ATOM at commit 26f23a0.
- Frozen Transformers 5.2.0 has no Mellum support; evalscope not installed. No dependency upgrades.
- Checkpoint contains 5631 tensors, no mtp keys or layers beyond 27.
- baseline.json has A/C numbers but HumanEval reference invalid/null.
- Started eager single-GPU adaptation with plain expert loop and explicit masked SDPA.
- First GPU smoke: raw code, prose, chat/thinking outputs matched the independent reference text (3 cases). Raw token IDs initially omitted by response serialization; added a typed diagnostic field before claiming token-level agreement.
- Fixed two adapter bugs isolated by smoke: uncached-prefill metadata has no block table; YaRN inverse frequencies must use the runner device. Retained failed logs.
- Final smoke: 3x16 raw IDs exactly match independent reference; boundaries 8x64 (through 16384 input tokens) all match. Concurrent API smoke 8/8 successful; not an official performance result.
- User confirmed no additional correctness materials; froze 200 public prompts (164 HumanEval + 36 MBPP test tasks), source/selection/hash recorded before inference.
- Official card reports HumanEval+ 91.5%, not original HumanEval. Chose to establish paired same-setting HF/ATOM HumanEval reference instead of comparing incompatible scores.
- Automatic approval rejected the command preparing/starting the public200 batch validation, without a specific reason. Command did not execute; full200 and HumanEval remain unvalidated.

## 2026-11-03 当前版本一致率实测（按用户要求冻结模型）
- 200条固定公开prompt，每条greedy生成64 token，ignore_eos=True；ATOM与参考均单请求推理，分别使用分配的两张GPU，非性能测试。
- token一致率：10773/12800 = 84.1640625%；整条64 token完全一致：136/200 = 68%。
- HumanEval组164条：83.67949695%；MBPP组36条：86.37152778%。这些是token一致率，不是pass@1。
- 参考仍为HF Mellum源码回移实现。原始输入、两侧token和源码SHA256见results/current_token_agreement.json。
- 200条ID均唯一，所有请求两侧均为64 token；全程源码和样本哈希保持不变。耗时2663.26秒（约44.4分钟）。
- 遵照最新要求，不以一致率为门槛继续修改模型，不改ATOM引擎，不启动其他阶段，不作Git提交。此前的适配改动保留在工作区，待用户审批。

- Installed EvalScope[perf] 1.14.1 in /workspace/venvs/evalscope; bare evalscope command available. Frozen /opt/venv pip freeze identical before/after. Perf smoke 2/2 succeeded, 8 tokens each. New CLI uses --stream flag. No ATOM changes or git commits.

- User confirmed Transformers need not remain frozen. Upgraded /opt/venv Transformers 5.2.0 -> 5.10.1; pip freeze confirms no other package changed. Native AutoConfig/MellumForCausalLM and 2 generation prompts passed; ATOM restarted and 8-token request passed; EvalScope CLI passed. No ATOM source edits or git commits in this upgrade. ATOM package metadata still pins 5.2.0. Earlier token agreement used the 5.2.0 source-backport reference and was not remeasured against native 5.10.1.


## 2026-11-03 原生 Transformers 一致率重测启动
- 按用户要求，仅重新测量一致率；不修改 ATOM 模型或引擎，不进行 Git 提交。
- 参考替换为安装包 Transformers 5.10.1 原生 MellumForCausalLM，bf16、SDPA，官方 generate，单请求 greedy 64 token、忽略 EOS。
- 固定 200 条公开 prompt，逐条核对文本哈希和输入 token 与旧测量完全一致。
- 两侧分别使用 GPU0/1，均 concurrency=1；不是性能测试。
- 测试工具：/workspace/tools/measure_native_token_agreement.py；进程记录：logs/native_token_agreement.pid。
- 结果逐条写入 results/native_token_agreement.json；日志：logs/native_token_agreement.log。完成状态以结果 summary.status 为准。


## 原生一致率评测暂停 2026-11-03T15:43:42.577643+08:00
- 用户明确要求暂停；已核对 PID 与命令后使用 SIGSTOP，保留进程内存和已完成结果。
- 已完成 109/200；阶段性 token 一致率 70.384174%，不是最终成绩。
- ATOM 服务继续运行，未修改源码或执行 Git 提交。恢复需用户后续指示。


## 原生 ATOM 组件复用重建 2026-11-03T16:24:40.696347+08:00
- 用户要求放弃先前改动，复用Qwen3-MoE骨干、MoE、norm及权重加载，沿用GPT-OSS/MiMo原生SWA接入。
- 先归档源码、工具、参考、结果和未提交diff，再git restore --source原始基准 --worktree；未改变HEAD或历史，未提交。
- 中间HF packed experts尝试的加载覆盖检查未通过，日志保留，后按用户要求整体放弃该路线，不作为当前实现证据。
- 新实现只涉及Mellum构造层和注册项；四个forward继承原生Qwen3-MoE，旧mellum_ops及HTTP token_ids扩展撤销。
- 原109条任务进程停止并归档；新验证仅8条短输入、混合批次、9个长度、HTTP并发/流式。
- 验证脚本曾误要求未安装的可选accelerate，修正为无该依赖时CPU加载再移至GPU；没有安装或升级依赖。
- 短样本HF eager为128/128、grouped_mm为119/128；9个长度均成功，20000输入经过分块prefill；不代表正式验收。
- 数据和原有Git提交保留；正式200条、长输入HF对比、HumanEval、A/B/C未启动。


## MTP入口预留与原生一致率重测
- 复用现有EagleProposer/验证框架预留Mellum注册入口，默认关闭；当前5631个tensor无MTP头，显式开启提前报错，尚无可运行预测器。
- 用户要求跑通后与Transformers验证token一致性，现启动新的固定200条、每条64 token，原生HF SDPA/grouped_mm；不是恢复旧109条进程。
- 暂停GPU0 HTTP服务用于同配置原生ATOM token诊断，控制进程在测量结束后恢复服务。未提交Git。


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


## 固定 HF eager 正确性参考，启动完整重测 2026-11-03
- 用户明确要求使用eager；本地正确性参考固定原生Transformers5.10.1、BF16、SDPA attention、eager experts，保持当前ATOM实现及MTP关闭配置不变。
- 新工具measure_mellum_reuse_eager_200.py，独立结果atom_reuse_mtp_off_eager_200；不覆盖grouped_mm历史结果。GPU0重新跑ATOM、GPU1跑HF，固定原200条及每条64 token。
- 运行控制run_eager_200_driver.py临时释放GPU0 HTTP服务，结束或失败后恢复；不作Git提交。此测量仍不是HumanEval pass@1或16k长上下文验收。


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

## 现有朴素版固定10条复测与两种非融合实现逐层诊断 2026-11-03
- 现有朴素版代码不变，固定原10条各64 token，全部重复一次：361/640=56.40625%，3/10完整一致，10/10重复相同，逐条复现以前结果；比较冻结HF eager专家/SDPA参考，没有跑200条。
- 复测测量完成后，恢复脚本缓存跨挂载点rename报错，已修正为缓存同目录归档；原始错误记录保留，实际生成与显存验证后恢复。证据results/naive_quick10_retest_20261103T204523+0800/。
- 用户要求选一条不一致prompt逐层比较现有朴素版，以及保留原生骨干只替换专家的版本。新增mellum_torch_experts.py，沿用原生容器/路由/加载器，用torch计算专家并跳过AITER shuffle；envs/model_runner登记ATOM_MELLUM_TORCH_EXPERTS，默认关闭；新增run_torch_experts.sh，不改默认run0。
- HumanEval/151输入173，朴素版6/64一致、首次分叉第6个；局部torch专家版62/64、首次分叉第55个，主测与补充四次相同。两版5631个源tensor全部逐值匹配。局部后端实际28层方法为TorchExpertMethod，融合专家禁止调用检查通过；固定HF路由的新专家输出等于HF、12次相同。
- 朴素版首差在L00 attention；固定Q/K/V隔离，HF causal与显式KV展开均走Flash且完全一致，等价显式mask改走Efficient并复现max_abs=0.0009765625。Prefill首次相对L2超过1%在L03.post_norm；该阈值仅诊断用途。
- 局部后端首差在L00原生norm；prefill首次相对L2超过1%在L00.moe_output。原生路由FP32与HF BF16约定保留，没有为对齐改公共算子。第55步在相同54-token前缀下，HF ':'=38.0、','=37.5，局部后端二者37.5并列，argmax正确选ID46逗号。
- HF首次device_map启动缺accelerate，已改普通加载；新后端最初错用Config.data_parallel_size，已修为parallel_config.data_parallel_size；失败启动日志保留，续跑复用已完成HF/朴素证据。没有升级依赖。
- 外部hooks捕获前6步全部28层，另补第55步并核对同输入前缀；报告MELLUM_VARIANT_COMPARISON.md，证据results/mellum_variant_comparison_20261103T210115+0800/及decode55/。单条96.875%不能推断新后端完整10条或200条，Step0仍未通过。
- 原Mellum/Qwen、公共MoE/norm/attention/Sampler源码本轮SHA不变，HEAD及完整历史不变，无Git提交。GPU0原生默认与GPU1朴素服务恢复到原配置，实际8-token生成和显存检查通过。

## 完整200条、BF16路由权重及用户停止要求 2026-11-03
- 原生骨干+torch专家完整200条单卡顺序测量完成：9439/12800=73.7421875%，99/200完整一致。HF为冻结的5.10.1、BF16、SDPA/eager experts参考；未重跑或替换HF输出，输入/源码逐条SHA冻结。该结果不是HumanEval pass@1，未达到90%。
- HumanEval/5从实测不一致结果抽取，GPU1重放7/64、首个分叉第8个，完整权重逐值匹配；共享HF路由的专家输出完全一致。仅转换路由权重FP32->BF16的独立配对生成变46/64、两次相同，未影响当时运行中的原全量任务。
- 随用户要求接入默认关闭的ATOM_MELLUM_ROUTE_WEIGHT_BF16；只修改mellum_torch_experts.py、envs.py并新增启动包装。旧源码/环境注册保留在before-route-weight-bf16-20261103T214958备份，Git历史不改。
- 用户要求一卡诊断、另卡BF16全量：GPU0运行BF16路由完整任务，GPU1复核HumanEval/5第47步。参数均BF16；相同输入下28层router/专家与HF一致，BF16 dense路由权重也一致；norm/QK norm、融合残差仍有差异，RoPE探针幅度尚未独立排除探针因素，不作为结论。
- 用户明确停止BF16全量，已验证进程组后SIGTERM停止，保留96条4489/6144，48/96完整一致；GPU0服务恢复，控制状态改为stopped_by_user，原预期终止异常记录保留，不记为完成200条或模型故障，不自动恢复。

## GPU0最差样本逐层Hidden States和Logits 2026-11-03
- 按用户最新要求仅用GPU0，选择当前BF16版本已完成96条中同位置一致数最少的HumanEval/61（123输入，5/64一致，前5相同），依次运行HF和ATOM。捕获六次预测全部28层，同输入前缀核对通过，ATOM两次稳定并复现停止任务结果。
- HF层输出与原生真实层边界residual对齐；没有额外torch加法构造原生边界。保存每层Hidden States及统一FP32 final norm/LM head投影，完整logits shape[2,6,28,98304]，真实最终BF16 logits另存。
- 首差L00.input_norm；第6步Hidden相对L2：L00 1.0075%、L07 6.9900%、L27 4.0344%。真实HF implement=30.0/complete=29.875，ATOM implement=29.625/complete=29.75。公共FP32投影末层仍翻转，证明差异已进入隐藏状态。
- 独立BF16转换实验：融合残差与FP32和截断RTZ逐值一致；RMSNorm归一化乘gamma后截断一次与原生99.998939%元素一致，仍有约3个元素差异，不宣称完全复现。
- 源码核对active dispatch为module_rmsnorm_quant，optCompilerConfig默认OPUS_FP32_to_BF16_DEFAULT=2；opus.hpp将2定义为truncate，当前无环境覆盖。没有把inactive CK宏当作当前路径，没有设置宏或重编译AITER。
- 报告GPU0_LAYER_LOGITS_HIDDEN.md、TORCH_EXPERTS_FULL_AND_PRECISION.md；证据results/layer_logits_hidden_gpu0_20261103T220716+0800/。源码SHA、HEAD不变，两默认服务恢复并实际生成8 token。BF16全量仍停止，无Git提交。

## 朴素版SDPA调用修复 2026-11-03
- 用户要求修复朴素版本。新增naive_sdpa选择可等价省略mask的HF causal/GQA调用；单token decode关闭causal；SWA边界和cached多token查询保留原显式mask。mellum_naive_ops.py是唯一计算逻辑改动，mellum_naive.py仅更新说明；专家/norm/路由/KV/Sampler未为此修改。
- 同HF Q/K/V重放HumanEval/151前6步×28层，168/168输出完全一致，短prefill实际走Flash。真实32Q/4KV/head_dim128的10个因果、SWA1023/1024/1025和cached偏移组件用例通过；最初人工head_dim32的Efficient异常另留证据，没有放宽阈值或升级依赖。
- 原固定10条，冻结原HF SDPA/eager专家参考，每条64token，两轮均640/640=100%，10/10完整一致、10/10重复相同。修复前361/640、3/10完整一致；HumanEval/151从6/64变64/64。没有重新选择样本、更换参考或重跑200条，不等于Step0通过。
- 工具check_naive_sdpa_fix.py、validate_naive_sdpa_fix.py、run_naive_sdpa_fix.py；证据results/naive_sdpa_fix_quick10_20261103T224720+0800/；初次组件检查失败材料保留在224530目录。修复前模型文件备份before-naive-sdpa-fix-20261103T224503+0800，结果目录保留review补丁和前后SHA。
- 控制任务complete，源码/Git HEAD冻结审计通过；GPU1朴素服务重启到修正版，两服务实际8token生成及约63%显存检查通过。没有Git提交，完整历史保留，BF16全量保持stopped_by_user。
- 报告NAIVE_SDPA_FIX.md，本地朴素版SDPA修复说明.md。

## Mellum局部RNE修正及负向回归 2026-11-03
- 用户要求修复前述norm截断问题。新增mellum_rne_norm.py，在已有torch专家模型中可选替换113处norm，保留原权重Parameter及原生加载器，不修改公共layernorm/AITER/装饰过的模型文件，不升级依赖。
- 两种模式：native保留FP32残差/RMS/gamma算术顺序，BF16输出用RNE；hf复现BF16残差及gamma分步舍入。全部5631源tensor/255目标参数逐值一致。
- hf 282项共享输入检查逐值等于HF；native 282项FP64参考检查通过，norm容差rtol=2^-7/atol=1e-8，其中56项残差精确。组件合格不代表端到端合格。
- 同配置BF16路由固定10条，原RTZ 356/640（3exact），hf RNE 327/640（3exact），native RNE 320/640（4exact）；两RNE各重复一次均10/10稳定。最差HumanEval/61由5/64改善为21/64与51/64，但整体退化，默认RNE=0，未部署为默认服务。
- 三组目录：rne_ablation_quick10_20261103T223746+0800、mellum_rne_fix_20261103T223606+0800、mellum_native_rne_fix_20261103T224434+0800。新外部工具和本地JSON副本保留，报告MELLUM_RNE_FIX.md。
- 末次恢复检查native成功、naive8001暂未就绪，原连接拒绝保留为initial_restore_error；随后两现有服务均实际生成8 token，未再重启。RNE冻结范围源码/参考/数据/AITER均不变；独立朴素SDPA的两文件同期变化单列记录，repo_unchanged=false、rne_scope_unchanged=true。
- 原始仓库与Git HEAD完整历史保持，未提交Git。已停止BF16全量仍停96条且stopped_by_user，无新200条。独立朴素SDPA修复640/640另记，不能冒充原生RNE版本成绩。

RNE收尾补充（22:55）：两服务恢复检查完成后，检测到另一路独立朴素run_mellum_naive_200.py使用GPU1并暂停8001。没有干预该任务；GPU0继续在线，原BF16路由96条仍停止。四文件审阅补丁及哈希见native RNE结果目录的rne_review.patch和rne_file_changes.json。

## 修复后朴素版完整200条验证 2026-11-03
- 用户要求完整200条。仅新增measure_mellum_naive_200.py和run_mellum_naive_200.py，GPU1、单卡、串行、BF16、eager/level0、MTP和prefix-cache关闭；每条greedy固定64token、ignore-EOS。模型及算子源码不变，冻结同一HF SDPA/eager专家参考。
- 完整结果12800/12800=100%，200/200条完整一致；HumanEval 10496/10496、164/164完整一致；MBPP 2304/2304、36/36完整一致。全量中的原固定10条逐条复现修复后输出。
- 运行审计为MellumNaiveForCausalLM，28层TorchExperts/NaiveSDPAAttention，5376个专家切片完整加载，无FusedMoE，参数均BF16。原始messages/chat-template输入ID和SHA逐条核对；每条之后核对源码、配置、数据和HF参考哈希；完整仓库/HEAD冻结通过。
- 另读原始JSONL独立聚合，核对200个唯一ID、输入、冻结HF目标与全部输出，结果一致，保存independent_review.json。状态及控制complete，测量耗时1617.829秒（含引擎加载，不是正式性能成绩）。GPU1朴素服务恢复，两HTTP实例实际8token生成和约63%显存验证通过。
- 证据results/mellum_naive_200_20261103T225453+0800/，本地同步JSON/JSONL；报告NAIVE_FULL200.md（本地朴素版全量测试.md）。达到本地冻结公开集90% token一致率门槛；尚未完成HumanEval功能pass@1、16k配对或其他Step0验收，原生复用版与朴素版成绩分开记录。
- Git HEAD仍6c86378f230d7d10734b574aa30988a349ebd2ff，无提交，完整历史保留。之前BF16路由全量保持stopped_by_user，没有恢复。本轮200条没有全部重复，不把前次10条重复检查扩写为200条稳定性验证。

## HumanEval/66原生RNE退化诊断 2026-11-03
- 仅GPU0，依次运行HF及RTZ/native RNE/HF顺序RNE；三组ATOM分别两次稳定复现64/64、5/64、39/64，全部5631检查点源tensor/255目标参数逐值一致。捕获共同输入前缀下的6步全部28层状态、真实logits及公共FP32 logit lens。
- 第6步：HF write=complete=31.0，低ID取write；RTZ 30.75/30.625取write；native RNE 30.875/31.0取complete；HF顺序RNE 31.0/31.0取write。RNE末层hidden L2误差更小（5.3987%→4.2496%），候选词排名仍翻转；同位置一致率不等于功能正确率。
- 同HF输入重放56组：分开QKV56/56精确，合并14/56，native合并等于torch合并56/56；o_proj、gate、BF16 dense路由权重、固定HF排列的专家计算均56/56。相同QKV的原生prefill attention 0/28精确，最大相对L2约0.06564%。实际decode内核未单独隔离。
- 原生路由排列的专家重放43/56精确，13组prefill有微小差异；补充独立进程route-order重放未复现HF，标为inconclusive且排除因果结论，没有宣称已证明专家顺序是根因。
- 新增4个外部工具diagnose_mellum66.py、run_mellum66_diagnosis.py、analyze_mellum66.py、probe_mellum66_route_order.py。首次元数据属性读取错误失败目录231232保留，修正外部工具后有效主结果231407 complete；补充探针两项失败保留，不归为模型故障。
- 主目录results/humaneval66_diagnosis_20261103T231407+0800，全部生产源码/数据/参考/AITER哈希不变；GPU0恢复RNE关闭的原生服务且再次实际生成8 token。GPU1独立朴素全量未被干预，原BF16路由全量保持stopped_by_user，HEAD和完整历史不变，无Git提交。
- 报告HUMANEVAL66_DIAGNOSIS.md，本地HumanEval66差异诊断.md；包含第6步全部28层hidden误差与共同投影margin。生产模型/算子及默认开关未改，不由单条诊断直接部署其他计算路径。

## 朴素与原生逐算子直接对比 2026-11-03
- 按用户要求以修复后朴素实现直接作参考，不新跑HF。选原生BF16路由已停96条最差HumanEval/61，123输入；朴素两次64/64、原生骨干+torch专家/BF16路由/RNE-off两次5/64，均复现旧ID。前6步输入相同。
- 同输入56组attention重放控制及专家内部拆解控制全部通过；实际KV写入读回K/V各336/336精确。 o_proj、router gate、LM head及固定路由的专家gate-up/SiLU/乘法/down/加权/index_add均精确。
- 独立差异：input norm 0/56、融合残差0/56、QKV合并9/56（分开56/56）、RoPE Q33/56 K40/56、attention0/56（最大L2约0.0852%）。SWA decode RoPE Q/K21/21精确，全局7层均不同，最大约0.234%/0.223%。实际decode覆盖Gluon21层及ASM7层。
- 共同router输入：FP32 dense权重1/56精确但最大abs2.384e-7，转BF16后56/56精确，专家集合56/56相同，有序ID仅21/56相同。改用原生排列后专家gate/up对齐token比较已出现微差，最终45/56精确；固定朴素排列全部内部步骤56/56。这是有效模型进程内证据，未将该差异声称为唯一致因。
- 发现旧native rotary模块hook只捕获7个全局层且共享对象导致7份覆盖，SWA21缺失。新trace包装实际rope_cache返回并按当前层过滤模块hook；两侧RoPE自身重放56/56验证通过。5个旧结果目录增加rope_trace_instrumentation_caveat.json。原norm/QKV/真实边界hidden/logits等独立证据保留。
- 主结果results/naive_native_operator_diagnosis_20261103T235149+0800 complete；记录了经模型/数据SHA核对后的朴素参考与29组算子复用。生产源码/数据/参考/AITER不变，CPU分析器在生成后修正head_input选行对齐，前后SHA及原失败控制保留，source_unchanged=false、model_and_data_unchanged=true。
- 外部调试失败233625/234144/234514/234852保留；LM head诊断误用批次metadata导致越界，替换为叶子tgemm.mm并独立验证选行。约102.9GB core按精确路径清理，大小/原因记录保留。最终两服务实际8token和显存检查通过，GPU1推理工作未被停止/重启；BF16全量保持stopped_by_user。
- 新增4个外部工具及NAIVE_NATIVE_OPERATOR_DIAGNOSIS.md，未改生产模型/公共算子或依赖，无Git提交，HEAD完整历史保持6c86378f230d7d10734b574aa30988a349ebd2ff。

## vLLM ROCm源码实现对照 2026-11-04
- 按用户纠正聚焦ROCm实际调用链，固定当前vLLM b14e2f2792ae和原生Mellum迁移前ea106db80691；查明当前注册表已转TransformersMoEForCausalLM，分开记录当前HF层forward与旧Qwen3-MoE骨干。
- 对照到叶子实现：ATOM小维norm经rmsnorm2d_fwd走OPUS/module_rmsnorm_quant，而vLLM AITER rms_norm入口在冻结AITER中为CK/module_rmsnorm；eager无Graph配置vLLM默认优先vllm_c HIP。融合残差BF16舍入时点不同，未将通用融合norm误写为当前Transformers Mellum必走路径。
- vLLM tgemm分支限制gfx950，ATOM gfx942非量化linear直接调用tgemm；packed QKV双方都存在。RoPE区分SWA融合、全局AITER、旧vLLM通用HIP和当前HF forward；attention下钻context_attention_fwd、paged Triton及csrc/rocm HIP，与ATOM FMHA/Gluon/ASM不同。
- MoE核对当前topk_gating适用条件、HF路由保留分支及BF16权重扩展FP32的区别、AITER融合专家接口。本机HF5.10.1是独立MellumTopKRouter；没有运行vLLM fuser，不臆测实际分支。当前wrapper的gemma_norm参数与冻结AITER签名不兼容，未复制代码或升级依赖。
- 只读参考快照放/workspace/reference/及本地server-step0/vllm-rocm-review；ATOM/AITER/HF共14文件另存SHA，vLLM成功下载逐项复核，失败路径保留。新报告VLLM_ROCM_IMPLEMENTATION_COMPARISON.md；更新状态/文件清单/草稿。
- 本轮无生产实现变更、vLLM安装、GPU运行、新全量、服务停止/重启或Git提交；原BF16全量保持停止。静态对照不能替代vLLM logits、正确性或性能实测。

## Mellum局部ROCm精度适配 2026-11-04
- 用户授权参考vLLM ROCm修改适配。新增mellum_rocm_precision.py，继承torch专家版，保留原生Qwen3-MoE骨干、FusedMoE参数及loader；更换局部norm/attention/gate/head，Parameter身份/名称不变，两次5631→255权重核验全相等。只修改model_runner/envs注册，公共算子及装饰过的模型文件不变。
- 根据vLLM AITER norm API接入冻结CK入口、BF16残差边界；当前Transformers backend组织方式用于独立RoPE及HF有序路由。SDPA采用本地已验证实现，没有移植或安装最新版vLLM HIP/Triton内核。CK普通norm最大相对L2约0.447%，torch路径与独立HF norm 339项精确；两类RoPE的cos/sin在6个位置精确。
- 首轮固定10条：CK/native418/640，torch/native431/640，torch/SDPA/packed465/640，split640/640（两轮10/10稳定）；额外H61为42、51、21、51/64。控制变量证实仅norm或attention修改不够。
- split附测H61第52token仍分叉，继续隔离保留的gate/head GEMM。只换gate变64/64，固定10条640/640；最终correctness用torch gate/head，固定10条两轮640/640、10/10精确且稳定，H61两轮64/64。native-head组本轮亦通过，未将LM head臆定为根因；前次早期56组gate精确不外推全部decode。
- 证据mellum_rocm_precision_20261104T070835+0800和mellum_rocm_precision_linears_20261104T071652+0800；每轮223文件冻结，独立输入/参考/输出汇总通过。首轮correctness名称对应native gate/head版本，完整source_snapshot归档。最终新配置不是复用朴素版旧200条成绩。
- 新启动run_rocm_precision.sh；新外部工具measure_mellum_rocm_precision.py、run_mellum_rocm_precision_checks.py、review_mellum_rocm_precision.py；报告MELLUM_ROCM_PRECISION.md。本地证据/补丁同步server-step0/mellum_rocm_precision_evidence，原始两文件备份070813、补充三个文件备份071648，最终7文件review patch/SHA保留待审批。
- 只控制GPU0，两轮恢复均实际生成8token并核验显存；GPU1未停/重启。默认run0仍原生融合版，旧BF16全量保持stopped_by_user，无200条/功能HumanEval/16k/正式性能或Git提交。HEAD与完整历史保持6c86378f230d7d10734b574aa30988a349ebd2ff。

## 新ROCm精度模式完整200条 2026-11-04
- 用户要求全量检查，使用同一最终correctness实现、冻结HF5.10.1 BF16/SDPA attention/eager experts参考和原200条数据，每条greedy、ignore-EOS、64token；GPU0单卡串行，eager/level0，BF16权重/KV，MTP及prefix cache关闭。没有修改生产实现或依赖。
- 完成12800/12800=100%，200/200完整一致；HumanEval10496/10496、164/164，MBPP2304/2304、36/36。固定10条全部复现前次最终输出，H61/H66各64/64。本轮耗时1333.670秒含引擎与加载审计，不是正式性能成绩。
- 运行检查实际MellumRocmPrecisionForCausalLM和correctness设置、113norm、28局部专家方法/标准布局/BF16；重新逐值检查5631源tensor→255参数全相等。全部200条messages模板/inputIDs/输入SHA核对；228文件每条后冻结检查及完整仓库/HEAD/工作区状态前后冻结通过。
- 新外部工具measure_mellum_rocm_precision_200.py、run_mellum_rocm_precision_200.py、review_mellum_rocm_precision_200.py。独立重新读取全部200条原始JSONL、冻结HF参考及数据顺序复算通过，mismatching_ids=[]；没有将旧朴素200条成绩冒充本轮成绩。
- 目录results/mellum_rocm_precision_200_20261104T075934+0800，status/control complete；保存完整冻结源码快照和下载SHA清单，本地同步JSON/JSONL/日志/关键源码。报告ROCM_PRECISION_FULL200.md，本地Mellum_ROCm精度版全量检查.md，更新状态/文件清单/草稿。
- GPU0恢复原生服务，GPU1 PID136002未停/重启，两实例实际生成8token并核对显存。旧BF16全量保持stopped_by_user。默认run0未切换，Git HEAD仍6c86378f230d7d10734b574aa30988a349ebd2ff，无提交，完整历史保留。
- 新路径独立达到本地公开集≥90% token门槛；完整200条没有重复两轮，功能HumanEval、16k配对和正式A/B/C仍待完成，不标记Step0整体验收通过。

## 朴素与复用精度版性能测量（进行中）2026-11-04
- 用户要求比较两个已通过200条一致性检查的实现与题方baseline，后续明确继续尝试A/C。目录`/workspace/results/mellum_perf_compare_20261104T150625+0800/`，物理GPU0按precision:B→naive:B/A/C→precision:A/C串行执行；GPU1服务保留、无测试负载。
- 生产模型、公共算子和启动脚本不变。新增外部serve_mellum_perf.py（通过已有加载后入口只读审计）、run_mellum_perf_compare.py（测量及服务恢复）、review_mellum_perf_compare.py（SQLite独立复算）、mellum_perf_status.py（只读进度）；全部单卡BF16/eager/level0，MTP及prefix cache关闭。
- B已完成并独立复核：两版各50/50成功、输入1024/输出256，原始prompt匹配；朴素8.4234tok/s，TTFT p50/p99=689.078/798.648ms，TPOT=115.994/120.881ms；复用9.9759tok/s，TTFT=486.413/590.059ms，TPOT=98.443/103.126ms。复用相对朴素输出快18.4%，题方没有B参考值。
- A使用全部原512条、并发32、输出256，C全部原200条、open-loop rate0.30、timeout180；保留原始prompt，不截断或更改数据。以Output Throughput对比题方638.282724tok/s/GPU；C如有超时，同时报告成功率，不用幸存请求分位数宣称加速。
- 截至16:07 HKT，朴素A阶段95条成功、0失败，阶段输出32.8153tok/s，完整结果仍待结束；复用A及两版C尚未完成。该阶段值不是最终成绩，不能作为512条完整结果。服务恢复和源码冻结审计在最终执行，当前没有Git提交。
- 后续用户明确要求“正式测试直接暂停”：停止驱动和客户端，终止测试服务并清理后续队列；新增外部stop_mellum_perf_compare.py。最终status及measurement_status均stopped_by_user，active_formal_benchmark=false，不自动恢复。终端约224/512，最后周期汇总220成功、0失败、31.9865tok/s，无完整A summary，仅保留阶段记录；复用A及两版C未运行。
- GPU0原生服务恢复为PID170319，GPU1 PID136002不变，两服务实际8token及显存核对通过；restore_status=complete，全部原生产源码/脚本/数据SHA与Git HEAD不变，无提交。两版B再次独立复核通过，原日志/数据库和本地报告保留。
- 用户随后明确要求“全部停止，不用恢复服务”：终止GPU0原生PID170319组及GPU1朴素PID136002组；已无ATOM/评测/推理spawn进程，8000/8001连接检查关闭，GPU两卡利用率0%，显存各297623552字节（约284MiB）。16:35:28 HKT控制状态记录services_status=stopped_by_user、restore_status=cancelled_by_user_all_stop；不自动恢复。新增外部mark_mellum_services_stopped.py只检查并记录状态，不启动服务；已有日志与结果完整保留，无生产模型改动或Git提交。

## 复用精度版短程性能剖析 2026-11-04
- 用户随后要求只跑复用版并抓性能定位差距；未恢复正式A/C。GPU0单卡correctness、BF16/eager/level0/MTP-off/prefix-off，在同一ATOM离线引擎使用A前32条、8token短程；GPU1始终停止。6个新增外部工具负责采集、CPU/GPU归因、算子回放及独立核对，未改生产模型/公共算子/启动脚本/依赖。
- profile目录mellum_reuse_profile_20261104T164236+0800：未开启profiler单请求decode中位101.030ms；32请求两轮701.948/702.187ms，平均681.689/681.791ms。输出8token及profile/非profile输出一致；不是正式512条×256token的A成绩。
- 32请求单步trace墙钟890.057ms：互不重叠CPU范围attention686.504ms=77.13%，MoE189.804ms=21.32%，外部norm10.763ms=1.21%。GPU工作472.694ms，attention418.924ms=88.63%、MoE51.424ms=10.88%；896次SDPA/分页KV读取，588次显式mask，1120次attention内hipMemcpyWithStream。GPU标注与CPU标注分开，独立分析不重复累加嵌套时间。
- 探索回放165345：SWA原算子19.645ms，先收集KV再裁剪的逐请求版本12.576ms，合并padding版本20.664ms（无收益），裁剪+合并11.283ms；全局原11.410ms而合并19.768ms。采集只返回原始输出，候选不参加后续生成。
- 补充同组织control目录mellum_reuse_attention_control_20261104T170717+0800：全历史19.550ms，裁剪1024窗口12.781ms=1.530倍；排除首轮循环组织变化影响。裁剪max_abs=0.00048828125、relL2=0.04439%，没有直接部署。不能将局部收益当成整模型加速，也不能假设题方基线用了融合专家或宣称已解释全部约19倍差距。
- 3次诊断均complete，生产/脚本/数据SHA、HEAD/工作区保持；独立重算中位数，控制回放整模型输出与未profile输出相同，8000/8001/8098关闭、两GPU利用率0%、各约284MiB显存。raw/compressed trace与报告复用版性能剖析.md保留，无Git提交、不恢复服务。

## GPU1朴素基线对齐实验进行中 2026-11-04
- 用户新授权仅在GPU1用朴素实现对齐题方宏观配置并测试性能。没有题方源码，不称源码级复现；没有恢复旧A/C队列或GPU0服务。
- 独立工具安装进程内可选组织：显式mask SDPA的GQA查询分组、批量分页KV读取、CPU专家ID/预组织专家输入、连续KV增量更新；仍是单卡BF16、eager/level0、torch循环MoE、MTP-off。生产模型/依赖/默认脚本不变，不提交Git。
- 组件MoE 9组重复5次全部精确一致；非连续分页读取检查通过。current+packed_loop固定10条仍640/640；全显式mask候选361/640，逐条等于旧显式朴素实现，不能继承current的200条正确性结论。
- 短程32请求decode：原朴素约783ms，批量SDPA+packed_loop约259ms；补充同进程vector265.878ms、连续KV185.290ms，112组有效KV精确核对通过。
- vector候选正式A因进一步去除全历史重排由实验主动结束，末次完整周期日志320成功、0失败、87.6833token/s，仅阶段值；原因、旧工具和全部日志保留。不是用户停止，不伪造完整A结果。
- 连续KV候选正式A/C正在运行：A原512条/并发32/256输出；C原200条/0.30请求每秒/180秒超时。结束后只关闭本次进程，不恢复原服务。材料GPU1朴素基线对齐实验.md（服务器NAIVE_BASELINE_ALIGNMENT_GPU1.md）；正式结果路径results/naive_baseline_alignment_gpu1_20261104T172349+0800/formal_dense_packed/。

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


## 2026-11-06 norm_cap8 formal A/B/C performance attempt

User authorized current precision candidate A/B/C. Same GPU1, frozen Torch2.10, BF16 eager level0, no MTP/prefix cache/chunked prefill, max32 sequences, token budget16384, memory utilization0.5. A512/closed-loop32 and B50/closed-loop1 completed, all outputs256. A: 45.9763 output tok/s; TTFT p50/p99 14835.5833/20687.6428 ms; TPOT p50/p99 645.5744/705.5120 ms. B: 16.4705 output tok/s; TTFT p50/p99 452.6172/480.9692 ms; TPOT p50/p99 58.3270/60.8850 ms.

C attempted unchanged after A/B at Poisson0.30/s, timeout180s. Backend OOM during C-0009 full prefill (15194 input tokens), specifically eager attention FP32 softmax: allocation27.52GiB, free24.01GiB. Backend exited but HTTP frontend kept accepting. Stopped client on fatal failure: 50 accepted, zero complete256, partial max155; NOT a full200 result and no valid C throughput/p99. No cold-start C countercheck or configuration workaround performed.

Verified419 source hashes and Git HEAD/status unchanged; no model edits, dependency installs, or Git commit. Added only experiment/evidence scripts and documentation. Own processes exited, ports8580/8581 closed, both GPUs idle. No HF token or HumanEval retest. Report /workspace/MELLUM_NORM_CAP8_ABC.md; evidence /workspace/results/norm_cap8_abc_20261105T160808.

## 2026-11-06 Independent optimization points, frozen Torch2.10

Current separate entry: /workspace/run_mellum_opt_points.sh. Flags: OPT_ATTENTION=bounded, OPT_MOE=off, OPT_DISPATCH=graph_bucket. Retain two points: bounded head-wise Torch softmax for large prefill temporary memory; local HIP graph dispatch of unchanged per-request norm/linear operators. Preserve Qwen3-MoE backbone and original cap8 Torch experts. MTP off. Original implementations/dependencies and all Git history preserved, no commit.

Formal HTTP same GPU1: A 512/512, 67.0115 output tok/s (baseline45.9763, +45.75%); B 50/50, 19.0020 tok/s, TPOT p50=51.2094ms (baseline58.3270ms); C 0/200, failed=200, output throughput=0.0 tok/s. C latency, if available, is successful requests only; do not hide timeout coverage.

A200 at actual concurrent32: 12800/12800=100.0000% vs native HF5.20.2 eager single requests on frozen Torch2.10; first64 of SAME256-token performance outputs. Four approx16k inputs passed256/256 after full bucket warmup. Full C coverage-qualified precision: {'expected': 200, 'requests_with_64': 1, 'matches_on_available': 388, 'positions_available': 388, 'full_denominator': 12800, 'exact_64': 1, 'full64_subset_percent': 100.0}.

Three-run fixed32/64-token Engine.generate medians: control33.976, bounded33.777, MoE-script33.509, graph42.419, bounded+graph43.115 tok/s. All passed2048/2048 per run. These are short ablations, not complete official ABC per-point ablations. TorchScript norm/projection, fused expert and Triton softmax candidates failed screening; detailed failures retained. The threshold remained90%, not100%.

Production entry generic warmup HTTP smoke 64/64. Tests finished, ports8660/8661/8680/8681 closed and own services stopped. Source419 hashes and Git HEAD/status unchanged. HumanEval not rerun. Report /workspace/MELLUM_OPT_POINTS.md; evidence /workspace/results/mellum_opt_points_20261105T172604.


## 2026-11-06 User-confirmed precision acceptance

The user explicitly confirmed A200 at concurrency32 plus a separate approximately16k input group. Compare first64 raw generated token IDs against native HF eager single-request references on frozen Torch2.10, threshold >=90%; do not average the long-input group into A200. Current fixed long-input group contains four cases. Formal performance outputs256 tokens and precision uses their first64. The completed optimization round already used this setup; this confirmation does not authorize or require a rerun by itself. No GPU test started and no Git commit.

