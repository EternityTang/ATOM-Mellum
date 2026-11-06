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
