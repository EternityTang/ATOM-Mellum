## 2026-11-06 User-confirmed precision acceptance

The user explicitly confirmed A200 at concurrency32 plus a separate approximately16k input group. Compare first64 raw generated token IDs against native HF eager single-request references on frozen Torch2.10, threshold >=90%; do not average the long-input group into A200. Current fixed long-input group contains four cases. Formal performance outputs256 tokens and precision uses their first64. The completed optimization round already used this setup; this confirmation does not authorize or require a rerun by itself. No GPU test started and no Git commit.

## 2026-11-06 Independent optimization points, frozen Torch2.10

Current separate entry: /workspace/run_mellum_opt_points.sh. Flags: OPT_ATTENTION=bounded, OPT_MOE=off, OPT_DISPATCH=graph_bucket. Retain two points: bounded head-wise Torch softmax for large prefill temporary memory; local HIP graph dispatch of unchanged per-request norm/linear operators. Preserve Qwen3-MoE backbone and original cap8 Torch experts. MTP off. Original implementations/dependencies and all Git history preserved, no commit.

Formal HTTP same GPU1: A 512/512, 67.0115 output tok/s (baseline45.9763, +45.75%); B 50/50, 19.0020 tok/s, TPOT p50=51.2094ms (baseline58.3270ms); C 0/200, failed=200, output throughput=0.0 tok/s. C latency, if available, is successful requests only; do not hide timeout coverage.

A200 at actual concurrent32: 12800/12800=100.0000% vs native HF5.20.2 eager single requests on frozen Torch2.10; first64 of SAME256-token performance outputs. Four approx16k inputs passed256/256 after full bucket warmup. Full C coverage-qualified precision: {'expected': 200, 'requests_with_64': 1, 'matches_on_available': 388, 'positions_available': 388, 'full_denominator': 12800, 'exact_64': 1, 'full64_subset_percent': 100.0}.

Three-run fixed32/64-token Engine.generate medians: control33.976, bounded33.777, MoE-script33.509, graph42.419, bounded+graph43.115 tok/s. All passed2048/2048 per run. These are short ablations, not complete official ABC per-point ablations. TorchScript norm/projection, fused expert and Triton softmax candidates failed screening; detailed failures retained. The threshold remained90%, not100%.

Production entry generic warmup HTTP smoke 64/64. Tests finished, ports8660/8661/8680/8681 closed and own services stopped. Source419 hashes and Git HEAD/status unchanged. HumanEval not rerun. Report /workspace/MELLUM_OPT_POINTS.md; evidence /workspace/results/mellum_opt_points_20261105T172604.


## 2026-11-06 norm_cap8 formal A/B/C performance attempt

User authorized current precision candidate A/B/C. Same GPU1, frozen Torch2.10, BF16 eager level0, no MTP/prefix cache/chunked prefill, max32 sequences, token budget16384, memory utilization0.5. A512/closed-loop32 and B50/closed-loop1 completed, all outputs256. A: 45.9763 output tok/s; TTFT p50/p99 14835.5833/20687.6428 ms; TPOT p50/p99 645.5744/705.5120 ms. B: 16.4705 output tok/s; TTFT p50/p99 452.6172/480.9692 ms; TPOT p50/p99 58.3270/60.8850 ms.

C attempted unchanged after A/B at Poisson0.30/s, timeout180s. Backend OOM during C-0009 full prefill (15194 input tokens), specifically eager attention FP32 softmax: allocation27.52GiB, free24.01GiB. Backend exited but HTTP frontend kept accepting. Stopped client on fatal failure: 50 accepted, zero complete256, partial max155; NOT a full200 result and no valid C throughput/p99. No cold-start C countercheck or configuration workaround performed.

Verified419 source hashes and Git HEAD/status unchanged; no model edits, dependency installs, or Git commit. Added only experiment/evidence scripts and documentation. Own processes exited, ports8580/8581 closed, both GPUs idle. No HF token or HumanEval retest. Report /workspace/MELLUM_NORM_CAP8_ABC.md; evidence /workspace/results/norm_cap8_abc_20261105T160808.


## 2026-11-05 Partial batching with HF token checks

New independent candidate: /workspace/run_mellum_parallel_precision.sh, default norm_cap8. Decode shares the inherited Qwen3-MoE backbone across requests; MoE groups requests by expert and caps each expert GEMM at 8 rows. Norm and sensitive projections remain per request; full prefill remains per request; original Attention component retains HF-compatible eager math. Actual decode batch32, largest pre-chunk expert group32, matrix chunk8. Fixed10 640/640; fixed32 passed twice at 2048/2048 and 32/32 exact against frozen Torch2.10 native HF5.20.2 single-request eager references.

GPU0 sequential short-test pair, other GPU idle: original serial 128.681 s, norm_cap8 61.262 s, 2.1005x. This is fixed32 real-A input/64 output, not formal A512 or 600 tok/s. Unrestricted expert batches pass10 but fail32 (975/2048), so not adopted. Full200/HTTP/performance queues not resumed. Existing implementation hashes and Git status/HEAD unchanged, no commit. Tests ended, ports closed, both GPUs idle. Report: /workspace/MELLUM_PARALLEL_PRECISION.md; evidence: /workspace/results/mellum_parallel_precision_20261105.


## 2026-11-05 Framework reuse / SWA stage check

New independent entry: /workspace/run_mellum_reuse_request.sh (default framework_eager). Reuses inherited Qwen3-MoE backbone/decoder/MoE container and existing per-layer Attention components. Retains HF-compatible eager math and per-request numerical isolation: fixed10 640/640; actual 32-request decode 2048/2048, 32/32 exact against frozen Torch 2.10 native HF 5.20.2 single-request references.

Original accelerated SWA kernels are NOT enabled by default: both phases 17/640, prefill-only 25/640, decode-only 339/640. These short ablations failed. Component reuse is distinct from kernel reuse. 21 layers use window 1024; 7 are full attention. Existing plain/precision/speed implementation and dependency hashes unchanged; no Git commit, no full200/HTTP/performance run. Short tests finished, ports closed, both GPUs idle. Evidence: /workspace/results/mellum_reuse_stages_20261105; report: /workspace/MELLUM_REUSE_STAGES.md.

## 2026-11-05 Standalone plain-request precision path; stopped by user

New entry: /workspace/run_mellum_plain_request.sh. Independent Torch numerical forward in /workspace/tools/mellum_plain_request/plain.py; ATOM supplies loaded Parameters, scheduling and KV pages. Whole prefill, no padding/chunking, per-request full model and LM head, frozen Torch 2.10, BF16, no MTP. Reference: native HF 5.20.2 eager/eager generated one request at a time.
Completed quick32: 2048/2048 token IDs, 32/32 exact; controlled old precision 596/2048. Diagnostic time ~204s vs ~34s, not formal performance acceptance.
USER REQUESTED NO FULL RUN: all expansion/reference processes and the pending HTTP queue have been stopped. At stop, ATOM 160/200 and HF 180/200 completed; 160 paired rows match 10240/10240. Partial result only. HTTP never started. Do not automatically resume tests. GPUs idle, ports closed. Old source/entries and Git HEAD/status unchanged; no commit.
Report: /workspace/MELLUM_PLAIN_REQUEST_A.md. Evidence: /workspace/results/mellum_plain_A_20261105/pause-review.json.

## 2026-11-05 Precision full-chain audit and chunked SWA correction

Reviewed input/scheduling, all 311 parameter tensors, norms/projections/RoPE, attention/KV, routing/experts, logits/greedy and output. Weight hashes match native HF 5.20.2 on frozen Torch 2.10. Same single/full execution: 3696 recorded tensors bitwise equal. Fixed only tools/mellum_eager_precision/install.py:eager_one to retain W-1 prior keys plus the query chunk. Same chunk execution then matches all 5082 recorded tensors. Ten A hard cases pass 640/640 for each matched prefill mode. HF full vs chunk itself matches 276/640. Batch shape sensitivity is independently reproduced in identical-input QKV and expert GEMMs. Post-fix offline 32-request smoke passes all requests but only 187/256 first-eight IDs match HF static batch. This is not full A/200 or performance acceptance.
Speed dependencies and Git status/HEAD unchanged; no commit. Evidence: /workspace/results/mellum_chain_audit_20261105. Detailed report: /workspace/MELLUM_CHAIN_AUDIT.md. Tests finished.

# Mellum / ATOM 当前进展与下一步

2026-11-04最新GPU1实验已结束：题面朴素约束下，连续KV候选完整A为512/512成功、111.8655输出token/s，仅为题方638.282724的17.526%，未对齐。其C发生OOM；增加32GiB连续缓存预算后，单独完整重测C为1成功、199超时，成功率0.5%，仍未达标。A和C属于明确标识的两个版本，不能包装为同版完整成绩。生产源码/依赖/默认脚本不变；固定10条全显式mask候选361/640不覆盖原朴素/复用精度版的200条100%结果。详见`GPU1朴素基线对齐实验.md`及`naive_baseline_alignment_gpu1_20261104T172349+0800/final_review.json`。两卡已空闲、显存各约284MiB，8000/8001关闭；没有恢复旧队列或GPU0服务，没有Git提交。

最新短程profiling已完成：仅启动复用精度correctness版，32请求未开启profiler的decode中位约702ms/步；采样帧attention占CPU墙钟77.13%、MoE21.32%。同组织SWA算子回放全历史19.550ms、裁剪1024窗口12.781ms，局部1.530倍但有微小数值差异，未应用生产；简单padding批量SDPA未提速。源码/HEAD不变，trace和独立复核已保存，详见`复用版性能剖析.md`。诊断结束后两卡进程均关闭；正式A/C仍暂停，不恢复服务，无Git提交。

最终运行状态更新：按用户“全部停止，不用恢复服务”的最新要求，两张卡上的推理服务、评测客户端及后续队列已全部停止，8000/8001关闭，GPU利用率0%，各约284MiB显存。不自动启动服务或恢复测试；全部已有结果和日志保留。下段“服务已恢复”为先前暂停步骤的历史状态。

最新性能状态：正式A/B/C测试已按用户要求暂停，不自动恢复。两版B各50/50完成并复核，朴素8.4234、复用精度9.9759输出token/s；朴素A中断（最后周期汇总220成功、31.9865token/s，仅阶段记录），复用A及两版C未运行。原服务已恢复并实际生成8token，GPU1 PID136002不变，生产源码/HEAD不变。详见`朴素版与复用精度版性能对比.md`。

更新时间：2026-11-04，Asia/Hong_Kong。已完成朴素版SDPA修复、原生骨干的局部RNE验证及vLLM ROCm源码对照，未提交Git。

最新完整200条：新`MellumRocmPrecisionForCausalLM / correctness`已按冻结HF、greedy、每条64token跑完，12800/12800=100%，200/200条完整一致；HumanEval 164/164、MBPP 36/36，原固定10条全部复现前次输出。独立重读200条原始JSONL并核对输入/参考/计数通过，228文件及完整仓库/HEAD冻结通过；本轮重新验证5631源tensor→255参数逐值相同。没有为全量修改模型或依赖。

结果`/workspace/results/mellum_rocm_precision_200_20261104T075934+0800/`，报告`Mellum_ROCm精度版全量检查.md`；GPU0恢复、GPU1未停/重启，两服务实际8token及显存检查通过。旧BF16全量仍停止，无Git提交。新路径已独立达到公开集token一致率门槛；HumanEval功能pass@1、16k配对及正式A/B/C仍未完成。本段优先于以下历史阶段的“尚未运行200条”描述。

最新精度适配：已新增Mellum局部`MellumRocmPrecisionForCausalLM`，保留Qwen3-MoE骨干、FusedMoE参数和原生loader；参数对象/名称不变，5631→255权重逐值核验通过。参考vLLM接入BF16残差边界、CK/torch norm、独立RoPE、可选原生/SDPA attention和有序HF路由；拆分QKV与gate/head GEMM分别可切换。最终精度配置固定10条两轮均640/640、10/10完整一致且重复相同；额外HumanEval/61两轮均64/64。只替换gate GEMM便将该附测51/64修复到64/64，记录了直接消融。

使用`/workspace/run_rocm_precision.sh`启用新路径，默认run0仍原生融合版。只动了Mellum局部新文件及两处注册/环境配置，没有修改公共算子或带编译装饰器的模型文件；本轮没有跑200条、功能HumanEval或16k。两轮测量和独立汇总完成，服务恢复并实际生成；旧BF16全量仍停止，无Git提交。详见`Mellum_ROCm精度适配与验证.md`。本段优先于以下保留的阶段记录。

最新ROCm对照：已沿vLLM实际入口核对norm/残差、GEMM、RoPE、attention叶子内核及MoE。当前ATOM小维norm走冻结AITER的OPUS，vLLM AITER普通norm入口对应CK；对应eager配置vLLM默认优先自身HIP norm。vLLM的tgemm分支限制gfx950，我们gfx942也直接调用；prefill/SWA/global decode内核也不同。当前vLLM Mellum已转Transformers backend，与旧原生Qwen3-MoE路径分开记录，未混用其RoPE/残差/路由语义。详见`vLLM_ROCm实现对照.md`。本轮只读源码及更新文档，没有生产改动、vLLM安装、GPU测试、服务重启或Git提交。

最新朴素版进展：修复SDPA调用后，固定10条两轮均为640/640，随后按用户要求完整200条也达到12800/12800、200/200完整一致（HumanEval 164/164、MBPP 36/36）。源码和冻结HF参考保持不变，GPU1服务已恢复，没有Git提交。该结果达到本地公开集90% token一致率门槛；HumanEval功能测试、16k配对和其他Step0检查仍未完成。详见`朴素版全量测试.md`及`朴素版SDPA修复说明.md`。

最新RNE进展：原生骨干+torch专家、BF16路由新增113处局部RNE norm，权重逐值核对通过。native顺序使HumanEval/61从5/64变51/64，但固定10条从356/640下降到320/640；hf顺序为327/640。两种RNE均10/10重复稳定，总体退化，因此默认关闭，未替换当前服务。详见`RNE归一化修复与回归结果.md`。

最新HumanEval/66诊断：GPU0复现RTZ 64/64、native RNE 5/64、HF顺序RNE 39/64，均重复稳定。第6步HF的write/complete均为31.0，native RNE为30.875/31.0，因此argmax翻转；native RNE末层hidden误差虽从5.3987%降至4.2496%，仍未保住排名。同输入确认合并QKV及原生prefill attention各自存在微小差异；没有修改生产代码或启用RNE。详见`HumanEval66差异诊断.md`。

最新逐算子诊断：改用修复后的朴素实现直接对照HumanEval/61，两次分别64/64与原生骨干+torch专家/BF16路由/RNE-off的5/64。56组attention及专家内部控制全部通过；KV读写336/336一致，o_proj、router线性层、LM head及固定路由专家全部一致。独立差异在norm/融合残差、合并QKV、全局层RoPE、prefill/实际decode attention及路由排列。旧原生RoPE子阶段捕获的SWA遗漏/共享模块覆盖已标注作废，层边界hidden与真实logits等独立证据保留。详见`朴素版与原生版逐算子对比.md`。生产代码未改，无Git提交。

服务状态：HumanEval/66诊断结束后，GPU0已恢复原配置并通过实际8-token生成；GPU1另一路独立朴素200条已完成，其结果与恢复记录单列。该任务没有被本轮诊断干预。原BF16路由全量96条仍为stopped_by_user，未恢复。

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
| 独立朴素版，Sampler修复后、SDPA修复前 | 361/640 = 56.40625% | 3/10 | 10/10相同 |
| 独立朴素版，SDPA调用修复后 | 640/640 = 100% | 10/10 | 10/10相同 |
| 原生复用版，Sampler及SWA修复后 | 310/640 = 48.4375% | 3/10 | 4/10相同 |

这是困难子集的开发结果，不能推断完整200条正确率。所有输入加输出均不足1024，不能把两版差距或分数变化归因于SWA修复；两版也不只是MoE不同，norm和attention等实现均有区别。

SDPA修复前的朴素版复测为361/640，10/10条两次输出相同，逐条与上次完全一致，详见`朴素版固定10条复测.md`。随后选HumanEval/151逐层比较：朴素版6/64、首差在第0层attention；原生骨干+torch专家62/64、首差在第0层norm。朴素版的显式mask使SDPA从Flash切换为Efficient，固定Q/K/V实验复现数值差异。新后端在本条生成四次相同，但不能由单条结果推断固定10条或整体成绩，详见`单条逐层对比与非融合专家方案.md`。最新朴素版SDPA修复后，该条及原固定10条已全部64/64一致。

历史完整200条曾得到：HF eager experts参考下8770/12800 = 68.515625%，81/200完整一致；HF grouped_mm参考下8967/12800 = 70.0546875%，95/200完整一致。这些均发生在确定性Sampler修复之前，保留作历史记录，不能当作当前版本的正式成绩。后续替换专家版另有完整200条测量，见本文开头及`替换专家版全量与精度检查记录.md`；最新朴素版SDPA修复后完整200条为12800/12800，独立记录在`朴素版全量测试.md`。

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
- 首差在第0层norm；独立实验发现原生融合残差完全等于BF16截断结果，RMSNorm约99.999%元素也可由截断复现。当前active norm走module_rmsnorm_quant，OPUS_FP32_to_BF16_DEFAULT默认2，源码定义为truncate。未修改或重新编译AITER；随后已实现Mellum局部RNE路径并完成单条/固定10条验证。
- RNE的native与hf两种顺序均通过各自数值约定的282项检查及全部权重验证，但固定10条整体退化，保持默认关闭。后续仍需定位其余计算差异，不能把RTZ转换问题已局部修正等同于整体正确性通过。当前不启动新的全量任务。详情见`RNE归一化修复与回归结果.md`。

本节及文首状态优先于前面按阶段保留的描述。当前默认run0仍为原生融合版，GPU1保持朴素服务；两者已通过实际8-token生成，Git HEAD及完整历史不变，没有提交。
