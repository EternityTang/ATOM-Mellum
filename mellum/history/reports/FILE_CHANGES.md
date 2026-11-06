
## 2026-11-05 Partial batching with HF token checks

New independent candidate: /workspace/run_mellum_parallel_precision.sh, default norm_cap8. Decode shares the inherited Qwen3-MoE backbone across requests; MoE groups requests by expert and caps each expert GEMM at 8 rows. Norm and sensitive projections remain per request; full prefill remains per request; original Attention component retains HF-compatible eager math. Actual decode batch32, largest pre-chunk expert group32, matrix chunk8. Fixed10 640/640; fixed32 passed twice at 2048/2048 and 32/32 exact against frozen Torch2.10 native HF5.20.2 single-request eager references.

GPU0 sequential short-test pair, other GPU idle: original serial 128.681 s, norm_cap8 61.262 s, 2.1005x. This is fixed32 real-A input/64 output, not formal A512 or 600 tok/s. Unrestricted expert batches pass10 but fail32 (975/2048), so not adopted. Full200/HTTP/performance queues not resumed. Existing implementation hashes and Git status/HEAD unchanged, no commit. Tests ended, ports closed, both GPUs idle. Report: /workspace/MELLUM_PARALLEL_PRECISION.md; evidence: /workspace/results/mellum_parallel_precision_20261105.


## 2026-11-05 Framework reuse / SWA stage check

New independent entry: /workspace/run_mellum_reuse_request.sh (default framework_eager). Reuses inherited Qwen3-MoE backbone/decoder/MoE container and existing per-layer Attention components. Retains HF-compatible eager math and per-request numerical isolation: fixed10 640/640; actual 32-request decode 2048/2048, 32/32 exact against frozen Torch 2.10 native HF 5.20.2 single-request references.

Original accelerated SWA kernels are NOT enabled by default: both phases 17/640, prefill-only 25/640, decode-only 339/640. These short ablations failed. Component reuse is distinct from kernel reuse. 21 layers use window 1024; 7 are full attention. Existing plain/precision/speed implementation and dependency hashes unchanged; no Git commit, no full200/HTTP/performance run. Short tests finished, ports closed, both GPUs idle. Evidence: /workspace/results/mellum_reuse_stages_20261105; report: /workspace/MELLUM_REUSE_STAGES.md.

# Mellum 原生 ATOM 复用版文件变更说明

更新时间：2026-11-04，Asia/Hong_Kong。本版替代之前的自定义模型计算、HF 专家模块和显式 SDPA 适配路线。

## 当前复用实现的有效改动速览

### 2026-11-04 精度版 decode 页索引优化

`atom/models/mellum_rocm_precision.py` 新增可选 profile `indexed_decode`。普通单 token decode 在进入模型时生成一次物理页索引、页内偏移、写入槽位和滑窗 mask，28 层共享这份计划。每层合并读取批内 K/V，再按原请求长度和原 SDPA 分支计算。计划仅在本次 forward 内有效。历史 KV 总长超过 131072 token 或不满足单步位置条件时使用原 SDPA 路径。默认 profile 为 `correctness`。服务器启用命令为 `ATOM_MELLUM_ROCM_PRECISION_PROFILE=indexed_decode bash /workspace/run_rocm_precision.sh`。

在同一 MI308X 上使用 A 前 32 条原始输入，每条生成 16 token，预热后以原版、新版、新版、原版的顺序测量。原版两轮 decode forward 中位数为 748.48 / 749.56 ms，新版为 568.80 / 569.41 ms，耗时下降约 24.0%。两轮各 512 个输出 token 均与原精度版相同，固定 10 条共 640 个 token 与冻结 HF 参考相同。另用非连续物理页和上下文长度 1、1023、1024、1025 检查了 SWA 与全局 attention，输出及写入的 K/V 均逐值一致。完整 200 条及正式 A/C 尚待验证。

验证工具为 `server-step0/check_decode_optimization.py` 和 `server-step0/check_indexed_decode_boundary.py`。原始数据在 `server-step0/indexed_decode_20261104T092811/`。另试过 SDPA prefill 直接接原生分页 decode，长输入仅 52/512 token 与精度版一致，该实验快照保存在 `server-step0/decode_opt_20261104T092346/`。本轮结束后两卡空闲，Git HEAD 为 `6c86378f`。

最新短程性能诊断：新增6个外部工具`profile_mellum_reuse.py`（实际forward计时及单帧CPU/GPU trace）、`launch_mellum_reuse_profile.py`（GPU0隔离执行、结束关闭进程）、`analyze_mellum_reuse_profile.py`（CPU范围与GPU correlation分析，排除同名GPU标注重复计数）、`replay_mellum_reuse_attention.py`（原Q/K/V的SWA/全局算子回放）、`replay_mellum_reuse_attention_control.py`（补充同组织全历史/窗口对照）、`review_mellum_reuse_replays.py`（独立复算、输出/SHA及关停核对）。本地在`server-step0/`，服务器在`/workspace/tools/`。本轮未修改生产模型/算子/启动脚本或依赖，诊断结束两卡均关闭，正式A/C仍暂停；报告`复用版性能剖析.md`，无Git提交。

最终运行状态：用户进一步要求全部停止、不恢复服务。两卡推理服务及正式测试/后续队列已全部停止，端口8000/8001关闭，两卡利用率0%、各约284MiB显存；不自动重启。新增外部只读检查/状态记录脚本`mark_mellum_services_stopped.py`；它不启动服务。下段恢复记录为先前历史状态，当前以本段为准。

最新性能状态：正式测试已按用户要求暂停，不自动恢复。新增外部`/workspace/tools/stop_mellum_perf_compare.py`用于停止客户端/整个后续队列并恢复原服务；未修改生产模型或公共算子。两版B完成，朴素A中断，复用A及两版C未运行；两服务实际生成和源码/HEAD冻结核对通过。下段“进行中”为暂停前的阶段记录，当前以本段为准。

2026-11-04性能测量进行中：本轮没有修改生产模型、公共算子或启动脚本。新增外部工具`/workspace/tools/serve_mellum_perf.py`（加载后只读配置审计）、`run_mellum_perf_compare.py`（同GPU串行执行两版A/B/C并恢复服务）、`review_mellum_perf_compare.py`（从SQLite独立核对请求、分位数及baseline比值）、`mellum_perf_status.py`（只读进度）。结果目录为`/workspace/results/mellum_perf_compare_20261104T150625+0800/`，本地报告为`朴素版与复用精度版性能对比.md`。两版B均50/50成功；完整A/C尚在运行，不将阶段吞吐标成最终成绩。本轮没有Git提交。

以下以原始ATOM `26f23a0bdbf50b9010c07ee3a71171189723e767` 为基准，优先于后文保留的早期阶段描述。最终通过完整200条的是 `MellumRocmPrecisionForCausalLM / correctness`，默认run0仍为原生融合配置。

| 范围 | 文件 | 当前作用 |
| --- | --- | --- |
| Mellum基础适配 | `atom/models/mellum.py` | 复用Qwen3-MoE骨干/加载逻辑，增加逐层SWA、RoPE、MLP类型和显式head_dim等构造适配 |
| 专家计算替换 | `atom/models/mellum_torch_experts.py` | 保留FusedMoE参数容器，用torch执行专家，跳过融合内核所需的权重shuffle；保留BF16路由权重开关 |
| 最终精度模式 | `atom/models/mellum_rocm_precision.py` | 局部替换113处norm、28处self-attention及gate/head；独立RoPE、SDPA、split QKV、HF有序路由和torch linear；全部Parameter对象/名称保持 |
| 共用数值辅助 | `atom/models/mellum_naive_ops.py` | 原为朴素版新增，精度复用版也直接使用其RoPE、旋转、SDPA选择及Torch分页KV读写；无需独立朴素骨干 |
| 模型选择和开关 | `atom/model_engine/model_runner.py`、`atom/utils/envs.py` | 选择原生、torch专家、朴素及ROCm精度模型，登记各开关/profile |
| 公共greedy修复 | `atom/model_ops/sampler.py` | 参考v0.1.7处理整批/混合批次greedy、EPS温度及argmax并列；两条路线共用 |
| 原生SWA边界修复 | `atom/model_ops/attention_mha.py`、`atom/plugin/attention_mha.py` | prefill使用W-1，冻结Gluon decode使用W+query_len；模型窗口仍1024；correctness的SDPA路径采用自身显式窗口逻辑 |
| 独立RNE实验 | `atom/models/mellum_rne_norm.py` | 早期局部RNE对照，默认关闭；最终精度模式使用自己的norm选择，不叠加该override |
| MTP入口/保护 | `atom/models/mellum_mtp.py`、`atom/config.py`、`atom/spec_decode/eagle.py` | 注册及能力/缺权重检查，MTP关闭；当前没有可运行的Mellum预测头 |
| 回归测试 | `tests/test_sampler_greedy.py`、`tests/test_swa_window.py` | 公共Sampler与原生SWA回归 |
| 启动和测量 | `/workspace/run0.sh`、`run_torch_experts.sh`、`run_torch_experts_route_bf16.sh`、`run_rocm_precision.sh`及`/workspace/tools/` | 原生/专家替换/精度启动，逐层逐算子追踪、权重审计、固定10条/完整200条及独立结果复核 |

相对原始基准，`qwen3_moe.py`、`gpt_oss.py`、`mimo_v2_flash.py`和公共`layernorm.py`、`linear.py`、`moe.py`、`embed_head.py`、`model_loader/loader.py`源码保持不变。这里的源码不变不表示精度模式仍执行这些类的原生计算：该模式通过Mellum局部模块选择实际forward，保留原参数和加载接口。调度、请求管理及KV内存分配沿用ATOM；精度模式的KV读写使用上述torch辅助代码。

相对当前HEAD看到的协议token_ids字段删除、旧mellum_ops/submission移除等，还包含撤销旧适配的历史工作区差异；它们不能全部算作相对原始ATOM新加的改动。当前无Git提交，完整历史保留。

最新汇总见`当前进展与下一步.md`。本文各节保留阶段记录：早期“原始组件不变”的清单已被后续经用户授权的Sampler/SWA修改更新。局部torch专家后端已完成原FP32路由200条及BF16路由精度检查；BF16全量按用户要求停止，GPU0逐层状态/logits诊断发现norm的BF16截断约定，详见第14节及`GPU0逐层Logits与HiddenStates对比.md`。

最新局部RNE实现及回归见第17节和`RNE归一化修复与回归结果.md`：113处norm、权重不变，单条改善但固定10条整体退化，开关默认关闭；HumanEval/66诊断见第18节。

**基础模型的适配仍是一个 Mellum 模型文件和一处注册项。** 最新另增加MTP框架入口及缺权重检查，默认关闭；预测头尚不可运行。其余核心组件通过组合和继承复用。完整提交历史保留，尚未新增Git提交。

## 1 对比基准

| 对象 | 值 |
| --- | --- |
| 原始基准 | `26f23a0bdbf50b9010c07ee3a71171189723e767` |
| 当前 HEAD | `6c86378f230d7d10734b574aa30988a349ebd2ff`，未改变 |
| 开发目录 | `/workspace/ATOM` |
| 原始仓库 | `/app/ATOM`，未修改 |
| 历史处理 | 仅恢复工作区后重建，没有 reset HEAD、rebase 或 squash；新提交由用户审批 |

相对原始基准与相对当前 HEAD 是两种不同的 diff：HEAD 中已有旧适配提交，因此当前工作区还包含对旧代码、诊断接口和旧 submission 快照的撤销。这些撤销没有删除历史提交。

## 2 当前有效源码

| 文件 | 相对原始基准 | 作用 |
| --- | --- | --- |
| `/workspace/ATOM/atom/models/mellum.py` | 新增 | 通过薄构造层接入 Mellum；Attention、Decoder、Model、CausalLM 的 forward 均继承 Qwen3-MoE；MoE 直接使用 Qwen3MoeSparseMoeBlock；使用原生 norm、权重加载与 attention backend。新增逻辑集中在逐层 SWA/RoPE 和 MLP 类型选择。 |
| `/workspace/ATOM/atom/model_engine/model_runner.py` | 修改 1 行 | 将 MellumForCausalLM 注册到上述模型类。 |
| `/workspace/ATOM/atom/models/mellum_mtp.py` | 新增 | 预留Mellum MTP注册入口，检查头权重，缺失或架构尚未支持时拒绝启用。不是已经实现的预测头。 |
| `/workspace/ATOM/atom/config.py` | MTP配置入口修改 | SpeculativeConfig识别Mellum的MTP请求并执行能力检查；正常MTP-off的AutoConfig路径不变。 |
| `/workspace/ATOM/atom/spec_decode/eagle.py` | 注册表增加1项 | 在现有DeepSeek/MiMo草稿框架中预留MellumMTPModel；提议、验证和接受逻辑未修改。 |

以下文件逐字节核对为原始基准版本：

- `/workspace/ATOM/atom/models/qwen3_moe.py`
- `/workspace/ATOM/atom/models/gpt_oss.py`
- `/workspace/ATOM/atom/models/mimo_v2_flash.py`
- `/workspace/ATOM/atom/model_ops/layernorm.py`
- `/workspace/ATOM/atom/model_ops/attention_mha.py`
- `/workspace/ATOM/atom/model_loader/loader.py`
- `/workspace/ATOM/atom/entrypoints/openai/protocol.py`
- `/workspace/ATOM/atom/entrypoints/openai/serving_completion.py`

GPT-OSS/MiMo 的复用点是共用的 `Attention(per_layer_sliding_window=...)` 接入方式和底层实现，不是直接套用它们含 sinks、bias 或 V 缩放的完整模型类。

以上“逐字节不变”列表不包含最新为MTP入口调整的config.py。新增模型文件和MTP入口均未提交，不能仅依赖git diff查看未跟踪文件。

## 3 已放弃和撤销的改动

| 旧内容 | 当前处理 |
| --- | --- |
| 自定义 torch Mellum 模型计算及 HF experts 包装 | 由原生 ATOM 复用版替换，旧源码归档 |
| `/workspace/ATOM/atom/models/mellum_ops.py` | 从有效源码移除，不再维护自定义 SDPA 或私有 KV 布局 |
| Mellum 通用 PretrainedConfig 特例 | 撤销，使用原始 AutoConfig 加载原生 MellumConfig |
| HTTP completion 的 token_ids 扩展 | 撤销，OpenAI 协议和响应代码恢复原样 |
| 旧 `/workspace/ATOM/submission` 快照 | 工作区中恢复为原始基准状态；旧快照仍在历史提交和归档中 |
| 旧回移参考和旧评测工具 | 移至归档；不再参与当前运行 |
| 暂停的 109 条一致率评测 | 进程已停止并归档控制状态；数据保留，不恢复该旧进程 |

归档目录：`/workspace/backups/abandoned-adapter-20261103T081334Z`。包括完整旧源码副本、未提交 patch、旧工具、参考及结果。此前更早的备份也保留。

## 4 当前运行和验证文件

| 文件 | 作用 |
| --- | --- |
| `/workspace/run0.sh` | 启动单卡 bf16、level 0、eager 的原生 ATOM 服务，端口 8000，最大输入输出总长度配置 32768，MTP 与 prefix cache 关闭。 |
| `/workspace/tools/validate_mellum_reuse.py` | 新增工程验证工具；包含 native HF 短参考、原生 ATOM token 诊断、长度边界、HTTP 并发和流式检查；不恢复完整 200 条任务。 |
| `/workspace/tools/measure_mellum_reuse_200.py` | 新增MTP-off完整200条比较工具；两侧各单请求、每条64 token，冻结源码及输入，分开保存原始ID。 |
| `/workspace/results/mellum_mtp_readiness.json` | 默认关闭与缺权重误开启检查；当前ready=false、head_tensor_count=0。 |
| `/workspace/results/atom_reuse_mtp_off_200*.json` 及 `.jsonl` | 新一轮完整token测量的控制状态、分侧结果及最终汇总；完成状态以对应文件为准。 |
| `/workspace/results/reuse_adapter_transition.json` | 记录原始基准、未改变的 HEAD、旧任务处理与备份位置。 |
| `/workspace/results/reuse_adapter_source_audit.json` | 核对原文件哈希、forward 继承关系、5631 个源 tensor 的名称映射和层配置。 |
| `/workspace/results/reuse_adapter_validation_status.json` | 验证的当前阶段和完成状态。 |
| `/workspace/results/reuse_adapter_reference.json` | 8 条短样本在原生 HF grouped_mm 和 eager experts 下的输入及输出 ID。 |
| `/workspace/results/reuse_adapter_atom.json` | 原生 ATOM 的短生成、混合批次和长度边界结果。 |
| `/workspace/results/reuse_adapter_server.json` | 未修改协议的 HTTP 并发与流式检查。 |
| `/workspace/results/reuse_adapter_summary.json` | 以上工程检查的汇总，不能作为完整正确性或性能成绩。 |
| `/workspace/logs/run0.log` | 当前 ATOM 服务日志。 |
| `/workspace/logs/reuse_adapter_*.log` | 各项验证日志，失败记录保留。 |
| `/workspace/report.md`、`/workspace/WORKLOG.md` | 当前报告和按时间追加的工作记录。 |

模型权重、题方 `/data/datasets` 数据、固定的 `/workspace/datasets/correctness_200.jsonl` 均保留。EvalScope 独立环境与已获授权的 Transformers 5.10.1 保留，本次未升级依赖。

工程验证状态为complete：8条短生成、8条混合批次、9个输入长度、HTTP并发与流式均完成。新一轮MTP-off的200条测量也已完成：8967/12800 token一致，70.0546875%，95/200条完整输出一致，未达到本轮90%门槛。HF参考为原生5.10.1、bf16、SDPA、grouped_mm；代码和输入全程冻结，服务已恢复。长输入与HF的配对比较、HumanEval及正式A/B/C仍待执行。

## 5 本地文档和副本

| 本地路径 | 作用 |
| --- | --- |
| `C:/Users/tang/Documents/ChatGPT/腾讯笔试/Mellum重新适配方案.md` | 当前复用方案；服务器副本为 `/workspace/MELLUM_ADAPTATION_PLAN.md`。 |
| `C:/Users/tang/Documents/ChatGPT/腾讯笔试/文件变更说明.md` | 当前文件说明；服务器副本为 `/workspace/FILE_CHANGES.md`。 |
| `C:/Users/tang/Documents/ChatGPT/腾讯笔试/server-step0/mellum.py` | 新模型文件的本地副本。 |
| `C:/Users/tang/Documents/ChatGPT/腾讯笔试/server-step0/validate_mellum_reuse.py` | 新验证工具的本地副本。 |
| `C:/Users/tang/Documents/ChatGPT/腾讯笔试/tmp/archives/before-atom-reuse/` | 上一版方案和文件说明，保留追溯。 |

`server-step0` 中其他旧下载结果和工具副本属于历史材料，不能据此判断服务器当前部署状态。

## 6 审阅命令

```bash
# 当前重建相对原始 ATOM 的差异
git -C /workspace/ATOM diff 26f23a0bdbf50b9010c07ee3a71171189723e767 --stat

# 待审批的全部工作区变化，包括撤销旧实现
git -C /workspace/ATOM status --short
git -C /workspace/ATOM diff --stat

# 保留的提交历史
git -C /workspace/ATOM log --oneline -5
```

## 7 单条最差样本诊断（2026-11-03）

本轮没有修改模型/引擎/采样器或 run0.sh，没有提交 Git。新增两个外部诊断工具 `/workspace/tools/diagnose_worst_mellum_prompt.py` 与 `/workspace/tools/probe_worst_mellum_operators.py`，分别记录逐层中间值和进行相同输入的算子控制实验；本地副本位于 `server-step0/`。

新增 `/workspace/results/worst_prompt_selected.json` 及 `/workspace/results/worst_prompt_humaneval104/` 证据目录；报告为该目录 `report.md`，本地为 `单条Prompt差异诊断.md`。本地只下载 JSON 和报告，较大张量留在服务器。

HumanEval/104 的两侧输出均复现上一轮：5/64 同位置一致，第6个 token 首次分叉；最早数值差异位于第0层 input RMSNorm。HF只切换 experts 后端为 eager 后，该样本为58/64一致。详细证据与结论边界见诊断报告；未据此改动正式200条测量结果。服务已恢复，8-token检查通过，记录的源码和数据哈希均未变化。

## 8 HF eager 完整重测（用户确认，2026-11-03）

- 新增 `/workspace/tools/measure_mellum_reuse_eager_200.py`：沿用固定200条数据，HF显式指定SDPA attention、eager experts，并检查实际配置；输出单独使用 `atom_reuse_mtp_off_eager_200` 前缀。
- 新增 `/workspace/tools/run_eager_200_driver.py`：GPU0/1分别启动ATOM/HF，临时释放HTTP服务，成功或失败后恢复；日志及进程状态独立保存。
- 新增 `/workspace/tools/check_mellum_atom_repeatability.py`：4条固定输入在同一个ATOM进程中各生成3次，记录输出是否稳定；不替换模型计算。结果为 `/workspace/results/atom_eager_reference_repeatability.json`。
- 新增 `/workspace/results/correctness_reference_policy.json`：明确记录用户确认的本地HF参考配置和适用范围。
- 没有修改ATOM模型、引擎、run0.sh或旧测量工具，也没有升级依赖或提交Git。旧grouped_mm结果保留。上述工具的本地副本位于 `server-step0/`。

本轮已完成：8770/12800=68.515625%，81/200条完整一致，未达到90%。同时确认部分样本在同一ATOM进程内重复greedy输出不稳定。详细测量报告及复核分别为 `/workspace/results/atom_reuse_mtp_off_eager_200.md`、`atom_reuse_mtp_off_eager_200_checks.json`；本地副本均在 `server-step0/`。服务已恢复并通过真实生成检查，Step0尚未通过。

## 9 独立朴素基线、greedy修正和结构审计（2026-11-03）

用户要求复现题面朴素配置，并把开发迭代改为固定最差10条。新增独立基线路径，默认复用版保留；以下为本轮实际改动，未提交Git。

| 文件 | 本轮作用 |
| --- | --- |
| `/workspace/ATOM/atom/models/mellum_naive.py` | 新增普通torch骨干、专家循环、权重加载映射及单卡/eager/level0/MTP-off约束。 |
| `/workspace/ATOM/atom/models/mellum_naive_ops.py` | 新增RMSNorm、RoPE、显式mask SDPA与ATOM分页KV的torch读写。 |
| `/workspace/ATOM/atom/model_engine/model_runner.py` | `ATOM_MELLUM_NAIVE_BASELINE=1`时选择独立基线，默认仍为原生复用模型。 |
| `/workspace/ATOM/atom/utils/envs.py` | 登记上述默认关闭的开关。 |
| `/workspace/run_naive_baseline.sh` | 一条命令启动单卡朴素基线；原run0.sh未改。 |
| `/workspace/ATOM/atom/model_ops/sampler.py` | 用户单独批准：新增4行，使all_greedy直接argmax，避免无过滤器时并列最高分随机选择。 |
| `/workspace/ATOM/tests/test_sampler_greedy.py` | 新增CPU回归测试；与环境及SamplingParams测试合计57项通过。 |
| `/workspace/tools/validate_mellum_naive.py` | 组件、固定最差10条、每条重复一次、混合批次、HTTP并发/SSE验证。 |
| `/workspace/tools/audit_mellum_compatibility.py` | 实际加载原生复用版，审计28层类型、RoPE、全部权重值和SWA边界。 |
| `/workspace/tools/probe_mellum_rope_actual.py`、`probe_mellum_window_boundary.py` | 独立核对RoPE缓存与窗口修正候选参数；不改模型计算源码。 |
| `/workspace/tools/probe_mellum_greedy_sampler.py`、`test_proposed_mellum_greedy_fix.py` | 同logits复现随机并列选择，以及应用补丁前的隔离验证。 |

修正后朴素基线的固定10条为361/640=56.40625%，3/10完整一致，10/10重复相同；不是完整验收。旧200条结果为采样器修正前的temperature=0路径记录，应保留但不能声称已验证严格argmax一致性。

原生复用版结构审计：5631源tensor/255目标参数全部逐值一致，MoE层选择正确；SWA prefill确认多看1个token，decode正确，原生attention源码尚未改。RoPE嵌套配置选择正确，但attention_factor未显式映射至AITER字段，当前值与默认公式一致；长位置有小幅三角表差异。最初RoPE审计存在单例轴广播错误，已独立复测并显式归档无效结果，未据此改模型。

报告为本地 `朴素基线实现说明.md`、`Mellum结构兼容性检查.md`；服务器对应 `/workspace/MELLUM_NAIVE_BASELINE.md`、`/workspace/MELLUM_COMPATIBILITY_AUDIT.md`。数据见 `results/mellum_naive_baseline/`、`mellum_compatibility_audit.json`；修正前证据完整保留在 `mellum_naive_baseline_before_greedy_fix/`。

GPU0:8000为原生复用版，GPU1:8001为独立基线，两者均通过真实生成检查。原mellum.py、Qwen源码、run0.sh等保护文件未变，Git HEAD及完整历史保留。

## 10 参考v0.1.7补齐Sampler greedy（2026-11-03）

按用户最新优先级仅处理Sampler，SWA暂不修改。参考ROCm/ATOM v0.1.7，固定commit `a526f0d557eeed08396670249af58bd85fa57338`，不升级ATOM或AITER。

- 修改 `/workspace/ATOM/atom/model_ops/sampler.py`：新增 `_apply_greedy_tokens`，用 `temperatures <= EPS` 与固定形状 `torch.where` 修正混合批次中的greedy行，覆盖无过滤、AITER及native fallback路径。整批greedy入口保留。当前AITER没有上游 `topk_select(tie="low")`，使用同样最小ID并列规则的 `torch.argmax`。
- 扩展 `/workspace/ATOM/tests/test_sampler_greedy.py`：新增混合温度与过滤路径回归。旧4行修复上新增用例10失败，修正后相关测试合计67项通过。
- 新增工具 `/workspace/tools/validate_sampler_v017_gpu.py`、`validate_mellum_naive_sampler_v017.py`，以及独立证据目录 `/workspace/results/sampler_v017/`；旧工具和结果不覆盖。
- 上游源码与SHA记录于 `/workspace/reference/atom-v0.1.7/`，仅作参考，不参与替换整个引擎。

32组GPU混合配置、4组整批greedy及greedy辅助函数的CUDA Graph检查通过。固定最差10条每条生成64 token、重复一次，全部稳定，且首轮输出与此前朴素基线逐条一致；仍为361/640，不能据此宣称通过完整正确性验收。本轮没有运行200条。

两个HTTP实例已重启并完成真实生成检查。SWA、Mellum模型文件及启动脚本哈希未变；没有Git提交。报告为本地 `Sampler_v0.1.7修复说明.md`，服务器为 `/workspace/SAMPLER_V017_BACKPORT.md`。

## 11 SWA窗口语义修复（2026-11-03，最新）

按用户要求修改SWA。扩大边界测试后，确认此前只测过期key而未测最早合法key，导致“decode正确”的初步结论不充分；本轮同时修正prefill和decode。

| 文件 | 作用 |
| --- | --- |
| `/workspace/ATOM/atom/model_ops/attention_mha.py` | varlen prefill传W-1左距离；冻结Gluon decode传W+query长度，保留完整的W个可见位置。Flash-layout和全局路径保持原语义。 |
| `/workspace/ATOM/atom/plugin/attention_mha.py` | 同步修正2处prefill/extend及1处Gluon decode参数换算。 |
| `/workspace/ATOM/tests/test_swa_window.py` | 新增26个前端参数/边界回归，与此前Sampler等测试合计93项通过。 |
| `/workspace/tools/validate_swa_gpu_boundaries.py`、`validate_swa_gluon_production.py` | 实际生产方法的12个prefill、9个decode和32个扩展GPU用例。 |
| `/workspace/tools/validate_swa_fix.py`、`run_swa_fix_checks.py` | 原生复用版固定10条及重复检查，并恢复HTTP服务；未运行200条。 |

实际模型窗口仍为1024。没有修改Sampler、Mellum原生/朴素模型、启动脚本或AITER源码；冻结内核SHA与换算依据保存在 `results/swa_fix/frozen_kernel.json`。这项修正不算优化点。

53个GPU窗口用例通过，覆盖过期/最早合法/当前/未来key、cached prefill、query长度1～4、KV heads1/4及两种V-cache视图。插件参数有单测，未声称插件端到端已验证；严格窗口1仍受当前AITER后端限制，详见报告。

原生复用版固定10条为310/640，3/10完整一致，重复只有4/10相同，仍存在数值/稳定性问题。它与朴素基线361/640、10/10重复稳定的结果分开记录；短样本未触及窗口，不把分数变化归因于本次修复。正式正确性尚未通过。

报告：本地 `SWA修复说明.md`，服务器 `/workspace/SWA_FIX.md`。完整证据为 `/workspace/results/swa_fix/`，只改prefill阶段的材料归档在 `prefill_only/`。原结构审计文档已更正，旧版保存在本轮备份目录。两个服务均已重启通过真实生成检查，没有Git提交。

## 12 原生输出不稳定定位（2026-11-03）

本轮只增加外部诊断工具和记录，没有修改模型、attention、MoE或Sampler源码，没有Git提交。

- 新增 `/workspace/tools/diagnose_mellum_repeat_logits_aligned.py`：通过原有debug-helper入口安装临时hooks，按实际prefill边界对齐请求，检查Sampler与逐层张量；固定输入、权重和路由重复调用原生融合MoE。
- 新增 `/workspace/tools/restore_after_repeat_diagnosis_aligned.py`：诊断结束恢复原生服务。两个HTTP实例已通过实际生成检查。
- 新增 `/workspace/results/native_repeat_diagnosis_aligned/` 和 `/workspace/NATIVE_REPEAT_DIAGNOSIS.md`；本地报告为 `原生输出不稳定定位.md`，JSON证据同步到 `server-step0/native_repeat_diagnosis_aligned/`。
- 更新 `/workspace/WORKLOG.md`、`/workspace/report.md`，记录结论及验证范围。

HumanEval/120重复16次，后15次prefill的首次数值变化均位于第0层MoE输出。96个被检查预测均等于当次argmax。进一步固定专家ID及路由权重，直接调用AITER融合专家函数24次，后23次均与首次不同，最大绝对差0.03125，模型参数不变。问题已定位至融合专家计算/合并路径；具体内部内核机制尚未证实，本轮没有实施修复。

初版工具按固定预测次数划分请求，遗漏了引擎额外的未返回预测；其跨请求逐层比较不作为最终证据。上述结论来自已按真实prefill边界对齐、且与客户端返回token逐一核对通过的复测。

## 13 固定10条复测及两种非融合版本逐层比较

按用户要求，先使用未修改的现有朴素版重测原固定10条：361/640=56.40625%，3/10完整一致，10/10条重复相同，逐条复现此前输出。工具为`/workspace/tools/retest_naive_quick10.py`、`run_naive_quick10_retest.py`；证据目录`results/naive_quick10_retest_20261103T204523+0800/`。恢复脚本的跨挂载点缓存rename错误已修正为同目录归档并保留日志。

随后按用户要求同时尝试现有朴素版和保留原生骨干、仅替换专家计算的版本，选HumanEval/151与HF逐层比较。

| 文件 | 本次作用 |
| --- | --- |
| `/workspace/ATOM/atom/models/mellum_torch_experts.py` | 新增局部专家后端：继承原生Mellum模型，保留FusedMoE参数/路由容器及加载器，把专家执行方法替换为torch循环，跳过AITER shuffle；仅支持当前单卡BF16、eager/level0、MTP/TBO-off范围。 |
| `/workspace/ATOM/atom/model_engine/model_runner.py` | 在原生/朴素已有选择中增加局部专家后端注册，默认仍走原生。 |
| `/workspace/ATOM/atom/utils/envs.py` | 登记默认关闭的`ATOM_MELLUM_TORCH_EXPERTS`。 |
| `/workspace/run_torch_experts.sh` | 设置新后端开关，复用run0启动参数。 |
| `/workspace/tools/compare_mellum_variants.py`、`run_mellum_variant_comparison.py` | 外部hooks捕获前6步及补充第55步全部28层；逐值检查两版全部权重；固定输入/路由专家探针；临时释放GPU并恢复服务。 |
| `/workspace/tools/probe_variant_sdpa.py` | 同一Q/K/V下隔离等价因果mask调用及实际SDPA dispatch。 |

本条朴素版6/64一致，两次相同，首差为第0层attention；同输入实验确认显式mask使SDPA由Flash转为Efficient。原生骨干+torch专家62/64一致、四次相同，首次输出分叉在第55个token，首次数值差异为原生第0层norm。两版5631个源tensor均逐值检查一致；固定HF输入和路由的新专家输出等于HF、12次相同。

原Mellum/Qwen模型文件和公共MoE/norm/attention/Sampler本轮没有修改，SHA保持。新后端默认关闭，GPU0/8000与GPU1/8001恢复原有配置并通过实际生成；没有新200条或新后端完整10条成绩，没有Git提交。HEAD及完整历史不变。

报告：`朴素版固定10条复测.md`、`单条逐层对比与非融合专家方案.md`；服务器主证据`/workspace/results/mellum_variant_comparison_20261103T210115+0800/`，补充第55步见其`decode55/`。

## 14 替换专家全量测试、BF16路由开关和GPU0逐层logits

原FP32路由替换专家版完整200条为9439/12800=73.7421875%，99/200完整一致。按用户要求继续验证路由精度：HumanEval/5中仅转BF16使7/64变46/64，随后将该转换作为默认关闭的开关接入。BF16全量在96条时按用户明确要求停止，4489/6144；没有恢复或伪记为200条结果。

| 文件 | 本阶段修改/新增与作用 |
| --- | --- |
| `/workspace/ATOM/atom/models/mellum_torch_experts.py` | 在已有局部专家后端新增route_weights_bf16属性，开关开启时仅在专家乘权前将路由权重转BF16，保留原生路由算法；默认FP32语义不变。 |
| `/workspace/ATOM/atom/utils/envs.py` | 注册默认关闭的ATOM_MELLUM_ROUTE_WEIGHT_BF16。 |
| `/workspace/run_torch_experts_route_bf16.sh` | 设置BF16路由开关，沿用现有torch专家启动脚本。 |
| `/workspace/tools/measure_mellum_torch_experts_200.py`、`run_mellum_torch_experts_200.py` | 冻结代码/输入/参考，运行原FP32路由完整200条及恢复服务。 |
| `/workspace/tools/measure_mellum_torch_experts_route_bf16_200.py`、`run_mellum_torch_experts_route_bf16_200.py` | 同一评测口径的BF16路由任务；已停止，结果保留。 |
| `/workspace/tools/diagnose_torch200_mismatch.py`、`run_torch200_mismatch_diagnosis.py` | 原FP32路由HumanEval/5的逐层对比、共享输入norm/attention/路由/专家探针。 |
| `/workspace/tools/probe_route_bf16_generation.py`、`run_route_bf16_probe.py` | 同一独立模型进程内，仅改变路由权重转换的配对生成；当时未改变运行中的原200条。 |
| `/workspace/tools/diagnose_bf16_route_followup.py`、`run_bf16_route_followup.py` | 在GPU1复核BF16路由第47步，逐层参数dtype及共享输入计算检查。 |
| `/workspace/tools/trace_layer_logits_hidden.py`、`analyze_layer_logits_hidden.py`、`run_layer_logits_hidden_gpu0.py` | 按最新要求仅用GPU0，对已完成96条中最差HumanEval/61比较全部28层真实边界状态、公共FP32诊断logits和实际最终logits；另有BF16舍入/截断独立实验。 |

最后一次GPU0诊断没有修改模型/算子源码。发现当前AITER norm路径默认BF16转换为截断，融合残差与RTZ逐值一致，norm约99.999%匹配RTZ计算；没有重编译AITER或据此宣称修复完成。原Mellum/Qwen、公共MoE/norm/attention/Sampler源码保持，未升级依赖，未提交Git。

旧局部专家源码及envs在加入BF16开关前备份于`/workspace/backups/before-route-weight-bf16-20261103T214958/`。报告为`替换专家版全量与精度检查记录.md`和`GPU0逐层Logits与HiddenStates对比.md`；最新证据目录`/workspace/results/layer_logits_hidden_gpu0_20261103T220716+0800/`，含完整每层词表投影tensor。两服务恢复通过实际生成，BF16全量状态为stopped_by_user。

## 15 朴素版SDPA调用修复

用户要求修复朴素版。只在`atom/models/mellum_naive_ops.py`增加`naive_sdpa`：可等价省略mask的prefill使用causal/GQA调用，单token decode关闭causal；滑窗达到边界和缓存多token查询保留显式mask。`mellum_naive.py`仅修改模块说明。没有为此修改专家、norm、路由、KV读写、Sampler或原生复用模型。

新增三个外部工具：`/workspace/tools/check_naive_sdpa_fix.py`检查共享HF张量和边界；`validate_naive_sdpa_fix.py`运行固定10条及全部重复；`run_naive_sdpa_fix.py`管理GPU1测量、源码审计和服务恢复。修改前文件在`/workspace/backups/before-naive-sdpa-fix-20261103T224503+0800/`；结果另存补丁和前后SHA供审批。

修复后固定10条两轮均为640/640，10/10完整一致，10/10重复相同；修复前361/640、3/10完整一致。HumanEval/151的前6步×28层共享HF Q/K/V重放168/168完全一致；真实模型head_dim=128的10个因果/SWA/缓存偏移组件用例通过。初次人工head_dim=32检查失败的目录与隔离证据保留，未放宽阈值或升级依赖。

证据：`/workspace/results/naive_sdpa_fix_quick10_20261103T224720+0800/`；报告：`朴素版SDPA修复说明.md`，服务器`/workspace/NAIVE_SDPA_FIX.md`。没有跑200条、恢复已停止的全量任务或提交Git；短prompt的100%不等于Step0整体验收通过。

## 16 修复后朴素版完整200条测试

按用户最新要求运行完整200条，本轮不修改模型/算子、启动脚本或依赖。新增`/workspace/tools/measure_mellum_naive_200.py`，复用既有固定数据、HF参考及串行64-token比较口径，增加朴素模型运行审计及前次修复后源码SHA核对；新增`run_mellum_naive_200.py`，管理GPU1测量、缓存归档、服务恢复及仓库冻结审计。

结果12800/12800=100%，200/200完整一致；HumanEval 10496/10496、164/164，MBPP 2304/2304、36/36。原固定10条逐条等于上一轮修复后输出。独立重新聚合JSONL并与冻结HF核对全部ID、输入及输出，通过。没有把该结果记作HumanEval功能pass@1或16k正确性通过。

证据`/workspace/results/mellum_naive_200_20261103T225453+0800/`，包含`comparison.json`、`atom_outputs.jsonl`、源码/模型/设备审计、`independent_review.json`及恢复检查。状态complete，源码/仓库/HEAD不变；GPU1服务恢复，两实例实际8-token生成及显存检查通过。原BF16路由全量仍为stopped_by_user，没有恢复。报告`朴素版全量测试.md`（服务器`NAIVE_FULL200.md`）；未提交Git。

## 17 Mellum局部RNE归一化实现与回归

新增`/workspace/ATOM/atom/models/mellum_rne_norm.py`，在`mellum_torch_experts.py`中按开关替换113处norm，保留原weight Parameter对象及加载器。`envs.py`登记`ATOM_MELLUM_RNE_NORM`（默认0）和`ATOM_MELLUM_RNE_NORM_STYLE`（默认native）；BF16路由启动脚本保留RNE默认关闭。支持native FP32残差/RMS/gamma顺序及HF分步舍入顺序，不修改公共norm、AITER或原生Mellum/Qwen forward。

全部5631个检查点源tensor与255个目标参数逐值核对通过。hf模式282项共享输入检查逐值等于HF；native模式282项对FP64参考检查通过（norm rtol=2^-7、atol=1e-8，56项残差逐值相等）。这两种验证标准不同，不能混称逐值等于HF。

同一固定10条、BF16路由：RNE关闭356/640，RNE hf 327/640，RNE native 320/640；完整一致分别3/10、3/10、4/10。两个RNE版本均10/10重复稳定。HumanEval/61单独从5/64改善到21/64和51/64，但整体退化，RNE保留为可选实现，默认关闭。

新增外部工具：`validate_mellum_rne_fix.py`、`run_mellum_rne_fix_checks.py`、`analyze_mellum_rne_fix.py`、`measure_rne_ablation_quick10.py`、`run_rne_ablation_quick10.py`、`validate_mellum_native_rne_fix.py`、`run_mellum_native_rne_fix_checks.py`，均位于`/workspace/tools/`。结果分别在`results/rne_ablation_quick10_20261103T223746+0800/`、`mellum_rne_fix_20261103T223606+0800/`、`mellum_native_rne_fix_20261103T224434+0800/`。

报告为`RNE归一化修复与回归结果.md`，服务器`/workspace/MELLUM_RNE_FIX.md`。RNE冻结源码/参考/数据/AITER哈希保持；同期独立朴素SDPA两文件发生修改，已在最终审计单独记录，不宣称全仓库哈希始终相同。两HTTP服务恢复且实际生成8 token；BF16全量保持stopped_by_user；无Git提交。

RNE收尾补充（22:55）：两服务恢复检查完成后，检测到另一路独立朴素run_mellum_naive_200.py使用GPU1并暂停8001。没有干预该任务；GPU0继续在线，原BF16路由96条仍停止。四文件审阅补丁及哈希见native RNE结果目录的rne_review.patch和rne_file_changes.json。

## 18 HumanEval/66的RNE退化诊断

本轮没有修改生产模型、公共算子、默认开关或依赖。新增`/workspace/tools/diagnose_mellum66.py`（共同输入6步全部28层、权重与同输入算子重放）、`run_mellum66_diagnosis.py`（仅GPU0顺序测量与恢复）、`analyze_mellum66.py`（共同FP32投影/真实logits/hidden/路由比较）、`probe_mellum66_route_order.py`（补充探针，未复现参考，结果排除因果结论）。

RTZ、native RNE、HF顺序RNE分别稳定复现64/64、5/64、39/64。第6步write/complete的实际logits在native RNE为30.875/31.0，HF为31.0/31.0，导致argmax分叉；RNE整体hidden误差同时变小。固定输入确认合并QKV和原生prefill attention仍有独立微小差异；尚未证明唯一致因组件，不自动修改生产路径。

有效证据`/workspace/results/humaneval66_diagnosis_20261103T231407+0800/`，初次外部工具错误及补充探针失败保留。报告`HumanEval66差异诊断.md`，服务器`/workspace/HUMANEVAL66_DIAGNOSIS.md`。主诊断complete，生产源码/权重/参考/数据冻结核对通过；GPU0恢复并通过实际8-token生成，GPU1未干预。没有Git提交。

## 19 以修复后朴素实现为参考的逐算子诊断

生产模型、算子及依赖本轮未改。新增4个外部工具：`trace_naive_native_operators.py`捕获28层真实状态、SWA融合RoPE及缓存；`naive_native_operator_probes.py`在模型进程内重放norm/残差/QKV/RoPE/KV/实际attention/路由/专家内部/LM head；`run_naive_native_operator_diagnosis.py`管理GPU0、冻结与复用验证、服务恢复；`analyze_naive_native_operators.py`作相同输入及相同选行语义的CPU汇总。均位于`/workspace/tools/`。

HumanEval/61两侧两次均复现：朴素64/64、原生骨干torch专家/BF16路由/RNE-off为5/64。56组attention控制及56组专家细分控制均通过，KV K/V各336/336一致；norm/残差、合并QKV、部分RoPE、attention及路由排列存在独立差异。固定路由下专家所有内部步骤56/56一致。实际decode对照包含21个Gluon和7个ASM层。

旧原生RoPE模块hook遗漏SWA并因共享模块覆盖，已修正外部追踪并为5个历史目录增加caveat，不用旧RoPE字段作依据；其他独立记录保留。新增报告`朴素版与原生版逐算子对比.md`，服务器`/workspace/NAIVE_NATIVE_OPERATOR_DIAGNOSIS.md`；主结果`results/naive_native_operator_diagnosis_20261103T235149+0800/`。

外部工具调试失败目录、显存core清理元数据及CPU选行汇总修正保留。主控制最终complete，model_and_data_unchanged=true；唯一生成后源码修订是CPU分析器，前后SHA单列，未伪称全清单不变。两个服务恢复并实际生成8token，原BF16全量仍停止，无Git提交。

## 20 vLLM ROCm源码实现对照

本轮没有修改生产模型/算子、依赖或启动配置，也没有新增评测代码。新增本地`vLLM_ROCm实现对照.md`，服务器副本`/workspace/VLLM_ROCM_IMPLEMENTATION_COMPARISON.md`；更新本文件、当前进展、WORKLOG及报告草稿。

新增只读参考材料：`/workspace/reference/vllm-source-b14e2f2792ae/`为固定当前vLLM源码，`vllm-native-mellum-ea106db80691/`为迁移前原生Mellum及相关ROCm源码，均有SOURCE和逐文件SHA manifest；`atom-rocm-comparison-20261104/`保存当前ATOM/AITER相关文件及HF模型文件共14份，含实际未提交修改的哈希。本地同步到`server-step0/vllm-rocm-review/`，另存verification.json核验完整性。404与下载错误保留，不作为有效源码。

具体差异包括OPUS/CK/HIP norm入口与残差舍入、gfx942 GEMM分派、RoPE融合位置和YaRN字段、Triton/HIP与Gluon/ASM attention、topk_gating与legacy topk_softmax、HF预计算权重与融合专家接口。当前Transformers Mellum和旧原生路径分别说明；这是静态源码对照，未运行vLLM或宣称其一致率。服务未被停止/重启，BF16全量未恢复，HEAD与历史保留，无Git提交。

## 21 参考vLLM ROCm接入局部精度路径

新增`/workspace/ATOM/atom/models/mellum_rocm_precision.py`，继承已有torch专家模型，保留原生Qwen3-MoE骨干、loader和全部Parameter对象/名称；113处norm、28处attention、gate/head的计算可局部选择CK/torch、native/SDPA和packed/split。路由按HF有序topk和BF16权重边界执行，专家仍复用已验证的torch方法，不重写专家权重映射。

修改`atom/model_engine/model_runner.py`的Mellum选择入口及`atom/utils/envs.py`，新增默认关闭的`ATOM_MELLUM_ROCM_PRECISION`与默认profile为correctness的`ATOM_MELLUM_ROCM_PRECISION_PROFILE`。公共norm、linear、attention、MoE、Sampler以及带编译装饰器的模型文件本轮不变。

新增`/workspace/run_rocm_precision.sh`，沿用run0公共参数，默认开启最终精度配置；新增外部工具`measure_mellum_rocm_precision.py`、`run_mellum_rocm_precision_checks.py`、`review_mellum_rocm_precision.py`。代码/启动/验证共7个文件：2修改、5新增。最终补丁与SHA在`/workspace/backups/before-mellum-rocm-precision-20261104T070813+0800/final_review.patch`、`final_changes.json`；补充gate/head修改前源码另存071648备份。

首轮ck/native、torch/native、SDPA/packed、SDPA/split分别418、431、465、640/640；split组固定10条两轮精确，但额外HumanEval/61仅51/64。只替换gate GEMM后附测64/64。最终gate/head均torch的correctness两轮640/640、10/10稳定，附测两轮64/64。113处norm与独立HF组件共339项精确，两种RoPE在六个位置cos/sin精确；两轮均验证5631源tensor→255参数逐值相同。

结果目录`mellum_rocm_precision_20261104T070835+0800`及`mellum_rocm_precision_linears_20261104T071652+0800`，每轮223文件冻结与独立聚合通过。首轮旧correctness名称对应当时保留native gate/head的版本，源码完整归档，不冒充最终版本。报告`Mellum_ROCm精度适配与验证.md`，服务器`MELLUM_ROCM_PRECISION.md`；本地证据`server-step0/mellum_rocm_precision_evidence/`。

GPU0恢复、GPU1未停止/重启，两服务实际8token及显存核验。默认run0仍原生融合版；本次没有新200条、功能HumanEval、16k、性能结论或Git提交。旧BF16全量仍停止，完整历史保留。

## 22 新ROCm精度路径完整200条检查

本轮只新增三个外部工具：`/workspace/tools/measure_mellum_rocm_precision_200.py`运行冻结200条并作模型/权重/输入/逐样本源码审计；`run_mellum_rocm_precision_200.py`只管理GPU0测量、缓存归档、仓库冻结和服务恢复；`review_mellum_rocm_precision_200.py`独立重新读取全部原始输出、输入和HF参考，复算计数。生产模型、公共算子、依赖及启动脚本本轮均未修改。

完整结果12800/12800=100%，200/200完整一致；HumanEval164/164、MBPP36/36，固定10条输出全部等于前次最终精度模式。独立复核通过；5631源tensor→255目标参数逐值验证通过。228文件逐条测量冻结及完整仓库/HEAD/原工作区状态冻结通过。

证据`results/mellum_rocm_precision_200_20261104T075934+0800/`，控制/状态complete，含原始JSONL、comparison、模型与权重审计、独立复核、源码快照及下载SHA清单；本地同步同名目录。新增报告`Mellum_ROCm精度版全量检查.md`，服务器`/workspace/ROCM_PRECISION_FULL200.md`；更新状态、日志和报告草稿。

GPU0原生服务恢复，GPU1 PID136002保持；两实例实际8token和显存验证通过。旧BF16路由全量仍为stopped_by_user，无Git提交，历史完整保留。该结果达到新精度路径本地公开集token门槛；功能HumanEval、16k及正式性能另测，未宣称完整200条重复稳定。

## 23 GPU1题面朴素基线约束下的数据组织与性能实验

按用户新授权仅使用GPU1，所有实现选择通过独立测试/HTTP进程的加载后入口安装；生产ATOM模型/算子、依赖、默认启动脚本和Git HEAD保持不变。新增外部工具11个，服务器位于/workspace/tools/，本地在server-step0/：

| 文件 | 作用 |
| --- | --- |
| bench_naive_alignment_sdpa.py | 显式mask、重复K/V与GQA查询分组的同输入组件计时及FP32复核 |
| naive_alignment_ops.py | 可选批量分页读取、分组SDPA、连续KV和32GiB缓存预算/回退，以及保持输入和累加顺序的torch专家循环组织 |
| test_naive_alignment_ops.py | 专家三种组织的9组重复精确检查、非连续页/异长batch KV收集检查 |
| run_naive_alignment_experiments.py、run_naive_dense_experiments.py | 单GPU1完整模型短程对照和固定10条检查；原始输出及逐forward耗时留档 |
| serve_naive_alignment.py | 真实ATOM HTTP server和模型/运行模式审计，仅实验进程启用 |
| benchmark_naive_alignment.py、benchmark_naive_dense_alignment.py、benchmark_naive_bounded_c.py | 分别管理vector候选、连续KV候选和缓存限额后C重测；保留原负载、全部请求及日志，收尾不恢复服务 |
| review_naive_alignment_bench.py、review_naive_bounded_c.py | 独立重读SQLite，核对请求参数/原始prompt、成功数、输出长度和分位数 |

组件9组torch专家输出各重复5次完全一致；32请求同输入SDPA约19.62→0.83ms。短程完整模型decode约783→259ms，补充同组织265.878→185.290ms，112组连续KV有效数据精确核对通过。current+专家重排固定10条640/640；全显式mask候选361/640，不能继承原current完整200条100%正确性结果。

vector候选A主动中断于阶段日志320成功、87.6833token/s，仅阶段记录。连续KV原候选完整A为512/512成功、111.8655token/s（题方638.282724的17.526%），C后续OOM。加入32GiB连续缓存预算后仅重测C：200条全部测完，1成功、199超时，成功率0.5%，无再次OOM。A与C修正版分开记录，未把二者当作同版完整A/C成绩；未改0.30到达速率或180秒超时。

结果根目录/workspace/results/naive_baseline_alignment_gpu1_20261104T172349+0800/，含组件、模型对照、各版本SQLite/命令/审计、失败材料、revision1/2旧工具快照、code_final最终工具manifest及final_review.json。完整A和C原始记录已独立核验；A旧源码通过归档SHA验证，C最终源码冻结通过，生产源码及HEAD不变。最终两卡空闲、显存各约284MiB，8000/8001关闭，无Git提交。报告GPU1朴素基线对齐实验.md（服务器NAIVE_BASELINE_ALIGNMENT_GPU1.md）。题方源码未提供，未宣称复刻或性能对齐成功。

## 2026-11-05 Precision full-chain audit and chunked SWA correction

Reviewed input/scheduling, all 311 parameter tensors, norms/projections/RoPE, attention/KV, routing/experts, logits/greedy and output. Weight hashes match native HF 5.20.2 on frozen Torch 2.10. Same single/full execution: 3696 recorded tensors bitwise equal. Fixed only tools/mellum_eager_precision/install.py:eager_one to retain W-1 prior keys plus the query chunk. Same chunk execution then matches all 5082 recorded tensors. Ten A hard cases pass 640/640 for each matched prefill mode. HF full vs chunk itself matches 276/640. Batch shape sensitivity is independently reproduced in identical-input QKV and expert GEMMs. Post-fix offline 32-request smoke passes all requests but only 187/256 first-eight IDs match HF static batch. This is not full A/200 or performance acceptance.
Speed dependencies and Git status/HEAD unchanged; no commit. Evidence: /workspace/results/mellum_chain_audit_20261105. Detailed report: /workspace/MELLUM_CHAIN_AUDIT.md. Tests finished.


## 2026-11-05 Standalone plain-request precision path; stopped by user

New entry: /workspace/run_mellum_plain_request.sh. Independent Torch numerical forward in /workspace/tools/mellum_plain_request/plain.py; ATOM supplies loaded Parameters, scheduling and KV pages. Whole prefill, no padding/chunking, per-request full model and LM head, frozen Torch 2.10, BF16, no MTP. Reference: native HF 5.20.2 eager/eager generated one request at a time.
Completed quick32: 2048/2048 token IDs, 32/32 exact; controlled old precision 596/2048. Diagnostic time ~204s vs ~34s, not formal performance acceptance.
USER REQUESTED NO FULL RUN: all expansion/reference processes and the pending HTTP queue have been stopped. At stop, ATOM 160/200 and HF 180/200 completed; 160 paired rows match 10240/10240. Partial result only. HTTP never started. Do not automatically resume tests. GPUs idle, ports closed. Old source/entries and Git HEAD/status unchanged; no commit.
Report: /workspace/MELLUM_PLAIN_REQUEST_A.md. Evidence: /workspace/results/mellum_plain_A_20261105/pause-review.json.

