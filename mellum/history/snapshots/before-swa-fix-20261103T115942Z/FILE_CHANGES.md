# Mellum 原生 ATOM 复用版文件变更说明

更新时间：2026-11-03，Asia/Hong_Kong。本版替代之前的自定义模型计算、HF 专家模块和显式 SDPA 适配路线。

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
