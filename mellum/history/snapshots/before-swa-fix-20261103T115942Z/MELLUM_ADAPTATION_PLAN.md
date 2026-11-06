# Mellum 原生 ATOM 复用方案

更新时间：2026-11-03，Asia/Hong_Kong。此前以 HF 小模块和自定义 SDPA 为主体的方案已放弃。本版按用户要求，以原始 ATOM 为基准，复用 Qwen3-MoE 骨干和 GPT-OSS/MiMo 所用的原生滑窗注意力接入方式。

目标：跑通模型并验证正确性，同时保留原生 ATOM 的 profiling、compile/graph、算子调优与缓存优化空间。HF 仅作为独立参考，不以逐层数值完全相同为实现目标。

## 1 代码基准与历史

- 原始基准：`26f23a0bdbf50b9010c07ee3a71171189723e767`。
- 开发目录：`/workspace/ATOM`；原始 `/app/ATOM` 保持原样。
- 当前 HEAD 保持 `6c86378f230d7d10734b574aa30988a349ebd2ff`。历史提交全部保留，没有 squash、rebase 或重置 HEAD。
- 通过工作区恢复及新改动替换旧实现，尚未执行新的 Git 提交；提交仍由用户审批。
- 原生 Transformers 5.10.1 保留，沿用已获确认的版本例外；不另行安装或升级依赖。

## 2 复用关系

| Mellum 模块 | 复用内容 | Mellum 专有部分 |
| --- | --- | --- |
| `MellumAttention` | 继承 `Qwen3MoeAttention.forward`；使用 ATOM packed QKV、输出投影、RMSNorm、原生 `Attention` 和 AITER `get_rope()` | 根据当前层选择 `rope_parameters[layer_type]`；SWA 层传窗口 1024，full 层传 None |
| `MellumDecoderLayer` | 继承 `Qwen3MoeDecoderLayer.forward`，直接复用残差与 norm 流程 | 按 `mlp_layer_types[layer]` 构造 MLP，而不套用 Qwen 的稀疏层推导字段 |
| MoE | 直接构造 `Qwen3MoeSparseMoeBlock`，使用现有 router、`FusedMoE` 和加载后处理 | 传入 Mellum 的 64 experts、top-8、intermediate 896 等配置 |
| `MellumModel` | 继承 Qwen3-MoE backbone 的 forward、embedding 接口、专家映射及中间张量约定 | 构造 Mellum decoder layers；原 ATOM 构造函数虽有 `layer_type` 形参，但内部固定构造 Qwen 层，故只重写构造过程 |
| `MellumForCausalLM` | 继承 Qwen3-MoE 的 forward、compute_logits、专家映射；使用原生 `ParallelLMHead` | 构造 MellumModel，并复用原生 packed 参数映射 |
| 滑窗接入 | GPT-OSS/MiMo 使用的 `Attention(per_layer_sliding_window=...)`，沿用同一套 ATOM KV 与 attention backend | Mellum 按 `layer_types` 逐层决定窗口；不引入 GPT-OSS 的 sinks/bias 或 MiMo 的 V 缩放等模型差异 |
| 配置 | 原始 ATOM 的 `AutoConfig` 路径 | Transformers 原生返回 MellumConfig，无需特殊绕行 |

新实现不再导入 HF 的模型计算模块，不使用自定义专家循环、HF packed experts 加载回调、自定义 KV 布局或自定义 SDPA backend。原生 attention backend 按既有路径管理 KV、mask、prefill 和 decode。

## 3 当前源码范围

相对原始 ATOM，只有两处源码变化：

1. 新增 `/workspace/ATOM/atom/models/mellum.py`：上述类的构造与连接。
2. 修改 `/workspace/ATOM/atom/model_engine/model_runner.py`：增加 Mellum 架构注册项。

Qwen3-MoE、GPT-OSS、MiMo、norm、attention backend、配置读取、通用权重加载及 OpenAI 协议文件均与原始基准一致。此前的 `mellum_ops.py` 和原始 token ID HTTP 扩展已移除。

## 4 必须保持的 Mellum 配置语义

- 明确使用 `head_dim=128`，32 Q heads、4 KV heads；不能用 hidden size 2304 除以 32 推导 head dim。
- 28 层，按检查点 `layer_types` 选择 3 层 SWA、1 层 full。0 起算 full 层为 3、7、11、15、19、23、27。
- 两类 RoPE theta 均取检查点中的 500000；full 层使用 YaRN factor 16、原始长度 8192，SWA 层使用 default RoPE。
- 每层 MoE 使用 64 个专家、top-8、intermediate 896，归一化 top-k 权重，无共享专家。
- embedding 与输出 head 不共享，词表 98304。
- 5631 个检查点 tensor 通过原生 QKV 与专家映射组织为 255 个目标参数名。该数量是映射审计，不等于逐值正确性已经通过。

## 5 运行配置与优化空间

`/workspace/run0.sh` 启动 ATOM OpenAI server，默认单卡 GPU0、端口 8000、bf16 KV、eager、level 0、MTP 关闭、prefix cache 关闭，服务最大长度 32768。

这是一版**复用现有 FusedMoE 和原生 attention 的接入配置**，已不同于之前的 torch 专家循环和显式 mask SDPA 配置。报告需如实标明，不能将旧实现的测量直接记到新配置上，也不能把现成能力的启用重新包装成新写的优化。

后续保留的空间包括：针对 Mellum 形状的 MoE kernel 调优、compile/graph、已有融合开关、SWA 缓存策略及调度开销。先 profiling 再选择最多 3 个自包含优化点，各自有开关、正确性检查及消融。

## 6 验证与证据

新工具：`/workspace/tools/validate_mellum_reuse.py`。

- 在未修改协议的 HTTP server 上检查真实生成、8 路并发和 SSE 流式响应。
- 用原生 ATOM `LLMEngine` 返回的 `token_ids` 做短样本诊断，不通过文本重新分词推断原始 ID。离线路径只用于诊断，交付入口仍是 ATOM server。
- 固定 8 条短样本，各输出 16 个 greedy token；分别记录 HF 原生 grouped_mm 与 eager experts 两种参考配置。
- 检查 32、1023、1024、1025、8191、8192、8193、16384、20000 token 输入，各生成 4 token；20000 输入覆盖实际分块 prefill。
- 原始 200 条一致率任务不恢复。短样本及边界运行检查不能代替完整 200 条、16k 正确性或 HumanEval pass@1。

证据：`/workspace/results/reuse_adapter_source_audit.json`、`reuse_adapter_validation_status.json`、`reuse_adapter_summary.json` 及对应阶段文件；每轮结果以实际存在且完成的文件为准。

本轮工程验证已完成：HTTP 8 路并发和 SSE、8 条混合批次及全部 9 个长度均成功。短样本原始 token 对比为 HF eager 128/128、grouped_mm 119/128；分别 8/8 与 7/8 条输出完全相同。该结果限定为 8 条、每条 16 token，不代表正式验收，也不作为继续微调实现以追求逐层相同的依据。

## 7 旧材料归档

服务器归档：`/workspace/backups/abandoned-adapter-20261103T081334Z`。包括先前源码、未提交 diff、旧工具、回移参考和历史结果。旧 109 条任务的进程已停止，控制记录标为 archived，保留结果且不再恢复。

旧材料只供追溯，不参与新实现运行。人类审批后可将工作区的撤销和新实现作为后续提交保存；已有完整 Git 历史不变。

## 8 MTP 接入边界和当前状态

按最新要求预留 Mellum 到 ATOM 现有 MTP 框架的入口，复用方向为 DeepSeek/MiMo 使用的 EagleProposer、target verification、拒绝采样及已有草稿缓存管理，不另写一套投机调度器。

当前状态必须区分：**已登记框架入口和缺权重检查，尚未实现可运行的 Mellum 预测头。** 检查点5631个tensor中未发现MTP头；官方模型卡也标注MTP头尚待发布。DeepSeek/MiMo的预测层结构不能直接视作Mellum结构，不能以随机权重或其他模型权重替代。

- 新增 `atom/models/mellum_mtp.py`，提供能力检查和拒绝误开启的入口。
- `SpeculativeConfig` 增加 Mellum MTP 类型识别；Eagle 草稿注册表预留 MellumMTPModel。
- run0保持不传 `--method`，默认 speculative_config为None，不构造draft、不分配draft KV、不执行投机推理。
- 当前显式启用时在配置阶段清楚报错，避免静默运行未装载的预测头。
- 获得正式MTP权重、结构和命名后再实现预测头适配及加载；验证 greedy MTP on/off完全一致后才开启性能测试。

本轮在基础模型和关闭路径跑通后，已完成新的固定200条、每条64 token的MTP-off原始token对比；参考固定为Transformers5.10.1、bf16、SDPA、grouped_mm experts。该任务使用新结果名 `atom_reuse_mtp_off_200`，不是恢复旧109条任务。

结果为8967/12800 token一致，即70.0546875%；95/200条完整输出一致，未达到本轮90%要求。两侧均完整输出64 token，输入相同，源码哈希全程不变；HTTP服务已恢复并通过真实生成检查。该结果不是HumanEval pass@1，差异原因尚未确定，不能简单归因于MTP或单一算子。测量未为提高分数更换参考后端或修改实现。

## 9 当前正确性参考口径（用户确认，2026-11-03）

用户明确要求使用 eager，以正确性验证为优先。后续本地正确性参考固定为 **Transformers 5.10.1 原生 Mellum、BF16、SDPA attention、eager experts**。attention 和 experts 是两个独立后端配置；ATOM 的 enforce_eager 则是另一项执行模式设置。

该配置是用户确认的本地参考，不宣称为题方另行提供的官方配置；仅切换 HF 后端不能代替适配正确性证明。ATOM 模型、引擎及采样实现不变，不为提高一致率替换其原生 norm 或 MoE。

独立的完整 200 条重测已完成，结果前缀为 `atom_reuse_mtp_off_eager_200`；GPU0 重新运行 ATOM，GPU1 运行 HF，每条 64 token、单请求 greedy、忽略 EOS，MTP 关闭。结果为 **8770/12800 = 68.515625%，81/200 条完整一致**，未达到90%。之前 grouped_mm 的70.0546875%保留为历史对照，不覆盖、不混用；单条样本的58/64不能推断全量成绩。

同时发现两轮ATOM完整输出仅93/200条相同。进一步在同一个ATOM进程内，4条固定样本各重复3次：HumanEval/5出现3种输出、HumanEval/12出现2种，其余两条稳定。当前已确认部分样本greedy输出不稳定，但具体原因未定位，不能直接归因于某个算子；源码及数据哈希没有变化。

固定上一轮ATOM输出对比本轮HF eager为68.6875%，也未达90%；该交叉分析不替代本轮主结果。后续优先诊断ATOM自身重复生成的首个差异，再完成HumanEval功能正确率与16k配对检查。当前不将Step0标为通过，不以更换参考后端代替正确性证明。HTTP服务已恢复，真实生成检查通过，未提交Git。

## 10 独立朴素基线及兼容性复核（最新状态）

用户要求实现题面朴素配置，故新增独立 `mellum_naive.py` / `mellum_naive_ops.py` 路径，开关 `ATOM_MELLUM_NAIVE_BASELINE=1`，启动入口 `/workspace/run_naive_baseline.sh`。它是按题面复现的本地基线，不能冒充出题方原始源码或替换官方计分数字；原Qwen3-MoE复用版仍为默认，run0.sh保留。

经同logits探针确认公共Sampler在无top-k/top-p时忽略all_greedy，最高分并列仍会随机选择。用户已明确批准公共模块的范围例外：在Sampler.forward开头新增all_greedy时直接argmax的4行逻辑，相关57项回归通过。其他采样路径不改；这项正确性修正不计优化点。旧200条结果保留为修正前temperature=0路径记录，不能作为严格argmax已经验收的证据。

开发迭代按用户要求固定使用上一轮eager结果最差10条，每条64 token；正式验收再跑完整200条。当前朴素基线为361/640=56.40625%，3/10完整一致；10条各重复一次全部相同，混合批次、并发及SSE通过。不能由该困难子集推断完整正确率。

原生复用版四项审计：按层读取RoPE及选择MoE的构造逻辑已适配；全部5631源tensor对应255参数逐值检查一致。RoPE的attention_factor未显式转换成AITER的attn_factor，当前实际幅度与默认公式一致，存在通用适配缺口；长位置三角表有小幅数值差异。SWA参数未丢失、没有变成全局attention，但实测native prefill窗口比HF多1个token，decode边界正确；应修正prefill实参语义，不能整体把模型窗口改为1023。原生attention此次未修改。

详见 `朴素基线实现说明.md`、`Mellum结构兼容性检查.md`；两条服务路径均已恢复，没有Git提交，HEAD和完整历史保持。

## 11 最新优先级：先完成Sampler v0.1.7 greedy回移

用户要求先处理Sampler，本轮SWA不改。已参考上游v0.1.7补齐混合温度批次的greedy行修正：按 `temperatures <= EPS` 判定，使用固定形状 `torch.where`；覆盖无过滤及两类过滤后端。上游的 `topk_select(tie="low")` 在冻结AITER中不存在，兼容使用 `torch.argmax`，不升级库。原有整批greedy快速入口保留，非greedy行沿用原采样结果。

相关67项单测、32组GPU混合配置、4组整批greedy及辅助函数CUDA Graph验证通过。仍只用固定最差10条模型回归：361/640，10/10重复稳定，10/10首轮输出与上一轮朴素基线相同，没有运行200条。SWA边界问题及RoPE显式适配事项继续保留，尚未因本轮修复而解决。

来源、改动和测试细节见 `Sampler_v0.1.7修复说明.md`；两条服务已重启通过真实生成检查，未提交Git。
