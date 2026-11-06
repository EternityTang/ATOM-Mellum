# Mellum 复用版四项结构兼容性审计

日期：2026-11-03。对象是默认原生复用版atom/models/mellum.py，而非新加的朴素基线。审计通过进程内外部诊断回调读取实际加载模型，未修改模型前向计算或原生attention源码。完整JSON为/workspace/results/mellum_compatibility_audit.json。

## 结论

之前没有把Qwen3-MoE完整构造函数直接套到Mellum：继承其forward，但MellumAttention、MellumDecoderLayer、MellumModel和MellumForCausalLM均有自己的构造逻辑。直接套用Qwen构造函数确实不兼容。

四项检查中，确认原生SWA prefill存在一个token的边界差异；MoE层选择和全部检查点权重映射通过。RoPE的嵌套读取、当前YaRN参数选择正确，但attention_factor字段存在通用适配缺口，三角表也有小幅长位置数值差异。

## 1 RoPE配置

- Mellum使用config.rope_parameters[layer_type]，没有执行Qwen构造函数中对顶层rope_theta的读取。
- 21个SWA层为default RoPE；7个full层为YaRN。两类theta均为500000，head_dim明确取128，没有用2304/32推导。
- full层YaRN factor16、原始长度8192、缓存长度131072；实测native_mscale=HF attention_factor=1.2772588722239782。
- 原生AITER get_rope识别attn_factor而不是HF的attention_factor；当前没有显式转换该字段。当前检查点值与AITER默认公式一致，因此当前幅度正确，但不能据此声称任意attention_factor都已支持。后续应增加映射/断言。
- 逐位置对比0、1、127、1023、1024、8191、8192、16384、32767的实际缓存：前六个位置本次检查均相同；长位置default sin最大差0.0009765625，YaRN cos最大差0.0078125、sin最大差0.0009765625。并非大范围结构错位；也不能宣称全部位置逐元素一致。数值来源仍需单独拆分。

首次审计脚本未压平AITER缓存的两个单例轴，比较时发生广播误报。已修正并用独立脚本复测，错误证据显式归档为mellum_compatibility_audit_rope_broadcast_invalid.json，未据此修改模型。正确证据为mellum_rope_probe_corrected.json。

## 2 滑动窗口：已扩大验证并修正

实际模型仍是21层窗口1024、7层全局attention。SWA参数没有丢失，但之前只检查decode排除过期key，缺少最早合法key测试，故“decode正确”的初步结论不充分。扩大覆盖后确认prefill多看一个位置，Gluon decode也会漏掉合法前缀。

本轮已在ATOM调用层修正：varlen prefill左距离取W-1；冻结Gluon decode传W+query长度，使实际每个query都看W个位置。模型配置仍为1024，Flash-layout路径与全局attention保持原样，AITER库未改。插件中的同类参数转换一并处理。

93项相关单测、12个prefill和9个decode边界用例、32个Gluon GPU用例通过。覆盖过期、最早合法、当前和未来token，以及query长度1～4、KV heads 1/4。完整依据见/workspace/SWA_FIX.md及results/swa_fix/。

原生复用版固定最差10条仍有数值/稳定性问题：310/640 token一致，重复仅4/10条完全相同；它们最长353个输入token，未触及1024窗口，不能将该分数变化归因于SWA边界。朴素基线结果分开保存，Step0仍未通过。

## 3 MoE层选择

MellumDecoderLayer.__init__自己按mlp_layer_types构造，不调用Qwen的mlp_only_layers/decoder_sparse_step判断。当前检查点28项均为sparse；实际28层均为Qwen3MoeSparseMoeBlock，64 experts、top8、moe_intermediate_size896，无共享专家，top-k权重归一化。

没有误用dense intermediate_size7168，也没有错误地按Qwen稀疏步长漏掉MoE层。forward继续继承Qwen的残差/norm流程。

## 4 专家及其他权重

对当前实际加载模型做了完整逐值检查，而不只是核对名称数量：

- 检查点5631个tensor全部参与校验，映射至255个ATOM参数。
- 28层Q/K/V按实际顺序合并后，与packed QKV参数逐元素一致。
- 全部28×64个专家的gate/up/down来源重组后，执行同一原生shuffle，与所有w13/w2参数逐元素一致。
- 其余norm、router、o_proj、embedding、lm_head也逐元素一致。
- unmapped_source_keys和unverified_target_parameters均为空，全部参数值一致。

因此没有发现当前检查点的专家格式错映射或漏加载。权重正确不代表MoE内核的数值归约与HF相同，后者是另一项检查。

## 本轮范围与后续

没有更改原生复用模型构造；共享attention调用层已按最新要求完成上述SWA修正。Sampler v0.1.7修复保留。RoPE attention_factor显式适配和原生路径数值稳定性仍待处理；开发阶段使用固定10条及边界测试，正式验收仍需200条、HumanEval和16k配对检查。

原生服务与朴素基线服务均已恢复并通过真实生成检查，Git未提交，历史保持不变。
