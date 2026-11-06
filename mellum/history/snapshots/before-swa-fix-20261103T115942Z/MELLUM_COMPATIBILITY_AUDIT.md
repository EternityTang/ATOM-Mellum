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

## 2 滑动窗口：确认边界问题

实际28层中，0起算3、7、11、15、19、23、27的effective_sliding_window=-1；其余21层为1024。参数没有丢失，SWA没有统一被当作全局attention。

但原生attention_mha.py的prefill_attention传入window_size=(1024,0,0)。AITER把左界解释为含端点的1024个历史位置，连当前token共1025个。HF的mask是kv_idx > q_idx - 1024，窗口总共1024个位置。

GPU边界实测：Q/K全0，只有V的key0为1，其他V为0，序列长度1026。

| query位置 | HF的1024窗口应输出 | 当前原生SWA prefill | 原生SWA decode |
| --- | ---: | ---: | ---: |
| 1023 | 1/1024 | 0.0009765625 | 未测此点 |
| 1024 | 0 | 0.000972747802734375 | 0 |
| 1025 | 0 | 0 | 0 |

同输入的full层在1025位置仍非零，进一步区分了局部窗口与全局attention。当前SWA prefill与decode边界不一致，不应以“复用原生接口”作为语义正确的证明。

独立控制实验将prefill kernel参数改为(1023,0,0)，位置1024输出变为0，符合HF窗口语义；没有把这项改动应用到共享attention源码。不能直接把模型的sliding_window整体改成1023，因为decode当前已经按1024正确工作。

本轮选定的10条prompt输入长度为116～353，连64个输出也未达到1024。这项边界问题影响长输入，不能解释这10条里的全部差异。证据为mellum_window_candidate_probe.json。

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

没有更改原生复用模型构造或共享attention代码。公共sampler的greedy修正已获得用户单独批准，测试与基线报告另行记录。下一步应修正/验证SWA prefill边界，并为RoPE的attention_factor增加显式兼容处理；再使用固定10条及窗口边界测试，而非每次完整200条。正式验收仍需200条、HumanEval和16k配对检查。

原生服务与朴素基线服务均已恢复并通过真实生成检查，Git未提交，历史保持不变。
