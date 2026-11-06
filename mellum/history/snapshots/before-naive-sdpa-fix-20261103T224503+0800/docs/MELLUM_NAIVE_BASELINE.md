# Mellum 朴素单卡基线实现

日期：2026-11-03。实现题面所述的单卡、eager、编译等级0、torch循环MoE、显式mask SDPA、不开MTP。该实现是按题面复现的本地朴素基线，不宣称与出题方未提供源码的基线逐行相同，也不能替换出题方公布的计分基线数字。

## 启动和切换

```bash
# 默认单卡GPU0、端口8000；已有服务占用时先停止对应实例。
bash /workspace/run_naive_baseline.sh

# 当前验证实例使用GPU1、端口8001。
BASELINE_GPU=1 SERVER_PORT=8001 bash /workspace/run_naive_baseline.sh
```

独立开关为ATOM_MELLUM_NAIVE_BASELINE=1，默认0仍使用原生Qwen3-MoE复用版。不能在运行中混用两个attention实现的KV缓存，切换需重启模型实例。原run0.sh保留，当前GPU0:8000是复用版；GPU1:8001是朴素基线，各自只使用1张卡。

| 项目 | 基线设置 |
| --- | --- |
| 引擎路径 | ATOM OpenAI server；保留原调度、分页分配和采样接口 |
| 执行 | 单卡、--enforce-eager、--level 0；模型不加support_torch_compile装饰器 |
| MoE | 显式torch专家循环，FP32 softmax/top-k，BF16路由权重；不调用FusedMoE |
| Attention | torch.scaled_dot_product_attention，显式因果和1024滑窗mask；不调用原生attention计算内核 |
| KV | ATOM分配BF16分页；torch索引读写已绑定的packed K/HDS V视图 |
| Norm及投影 | 普通torch RMSNorm、nn.Linear；不调用AITER融合norm或packed-QKV投影 |
| RoPE | FP32频率、分层default/YaRN；YaRN参数初始化复用Transformers公开工具 |
| MTP/量化/prefix cache/TBO | 全部关闭；不符合约束的配置在模型构造阶段拒绝 |
| 其他显式配置 | max_model_len=32768，gpu_memory_utilization=0.65，KV块16；沿用当前ATOM默认chunked prefill及16384 token预算 |

## 文件

- /workspace/ATOM/atom/models/mellum_naive.py：普通torch骨干、专家循环、加载映射、运行约束。
- /workspace/ATOM/atom/models/mellum_naive_ops.py：RMSNorm、RoPE、显式mask SDPA及分页KV读写。
- /workspace/ATOM/atom/model_engine/model_runner.py：按独立开关选择模型类。
- /workspace/ATOM/atom/utils/envs.py：登记开关，默认关闭。
- /workspace/run_naive_baseline.sh：一条命令启动基线。
- /workspace/tools/validate_mellum_naive.py：组件检查、固定最差10条、重复检查、混合批次及HTTP验证。
- /workspace/ATOM/atom/model_ops/sampler.py：用户单独批准的公共greedy正确性修正，详见下文。
- /workspace/ATOM/tests/test_sampler_greedy.py：防止并列最高分再次进入随机路径的回归测试。

模型数学参考Transformers v5.10.1 modeling_mellum.py（Apache-2.0），新模型文件标注来源及许可证；没有用HF CausalLM替换ATOM引擎。上游来源：https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/mellum/modeling_mellum.py。

## 快速正确性结果

按用户要求，开发迭代固定使用上一轮HF eager对比中差异最大的10条，排序为matching_tokens升序、共同前缀升序、ID字典序；选择在本轮生成前冻结，不修改prompt。HF参考为已保存的原生Transformers5.10.1、BF16、SDPA attention、eager experts输出。

修正采样器后：**361/640 = 56.40625%**，3/10条完整64 token一致。相同10条各重复一次，**10/10输出相同**。另有10条混合批次各16 token生成成功、HTTP8路并发及SSE通过。该极难子集用于快速诊断，不是完整200条验收，不是HumanEval pass@1，也没有完成16k配对正确性或正式A/B/C测量。

| 样本 | 上轮复用版与HF一致数（用于选样） | 本轮朴素基线与HF一致数 | 重复一次 |
| --- | ---: | ---: | --- |
| HumanEval/135 | 5/64 | 64/64 | 相同 |
| HumanEval/66 | 5/64 | 63/64 | 相同 |
| MBPP/25 | 5/64 | 64/64 | 相同 |
| HumanEval/120 | 6/64 | 64/64 | 相同 |
| HumanEval/151 | 6/64 | 6/64 | 相同 |
| HumanEval/94 | 6/64 | 6/64 | 相同 |
| HumanEval/124 | 6/64 | 7/64 | 相同 |
| HumanEval/137 | 6/64 | 55/64 | 相同 |
| HumanEval/142 | 6/64 | 26/64 | 相同 |
| HumanEval/147 | 6/64 | 6/64 | 相同 |

原生复用版的历史输出来自采样器修正前，不能将这张表视为只改变单个因素的消融。当前尚不能宣称通过90%正确性门槛。

## 公共greedy修正

原Sampler.forward无top-k/top-p时绕过了all_greedy判断，仍进入带指数随机噪声的路径。相同logits、三个并列最大token（2339、2468、3559）连续32次调用会得到不同token，而torch.argmax固定选2339。最高分唯一的对照则稳定。

经用户明确批准，在Sampler.forward分支前新增all_greedy时直接torch.argmax的4行逻辑。首次4行修复只覆盖整批greedy。最新已参照v0.1.7补齐混合温度批次的greedy行处理，详见SAMPLER_V017_BACKPORT.md；非greedy行保留既有采样结果。

旧代码回归测试复现4项失败；修正后新增回归、环境变量及SamplingParams相关测试共57项通过。隔离GPU测试覆盖FP32/BF16、单行/多行、带/不带过滤器，并验证greedy不消耗RNG、相同种子的非greedy输出保持一致。该修正属于正确性修正，不作为优化点。

修正前基线证据保存在results/mellum_naive_baseline_before_greedy_fix/，修正后在results/mellum_naive_baseline/。之前70.05%和68.52%保留为原始temperature=0路径的历史测量，不能作为确定性argmax路径已经验证的证据。

## 复核与限制

组件检查覆盖RoPE、1024边界mask、非连续物理页映射、BF16 KV读写、整段/分块SDPA误差及torch MoE对HF eager；5376个专家切片完整加载。分块SDPA数值检查使用容差，最大绝对差0.00390625，不宣称bitwise相同。组件成功不能替代完整模型验收。

所有记录的源码/数据哈希在快速测量期间不变。当前原生复用模型mellum.py、Qwen源码、run0.sh均保持原样；新增独立基线入口和已批准采样修正。两个HTTP实例已通过真实8-token生成，GPU显存非零。Git HEAD仍为6c86378f230d7d10734b574aa30988a349ebd2ff，没有提交。

完整日志及JSON位于/workspace/results/mellum_naive_baseline/。不根据这10条推断全量正确率，也没有为这轮快速检查重跑200条。


## 最新采样器状态
已按用户要求回移v0.1.7相关greedy逻辑，67项单测、32组GPU混合配置及4组整批greedy验证通过。固定10条模型输出与本报告先前记录逐条一致，重复仍稳定；本轮未修改SWA。详见/workspace/SAMPLER_V017_BACKPORT.md。
