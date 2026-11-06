# SWA 窗口边界修复

日期：2026-11-03（Asia/Hong_Kong）。模型窗口保持1024。本轮修改ATOM调用层，未更改Mellum构造、模型权重、Sampler、启动配置或AITER库，没有Git提交。

## 修复内容

此前审计只检查了decode排除过期key，未检查最早合法key，因此“decode正确”的结论不充分。本轮扩大边界覆盖后，确认prefill与decode各有一个换算问题：

| 路径 | 旧行为 | 修正 |
| --- | --- | --- |
| AITER varlen prefill | 传左距离W，连当前token实际看W+1个位置 | 左距离传W-1 |
| 冻结Gluon decode | 内核按L-kernel_window+query_index+1设左界，直接传W会漏掉合法前缀；query长度K时会少K个位置 | 传kernel_window=W+K，使每个query都只看W个位置 |
| Flash-layout unified路径 | 已经使用W-1的左距离 | 保持不变 |
| 全局attention | 不使用SWA窗口 | 保持原参数 |

W始终指模型配置的窗口长度，不把config.sliding_window改为1023或1025。Gluon换算依赖当前冻结内核的实际mask定义，内核源码SHA记录于frozen_kernel.json；将来升级AITER必须重新验证这一换算，不能盲目沿用。

具体修改：

- /workspace/ATOM/atom/model_ops/attention_mha.py：prefill_attention的左距离修正；paged_attention_triton在非flash的Gluon调用处修正query-span参数。
- /workspace/ATOM/atom/plugin/attention_mha.py：同步修正两个prefill/extend分支和一个Gluon decode分支，避免同类错误遗留。
- /workspace/ATOM/tests/test_swa_window.py：参数契约、因果/滑窗边界及query长度1～4回归。

未改装饰过的模型源码，未改AITER源码；模型21层SWA仍为1024，7层full仍为-1。修复属于正确性修正，不计优化点。

## 关键边界

对Q/K全0、只有指定key的V为1的输入，注意力结果有可直接计算的闭式参考。

| query位置与标记key | 正确行为 | 修正前 | 修正后 |
| --- | --- | --- | --- |
| q=1024、key=0，prefill | 排除，输出0 | 非零，多看一个过期key | 0 |
| q=1023、key=0，decode | 保留最早合法key | 0，被错误排除 | 约1/1024 |
| q=1024、key=1，decode | 保留最早合法key | 0，被错误排除 | 约1/1024 |
| 当前query对应key | 可见 | 需独立核对 | 保留 |
| 将来的key | 不可见 | 需独立核对 | 0 |

## 验证

- 旧prefill/插件参数用例10失败；扩展decode测试又复现4项失败。
- 修正后SWA、Sampler、环境变量及SamplingParams相关单测合计 **93通过**。
- 实际生产方法的 **12个prefill用例、9个decode用例**通过，覆盖完整prefill、带历史KV的prefill、过期/最早合法/当前/未来token以及全局attention对照。
- 实际生产Gluon路径另有 **32个GPU用例**通过，覆盖KV heads 1/4、query长度1～4，验证逐query因果及1024窗口语义。HDS和SHUFFLE两种V-cache视图均有覆盖。
- 插件分支有参数换算单测和同类扫描；未将其描述为完整vLLM/SGLang插件端到端验收。
- 原生GPU0:8000与朴素GPU1:8001均重启并通过真实8-token生成检查。

扩展测试还发现冻结AITER无法处理严格窗口1（左距离0）的某个fmha_v3分派，返回invalid argument；额外全局cached-prefill形状通过。本轮Mellum窗口为1024，正式GPU边界集全部通过，不宣称窗口1或所有其他模型配置都已获支持。失败日志与extra_backend_shapes.json保留，未为绕过它而修改库或模型配置。

## 固定10条快速回归及剩余问题

按用户要求继续使用上一轮冻结的最差10条，每条64 token，不运行200条。本轮检查的是 **原生Qwen3-MoE/FusedMoE复用版**，与此前朴素torch循环版分开记录：

- 与HF原生5.10.1、BF16、SDPA/eager experts参考：**310/640 = 48.4375%**，3/10条完整一致。
- 每条重复一次，只有 **4/10条完全相同**。仍存在原生复用路径的输出不稳定，原因尚未定位；不能因Sampler已确定性就假设模型中间计算必然稳定。
- 这10条输入最长353 token，连64个输出也未触及1024窗口边界。不能把本轮与此前统计的变化当作SWA修复的收益或退化证据；边界修复的证据是上述独立GPU检查。
- 这是困难子集快速诊断，不是完整200条正确性、HumanEval pass@1或16k配对验收。Step0仍不能标为通过。

只修prefill阶段的初步结果及旧“decode不变”审计保存在prefill_only/；扩展测试发现问题后修正了decode，最终结果是本目录根下quick10.json及GPU验证JSON。不会拿初步结果冒充最终结果。

## 证据与状态

证据目录：/workspace/results/swa_fix/。主要文件：regression_after.log、gpu_boundaries.json、gluon_production_gpu.json、method_audit.json、quick10.json、model_audit.json、final_checks.json。

GPU工具为validate_swa_gpu_boundaries.py、validate_swa_gluon_production.py；原生快速检查为validate_swa_fix.py native，由run_swa_fix_checks.py负责临时释放GPU0和恢复服务。原validate_swa_fix.py的探索性gpu模式包含上述窗口1限制用例，正式边界验证使用单独GPU工具。

Sampler v0.1.7修复、Mellum原生/朴素模型和两份启动脚本SHA均未变；快速测量期间源码及输入哈希一致。HEAD仍为6c86378f230d7d10734b574aa30988a349ebd2ff，没有提交，完整历史保留。
