# Mellum RNE归一化修复与回归结果

日期：2026-11-03，Asia/Hong_Kong。实现和GPU验证已完成，但固定10条总体退化，开关默认关闭；Step0仍未通过。没有提交Git，没有恢复已停止的BF16全量测试。

## 修改内容

此前GPU0逐层诊断确认：当前AITER norm实际走`module_rmsnorm_quant`，`OPUS_FP32_to_BF16_DEFAULT=2`代表截断（RTZ）。融合残差与RTZ求和逐值一致，RMSNorm约99.999%的元素可由RTZ复现。HF使用的BF16舍入约定不同。

新增Mellum局部的torch RNE（就近舍入、相等取偶）norm实现，供保留原生骨干的torch专家版本选择。共替换113处norm：28层各自的input、post-attention、Q、K norm以及最终norm。每处保留原来的weight Parameter对象，继续使用原生加载器。

| 文件 | 本轮作用 |
| --- | --- |
| `/workspace/ATOM/atom/models/mellum_rne_norm.py` | 新增局部RMSNorm、融合残差实现及模块替换函数；支持native与hf两种算术顺序 |
| `/workspace/ATOM/atom/models/mellum_torch_experts.py` | 在已有局部专家后端中按开关安装113处norm，并检查style取值；专家算法不变 |
| `/workspace/ATOM/atom/utils/envs.py` | 登记`ATOM_MELLUM_RNE_NORM`，默认0；登记`ATOM_MELLUM_RNE_NORM_STYLE`，默认native |
| `/workspace/run_torch_experts_route_bf16.sh` | 保留BF16路由设置，显式保留RNE默认关闭及环境变量覆盖能力 |

本轮没有修改原生`mellum.py`、`qwen3_moe.py`、共享`layernorm.py`、MoE、Sampler或attention，也没有修改/重编译AITER或升级依赖。默认`run0.sh`原生融合专家服务不受RNE开关影响。

该路径仅验证单卡、BF16、eager、编译等级0、MTP/TBO关闭。量化、padding和多卡融合配置不在支持范围内。

## 两种舍入模式

| 模式 | 残差与归一化顺序 |
| --- | --- |
| native | 残差以FP32相加，未舍入的FP32和参与RMS计算；保存的BF16残差和最终norm输出使用RNE；gamma在FP32乘入 |
| hf | 残差先在BF16相加并舍入，再转FP32计算RMS；归一化结果先转BF16，再乘BF16 gamma，复现HF分步顺序 |

native保留原生的算术顺序，但计算由torch完成；FP32归约与AITER内核并非已证明逐bit相同。因此它是局部RNE参考路径，不是仅替换AITER转换指令的内核补丁。

## 数值与权重验证

- 两种模式均验证全部5631个检查点源tensor到255个目标参数的映射及转换后逐值一致，没有遗漏或未验证参数。
- hf模式：在prefill与共同decode第6步，使用同一输入，28层input/post norm、融合残差、Q/K norm及最终norm共282项检查，全部逐值等于HF参考。
- native模式：同样282项检查全部通过。norm对照FP64计算结果，容差为`rtol=2^-7, atol=1e-8`；其中56项残差检查要求逐值相等。此处不能称为282项逐值等于HF。
- 两种模式均对固定10条各重复生成一次，10/10输出完全稳定；HumanEval/61也各运行两次，输出分别稳定复现。

组件验证通过说明实现符合选定的算术约定，不能替代端到端token一致率。

## 固定10条配对结果

参考保持Transformers 5.10.1、BF16、SDPA attention、eager experts、greedy 64 token、MTP关闭。三组ATOM均为原生骨干+torch专家、BF16路由；输入、原固定10条及HF参考不变。

| norm路径 | 同位置token一致 | 64-token完整一致 | 重复稳定性 | HumanEval/61一致/连续相同前缀 |
| --- | ---: | ---: | --- | ---: |
| 原生AITER RTZ，RNE关闭 | 356/640 = 55.625% | 3/10 | 本轮只测一轮10条 | 5/64，前缀5 |
| RNE，hf顺序 | 327/640 = 51.09375% | 3/10 | 10/10相同 | 21/64，前缀20 |
| RNE，native顺序 | 320/640 = 50% | 4/10 | 10/10相同 | 51/64，前缀51 |

| 固定样本 | 原生RTZ | RNE hf | RNE native |
| --- | ---: | ---: | ---: |
| HumanEval/135 | 63 | 5 | 64 |
| HumanEval/66 | 64 | 39 | 5 |
| MBPP/25 | 64 | 64 | 64 |
| HumanEval/120 | 48 | 48 | 6 |
| HumanEval/151 | 8 | 6 | 64 |
| HumanEval/94 | 6 | 6 | 6 |
| HumanEval/124 | 64 | 64 | 64 |
| HumanEval/137 | 7 | 64 | 7 |
| HumanEval/142 | 26 | 25 | 34 |
| HumanEval/147 | 6 | 6 | 6 |

每格为64个输出中同位置相同的token数。HumanEval/61不属于上述固定10条，单独用于原差异点验证。

native RNE将HumanEval/61的首次分叉从第6个输出延后到第52个，但固定10条少了36个相同token，下降5.625个百分点。hf顺序也未改善总体结果。不能用单条改善宣称整体正确性已修好，也不能仅凭完整一致条数增加判断总体一致率改善。

独立朴素版的SDPA修复另有固定10条640/640结果，见`朴素版SDPA修复说明.md`。其attention、norm和骨干计算路径不同，不能并入本轮RNE三组对照。

## 当前开关与运行状态

RNE实现保留，默认关闭。显式验证native模式时可使用：

```bash
ATOM_MELLUM_RNE_NORM=1 ATOM_MELLUM_RNE_NORM_STYLE=native \
  bash /workspace/run_torch_experts_route_bf16.sh
```

将style改为`hf`可重现HF分步舍入模式。该命令供后续受控验证使用，本轮没有将其部署为默认HTTP服务。

RNE收尾恢复时，GPU0:8000原生服务、GPU1:8001独立朴素服务均已通过实际8-token生成；当时显存占用均约130GB。末次恢复检查曾在8001暂未就绪时连接被拒绝，原始错误保留于`control.json.initial_restore_error`，后续两服务的真实生成记录在`final_recovery`中。

随后在22:54:52（香港时间）检测到另一路`run_mellum_naive_200.py`启动，使用GPU1并暂停8001；GPU0:8000继续可用。这是独立朴素版任务，本轮RNE工作没有启动、停止或改变它。该外部活动另记于`control.json.post_validation_external_activity`，不能把此前服务恢复记录当作其运行期间8001仍在线的证明。

RNE测量期间，独立朴素版的两个SDPA相关文件发生修改：`mellum_naive.py`、`mellum_naive_ops.py`。因此不宣称整个仓库哈希始终不变；本轮冻结的全部RNE/原生计算源码、AITER源码、数据和HF参考哈希均未变，`rne_scope_unchanged=true`。独立朴素文件不在本轮原生torch专家模型的计算链上。

原BF16路由全量任务仍为`stopped_by_user`，停在96条；本轮RNE工作没有启动新200条，上述独立朴素任务与它分开记录。Git HEAD仍为`6c86378f230d7d10734b574aa30988a349ebd2ff`，历史完整，所有新增改动尚未提交。

## 证据和外部工具

| 服务器目录 | 证据 |
| --- | --- |
| `/workspace/results/rne_ablation_quick10_20261103T223746+0800/` | RNE关闭的同配置对照、模型审计和恢复状态 |
| `/workspace/results/mellum_rne_fix_20261103T223606+0800/` | hf舍入模式：282项精确参考检查、全部权重核对、单条及固定10条重复生成、逐层投影 |
| `/workspace/results/mellum_native_rne_fix_20261103T224434+0800/` | native舍入模式：FP64参考检查、全部权重核对、单条及固定10条重复生成、源码核对及最终恢复状态 |

外部工具位于`/workspace/tools/`：`validate_mellum_rne_fix.py`、`run_mellum_rne_fix_checks.py`、`analyze_mellum_rne_fix.py`、`measure_rne_ablation_quick10.py`、`run_rne_ablation_quick10.py`、`validate_mellum_native_rne_fix.py`、`run_mellum_native_rne_fix_checks.py`。工具与JSON证据的本地副本位于`server-step0/`；完整张量保留在服务器。

本轮四个有效文件的独立审阅补丁和前后哈希位于native结果目录的`rne_review.patch`、`rne_file_changes.json`；新增文件也包含在补丁中，便于后续审批。

修改前备份：`/workspace/backups/before-mellum-rne-norm-20261103T223558/`；切换到native顺序前的备份：`/workspace/backups/before-native-order-rne-trial-20261103T224425/`。

后续应继续定位剩余attention、残差边界及近分数token的误差传播；先做共享输入/固定10条验证，确认总体改善后再决定是否替换默认路径。当前局部转换修正可用，整体正确性尚未解决。

后续追踪修正：旧原生RoPE子阶段存在21个SWA层遗漏及7个共享rotary模块记录覆盖的问题。相关结果目录已增加rope_trace_instrumentation_caveat.json；旧native rope_query/rope_key比较及由其选出的首个大误差阶段不再作为依据。norm、QKV、真实层边界hidden和最终logits为独立记录，结论不受该hook问题影响。最新已用朴素实现直接对照、捕获实际rope_cache返回并完成自身重放控制，见`朴素版与原生版逐算子对比.md`。
