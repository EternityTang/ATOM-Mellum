# 修复后朴素版：固定200条token一致率

日期：2026-11-03，Asia/Hong_Kong。按用户要求运行完整200条；本轮不修改模型或算子源码，不提交Git。

**已完成：12800/12800 token一致，200/200条完整64-token一致，均为100%。** 达到本地冻结公开集的90% token一致率门槛；这不等于Step0整体验收完成。

| 数据来源 | 样本数 | 一致token / 总token | Token一致率 | 完整序列一致 |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 164 | 10496/10496 | 100% | 164/164 |
| MBPP | 36 | 2304/2304 | 100% | 36/36 |
| 合计 | 200 | 12800/12800 | 100% | 200/200 |

原固定10条在本轮全量中的输出也逐条等于上一轮修复后结果（640/640）。本轮独立重新聚合逐条JSONL，核对200个唯一ID、输入ID/SHA、冻结HF目标和全部输出，得到相同结果，见`independent_review.json`。

## 固定配置与比较口径

| 项目 | 设置 |
| --- | --- |
| 模型 | `/data/models/Mellum2.1-12B-A2.5B-Thinking` |
| ATOM实现 | `MellumNaiveForCausalLM`，修复SDPA调用选择后的独立朴素版 |
| GPU与并发 | 物理GPU1，单卡，每次一条 |
| 计算 | BF16权重/KV、普通torch norm/linear/专家循环；SDPA causal/GQA或必要的显式mask |
| 执行 | eager、level0；MTP及prefix cache关闭 |
| 生成 | temperature=0、argmax、ignore_eos=true，每条固定64 token |
| 数据 | 固定164条HumanEval及36条MBPP，共200条，输入token范围66～449 |
| HF参考 | 已冻结的Transformers 5.10.1、BF16、SDPA attention、eager experts；没有本轮重生成或更换参考 |

数据文件`/workspace/datasets/correctness_200.jsonl`，SHA256为`187eb968948791becd32979a14c2adb0d27622f8a1797ab5c799827f5d562d0e`。

HF参考`/workspace/results/atom_reuse_mtp_off_eager_200.json`，SHA256为`357893f7244c23d873371f1e332f3b2e880506757e8f7b62881ff41095763f4c`。运行前逐条将原始messages应用相同chat template，核对输入ID和prompt SHA；每条测量后检查源码、配置、数据及参考SHA未变。

Token一致率按同一输出位置的token ID比较，分母固定12800；完整序列一致要求64个位置全部相同。这不是HumanEval功能测试pass@1，也不能替代长序列配对、性能或Step0整体验收。

## 运行审计与文件

实际加载28层`TorchExperts`和28层`NaiveSDPAAttention`，5376个专家权重切片加载检查通过，没有`FusedMoE`；参数均为BF16，eager/level0、MTP-off等运行约束通过。

| 文件 | 作用 |
| --- | --- |
| `/workspace/tools/measure_mellum_naive_200.py` | 复用既有全量比较口径，核对修复后10条测试的源码SHA、运行审计、逐条保存输出及统计 |
| `/workspace/tools/run_mellum_naive_200.py` | 仅释放GPU1朴素服务；运行完整测量、归档缓存、恢复服务并检查真实生成、显存及仓库SHA |

证据目录：`/workspace/results/mellum_naive_200_20261103T225453+0800/`。逐条输出`atom_outputs.jsonl`，进度`status.json`，控制与恢复状态`control.json`，最终聚合结果`comparison.json`。

`status.json`及`control.json`均为`complete`，源码/数据/参考哈希冻结检查和完整仓库审计通过。Git HEAD仍为`6c86378f230d7d10734b574aa30988a349ebd2ff`，没有提交。GPU1朴素服务恢复，两个HTTP实例均通过实际8-token生成，显存占用均约63%。

记录的测量耗时为1617.829秒（约26分58秒，含本轮引擎加载）；它是串行正确性检查时间，不是正式A/B/C性能成绩。JSON及逐条JSONL已同步本地`server-step0/mellum_naive_200_20261103T225453+0800/`。

本轮是一次完整200条测量；此前固定10条的重复稳定性记录保留，不伪记为本轮200条全部重复。全量集所有输入加64-token输出均不足1024，因此本轮不触及SWA窗口边界，也不覆盖16k长序列。先前被用户停止的BF16路由全量任务保持停止，不与本轮朴素任务混记。
