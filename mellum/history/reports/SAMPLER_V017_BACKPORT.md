# Sampler greedy：参考 ATOM v0.1.7 的兼容回移

日期：2026-11-03（Asia/Hong_Kong）。用户最新要求先处理Sampler，SWA暂不修改。本轮没有升级ATOM/AITER或其他依赖，没有Git提交。

## 来源与范围

- 上游仓库：ROCm/ATOM，tag v0.1.7，commit `a526f0d557eeed08396670249af58bd85fa57338`。
- 源码：https://github.com/ROCm/ATOM/blob/a526f0d557eeed08396670249af58bd85fa57338/atom/model_ops/sampler.py。
- 原始文件及SHA256记录：`/workspace/reference/atom-v0.1.7/sampler.py`、`SOURCE.json`。
- 仅回移与greedy直接相关的逻辑，保留当前调用接口及依赖版本。镜像AITER没有上游新增的`topk_select(tie="low")`，因此使用同样按最小token ID打破并列的`torch.argmax`作为兼容实现。

## 实际修正

1. 保留整批all_greedy时直接argmax的入口，忽略过滤器配置，输出固定长度int32，不抽取随机数。
2. 增加`_apply_greedy_tokens`，参照v0.1.7用`temperatures <= SAMPLER_EPS`识别混合批次的greedy行。ModelRunner会将请求温度0钳制为EPS，旧的`temperatures == 0`会漏掉这些行。
3. 无top-k/top-p路径采样后，按行覆盖greedy结果。
4. AITER和PyTorch fallback的过滤路径同样按行覆盖，替换原有的mask.any()分支和布尔索引赋值。使用固定形状torch.where；非greedy行保留已有采样结果。

涉及源码：`/workspace/ATOM/atom/model_ops/sampler.py`。测试：`/workspace/ATOM/tests/test_sampler_greedy.py`。SWA、Mellum原生/朴素模型文件、两份启动脚本均通过哈希核对，未改动。

## 验证

| 项目 | 结果 |
| --- | --- |
| 新混合批次用例在旧4行修复上 | 10失败、18通过，确认缺口可以复现 |
| 修正后Sampler、环境变量及SamplingParams相关单测 | **67通过** |
| 实际GPU混合温度检查 | **32组通过**：FP32/BF16、AITER/PyTorch fallback、无过滤/top-k/top-p/两者、共享/独立噪声 |
| 实际GPU整批greedy | **4组通过**，并列取最小ID，不消耗RNG；使用已钳制的合法温度输入 |
| 混合批次无过滤路径 | 正温度行在同种子下与修改前输出相同 |
| CUDA Graph | 固定形状的greedy行修正函数捕获及重放通过；不代表完整引擎graph已验收 |
| 固定最差10条模型检查 | **361/640 = 56.40625%**，3/10完整一致；每条重复一次10/10相同 |
| 与之前朴素基线的整批greedy输出比较 | **10/10完全相同**，说明本轮没有改变原已修复路径的结果 |
| 服务恢复 | 原生GPU0:8000、朴素GPU1:8001均完成真实8-token生成 |

本轮仅用固定10条，不跑200条；该困难子集不是完整正确性验收，也不是HumanEval pass@1。完整模型仍有数值差异，SWA已发现的prefill边界问题仍待下一步处理。

## 测试脚本问题与记录

首次GPU脚本把未经钳制的0温度传给内部Sampler接口，过滤路径产生无效概率并触发测试进程GPU异常。该次结果无效，日志单独保留；按调试指南检查并清理生成的GPU core文件。修正为实际ModelRunner使用的EPS输入后，相同环境下32组检查全部完成，未修改依赖或随机采样内核。

另有旧测试断言返回张量的对象身份；torch.where合法地返回同值新张量，已改为核对输出值和dtype，保留失败日志。测试结果不依赖这类实现细节。

证据目录：`/workspace/results/sampler_v017/`，含regression_before.log、regression_after.log、gpu_tests.json、quick10/validation.json、final_checks.json。工具为`validate_sampler_v017_gpu.py`和`validate_mellum_naive_sampler_v017.py`，原测量工具及结果未覆盖。

Git HEAD保持`6c86378f230d7d10734b574aa30988a349ebd2ff`；完整历史保留，提交仍由用户审批。
