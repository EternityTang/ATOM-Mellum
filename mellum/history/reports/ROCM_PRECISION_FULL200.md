# Mellum ROCm 精度版完整 200 条检查

日期：2026-11-04，Asia/Hong_Kong。

本次检查新接入的 `MellumRocmPrecisionForCausalLM`，profile 固定为 `correctness`。完整数据仍是原先冻结的 200 条：HumanEval 164 条、MBPP 36 条。HF 参考、消息模板、输入 IDs 和每条 64 token 的生成口径保持不变。

## 结果

**完整结果为 12800/12800 token 一致（100%），200/200 条完整一致。**

| 数据来源 | 条数 | 同位置 token 一致 | 完整一致 |
| --- | --- | --- | --- |
| HumanEval | 164 | 10496/10496，100% | 164/164 |
| MBPP | 36 | 2304/2304，100% | 36/36 |
| 总计 | 200 | 12800/12800，100% | 200/200 |

原固定 10 条在本次完整数据中的输出逐条等于前次最终精度配置输出，仍为 640/640。此前单独重点检查的 HumanEval/61、/66 在本次完整数据中也均为 64/64。没有重新选择样本、更换 HF 参考、修改输入或在运行途中修改模型。

本轮独立达到本地公开集 token 一致率不低于 90% 的门槛。这个完整成绩属于新 ROCm 精度路径，不是沿用朴素版此前的完整成绩。测量耗时 1333.670 秒，包含引擎加载与加载审计，不作为正式吞吐或性能成绩。

## 检查方法

| 项目 | 设置 |
| --- | --- |
| 运行模型 | 原生 Qwen3-MoE 骨干和加载器；Mellum 局部精度模式 |
| norm / residual | torch norm，显式 BF16 残差边界 |
| attention / QKV | SDPA；拆分 Q/K/V 的 torch linear |
| 路由 / 专家 | HF 有序 top-k、BF16 路由权重；torch 专家循环，原生参数容器 |
| gate / LM head | torch linear，保留原参数及 prefill 选行语义 |
| 精度 | BF16 权重与 KV |
| 执行 | 单卡 GPU0、单请求串行、eager、level 0 |
| 解码 | greedy，temperature=0，ignore EOS，每条 64 token |
| 其他 | MTP 和 prefix cache 关闭 |
| HF 参考 | Transformers 5.10.1，SDPA attention、eager experts，原冻结参考 |

开始前逐项核对上次固定 10 条验证的全部源码哈希，确保使用同一版本。模型加载后检查实际类型、profile、113 处 norm、28 个专家方法及标准权重布局、全部参数 dtype，并重新逐值核验 5631 个检查点源 tensor 到 255 个目标参数。

全部 200 条原始 messages 重新应用同一个 tokenizer/chat template，核对 prompt SHA 与 input IDs。生成中每条完成后检查 228 个源码、模型配置、数据和参考文件的 SHA。独立 CPU 复核工具再读取全部原始 JSONL，重新计算匹配数量、完整一致数量、相同前缀长度、两个数据源汇总，以及固定 10 条与前次输出的逐条复现情况。

本轮只增加测量、控制和独立复核工具，没有修改生产模型或公共算子。GPU0 原生服务在测量期间暂停，结束后恢复；GPU1 朴素服务保持运行。旧 BF16 路由全量仍保持用户要求的停止状态，当前任务使用独立结果目录。

## 文件与复现材料

- `/workspace/tools/measure_mellum_rocm_precision_200.py`：完整 200 条、加载审计、输入/参考与逐样本源码冻结检查。
- `/workspace/tools/run_mellum_rocm_precision_200.py`：GPU0 测量控制、缓存归档、仓库/HEAD 冻结以及服务恢复。
- `/workspace/tools/review_mellum_rocm_precision_200.py`：独立重算全部样本和汇总。
- `/workspace/results/mellum_rocm_precision_200_20261104T075934+0800/`：本次结果目录。
- `/workspace/ROCM_PRECISION_FULL200.md`：服务器报告副本。

结果目录包含 `atom_outputs.jsonl`、`comparison.json`、`status.json`、`model_audit.json`、`native_torch_weights.json`、`source_sha256.json`、`repo_sha256_before.json`、`control.json` 和 `independent_review.json`。结束后保存冻结源码快照和下载用的逐文件 SHA 清单。

这是一次完整的公开集 token 一致性检查。固定 10 条在完整数据中重现此前输出，不等于把完整 200 条重复运行两轮。HumanEval 功能 pass@1、16k 配对和正式 A/B/C 性能仍需分别完成。

## 独立复核与恢复

`independent_review.json` 已通过：重新核对全部 200 个唯一 ID、数据集原始顺序、input IDs、输入 SHA、冻结 HF reference IDs、64-token 输出、同位置匹配数、相同前缀长度及完整一致数量，并复算两个数据源汇总。差异样本列表为空，原固定 10 条逐条复现。

运行审计确认 `MellumRocmPrecisionForCausalLM / correctness`，设置为 torch norm、split QKV、SDPA、HF 路由、torch gate/head；113 处 norm、28 层局部专家方法及 BF16 dtype 均核对。全部 5631 个源 tensor 与 255 个目标参数逐值相等，无遗漏源 tensor 或未验证参数。

状态和控制均为 `complete`。228 文件冻结通过，完整 ATOM 仓库文件、Git HEAD 和原工作区状态在测量前后均保持相同。HEAD 仍为 `6c86378f230d7d10734b574aa30988a349ebd2ff`，没有 Git 提交。Python 语法检查通过，没有修改模型或依赖。

GPU0 已恢复原生服务，GPU1 朴素服务 PID 仍为 136002，没有被停止或重启。两个实例均实际生成 8 token，并检查 GPU 显存占用。旧 BF16 路由 96 条任务保持 `stopped_by_user`。默认 run0 仍原生融合版；新精度模式通过 `/workspace/run_rocm_precision.sh` 启用。

本地证据副本：`C:/Users/tang/Documents/ChatGPT/腾讯笔试/server-step0/mellum_rocm_precision_200_20261104T075934+0800/`，含原始 JSONL、汇总、运行和独立审计、日志及关键源码；`DOWNLOAD_MANIFEST.json` 用于复核下载完整性。完整的 228 文件冻结快照保存在服务器结果目录 `source_snapshot/`。
