# 交付证据索引

原始trace为ROCm torch profiler导出的Chrome Trace JSON，gzip压缩。解压后可在Perfetto或Chrome Trace导入。analysis.json保存按采样步归类的耗时，原始trace可用于独立核对。

| 文件 | 对应路径与作用 |
| --- | --- |
| [trace-eager.json.gz](trace-eager.json.gz) | 固定32条、1024输入，未开Graph |
| [trace-graph.json.gz](trace-graph.json.gz) | 同负载，原生decode Graph |
| [trace-expert-fp8.json.gz](trace-expert-fp8.json.gz) | Graph加专家FP8 |
| [trace-final-fixed.json.gz](trace-final-fixed.json.gz) | 最终融合及V视图修复，固定负载 |
| [trace-before-mixed.json.gz](trace-before-mixed.json.gz) | 变长prefill，前两点配置 |
| [trace-final-mixed.json.gz](trace-final-mixed.json.gz) | 同一组变长输入，最终融合及修复 |
| trace-*-analysis.json | 与上述trace对应的GPU跨度、类别耗时及kernel列表 |
| [abc-final-summary.json](abc-final-summary.json) | 四配置A512/B50/C200原始统计、分位数与输入集合核对 |
| [humaneval-seed43-summary.json](humaneval-seed43-summary.json) | 报告采用的148/164复测，以及前一轮原始结果 |
| [value-cache-check.json](value-cache-check.json) | 七组融合写入与Gluon读取的组件检查 |
| [baseline.json](baseline.json) | 题方提供的单卡朴素基线，原样保留空参考字段 |
| [manifest.json](manifest.json) | 文件来源、大小和SHA-256 |

## 完整实验归档

ABC原始数据库与脚本位于服务器 /workspace/results/mellum_speed_abc_20261107T064400Z/，归档SHA-256为0f800aec96b48f9ebdee75cd56988f507281843693a1439aa331304cc3aadfd3。

HumanEval采用轮次位于 /workspace/results/mellum_speed_humaneval_111_seed43_20261107T123400Z/，归档SHA-256为e5dbfaf5f31d95c8699fe39249a9df19f82ee34bede53aa11bb57375bc1fb689。

前两点与原生融合的详细消融位于 /workspace/results/mellum_native_ablation_20261106T103715Z/，V修复与复测位于 /workspace/results/mellum_speed_moe_replay_20261106T130400Z/。四组服务均使用同一冻结环境，报告保留每轮实际测量口径。

启动入口的实际HTTP验证见 ../validation/summary.json。四个入口分别完成单请求64输出token和32并发各8输出token，合计132请求。模型与原有启动源码在这轮检查前后哈希一致。
