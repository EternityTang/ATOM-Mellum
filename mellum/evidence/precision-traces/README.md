# 精度版原始 Trace 与分析索引

2026-11-08。按用户要求，将此前仅在本地/服务器保存的精度开发 trace 原文纳入 `codex/mellum-precision`。这是证据归档提交，模型计算、默认入口和依赖保持原有实现。

共 **57 份唯一原始记录**：47 份 GPU/CPU 算子 profiler、9 份请求/调度时间线、1 份层级诊断；另附 698 份分析、运行元数据和计时记录。原文合计 3.24 GiB，逐文件 gzip 后 182.61 MiB，最大单文件 21.29 MiB。

## 阅读和校验

- 下表每个链接都指向仓库内的完整 `.json.gz`，没有抽样删事件或只存摘要。可先下载并解压为 JSON，再导入 Perfetto/Chrome trace viewer。GitHub 网页预览不是读取完整 trace 的方式。
- “层级诊断记录”不是 Chrome 时间线，使用 JSON 工具阅读；“请求/调度时间线”不等于 GPU kernel profiler。
- [manifest.json](manifest.json) 保存压缩文件及解压原文的大小、SHA256、事件/类别计数和全部来源别名。逐字节相同的记录只存一份，所有来源继续登记。
- `python3 mellum/evidence/precision-traces/verify.py` 核对全部文件和解压原文；加 `--parse-json` 可独立复算事件/类别计数。只用 Python 标准库，不占用 GPU。
- [analysis/](analysis/) 是相应阶段的分析、命令、runtime、profile 表及调度记录；[reports/](reports/) 保留当时的中文解释。历史报告里的原服务器/本地路径是来源，不保证克隆后可直接访问；本索引与 manifest 给出的仓库路径为本次交付入口。

## 版本与结果边界

该分支此前模型 HEAD 为 `0bb1b726`。后期 trace 来自独立候选快照，保存它们不表示那些候选的运行代码已经全部合入默认模型。各条 trace 的原始 source 路径/归档成员、运行开关和对应报告用于区分版本。

- 当前归档含有效方案与失败/退化候选，也含为解释精度版差距而采集的 vLLM、朴素实验版和旧速度版比较对象；不能把比较对象当作精度版成绩。
- profiler 的单步累计时长不作为无 profiler 的完整 A/B/C 吞吐；同父版输出一致也不等于对 HF 一致。
- 旧代码题/SDPA、冻结 Torch2.10 的 HF eager、A200 并发32和长输入等不同口径分别读各阶段报告。
- ABC 工程消融的四组 C 全超时；后续 C 修复快照才达到 200/200，A200 90.75%、约16k100%。后者 C p99 延迟仍高，不能宣称全部性能门槛通过；也不能将这一成绩写成当前默认入口已实现。
- `mellum-exact-decode`、`precision-attention`、`precision-moe-pack` 这些扫描阶段没有发现另存的独立 Chrome 原始 trace；相关分析保存在 analysis 中。未采集或未留存的原始事件不能用摘要重建。manifest.skipped 列出格式不属于 Chrome trace 的来源，不冒称原始 profiler。

## 阶段目录

| 阶段 | 唯一原始记录数 | 目的 |
|---|---:|---|
| mellum_reuse_profile_20261104T164236+0800 | 2 | 早期复用精度版 batch1/batch32 性能剖析 |
| fixed32-three-traces | 6 | vLLM、朴素实验版、复用精度版三方对照；含比较对象 |
| mellum-step-cache | 4 | 步内 RoPE/KV 地址复用；含速度比较对象 |
| mellum-gap-trace | 6 | 早期分页/分组路径的发射间隙对照；含 vLLM/速度比较对象 |
| mellum-exact-decode | 0 | 相关分析/组件记录，未发现独立原始 Chrome trace |
| mellum-eager-precision | 6 | 冻结 Torch 2.10 原生 HF eager 参考重建阶段；含旧 packed 比较对象 |
| mellum-opt-points | 2 | 有界 softmax 与局部 Graph 的最初前后对照 |
| precision-points | 1 | Norm 逐元素合批的初始瓶颈采样 |
| precision-attention | 0 | 相关分析/组件记录，未发现独立原始 Chrome trace |
| precision-moe-pack | 0 | 相关分析/组件记录，未发现独立原始 Chrome trace |
| precision-trace-ablation | 9 | 局部 Graph、Norm、KV、滑窗及 prefill 候选的逐阶段采样 |
| precision-prefill-stack | 2 | 稳定路由排序的 prefill 前后对照 |
| precision-speed-transfer | 3 | GQA 共享读取与被淘汰的扩大 Attention Graph |
| precision-bold-reuse | 6 | 90% 门槛下 Norm/原生专家筛选及失败多 stream 候选 |
| precision-abc-ablation | 4 | 000/001/011/111 工程消融的同位置 decode32 |
| precision-cfix | 5 | C 精确 softmax 的长 prefill/decode 前后对照与最终 C200 请求时间线 |
| native_repeat_diagnosis_aligned | 1 | 早期融合专家自身不稳定的对齐诊断，非 Chrome profiler |

## 原始记录逐份索引

| 阶段 / 原记录 | 类型 | gzip MiB | 事件数 |
|---|---|---:|---:|
| [mellum_reuse_profile_20261104T164236+0800 / trace_bs1.trace.json.gz](raw/mellum_reuse_profile_20261104T164236+0800/trace_bs1.trace-b95ea0fdf3.json.gz) | GPU/CPU 算子 profiler | 1.49 | 64082 |
| [mellum_reuse_profile_20261104T164236+0800 / trace_bs32.trace.json.gz](raw/mellum_reuse_profile_20261104T164236+0800/trace_bs32.trace-994c1f99d0.json.gz) | GPU/CPU 算子 profiler | 7.32 | 310068 |
| [fixed32-three-traces / dense/control-timeline.json](raw/fixed32-three-traces/dense-control-timeline-e5f5e01410.json.gz) | 请求/调度时间线 | 0.01 | 840 |
| [fixed32-three-traces / dense/decode.trace.json.gz](raw/fixed32-three-traces/dense-decode.trace-3bb75e422a.json.gz) | GPU/CPU 算子 profiler | 1.50 | 65780 |
| [fixed32-three-traces / precision/control-timeline.json](raw/fixed32-three-traces/precision-control-timeline-f8522d2e64.json.gz) | 请求/调度时间线 | 0.01 | 840 |
| [fixed32-three-traces / precision/decode.trace.json.gz](raw/fixed32-three-traces/precision-decode.trace-24e8f48104.json.gz) | GPU/CPU 算子 profiler | 7.02 | 296534 |
| [fixed32-three-traces / vllm/control-timeline.json](raw/fixed32-three-traces/vllm-control-timeline-dd1a811685.json.gz) | 请求/调度时间线 | 0.01 | 831 |
| [fixed32-three-traces / vllm/decode.trace.json.gz](raw/fixed32-three-traces/vllm-decode.trace-c3111bb788.json.gz) | GPU/CPU 算子 profiler | 0.24 | 10596 |
| [mellum-eager-precision / evidence/baseline-bench/cropped-trace.json](raw/mellum-eager-precision/evidence-baseline-bench-cropped-trace-83bb8307f3.json.gz) | GPU/CPU 算子 profiler | 0.96 | 53086 |
| [mellum-eager-precision / evidence/bmm-trace.json](raw/mellum-eager-precision/evidence-bmm-trace-ce63dca2cf.json.gz) | GPU/CPU 算子 profiler | 0.57 | 29386 |
| [mellum-eager-precision / evidence/candidate-bench/exact_fast-trace.json](raw/mellum-eager-precision/evidence-candidate-bench-exact_fast-trace-75b18e738b.json.gz) | GPU/CPU 算子 profiler | 0.91 | 51378 |
| [mellum-eager-precision / evidence/cropped-trace.json](raw/mellum-eager-precision/evidence-cropped-trace-7eac0c8b8b.json.gz) | GPU/CPU 算子 profiler | 0.96 | 53086 |
| [mellum-eager-precision / evidence/gather-trace.json](raw/mellum-eager-precision/evidence-gather-trace-d3606e834e.json.gz) | GPU/CPU 算子 profiler | 0.90 | 50062 |
| [mellum-eager-precision / evidence/speed-packed/packed-trace.json](raw/mellum-eager-precision/evidence-speed-packed-packed-trace-54a2124e9c.json.gz) | GPU/CPU 算子 profiler | 0.19 | 10660 |
| [mellum-opt-points / evidence/combined-profile32/trace.json.gz](raw/mellum-opt-points/evidence-combined-profile32-trace-58b021e28a.json.gz) | GPU/CPU 算子 profiler | 4.16 | 255879 |
| [mellum-opt-points / evidence/control-profile32/trace.json.gz](raw/mellum-opt-points/evidence-control-profile32-trace-b74728279a.json.gz) | GPU/CPU 算子 profiler | 8.09 | 460325 |
| [precision-points / profile-control32/trace.json](raw/precision-points/profile-control32-trace-220f464e0a.json.gz) | GPU/CPU 算子 profiler | 3.80 | 225457 |
| [native_repeat_diagnosis_aligned / trace.json](raw/native_repeat_diagnosis_aligned/trace-a71c4d9629.json.gz) | 层级诊断记录 | 0.00 | — |
| [mellum-step-cache / trace/combined/control-timeline.json](raw/mellum-step-cache/trace-combined-control-timeline-476bcf00af.json.gz) | 请求/调度时间线 | 0.01 | 840 |
| [mellum-step-cache / trace/combined/decode.trace.json.gz](raw/mellum-step-cache/trace-combined-decode.trace-54cc39a72a.json.gz) | GPU/CPU 算子 profiler | 1.53 | 67835 |
| [mellum-step-cache / trace/fast_combined/control-timeline.json](raw/mellum-step-cache/trace-fast_combined-control-timeline-fc26c8dc0e.json.gz) | 请求/调度时间线 | 0.01 | 840 |
| [mellum-step-cache / trace/fast_combined/decode.trace.json.gz](raw/mellum-step-cache/trace-fast_combined-decode.trace-240d840b65.json.gz) | GPU/CPU 算子 profiler | 0.47 | 19377 |
| [mellum-gap-trace / vllm/control-timeline.json](raw/mellum-gap-trace/vllm-control-timeline-460321959d.json.gz) | 请求/调度时间线 | 0.01 | 831 |
| [mellum-gap-trace / vllm/decode.trace.json.gz](raw/mellum-gap-trace/vllm-decode.trace-c159a9cecc.json.gz) | GPU/CPU 算子 profiler | 0.24 | 10596 |
| [mellum-gap-trace / mixed_packed/control-timeline.json](raw/mellum-gap-trace/mixed_packed-control-timeline-f415f9b036.json.gz) | 请求/调度时间线 | 0.01 | 840 |
| [mellum-gap-trace / mixed_packed/decode.trace.json.gz](raw/mellum-gap-trace/mixed_packed-decode.trace-b5c7d986d5.json.gz) | GPU/CPU 算子 profiler | 1.78 | 77707 |
| [mellum-gap-trace / paged_grouped/control-timeline.json](raw/mellum-gap-trace/paged_grouped-control-timeline-63d7e34064.json.gz) | 请求/调度时间线 | 0.01 | 840 |
| [mellum-gap-trace / paged_grouped/decode.trace.json.gz](raw/mellum-gap-trace/paged_grouped-decode.trace-86b7c05391.json.gz) | GPU/CPU 算子 profiler | 0.76 | 29614 |
| [precision-trace-ablation / s0-trace/trace.json](raw/precision-trace-ablation/s0-trace-trace-531af42818.json.gz) | GPU/CPU 算子 profiler | 5.89 | 367183 |
| [precision-trace-ablation / s1-trace/trace.json](raw/precision-trace-ablation/s1-trace-trace-f4a5c6fb35.json.gz) | GPU/CPU 算子 profiler | 3.29 | 225457 |
| [precision-trace-ablation / s2-trace/trace.json](raw/precision-trace-ablation/s2-trace-trace-e6ca62b91a.json.gz) | GPU/CPU 算子 profiler | 2.69 | 172831 |
| [precision-trace-ablation / s3-trace/trace.json](raw/precision-trace-ablation/s3-trace-trace-639b00640b.json.gz) | GPU/CPU 算子 profiler | 2.21 | 142339 |
| [precision-trace-ablation / s3-prefill-trace/trace.json](raw/precision-trace-ablation/s3-prefill-trace-trace-2e9e7ee913.json.gz) | GPU/CPU 算子 profiler | 21.29 | 1133457 |
| [precision-trace-ablation / s2-prefill-pack-trace/trace.json](raw/precision-trace-ablation/s2-prefill-pack-trace-trace-a4b8cf58ce.json.gz) | GPU/CPU 算子 profiler | 12.39 | 670904 |
| [precision-trace-ablation / legacy0-trace/trace.json](raw/precision-trace-ablation/legacy0-trace-trace-1d7856a315.json.gz) | GPU/CPU 算子 profiler | 6.16 | 389765 |
| [precision-trace-ablation / legacy1-trace/trace.json](raw/precision-trace-ablation/legacy1-trace-trace-cfcd2d2115.json.gz) | GPU/CPU 算子 profiler | 3.59 | 248039 |
| [precision-trace-ablation / legacy2-trace/trace.json](raw/precision-trace-ablation/legacy2-trace-trace-661f984b37.json.gz) | GPU/CPU 算子 profiler | 3.01 | 195413 |
| [precision-prefill-stack / control-prefill-trace/trace.json](raw/precision-prefill-stack/control-prefill-trace-trace-6e1bbfdfbc.json.gz) | GPU/CPU 算子 profiler | 21.16 | 1133457 |
| [precision-prefill-stack / pack-prefill-trace/trace.json](raw/precision-prefill-stack/pack-prefill-trace-trace-7846c251ea.json.gz) | GPU/CPU 算子 profiler | 12.38 | 670904 |
| [precision-speed-transfer / grouped-trace/trace.json](raw/precision-speed-transfer/grouped-trace-trace-db4ee830cb.json.gz) | GPU/CPU 算子 profiler | 1.80 | 116290 |
| [precision-speed-transfer / shared-trace/trace.json](raw/precision-speed-transfer/shared-trace-trace-6368a4f8ad.json.gz) | GPU/CPU 算子 profiler | 2.21 | 142339 |
| [precision-speed-transfer / control-trace/trace.json](raw/precision-speed-transfer/control-trace-trace-01211c01e0.json.gz) | GPU/CPU 算子 profiler | 2.21 | 142339 |
| [precision-bold-reuse / trace-native-fp8/trace.json](raw/precision-bold-reuse/trace-native-fp8-trace-01686ba2bb.json.gz) | GPU/CPU 算子 profiler | 2.13 | 137747 |
| [precision-bold-reuse / trace-triton-experts/trace.json](raw/precision-bold-reuse/trace-triton-experts-trace-504cbdc4e3.json.gz) | GPU/CPU 算子 profiler | 1.83 | 116463 |
| [precision-bold-reuse / trace-candidate/trace.json](raw/precision-bold-reuse/trace-candidate-trace-e8abc1b79e.json.gz) | GPU/CPU 算子 profiler | 2.11 | 135211 |
| [precision-bold-reuse / trace-control/trace.json](raw/precision-bold-reuse/trace-control-trace-14c01a5bc7.json.gz) | GPU/CPU 算子 profiler | 2.21 | 142339 |
| [precision-bold-reuse / trace-native-norm/trace.json](raw/precision-bold-reuse/trace-native-norm-trace-08e2be117b.json.gz) | GPU/CPU 算子 profiler | 2.18 | 139155 |
| [precision-bold-reuse / supplemental/trace-streams4/trace.json](raw/precision-bold-reuse/supplemental-trace-streams4-trace-79750d1430.json.gz) | GPU/CPU 算子 profiler | 2.24 | 143011 |
| [precision-abc-ablation / 000/A/trace.json](raw/precision-abc-ablation/000-A-trace-f2d7fed385.json.gz) | GPU/CPU 算子 profiler | 7.24 | 449840 |
| [precision-abc-ablation / 001/A/trace.json](raw/precision-abc-ablation/001-A-trace-1e8fc39d5a.json.gz) | GPU/CPU 算子 profiler | 5.43 | 344364 |
| [precision-abc-ablation / 011/A/trace.json](raw/precision-abc-ablation/011-A-trace-a0fef02bf9.json.gz) | GPU/CPU 算子 profiler | 2.39 | 152640 |
| [precision-abc-ablation / 111/A/trace.json](raw/precision-abc-ablation/111-A-trace-9bae052030.json.gz) | GPU/CPU 算子 profiler | 2.22 | 142884 |
| [precision-cfix / ageguard-http/age140-fair4-C/request-timeline.json](raw/precision-cfix/ageguard-http-age140-fair4-C-request-timeline-c54b13eb39.json.gz) | 请求/调度时间线 | 0.01 | 200 |
| [precision-cfix / ageguard-http/decode-control/trace.json](raw/precision-cfix/ageguard-http-decode-control-trace-76ffab8133.json.gz) | GPU/CPU 算子 profiler | 2.02 | 130928 |
| [precision-cfix / ageguard-http/decode-exact/trace.json](raw/precision-cfix/ageguard-http-decode-exact-trace-a596dd0097.json.gz) | GPU/CPU 算子 profiler | 1.91 | 124768 |
| [precision-cfix / hfsoftmax-http/profile-hfsoftmax-long/trace.json](raw/precision-cfix/hfsoftmax-http-profile-hfsoftmax-long-trace-f2dea36fa1.json.gz) | GPU/CPU 算子 profiler | 2.62 | 143058 |
| [precision-cfix / tile-work/profile-control-long/trace.json](raw/precision-cfix/tile-work-profile-control-long-trace-60070027cc.json.gz) | GPU/CPU 算子 profiler | 2.82 | 153538 |

## 阶段说明

- [原生输出不稳定定位.md](reports/原生输出不稳定定位.md)
- [复用版性能剖析.md](reports/复用版性能剖析.md)
- [固定32并发三版本算子与调度Trace.md](reports/固定32并发三版本算子与调度Trace.md)
- [Mellum解码RoPE与KV跨层复用实验.md](reports/Mellum解码RoPE与KV跨层复用实验.md)
- [Mellum精度版提速与速度版差距.md](reports/Mellum精度版提速与速度版差距.md)
- [Mellum优化点消融与性能验收.md](reports/Mellum优化点消融与性能验收.md)
- [精度版逐项提速实验记录.md](reports/精度版逐项提速实验记录.md)
- [精度版三点Trace与消融.md](reports/精度版三点Trace与消融.md)
- [精度版滑窗Attention合批验收.md](reports/精度版滑窗Attention合批验收.md)
- [精度版参考速度版的新一轮优化.md](reports/精度版参考速度版的新一轮优化.md)
- [精度版放宽一致率后的原生组件复用.md](reports/精度版放宽一致率后的原生组件复用.md)
- [精度版三点ABC工程消融.md](reports/精度版三点ABC工程消融.md)
- [精度版C场景修复尝试.md](reports/精度版C场景修复尝试.md)

历史采样未重新运行；只恢复原始事件并做校验。既有提交历史完整保留，没有 squash。

完整开发过程回顾：[Mellum精度版开发全记录.md](reports/Mellum精度版开发全记录.md)。
