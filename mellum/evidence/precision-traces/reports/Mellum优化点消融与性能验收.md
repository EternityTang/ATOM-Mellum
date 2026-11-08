# Mellum 优化点消融与性能验收

2026-11-06。目标是 token 一致率 ≥90% 条件下提高性能，不要求完全一致。原精度版本、公共框架与依赖保留，无 Git 提交。

## 当前结论

第一轮三个方向中，局部 graph 是已测短吞吐收益最大的合格候选；有界 softmax 解决已复现的长 prefill OOM；MoE 候选没有得到同时满足精度和收益的方案。因此保留两个点，不凑满三个点。正式结果与短测分开列出。

## 参考与验收

- 冻结 Torch 2.10，原生 Transformers 5.20.2，BF16，HF attention=eager、experts=eager、greedy、逐条、整段 prefill、不 padding。比较自由生成的前64个原始token ID。
- 用户于2026-11-06明确确认沿用A200并发32＋16k长输入组：对HF逐条参考比较A前200条，共12800位置，至少11520位相同达到90%；约16k长输入组单独验收、不与A合并平均，当前固定四条。正式生成256 token，再比较前64位。上一轮已按该口径执行，无需因本次确认重跑。HumanEval本轮未重跑，MTP保持关闭。
- 复用原180条A参考并补齐20条，检查模型源码哈希、版本、输入哈希、生成配置与argmax。上一轮真实HTTP A512的前200条，独立复算为12800/12800、200/200完整一致。
- 固定难例10条用于筛选，32条防止小批误判；它们不能替代完整200，也不能据短集未过线宣称某失败候选完整200必然低于90%。
- 所有性能比较在同卡、其他GPU空闲时进行；最初10条与补HF参考存在重叠，仅作初筛。正式HTTP与三轮32条比较没有并行GPU任务。

## 三个候选方向及独立开关

| 候选 | 瓶颈与改动 | 开关 | 选择 |
|---|---|---|---|
| P1：attention临时显存 | 15194-token prefill的FP32 softmax申请27.52GiB失败；保持QK/PV矩阵形状及完整key长度，只对大矩阵按4个head分段做Torch softmax | OPT_ATTENTION=off/bounded | 入选点1；主要收益为长输入可运行 |
| P2：MoE调度/内核 | 比较原cap8 Python派发、同形状TorchScript派发、现有AITER Triton融合专家kernel | OPT_MOE=off/script/triton | 不入选；前者无稳定提速，融合候选精度不合格 |
| P3：局部graph | norm、QKV/o_proj/router/LM head保持逐请求算术形状，捕获这些局部算子，降低CPU逐kernel派发 | OPT_DISPATCH=off/graph/graph_bucket | 入选点2 |

graph_bucket仅对独立逐请求算术使用1/2/4/8/16/32档位，取回实际请求的行；不padding整个骨干，不扩大MoE路由输入，不缓存prompt或生成结果。ROCm通过PyTorch CUDAGraph接口使用HIP Graph。引擎仍level0/enforce-eager，但适配包内部开启局部graph，不能称为完全不开graph。

## 单项与组合消融：三轮短测

固定32条真实A输入、每条64输出，Engine.generate，预热4 token后计时，模型加载和profile不计入。以下数值不是正式场景A的tokens/s/GPU。

| 配置 | 吞吐中位数(token/s) | 相对对照 | 三轮token一致数 |
|---|---:|---:|---|
| 对照 norm_cap8 | 33.976 | +0.00% | 2048/2048、2048/2048、2048/2048 |
| 仅P1 bounded | 33.777 | -0.59% | 2048/2048、2048/2048、2048/2048 |
| 仅P2 script | 33.509 | -1.38% | 2048/2048、2048/2048、2048/2048 |
| 仅P3 graph | 42.419 | +24.85% | 2048/2048、2048/2048、2048/2048 |
| P1＋P3 | 43.115 | +26.90% | 2048/2048、2048/2048、2048/2048 |

P1在这组短输入上的吞吐变化接近噪声，不能宣称它有明显A提速。组合与仅graph之间的少量差异也不作为独立P1加速证据。正式候选使用graph_bucket以支持C变化的批大小，其短测检查另列；三轮graph消融用graph模式。

## 失败和放弃的候选

| 尝试 | 数据集 | 一致率 | 处理 |
|---|---|---:|---|
| 投影/norm TorchScript | 10条×64 | 79.688% | 未进入正式候选 |
| 融合MoE初筛 | 10条×64 | 30.156% | 未进入正式候选 |
| 融合MoE扩大 | 32条×64 | 39.795% | 未进入正式候选 |
| Triton softmax全部层 | 4条×64 | 69.141% | 未进入正式候选 |
| 仅global层Triton softmax | 4条×64 | 48.828% | 未进入正式候选 |
| 仅SWA层Triton softmax | 4条×64 | 41.406% | 未进入正式候选 |

- 首次Triton softmax原型的row×length偏移使用int32，16k矩阵超出范围，发生GPU地址错误；已改为int64并重测。地址问题修正后仍未达精度门槛，未采用。失败源码、日志与GPU dump保留，dump仅保存在服务器。
- TorchScript并不能自动保证数值路径相同。投影/norm脚本化候选的短集出现差异，本轮未逐层证明其唯一误差来源；原算子局部graph方案通过了后续检查。
- 最终仍复用Qwen3-MoE的骨干、MoE容器、参数和加载；容器名FusedMoE不代表启用了融合专家计算，最终OPT_MOE=off沿用Torch cap8专家。
- 初次测试脚本spawn子进程重复创建目录失败，已修正为允许已存在目录，保留失败记录；这不属于模型优化收益。

## 长输入与变化批大小

固定长输入：C-0009（15194）、C-0049（16019）、C-0089（16003）、C-0158（15965）token。原C故障输入和最长三条事先选定；推理代码不读取这些ID。

| 检查 | 一致数 | 说明 |
|---|---:|---|
| P1有界softmax长输入 | 256/256 | 原始token；不代表完整C200性能 |
| P1＋graph长输入 | 256/256 | 原始token；不代表完整C200性能 |
| graph_bucket 10条 | 640/640 | 原始token；不代表完整C200性能 |
| graph_bucket 17条 | 1088/1088 | 原始token；不代表完整C200性能 |
| 预热全部6档后长输入 | 256/256 | 原始token；不代表完整C200性能 |

预热全部graph档位后，长输入检查记录到1524个局部graph、fallback=0。PyTorch峰值allocated=133.638GiB，reserved=166.695GiB；这些是该检查的分配器指标，不是正式C的连续显存监控。

## Profiling证据

捕获固定32请求的一次模型decode forward，单独执行，不将profile耗时算入性能。范围不含外层调度、独立LM head与采样流程，因此不能当作完整请求端到端耗时。

| 指标 | 对照 | P1＋graph |
|---|---:|---:|
| hipLaunchKernel | 38968 | 7939 |
| hipExtModuleLaunchKernel | 8772 | 4292 |
| hipModuleLaunchKernel | 1820 | 1820 |
| 上述普通launch合计 | 49560 | 14051 |
| hipGraphLaunch | 0 | 253 |
| profile Self CPU合计 | 896.806ms | 520.135ms |
| profile Self CUDA合计 | 316.482ms | 355.714ms |

graph减少的是CPU发起次数，内部GPU算子仍执行，并有输入缓冲复制成本；不能声称GPU计算工作量同比减少。GPU profile合计甚至有所增加。组合中attention仍占较多CPU时间，归一化/投影仍保持逐请求算术，不能据此推断可达到vLLM吞吐。上述profile有额外开销，正式收益以无profile测量为准。

## 正式HTTP A/B/C

与上轮norm_cap8使用同一GPU1；数据、并发、输出256、seed42、greedy、ignore_eos、C泊松0.30/s及180秒请求超时保持。BF16、无MTP、prefix cache关闭、不分块prefill、max-num-seqs32、token budget16384、显存比例0.5。预热graph档位及A最长32条，不计入测试时间；客户端始终对接ATOM OpenAI server。

| 场景 | 成功/请求 | 输出token/s | TTFT p50(ms) | TTFT p99(ms) | TPOT p50(ms) | TPOT p99(ms) |
|---|---:|---:|---:|---:|---:|---:|
| A | 512/512 | 67.011 | 15333.208 | 20685.920 | 421.755 | 465.153 |
| B | 50/50 | 19.002 | 466.333 | 485.806 | 51.209 | 51.892 |
| C | 0/200 | 0.000 | — | — | — | — |

正式A相对上轮45.9763 token/s的变化为+45.75%。上轮B为16.4705 token/s；上轮C在第10条长输入OOM，不能用于计算有效C加速比。

### 与本轮优化前的正式结果比较

| 指标 | norm_cap8 对照 | 当前候选 | 变化 |
|---|---:|---:|---:|
| A 输出吞吐(token/s) | 45.9763 | 67.011 | +45.75% |
| B TPOT p50(ms，越低越好) | 58.3270 | 51.209 | -12.20% |
| B 输出吞吐(token/s) | 16.4705 | 19.002 | +15.37% |
| C 完整请求 | 长输入OOM，中止，0条完整 | 0/200 | 不计算加速比 |

这张表比较同一精度路线的优化前后；不能代替题方基线的正式得分。单项完整ABC消融尚未全部运行，现有单项证据为32条三轮筛选；不将组合收益冒充每个点的独立正式收益。

正式A前200条：12800/12800=100.0000%，完整相同200/200。来自同一次256-token性能生成的前64 token，对HF逐条参考，未另换精度配置。

C共200条，仅1条拿到至少64 token、完整前64一致1条。所有流最多取前64后，共观测到388/12800个位置，其中388个匹配；缺少的位置不能算匹配，也不能把这一小部分样本包装成C200一致率通过。完整256-token成功率仍以上表为准。

C失败请求数：200；错误消息汇总：{'TimeoutError': 200}。分位数若存在仅反映评测器的成功样本统计。

## 独立启动入口检查

生产入口 /workspace/run_mellum_opt_points.sh 使用通用32-token输入预热6档graph，不读取评测prompt或HF参考。启动后通过HTTP检查64个token，其中64个匹配HF；这是启动复核，完整精度结论仍来自正式A200。

启动检查结束后已停止本轮服务，8660/8661/8680/8681关闭；最终两卡均空闲、显存各约284MiB。清理前后记录保存在shutdown.json与shutdown-final.json。

## 下一步瓶颈与结论边界

本轮可保留的最大吞吐收益来自局部graph，减少逐算子CPU发起成本；它没有将每请求norm/投影和专家cap8合并成任意大矩阵。attention仍保留HF eager兼容的全矩阵及mask路径，长prefill二次方工作量和调度阻塞没有消失。因此该候选离既有速度版或vLLM仍有较大差距。
另外检查了原调度器：atom/model_engine/scheduler.py:633明确优先prefill；802行起在有prefill时直接返回，之后才进入decode分支。持续到达的整段长prefill会推迟已有请求的decode，这是C超时的一个可解释机制，尚未做独立调度消融，不能认定为唯一原因。原文件快照和审查记录保存在code/scheduler-audit-original.py.txt与C-scheduling-audit.json。
HTTP流的正常清理也不在finally中，所读清理函数仅删除前端映射、未显式取消core请求；这提示超时后仍可能保留无效工作，但本轮没有独立取消实验，不能给它分配确定的耗时占比。相关原始源码也已保存，未修改公共框架。
后续可分别实验保持BF16边界的attention分块/专用softmax、prefill/decode调度公平性与取消处理、选择性扩大专家小批、扩大graph捕获区域；每个变体仍以≥90%筛选。已失败的Triton softmax和融合专家不能直接当成可用优化，当前没有证据保证这些后续方向一定提速或过线。

## 文件、复现与边界

- 新实现包：/workspace/tools/mellum_opt_points/；install.py安装各独立开关，ops.py保留TorchScript候选，graph.py实现局部graph缓存及档位，softmax.py保留未采用的Triton候选，serve.py接入ATOM服务，serve_entry.py用通用输入完成启动预热；新增 /workspace/run_mellum_opt_points.sh 为独立入口。
- 实验目录：/workspace/results/mellum_opt_points_20261105T172604。implementation-v1至v5保存探索源码；各screen日志与原始token记录保留，http-best/implementation为正式服务使用的快照。
- analyze_points.py独立复算所有短测token、正式请求参数/覆盖、指标、源文件哈希与Git状态；make_report.py生成本报告。
- 本轮没有修改419个原有源文件或安装依赖，没有Git提交。已创建的候选文件和实验脚本独立于原实现。
- 本报告是本轮实验记录；未将未经正式测量的单项填成题面最终完整ABC消融成绩。单项三轮短测与最终HTTP结果明确分开；HumanEval尚未在此候选重测。
- GPU故障dump体积较大，仅留服务器failure-coredumps/；证据归档采用逐文件SHA256清单，完整trace另以gzip无损压缩保留，排除项记录在archive-omissions.json。
