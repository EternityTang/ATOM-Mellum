# Mellum 原生三点速度版

本分支基于 ATOM `26f23a0b` 重建，复用 Qwen3-MoE 骨干、MoE、norm 和权重加载，通过原生逐层接口接入 Mellum 的滑窗与全局 attention。三个加速点为 decode HIP Graph、专家在线 FP8、QK norm/RoPE/KV 写入融合。滑窗 decode 使用与融合写入一致的 SHUFFLE V 视图。

## 启动

```bash
CANDIDATE_GPU=0 bash mellum/run_speed_ablation.sh
```

运行依赖实验容器提供的 `/opt/venv`，模型为 `/data/models/Mellum2.1-12B-A2.5B-Thinking`。默认单卡、Torch 2.10、编译等级 0、BF16 KV，MTP 与前缀缓存关闭。默认服务端口8830、引擎端口8831，可分别通过 `SERVER_PORT`、`ENGINE_PORT` 指定。

| 优化点 | 环境变量 | 0 的行为 |
| --- | --- | --- |
| decode HIP Graph | `MELLUM_SPEED_GRAPH` | eager 执行 |
| 专家在线 FP8 | `MELLUM_SPEED_EXPERT_FP8` | BF16 专家 |
| QK norm/RoPE/KV 写入融合 | `MELLUM_SPEED_QK_FUSION` | 非融合路径 |

三个变量默认1，均接受0或1。以下命令复现前两点。

```bash
MELLUM_SPEED_GRAPH=1 MELLUM_SPEED_EXPERT_FP8=1 MELLUM_SPEED_QK_FUSION=0 \
  CANDIDATE_GPU=0 bash mellum/run_speed_ablation.sh
```

## 验收

2026-11-07，各场景各配置各测一轮，包含 V 缓存视图修复。A/B 在 GPU1，C 在 GPU0，各场景四组使用同一张卡。全部完成 A512、B50、C200，每条输出256 token。

| 开关 | A tok/s/GPU | B TPOT p99 ms | C TTFT p99 ms | C TPOT p99 ms |
| --- | ---: | ---: | ---: | ---: |
| 000 | 674.50 | 29.53 | 1225.52 | 42.06 |
| 100 | 934.12 | 5.19 | 1064.32 | 13.46 |
| 110 | 1505.64 | 6.16 | 740.07 | 12.41 |
| 111 | 2038.47 | 5.95 | 695.48 | 11.67 |

最终111另跑 A200 并发32，自然生成256 token，前64对冻结 Torch2.10 / 原生 Transformers5.20.2 eager 逐条参考为 **3030/12800，23.6719%**，11/200条完整一致，首 token 199/200。约16k组为110/256，42.9688%。两组均低于90%门槛。HF 参考重放的两个样本均复现已保存输出。

[完整开发过程](history/native-speed/development.md)记录尝试、淘汰、错误及复核。[ABC 与 token 报告](history/native-speed/abc-token-report.md)和[机器可读结果](history/native-speed/abc-final-summary.json)保留实际测试口径。

## 历史来源

本分支按真实提交时间，从保存的源码快照重建四个阶段，提交说明记录来源。原生重建基于适配前的 ATOM，历史 packed 分支 `codex/mellum-speed` 仍保留在 `27fd6d9b`。精度路线有独立分支。

原始评测脚本、数据库、输出 token 和日志位于服务器 `/workspace/results/mellum_speed_abc_20261107T064400Z/`。证据归档 SHA-256 为 `0f800aec96b48f9ebdee75cd56988f507281843693a1439aa331304cc3aadfd3`。本分支收录文本报告、汇总数据和来源清单。
