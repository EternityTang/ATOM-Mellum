# Mellum 精度版提速与速度版差距

2026-11-05。当前验收参考统一为 **冻结的 Torch 2.10 + 原生 Transformers 5.20.2，attention=eager、experts=eager**。没有升级或修改 Torch。

本轮把精度版修正到这一参考后，再将吞吐从 **204.6 提升至 309.0 output token/s，提升 51.0%**。公开 C 的 200 条逐条生成全部通过；32 并发对 HF 单条参考仍未达到 90%，不能把逐条结果推广到并发。原速度版约 662.5 token/s，但没有通过本轮新的精度参考。

## 1. 最终对比

性能为同一张 MI308X（GPU1）串行运行，固定 32 条不同的 1024-token 输入，每条输出 256 token；每轮先以同批输入预热 4 token，再计时完整的 8192-token 输出。时间包含 prefill，使用离线队列、最大活动序列 32。BF16、eager、引擎编译等级 0、MTP 关闭、prefix cache 关闭、显存比例 0.5。以下不是正式 HTTP 场景 A/B/C 成绩。

| 版本 | 三轮 output token/s | 中位数 | 固定 10 条对新 HF | C 200 条逐条对新 HF |
| --- | --- | ---: | ---: | ---: |
| 修正后的 eager 精度父版 `cropped` | 204.61 / 204.65 / 204.51 | 204.61 | 640/640，100% | 12800/12800，200/200 条完整一致 |
| **新精度版 `exact_fast`** | **309.51 / 308.98 / 300.77** | **308.98** | **640/640，100%** | **12800/12800，200/200 条完整一致** |
| 原速度版 `packed` | 662.46 / 652.07 / 662.51 | 662.46 | 196/640，30.63%；2/10 条完整一致 | 本轮未重跑 |

这里“完整一致”仅指本次生成的全部 **64 token**。固定 10 条包含 A/C 的困难样本。C 200 条与原始 `/data/datasets/scenario_c_prompts.jsonl` 的 prompt 逐条相同。HF 的每个输出位置均检查了原始 logits 的 argmax；没有使用采样解码。

历史 `/workspace/run_mellum_exact_decode.sh` 的约 315–321 token/s 对应旧 SDPA 路径，不能作为本次原生 eager 精度目标的有效基线。此前 Torch 2.13 的 HF 结果保留为历史记录，不参与当前验收。

### 32 并发精度单独报告

同一固定 32 条输入，每条生成 256，取前 64，与冻结 Torch 2.10 下 HF **逐条生成**的参考比较：

| 版本 | 相同 token / 2048 | 一致率 | 64 token 全部相同的请求 |
| --- | ---: | ---: | ---: |
| eager 精度父版 | 925 | 45.17% | 10/32 |
| 新精度版 | 925 | 45.17% | 10/32 |
| 原速度版 | 955 | 46.63% | 11/32 |

各版三轮结果均重复。新精度版与父版在整批 **8192/8192 输出 token** 上一致，父版三轮均通过这个比较。因此，这次提速没有引入观察到的并发差异；但并发结果对 HF 单条参考确实未过线。批大小会改变矩阵运算与专家分组的形状，这是进一步排查的方向，本轮没有把并发分叉逐层定位到某个唯一算子。

沿用此前“按题面分别验收”的决定，逐条正确性与并发性能分别列出。**本轮尚未实现 600 token/s 的新精度版，也未解决并发 token 一致率。** 没有重跑正式 A/B/C 性能或 HumanEval。

## 2. 先修正了什么

1. **参考环境**：最新 HF 官方 wheel 通过隔离目录加载，解释器及 Torch 始终来自原镜像 `/opt/venv`。审计的 2765 个 HF 文件全部匹配官方 wheel，没有改写 Mellum 源码。
2. **eager 注意力边界**：保留 BF16 QK 矩阵乘、缩放、加 mask，再以 FP32 softmax 计算概率、转回 BF16，最后执行 PV 矩阵乘。prefill/decode 都不使用 SDPA。
3. **SWA decode**：只取末尾 1024 个 K/V，匹配原生 HF sliding cache；“读取全历史再遮住旧 token”虽然数学等价，但改变 softmax 和 GEMM 的形状，不适合作为当前数值参考。
4. **RoPE**：按新版 HF 的 FP32 position × inv_freq 元素乘形式构造频率，保留原 Torch 旋转与精度边界。

先得到 200 条全部一致的 eager 父版，随后再验证提速版，两个阶段的原始输出分别保存。

## 3. 提速来自哪里

这轮有效优化保留浮点计算组织，主要减少数据搬运和 CPU 调度：

- 直接从分页 KV 中读取所需窗口的 **4 个 KV 头**，组织成连续布局；再用专门的 Triton kernel 按位复制为 32 个注意力头。没有将注意力矩阵乘改成另一种 GQA 形状。
- 同一 forward 的 RoPE 频率与 mask 跨层共享，减少重复构造。
- 对足够大的路由批使用 TorchScript 调度原来的逐专家 `F.linear`，保留每个 GEMM 的实际行数、顺序和 BF16 边界；小批仍使用原循环。
- 在已限定的单卡普通 Tensor forward 范围中临时移除 Torch 默认设备的 Python 拦截；算子的设备来自明确的参数或输入 Tensor。

组件验证包含随机非连续物理页，以及 1023/1024/1025 等窗口边界，7 组 KV 检查全部按位一致。单独的 K/V 头复制在 `N=32,L=1280` 时从约 **2.074 ms 降至 0.383 ms**，输出按位一致。HTTP 新入口通过单请求 64-token 文本对照及四并发冒烟检查。

### 同条件 trace

三版均抓取一次真实 decode forward：32 行、上下文长度全部 1025、28 层。表中模块 GPU 时间取 CPU range 对应的归因时间；kernel 总时长只统计原始 trace 的 `kernel` 事件，排除了同名 GPU annotation 的重复计数。Profiler 本身有开销，以下用于定位，不替代无 profiler 吞吐。

| 指标 | eager 父版 | 新精度版 | 速度版 |
| --- | ---: | ---: | ---: |
| GPU kernel 数 | 4304 | 4220 | 888 |
| GPU kernel 时长总和 | 101.68 ms | 61.87 ms | 15.04 ms |
| 注意力模块 GPU 归因时间 | 73.77 ms | 28.57 ms | 4.09 ms |
| MoE 模块 GPU 归因时间 | 21.22 ms | 20.74 ms | 10.05 ms |
| `aten::copy_` GPU 时间 | 48.75 ms | 3.17 ms | 0.92 ms |
| `aten::mm` 次数 | 1220 | 1220 | 84 |
| `aten::bmm` 次数 | 168 | 168 | 2 |
| MoE 模块 CPU 时间 | 91.36 ms | 57.22 ms | 19.00 ms |

`copy_` 时间下降不能全部视为消失：新候选另有 `_gather_heads` 约 6.26 ms、`_repeat` 约 6.31 ms。整体仍明显降低数据重排成本。MoE 的 GPU 计算时间变化小，但 CPU 调度时间下降。这解释了为何本轮可在保留父版输出的情况下提速。

## 4. 为什么仍比速度版慢

| 环节 | 新精度版 | 速度版 |
| --- | --- | --- |
| 注意力 | 物化 K/V、复制 KV 头、分步执行 QK/softmax/PV | 直接分页读取，在融合内核内执行注意力 |
| 专家计算 | 保留逐专家 GEMM 形状，仍有大量小矩阵调用和路由同步 | GPU 对齐路由，两次分组 MoE GEMM；trace 中共 56 次融合专家 kernel |
| QKV、norm、RoPE | 分开的 Q/K/V 投影与原 Torch norm/旋转 | 合并 QKV，并融合 QK norm/RoPE、部分 norm 等计算 |
| CPU 开销 | TorchScript 降低了一部分循环开销，仍需逐专家提交运算 | 融合后提交的算子明显更少 |

差距有直接的算子和 CPU 调度证据，不能简单归因于请求调度器。速度版在小批 decode（1–7 行）和大批 decode 的计算路径不同；新精度版采用统一的 eager 数值规则，专家循环仅改变调度方式。

本轮也尝试了更激进的路径，但未采用：融合 norm/RoPE 加脚本调度的组合只有 406/640；分组 Torch BMM 专家只有 300/640；改变 GQA 矩阵形状的快速路径只有 413/640。它们均低于 90%。这些是各组合的实测结果，不能将组合差异全部归因于其中某一个改动。

下一步应分别处理两件事：定位同一实现的单条/并发分叉；开发保持明确 BF16 边界的注意力与专家内核，再逐项验证。不能直接把当前速度版内核全部替换进来并继承新精度版的 100% 结论。

## 5. 新增文件与启动方式

服务器新增候选目录 `/workspace/tools/mellum_eager_precision/`：

| 文件 | 作用 |
| --- | --- |
| `install.py` | eager/SWA/RoPE 修正、跨层复用及调度接入，检查单卡 BF16/eager/level 0/MTP-off 等前提 |
| `safe_ops.py` | 按物理页读取窗口 KV；同形状逐专家 GEMM 的 TorchScript 循环 |
| `repeat_ops.py` | 按位一致的连续 K/V 头复制 |
| `serve.py` | 模型加载后安装候选，启动原有 OpenAI 兼容服务 |
| `__init__.py` | 导出安装函数 |
| `/workspace/run_mellum_eager_precision.sh` | 固定环境和运行约束的独立入口 |

```bash
CANDIDATE_GPU=1 SERVER_PORT=8468 bash /workspace/run_mellum_eager_precision.sh --port 8469
```

`SERVER_PORT` 为 HTTP 端口，`--port` 为内部引擎通信端口。入口复用现有 ATOM 和外部工具，依赖快照与哈希保留在证据目录。内部部分旧函数名仍含 `sdpa`，仅作为接入点；实际新候选的 prefill/decode 都执行显式 eager，decode trace 中 SDPA 调用数为 0。

初期显存比例 0.65 的长输入实验曾 OOM；后续统一为 0.5 并通过完整 C200。显式 eager 的长 prefill 工作空间较大，本轮结果不能保证任意 32K 请求组合都能容纳。失败日志保留。HTTP 初次冒烟使用了当前接口不接受的 token-ID prompt，返回 422；改为接口支持的文本 prompt 后通过，没有修改协议。

没有修改既有 ATOM 生产源码、旧启动脚本或已安装库，没有替换 `/workspace/run0.sh` 等交付入口。Git HEAD 仍为 `6c86378f230d7d10734b574aa30988a349ebd2ff`，工作区状态与开测前相同，**没有 Git 提交**。

## 6. 复现证据

服务器：`/workspace/results/mellum_eager_opt_4k8RQUX2/`；本地：[证据与源码](server-step0/mellum-eager-precision/README.md)。

重点文件：`final-summary.json`、`final-review.log`、`runtime-manifest.json`、`source-audit.json`、`final-source-check.json`；`hf-frozen-C200.jsonl` 是新参考，`baselineC200/` 和 `candidateC200/` 是两个完整输出集；`baseline-bench/`、`candidate-bench/`、`speed-packed/` 含三轮性能、原始 token、shape 记录及 trace。`code/final_review.py` 独立重算各项指标，检查输入、长度、哈希和父版输出保持。

参考 HF wheel SHA256：`7ae75dbc1fb3922aff1b6c97b9e5479d6cf26fd86a8866bfb71324190479f807`。C200 新参考 SHA256：`d05a173791a4496cef736cf060654a9431f72b1fb87e1b507feacd721599743f`。

测试正常结束后，两 GPU 利用率均为 0%，显存各约 284 MiB；未恢复旧服务或队列。
