# 交付入口

源码基准为 codex/mellum-native-speed-final 的372601f7。本目录对应题面交付物1、2。根目录run0.sh～run3.sh和report.md为交付入口与报告。

## 一条命令启动 ATOM 推理服务

    CANDIDATE_GPU=0 bash /workspace/run3.sh

仓库内也可执行bash run3.sh。四个脚本分别固定000、100、110、111，均调用本仓库mellum/run_speed_ablation.sh，透传后续ATOM参数。脚本通过真实文件路径定位仓库，支持从其他目录及符号链接调用。默认8830为HTTP端口、8831为引擎端口，可用环境变量覆盖。

单独消融可设置MELLUM_SPEED_GRAPH、MELLUM_SPEED_EXPERT_FP8、MELLUM_SPEED_QK_FUSION后调用bash mellum/run_speed_ablation.sh。三个变量默认1，接受0/1。评测按声明的单卡配置运行。

运行使用镜像的/opt/venv/bin/python和既有模型目录。依赖环境、权重及原始数据由题面容器提供。新增根目录脚本的服务HTTP验证记录保存在validation目录。

## 报告

[正式报告](../report.md)按四页结构整理，三个PAGEBREAK标记用于分页。页面包括运行与开关、profiling与优化、ABC消融、HumanEval与优化取舍。实际排版页数经独立渲染校验，Markdown为主要交付格式。

最新性能为A2038.47 tok/s/GPU，HumanEval pass@1为90.24%（148/164），采样协议见报告第四页。本报告记录速度版的性能优化与HumanEval功能评测。

## 版本与证据

[开发全记录](../mellum/history/native-speed/development.md)与仓库提交保留阶段历史。[证据索引](evidence/README.md)列出各文件来源、用途和校验值。完整数据库和输出归档位于记录的服务器实验路径。交付文件与测量来源见交付清单。
