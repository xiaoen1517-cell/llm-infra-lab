# llm-infra-lab

52 周 AI Infra 学习计划的实验仓库。战略见 `01.md`，逐周执行手册见 `02.md`。

## 目录约定

| 目录 | 放什么 |
|------|--------|
| `notes/` | 每周笔记（结论用英文写）；`glossary.md` 术语本 |
| `code/` | 实验代码 |
| `bench/` | 压测数据（csv / md） |
| `runs/` | 训练日志与 README |
| `scripts/` | 部署脚本 |
| `configs/` | 推理引擎配置 |
| `cpp/` | C++ 练习 |
| `cuda/` | CUDA kernel |
| `deploy/` | Docker / K8s |

## 每周纪律

1. 每周日结束前：交付物进仓库 + 更新 `WEEKLY.md`（结果表 + 可复现命令，缺一视为未完成）
2. 公式一律翻译成 numpy 代码验证，对上数字才算懂
3. 卡住 2 小时：记进 WEEKLY.md「卡点」区，跳过或求助，不死磕
