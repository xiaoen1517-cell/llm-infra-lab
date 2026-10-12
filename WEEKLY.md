# WEEKLY

每周追加一节，模板如下。没有「结果表」或「可复现命令」的周，视为没完成。结论部分用英文写。

```text
## Week XX
- GPU: 机型 / CUDA / 模型
- 目标:
- 命令与配置:（可复现）
- 结果表:
- 结论（3 句）:
- 卡点:（卡住超 2h 的记这里）
- 下周只改一个变量:
```

**进度协作约定：** 你描述完成了什么 → 我更新本文件 + 对应交付物路径。当前周：`Week 3`。

---

## Week 0
- GPU: 延后（等 Week 4 / 真正需要时再开）
- 目标: 环境准备 + 数学扫盲启动
- 状态: ✅ 本地验收通过（云 GPU / torch.cuda 延后，不挡进 Week 1）

### 最短 checklist

| # | 项 | 状态 |
|---|----|------|
| 1 | 注册云 GPU，开实例，`nvidia-smi` 有输出 | ⏸️ 延后 |
| 2 | `python -c "import torch; print(torch.cuda.is_available())"` → `True` | ⏸️ 延后 |
| 3 | 本地确认本仓库目录齐全（已就绪） | ✅ |
| 4 | Jupyter/notebook 可用 | ✅ web 版已在用 |
| 5 | 数学扫盲：3B1B《线性代数的本质》1–4 集 | ✅ 已看完（直觉靠后续代码加深） |
| 6 | numpy 验算：点积、矩阵乘、广播；`(B,T,C)@(C,C)` 输出形状 | ✅ 7 题全对 |
| 7 | `notes/glossary.md` ≥ 5 个真实术语 | ✅ 7 条 |

- 命令与配置: 本机 Jupyter web；无云实例命令
- 结果表:

| 检查项 | 结果 |
|--------|------|
| nvidia-smi | 延后 |
| torch.cuda.is_available() | 延后 |
| 3B1B 1–4 | 已看完 |
| numpy 形状自检 | 7/7 正确 |
| glossary 条数 | 7+（W1 继续追加） |

- 结论（3 句）:
  - Finished Week 0 local track: shapes (matmul/broadcast/batch) and a starter glossary.
  - Cloud GPU and `torch.cuda` are deferred until Week 4 / real GPU need.
  - Ready for Week 1: tensor shapes and AttentionHead Q/K/V + causal mask.
- 卡点: 3B1B 理解不深——已用形状题过关；云 GPU 主动延后
- 下周只改一个变量: 进入 Week 1，只盯 Attention 张量流向与形状注释

---

## Week 1
- GPU: 本机 / Jupyter web（无云）
- 目标: 张量与形状；搞懂 Attention 数据流（Q/K/V + mask）
- 状态: ✅ 完成

### 本周总 checklist（15h）

| 块 | 内容 | 约时 | 状态 |
|----|------|------|------|
| A | 学习：语言模型流水线概念 + 术语 | 1.5–2h | ✅ |
| B | PyTorch broadcasting / einsum | — | ⏸️ 并入 C |
| C | 三种 attention scores | 3h | ✅ |
| D | AttentionHead + causal mask + loss 下降 | 4–5h | ✅ |
| E | `notes/w01-shapes.md` + 多头流向（代码理解拆头/view/concat） | 3h | ✅ 10/12 默写通过（形状全对，含两次转置）；10/8 曾错 scores 形状。两次都漏写 causal mask，需重点记 |

- 命令与配置: `code/w01_attention_scores.py` / `w01_attention_head.py` / `w01_multihead_attention.py`
- 结果表: scores 三法一致；TinyLM loss→0.0016；多头 shape 链已通
- 结论（3 句）:
  - Can explain Q/K/V, scores (B,T,T), causal mask (-inf before softmax), and attn@V.
  - Multi-head = split C into H×D, attend per head as (B,H,T,D), concat back to (B,T,C) via view.
  - Shape notes in `notes/w01-shapes.md`; ready for Week 2 structural pieces (PE/residual/LN).
- 卡点: mask 曾误以为「变成负数」——已纠正为 softmax(exp(-inf))=0
- 下周只改一个变量: Week 2 只加结构组件（PE/残差/LN），不重写 Attention

---

## Week 2
- GPU: 本机 / Jupyter web
- 目标: 对照标准 Transformer——位置编码、残差、LayerNorm、FFN；画清数据流
- 状态: ✅ 10/12 完成（块 B 阅读延后补）
- 备注: 9/24 后暂停，国庆假期休息一周（约 10/1–10/7），10/8 恢复；用掉了之前超前的一周缓冲
- 10/12 检查: 日历上应在 Week 3，实际落后约 1 周；当天完成 Week 2 剩余部分，追回大半

### 本周总 checklist（15h）

| 块 | 内容 | 约时 | 状态 |
|----|------|------|------|
| A | 四个组件人话：PE / 残差 / LayerNorm / pre vs post；跑通带它们的代码 | 2h | ✅ 10/8；pre-norm 顺序答反已纠正（先 LN 再 Attention） |
| B | 扫读 The Illustrated Transformer（DeepL 可开；术语进 glossary） | 2–3h | ⏸️ 10/12 按你的选择延后；文章核心（FFN）已在块 C 代码里补上，三问待读完再答 |
| C | 补全 GPT block（多头 + FFN + 两次残差 + 叠层训练） | 3–4h | ✅ 10/12 `code/w02_gpt_block.py`，2 层 loss 3.19→0.0054；三问纠正后通过 |
| D | 数据流图 + `notes/w02-forward.md`（每个组件「没有它会怎样」）+ 验收两问 | 3h | ✅ 10/12 验收两问在填空提示下通过 |

- 命令与配置: `python code/w02_block.py`；`python code/w02_gpt_block.py`（CPU，约 5 秒）
- 结果表:

| 检查项 | 结果 |
|--------|------|
| w02_block.py 形状 | after PE / after block 为 (B,T,C)，logits 为 (B,T,V) |
| w02_gpt_block.py（2 层，C=32，H=4，300 步） | 参数量 28,564；loss 3.1913 → 0.0054 |
| 「去掉它会怎样」5 条 | 4 条直接通过；FFN 一条重答后通过 |
| 验收两问 | 填空提示下通过 |

- 结论（3 句）:
  - A GPT block is two pre-norm sublayers with residuals: attention gathers info across tokens, FFN processes each token on its own.
  - Residuals are structure (2 per block), not tied to training steps; they let gradients reach early layers so deep stacks train.
  - Causal mask hides future positions; padding mask hides `[PAD]` filler positions; both set scores to -inf before softmax.
- 卡点: 残差三次与「训练」混淆（10/8、10/12 两次）——根源是把「层」和「训练步」混为一谈，已用对照表区分；FFN 曾写成 (B,H,T,D) 并一度说成「看其他 token」
- 下周只改一个变量: Week 3 不加新结构，只对现有模型算 FLOPs 和 KV 显存
- 待复查: 进 Week 3 前，不看笔记再答一次「残差为什么让深层网络可训练」
