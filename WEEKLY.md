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

**进度协作约定：** 你描述完成了什么 → 我更新本文件 + 对应交付物路径。当前周：`Week 2`。

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
| E | `notes/w01-shapes.md` + 多头流向（代码理解拆头/view/concat） | 3h | 🔄 笔记已写；合书默写 10/8 未过（scores 写成 (B,H,T,D)），10/9 重默 |

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
- 目标: 对照标准 Transformer——位置编码、残差、LayerNorm；画清数据流
- 状态: 进行中 — 块 A：四个新组件（人话 + 代码）
- 备注: 9/24 后暂停，国庆假期休息一周（约 10/1–10/7），10/8 恢复；用掉了之前超前的一周缓冲

### 本周总 checklist（15h）

| 块 | 内容 | 约时 | 状态 |
|----|------|------|------|
| A | 四个组件人话：PE / 残差 / LayerNorm / pre vs post；跑通带它们的代码 | 2h | ✅ 10/8；pre-norm 顺序答反已纠正（先 LN 再 Attention） |
| B | 扫读 The Illustrated Transformer（DeepL 可开；术语进 glossary） | 2–3h | ⬜ 下一块 |
| C | 对照差异清单 + 画数据流图进仓库 | 3–4h | ⬜ |
| D | 写作 `notes/w02-forward.md` + 验收两问 | 3h | ⬜ |

### 块 A 要建立的四个直觉（先读再跑代码）

1. **位置编码 PE**：Attention 本身不知道谁在前谁在后；给每个位置加一个「座位号」向量。
2. **残差**：`x + f(x)`，给信息留一条不经变换的直达通道，深层更好训。
3. **LayerNorm**：把每个 token 的特征向量「拉回」合适尺度，训练更稳（公式先不抠）。
4. **pre-norm vs post-norm**：Norm 在子层前还是后；现代 GPT 多用 **pre-norm**。

文件：`code/w02_block.py`（在 W1 TinyLM 上加上 PE + 残差 + pre-LN）

做完块 A：跑通代码，用自己的话各用一句解释上面 4 个词，发我。

- 命令与配置:
- 结果表:
- 结论（3 句）:
- 卡点:
- 下周只改一个变量:
