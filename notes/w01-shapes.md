# Week 1 — Forward 形状变化表

来源：`code/w01_attention_head.py`（单头）+ 多头流向（验收用）。

## 符号

| 符号 | 含义 |
|------|------|
| B | batch，一次几条样本 |
| T | 序列长度（上下文长度） |
| C | 通道 / embedding 维 |
| V | 词表大小 |
| H | 头数（multi-head） |
| D | 每个头的维度，通常 `D = C / H` |

本周玩具设定示例：`B=4, T=8, C=32, V=20`。

---

## 1. TinyLM 端到端

| 步骤 | 运算 | 形状 |
|------|------|------|
| 输入 id | `idx` | `(B, T)` |
| Embedding | `embed(idx)` | `(B, T, C)` |
| AttentionHead | 见下一节 | `(B, T, C)` |
| lm_head | `Linear(C → V)` | `(B, T, V)` logits |
| 标签 | `y`（与 x 同长、错开一位） | `(B, T)` |
| loss | `cross_entropy` | 标量 |

---

## 2. AttentionHead.forward（单头）

| 步骤 | 运算 | 形状 |
|------|------|------|
| 输入 | `x` | `(B, T, C)` |
| Q | `Wq(x)` | `(B, T, C)` |
| K | `Wk(x)` | `(B, T, C)` |
| V | `Wv(x)` | `(B, T, C)` |
| scores | `Q @ Kᵀ`，再 `/ √C` | `(B, T, T)` |
| causal mask | 上三角（未来）→ `-inf` | 仍 `(B, T, T)` |
| attn 权重 | `softmax(scores, dim=-1)` | `(B, T, T)` |
| 输出 | `attn @ V` | `(B, T, C)` |

要点：

- scores 的 `(T, T)`：每个 query 位置对每个 key 位置一个分数。
- mask 必须在 **softmax 之前** 设 `-inf`，之后权重才≈0。
- `attn @ V`：用权重对 V 加权求和。

---

## 3. Multi-head 张量流向（验收默写）

单头把全部 `C` 维拿去算注意力；多头是把 `C` 拆成 `H` 份，每份维度 `D=C/H`，各算各的，再拼回来。

```text
x:           (B, T, C)
  → 投影 QKV: (B, T, C)  各一份
  → 拆成多头: (B, H, T, D)   其中 C = H * D
  → scores:   (B, H, T, T)   每个头各自 Q@Kᵀ
  → mask:     每个头同样挡未来
  → softmax:  (B, H, T, T)
  → @ V:      (B, H, T, D)
  → concat:   (B, T, H*D) = (B, T, C)
  → 可选再经一层 Linear(C→C)
```

一句话：**B,T,C → 拆 heads → 每头 scores → mask → softmax → 加权 V → concat 回 C。**

---

## 4. 本周做过的代码

- `code/w01_attention_scores.py` — 三种 scores 写法一致
- `code/w01_attention_head.py` — QKV + causal mask + loss 下降
- `code/w01_multihead_attention.py` — 多头拆分 / scores / concat 逐步打印 shape
