"""Week 1 块 D：AttentionHead（Q/K/V + causal mask）+ 最小训练看 loss 下降。

流水线（先建立这张图，再看代码）：
  x: (B, T, C)
    → Q,K,V = x @ Wq / Wk / Wv     各 (B, T, C)   【线性投影】
    → scores = Q @ K^T              (B, T, T)     【你块 C 已做过】
    → scores 上三角置 -inf           (B, T, T)     【causal mask：不能看未来】
    → attn = softmax(scores, dim=-1)(B, T, T)     【每行变成权重，和为 1】
    → out = attn @ V                (B, T, C)     【用权重对 V 加权求和】

在 Jupyter 里整文件跑即可（CPU 即可，约 1–2 分钟）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ---------- 超参（先不用纠结数字从哪来）----------
B = 4          # batch
T = 8          # 上下文长度
C = 32         # 通道 / embedding 维
V = 20         # 词表大小（玩具数据）
steps = 200    # 训练步数


class AttentionHead(nn.Module):
    """单头自注意力。"""

    def __init__(self, C: int):
        super().__init__()
        # 三个独立线性层：把 x 投成 Q / K / V
        self.Wq = nn.Linear(C, C, bias=False)
        self.Wk = nn.Linear(C, C, bias=False)
        self.Wv = nn.Linear(C, C, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        Q = self.Wq(x)  # (B, T, C)
        K = self.Wk(x)  # (B, T, C)
        V = self.Wv(x)  # (B, T, C)

        # 块 C：scores (B, T, T)
        scores = Q @ K.transpose(-2, -1)  # (B, T, T)
        scores = scores / (C ** 0.5)      # 缩放，训练更稳；先当「除个常数」

        # causal mask：下三角可见，上三角（未来）屏蔽
        # tril 例子 T=4:
        # [[1,0,0,0],
        #  [1,1,0,0],
        #  [1,1,1,0],
        #  [1,1,1,1]]
        mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))  # 未来位置 → -inf

        attn = F.softmax(scores, dim=-1)  # (B, T, T)；-inf 经 softmax 变 0
        out = attn @ V  # (B, T, T) @ (B, T, C) → (B, T, C)
        return out


class TinyLM(nn.Module):
    """embedding → AttentionHead → 线性层出 logits。极简，只为看 loss。"""

    def __init__(self, vocab_size: int, C: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, C)  # id → 向量 (B,T)→(B,T,C)
        self.attn = AttentionHead(C)
        self.lm_head = nn.Linear(C, vocab_size)   # 向量 → 词表分数 (B,T,C)→(B,T,V)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) 整数 id
        x = self.embed(idx)       # (B, T, C)
        x = self.attn(x)          # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)
        return logits


def get_batch(data: torch.Tensor, B: int, T: int):
    """从一长串 token 里随机切 (x, y)，y 是 x 错开一位。"""
    ix = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i : i + T] for i in ix])
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
    return x, y  # 都是 (B, T)


# ---------- 玩具数据：重复模式，模型容易学 ----------
# 例如 0,1,2,...,19,0,1,2,... 学会「下一个是 +1」loss 就会降
raw = torch.arange(V).repeat(200)  # 很长一串

model = TinyLM(V, C)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

print("step | loss")
for step in range(1, steps + 1):
    xb, yb = get_batch(raw, B, T)
    logits = model(xb)  # (B, T, V)

    # 交叉熵：把 (B,T,V) 和 (B,T) 展平再算；不会公式也行，只看数字降不降
    loss = F.cross_entropy(logits.view(-1, V), yb.view(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step == 1 or step % 50 == 0:
        print(f"{step:4d} | {loss.item():.4f}")

print("done — 若 loss 从第一步到最后明显下降，块 D 训练部分过关")

# 自检（回复我）：
# 1) causal mask 为什么要在 softmax 之前把未来位置设成 -inf？
# 2) attn @ V 的 shape： (B,T,T) @ (B,T,C) → ?
# 3) 把你看到的 step1 loss 和最后一次打印的 loss 发我
