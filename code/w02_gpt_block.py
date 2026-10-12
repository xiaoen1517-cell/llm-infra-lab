"""Week 2 块 C：完整的 GPT block = 多头注意力 + FFN，各带一个残差；叠 2 层再训练。

一个 block（pre-norm）：
  x = x + attn(ln1(x))     # 注意力：token 之间交换信息
  x = x + ffn(ln2(x))      # FFN：每个 token 自己加工，不看别的 token

FFN = 两层线性 + 中间一个激活函数：
  (B, T, C) → Linear → (B, T, 4C) → GELU → Linear → (B, T, C)

在 Jupyter 里整文件跑即可（CPU，约 1 分钟）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

B, T, C = 4, 8, 32
H = 4            # 头数，D = C / H = 8
V = 20           # 词表大小
N_LAYERS = 2     # 叠几个 block
steps = 300


class MultiHeadAttention(nn.Module):
    """与 w01_multihead_attention.py 相同，去掉了 print。"""

    def __init__(self, C: int, H: int):
        super().__init__()
        self.H, self.D = H, C // H
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.H, self.D).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]                       # 各 (B, H, T, D)
        scores = (Q @ K.transpose(-2, -1)) / (self.D ** 0.5)  # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ V                    # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """新东西：对每个 token 单独做「放大 → 激活 → 缩回」。"""

    def __init__(self, C: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4 * C),   # (B, T, C)  → (B, T, 4C)  放大到 4 倍宽
            nn.GELU(),             # 激活函数：加入非线性，形状不变
            nn.Linear(4 * C, C),   # (B, T, 4C) → (B, T, C)   缩回原宽度
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """一个完整的 GPT block：两个子层，各自 pre-norm + 残差。"""

    def __init__(self, C: int, H: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = MultiHeadAttention(C, H)
        self.ln2 = nn.LayerNorm(C)
        self.ffn = FeedForward(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # 残差 1
        x = x + self.ffn(self.ln2(x))   # 残差 2
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, C: int, H: int, n_layers: int, max_T: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, C)
        self.pos_embed = nn.Embedding(max_T, C)
        # 叠 n_layers 个 block：上一层的输出就是下一层的 x
        self.blocks = nn.Sequential(*[Block(C, H) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        x = self.blocks(x)               # (B, T, C)，每层进出形状都一样
        return self.lm_head(self.ln_f(x))  # (B, T, V)


def get_batch(data: torch.Tensor, B: int, T: int):
    ix = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i : i + T] for i in ix])
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
    return x, y


# ---------- 1) 先看形状 ----------
model = TinyGPT(V, C, H, N_LAYERS)
xb = torch.randint(0, V, (B, T))
ffn = FeedForward(C)
print("FFN 中间层  ", ffn.net[0](torch.randn(B, T, C)).shape)  # (B, T, 4C)
print("FFN 输出    ", ffn(torch.randn(B, T, C)).shape)         # (B, T, C)
print("logits      ", model(xb).shape)                         # (B, T, V)
print("参数量      ", sum(p.numel() for p in model.parameters()))

# ---------- 2) 再训练，看 loss 降 ----------
raw = torch.arange(V).repeat(200)  # 0,1,...,19,0,1,... 学会「下一个 = 当前 + 1」
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

print("\nstep | loss")
for step in range(1, steps + 1):
    xb, yb = get_batch(raw, B, T)
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, V), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step == 1 or step % 50 == 0:
        print(f"{step:4d} | {loss.item():.4f}")

# 自检（各用一句话回我）：
# 1) FFN 的输入和输出形状分别是什么？它会让不同 token 之间交换信息吗？
# 2) 一个 Block 里一共有几次残差相加？
# 3) 叠 2 层时，第 2 层 Block 的 x 是从哪来的？
