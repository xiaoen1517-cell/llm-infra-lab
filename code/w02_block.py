"""Week 2 块 A：在 Attention 外包一层「标准积木」

W1 只有：embed → attn → lm_head
这里补上：位置编码 + pre-LayerNorm + 残差

块结构（GPT 常用 pre-norm）：
  x
   → ln1 = LayerNorm(x)
   → attn_out = Attention(ln1)
   → x = x + attn_out          # 残差
   （本文件先不加 FFN，下一块再加也行）

位置：embed(token) + embed(position) 再送进块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

B, T, C = 2, 8, 32
V = 20  # 词表
assert T <= 64


class AttentionHead(nn.Module):
    """与 W1 相同的单头，略去 print。"""

    def __init__(self, C: int):
        super().__init__()
        self.Wq = nn.Linear(C, C, bias=False)
        self.Wk = nn.Linear(C, C, bias=False)
        self.Wv = nn.Linear(C, C, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = (Q @ K.transpose(-2, -1)) / (C ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return attn @ V


class TransformerBlock(nn.Module):
    """pre-norm + Attention + 残差。"""

    def __init__(self, C: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = AttentionHead(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # pre-norm：先规范化，再进 Attention
        attn_out = self.attn(self.ln1(x))  # (B, T, C)
        x = x + attn_out  # 残差：原路信息 + 注意力结果
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, C: int, max_T: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, C)   # token id → 向量
        self.pos_embed = nn.Embedding(max_T, C)          # 位置 0..T-1 → 向量
        self.block = TransformerBlock(C)
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        B, T = idx.shape
        tok = self.token_embed(idx)  # (B, T, C)
        pos = self.pos_embed(torch.arange(T, device=idx.device))  # (T, C)
        # 广播： (B,T,C) + (T,C) → (B,T,C)   ← W0 学过的广播
        x = tok + pos
        print("after PE        ", x.shape)

        x = self.block(x)
        print("after block     ", x.shape)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        print("logits          ", logits.shape)
        return logits


model = TinyGPT(V, C)
idx = torch.randint(0, V, (B, T))
logits = model(idx)

# 自检（各用一句话回我）：
# 1) 为什么要加 pos_embed？只有 token_embed 会缺什么？
# 2) x = x + attn_out 里，加号两边的 shape 为什么能加？
# 3) pre-norm 是先 LN 再 Attention，还是先 Attention 再 LN？
