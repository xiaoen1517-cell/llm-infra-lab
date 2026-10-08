"""Week 1 补充：Multi-Head Attention（带逐步 shape 注释）

对照单头：单头用满整个 C；多头把 C 拆成 H 份，每份 D=C/H，各算各的，再拼回。

跑完看打印的 shape；能对上 notes/w01-shapes.md 第 3 节即可。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

B, T, C = 2, 4, 8   # batch, seq, channels
H = 2               # 头数
D = C // H          # 每头维度；必须整除：8/2=4
assert C % H == 0

x = torch.randn(B, T, C)
print("0 输入 x          ", x.shape)  # (B, T, C)


class MultiHeadAttention(nn.Module):
    def __init__(self, C: int, H: int):
        super().__init__()
        assert C % H == 0
        self.H = H
        self.D = C // H
        # 一次投影出 Q,K,V 再拆头；也有人写成 3 个 Linear，一样
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)  # concat 后再混一次

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.H, self.D

        # 1) 投影出 Q,K,V 拼在一起，再切开
        qkv = self.qkv(x)  # (B, T, 3C)
        print("1 qkv             ", qkv.shape)

        qkv = qkv.reshape(B, T, 3, H, D)  # (B, T, 3, H, D)
        print("2 reshape 3,H,D   ", qkv.shape)

        # 把「头」维挪到前面，方便每个头做 (T,D)@(D,T)
        # (B, T, 3, H, D) → (3, B, H, T, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        print("3 permute         ", qkv.shape)

        Q, K, V = qkv[0], qkv[1], qkv[2]  # 各 (B, H, T, D)
        print("4 Q               ", Q.shape)

        # 5) 每个头各自算 scores（最后两维矩阵乘；B,H 当 batch）
        # (B,H,T,D) @ (B,H,D,T) → (B,H,T,T)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (D ** 0.5)
        print("5 scores          ", scores.shape)

        # 6) causal mask（和单头一样，挡未来）
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        print("6 after mask      ", scores.shape)

        # 7) softmax → 权重
        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
        print("7 attn            ", attn.shape)

        # 8) 加权 V
        out = attn @ V  # (B,H,T,T)@(B,H,T,D) → (B,H,T,D)
        print("8 out per head    ", out.shape)

        # 9) concat：把 H 个头拼回 C=H*D
        # (B, H, T, D) → (B, T, H, D) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        print("9 concat          ", out.shape)

        # 10) 输出投影
        out = self.proj(out)  # (B, T, C)
        print("10 proj           ", out.shape)
        return out


mha = MultiHeadAttention(C, H)
y = mha(x)
print("最终输出            ", y.shape)

# 自检（对照打印）：
# 默写链： (B,T,C) → 拆头 (B,H,T,D) → scores (B,H,T,T)
#          → mask → softmax → @V (B,H,T,D) → concat (B,T,C)
