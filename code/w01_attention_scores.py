"""Week 1 块 C：三种方式算 attention scores，结果必须一致。

在 Jupyter 里可整文件跑，或按格复制。
公式（先当翻译题，不抠推导）：
  scores[b, i, j] = Q[b, i] · K[b, j]
  即每个 query 位置 i，和每个 key 位置 j 做点积 → 形状 (B, T, T)
"""

import torch

torch.manual_seed(0)

B, T, C = 2, 4, 8  # batch, 序列长, 通道
Q = torch.randn(B, T, C)
K = torch.randn(B, T, C)

# ---------- 方式 1：双重循环 ----------
scores_loop = torch.zeros(B, T, T)
for b in range(B):
    for i in range(T):  # query 位置
        for j in range(T):  # key 位置
            scores_loop[b, i, j] = torch.dot(Q[b, i], K[b, j])

# ---------- 方式 2：矩阵乘 ----------
# (B, T, C) @ (B, C, T) -> (B, T, T)
scores_matmul = Q @ K.transpose(-2, -1)

# ---------- 方式 3：einsum ----------
# 读法：对 C 维求和，留下 b,i,j
scores_einsum = torch.einsum("bic,bjc->bij", Q, K)

print("loop   ", scores_loop.shape)
print("matmul ", scores_matmul.shape)
print("einsum ", scores_einsum.shape)

assert torch.allclose(scores_loop, scores_matmul, atol=1e-5)
assert torch.allclose(scores_matmul, scores_einsum, atol=1e-5)
print("OK: three methods match")

# 自检（跑完在聊天里回答）：
# 1) scores 的 shape 为什么是 (B, T, T)？
# 2) K.transpose(-2, -1) 之后 shape 是什么？
