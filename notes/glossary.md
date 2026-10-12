# Glossary 术语本

格式：`术语 | 一句话人话解释 | 在哪遇到的`
只记卡住了你的词，不抄词典。

| 术语 | 一句话解释 | 出处 |
|------|-----------|------|
| 点积 | 两个向量「有多同向」：越大越像，垂直≈0，反向为负 | W0 3B1B / numpy 练习 |
| 矩阵乘法 | `(m,k)@(k,n)→(m,n)`：结果格 = 左行 · 右列，不是逐元素乘 | W0 形状练习 |
| 广播 | 把长度为 1 的维复制对齐后，再做逐元素加减乘 | W0 `(8,1)+(1,4)` 例子 |
| shape | 张量每个维度有多长，如 `(B,T,C)` | W0 / Week 1 前置 |
| batch (B) | 一次打包一起算的样本数；锅里一次下几条数据 | W0 问答 |
| 向量 | 一串有序数字，可看成箭头/方向 | W0 3B1B |
| 相似度直觉 | 点积可当作「有多相关」的粗指标（Attention 里 Q·K 同理） | W0 点积讨论 |
| token | 模型读写的最小单位（Karpathy 这期里常是字符） | W1 视频 |
| embedding | 用 token id 查表，取出对应向量；不是位置编码 | W1 问答 |
| logits | 还没变成概率的分数；最后一维通常 = 词表大小 | W1 视频 |
| 训练标签 y | 与 x 同长，是 x 向右错开一格；每个位置要猜的「下一个 token」 | W1 问答 |
| attention scores | 每个 query 位置对每个 key 位置做点积，得到 (B,T,T) 相关分数表 | W1 块 C |
| causal mask | 挡住「未来」位置：softmax 前设 -inf，softmax 后权重≈0，不能偷看后面的 token | W1 块 D |
| softmax | 把一行分数变成和为 1 的权重；输入 -inf 的位置权重≈0 | W1 块 D |
| Q / K / V | 同一个 x 分别乘三张不同的权重矩阵得到；Q=要找什么，K=被怎么找到，V=提供的内容 | W1 复习 |
| 权重（参数） | 模型里的数字表，开始是随机数，训练按 loss 一点点调整到能用；Wq/Wk/Wv、embedding 表都是 | W1 复习 |
| 多头注意力 | 把 C 拆成 H 个 D 维（C=H×D），各算注意力再 concat 回 C | W1 多头代码 |
| 位置编码 PE | 给每个位置一个向量，加到 token 向量上，让模型知道先后顺序 | W2 块 A |
| 残差连接 | x + f(x)：保留原信息直达通道，深层更好训 | W2 块 A |
| LayerNorm | 对每个 token 的特征向量做归一化，稳定训练（先记作用） | W2 块 A |
| pre-norm | 子层（如 Attention）之前先 LayerNorm；现代 GPT 常用。代码 `attn(ln1(x))` 从里往外读 | W2 块 A |
| post-norm | 先 Attention 再 LayerNorm；2017 原版 Transformer 用，深层难训 | W2 块 A |
| FFN（Feed Forward） | 每个 token 单独过两层线性：C→4C→激活→C；不在 token 之间交换信息 | W2 块 C |
| GELU | 激活函数，给模型加入非线性；形状不变 | W2 块 C |
| GPT block | x = x + attn(ln1(x)); x = x + ffn(ln2(x))；可叠很多层 | W2 块 C |
| 位置编码两种做法 | sin/cos 固定公式（原版论文）vs 可学习查表（GPT-2，本仓库代码用这种） | W2 块 A |
