import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pe = torch.zeros(max_len, embed_dim) # l*d 附加 position embs
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (l, 1) idx
        # position embs 中对应的都是偶数的 j (相邻两个使用的 j 是相同的)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term) # 0::2 偶数列
        pe[:, 1::2] = torch.cos(position * div_term) # 1::2 奇数列
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # (N, S, D) + (1, max_len, embed_dim)
        # 只需要在 len 上进行裁剪, emb dim 是固定的
        x = x + self.pe[:, :S, :]
        output = self.dropout(x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).

        # key query value 都是 multihead 的矩阵的拼接
        # (d, d/h) *h 的矩阵 (具体是在哪个维度进行划分的? 可以用于同时计算吗?)

        # 不能交换参数初始化顺序, 会影响随机数初始化结果, 导致结果不一致
        # Layer 中使用了 bias, 但并不是严格必要的 (没有看懂什么意思)

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout) # dropout 的比例
        # droput 相当于训练多个相同的子网络然后再混合 
        # 训练的过程中需要保证权值的比例, 所以需要进行放缩

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """

        """
        在下面的形状定义中, N 是批次大小, S 是源序列长度, T 是目标序列长度, E 是嵌入维度

        输入:
        - query: 用作查询的输入数据, 形状为 (N, S, E)
        - key: 用作键的输入数据, 形状为 (N, T, E)
        - value: 用作值的输入数据, 形状为 (N, T, E)
        - attn_mask: 形状为 (S, T) 的数组, 其中 mask[i,j] == 0 表示源序列中的第 i 个 token 不应影响目标序列中的第 j 个 token

        返回:
        - output: 形状为 (N, S, E) 的张量
        - 表示根据使用 key 和 query 计算出的注意力权重对 value 中的数据进行加权组合的结果
        """

        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 将输入从 (N, S, E) 拆分为 (N, S, H, E/H)
        # torch.matmul 对于 matrix batch 进行矩阵乘法
        # (N, H, T, E/H) * (N, H, E/H, T) -> (N, H, T, T)
        # mask 要在哪一步起作用, masked_fill 的作用?

        # (N, S, E) * (E, E), (N, S, E) * (E, H, E/H)
        # 先进行矩阵乘法再划分, 和先划分再进行乘法相同 (分块矩阵)

        H = self.head_dim
        # 得到 (N, S, E), 由于分块矩阵是 (E, H, E/H), 顺序为 (N, S, H, E/H)
        # 需要进行转置将 H 提到前面来
        Q = self.query(query).view(N, S, self.n_head, self.head_dim).transpose(1, 2)
        K = self.key(key).view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        V = self.value(value).view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        # query 和 key/value 的序列长度可能不同 (self attention/cross attention)

        # (N, H, S, E/H) * (N, H, T, E/H) -> (N, H, S, T)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # H 对应的是 head_dim, 而不是 n_head
        # 先前写成 E // H, 应该是 head_dim = E // n_head
        # DeepSeek 的深度思考竟然找出来了这个错误!
        scores = scores / (self.head_dim ** 0.5) # 点乘的 d = E//H, 除以 sqrt(d) 归一化

        if attn_mask is not None:
            # masked_fill 根据 mask 将对应的位置设置为指定值
            # 在 softmax 之前就进行设置, 防止在 softmax 中产生影响
            # QA 这个尺寸是如何进行对齐的?
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1) # (N, H, S, T) 的最后一维
        attn_weights = self.attn_drop(attn_weights) # 要求使用 dropout
        # (N, H, S, T) * (N, H, T, E/H) -> (N, H, S, E/H)
        # 权重矩阵和 value 之间是矩阵乘法
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(N, S, E)
        output = self.proj(output)
        # contiguous 用于将 tensor 转换为连续存储的形式, 加快访问速度, 也保证能够 reshape

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


