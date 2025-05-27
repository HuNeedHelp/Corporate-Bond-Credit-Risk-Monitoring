import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()    # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)


    def forward(self, x):
        """
        Parameters: x (Tensor)
        Returns: Positional Encoding (Tensor, should plus with input)
        """
        seq_len = x.size(1)
        return self.encoding[:, :seq_len, :].to(x.device)



class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()


    def update_mask(self, mask, seq_len, mask_now):
        """
        既考虑原始数据的mask, 也考虑上三角的mask
        Parameters: -mask, -seq_len (int) 
        Returns: -mask (Tensor)
        """
        if mask_now:
            triu_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=-1).to(mask.device)   # 对角线为-1, 看不到现在的数据
        else:
            triu_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).to(mask.device)  # 创建上三角矩阵，未来部分为0, 把0的位置mask
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(0)  # 增加 batch 和 head 维度. (1, 1, seq_len, seq_len)

        # resize the original mask: (batch, seq_len, features) --> (1, 1, seq_len, seq_len)
        resize_mask = mask[:, :, 0].unsqueeze(-1).unsqueeze(0).transpose(1, 0).repeat(1, 1, 1, seq_len) # (batch, 1, seq_len, seq_len)
        mask = triu_mask * resize_mask # 按元素相乘
        return mask  


    def forward(self, query, key, value, mask=None, mask_now=False):
        """
        Input: query, key, value (Tensor, shape: (batch_size, n_heads, seq_len, d_k))
        Returns: output, attention_scores (Tensor, shape: (batch_size, n_heads, seq_len, d_k))
        """
        d_k = query.size(-1)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)   # shape: (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            # 注意mask这里的mask是从最开始的dataloader传进来的
            mask = self.update_mask(mask, mask.size(1), mask_now).to(attn_scores.device)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_scores = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_scores, value)
        return output, attn_scores





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attention = ScaledDotProductAttention()


    def forward(self, query, key, value, mask, mask_now):
        """
        Input: query (Tensor), key (Tensor), value (Tensor), mask (Tensor)
        Returns: output (Tensor), attn (Tensor)
        """
        batch_size = query.size(0)

        # Linear projections
        query = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) ## (batch_size, seq_len, n_heads * d_k) --> (batch_size, seq_len, n_heads, d_k)然后将第1维和第2维进行转置
        key = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Apply attention
        output, attn = self.attention(query, key, value, mask, mask_now)
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        # Final linear layer
        output = self.linear_out(output)
        return output, attn






class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))




class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attnBlock = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, query, key, value, mask, mask_now):
        # Attention
        attn_output, _ = self.attnBlock(query, key, value, mask, mask_now)
        query = query + self.dropout1(attn_output)
        query = self.norm1(query)

        # Feed-forward
        ff_output = self.feed_forward(query)
        query = query + self.dropout2(ff_output)    # Residual connection
        query = self.norm2(query)
        return query




class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, dropout):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEmbedding(d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(d_model, d_model)


    def forward(self, query, key, value, mask=None, mask_now=False):
        # Input embedding + positional encoding
        query = query + self.positional_encoding(query)
        key = key + self.positional_encoding(key)
        value = value + self.positional_encoding(value)

        for layer in self.layers:
            x = layer(query, key, value, mask=mask, mask_now=mask_now)
        output = self.fc_out(x)
        return output
    