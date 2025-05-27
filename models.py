from turtle import forward
from networkx import dominance_frontiers
from regex import D
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PositionalEmbedding, Transformer

class MLP_0(nn.Module):
    def __init__(self, feature_size, d_model, dropout):
        """
        参数：
            - feature_size (int): 输入数据的特征维度。
            - d_model (int): 模型中间层的特征维度，第一层输出的维度。
            - dropout (float): Dropout 层的丢弃率，用于防止过拟合。
            - mask (tensor): 用于处理填充（padding）的掩码（mask），用于避免在有填充的地方进行计算。
        """
        super().__init__()
        self.d_model = d_model
        self.Linear1 = nn.Linear(feature_size, d_model)
        self.Linear2 = nn.Linear(d_model, 2*d_model)
        self.Linear3 = nn.Linear(2*d_model, d_model)
        
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(2*d_model)
        self.LayerNorm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, data, mask):
        """
        参数：
            - data
        返回:
            - data (tensor): 经过 MLP 处理后的数据.
        """
        # mask的shape: (batch_size, seq_len, features)， 需要调整最后一个维度
        data = self._apply_layer(data, self.Linear1, self.LayerNorm1, mask, target_size=self.d_model)
        data = self._apply_layer(data, self.Linear2, self.LayerNorm2, mask, target_size=2*self.d_model)
        data = self._apply_layer(data, self.Linear3, self.LayerNorm3, mask, target_size=self.d_model)
        return data
    
    def _apply_layer(self, data, linear_layer, layer_norm, mask, target_size):
        data = linear_layer(data)
        # 调整mask的维度
        mask = self.mask_resize(mask, target_size)
        data = data * mask  # mask掉padding的部分
        data = layer_norm(data)
        data = nn.Sigmoid()(data)
        data = self.dropout(data)
        return data
    
    def mask_resize(self, mask, target_size):
        """
        调整mask的维度
        """
        mask = mask[:, :, 0].unsqueeze(-1).repeat(1, 1, target_size) # shape: (batch_size, seq_len, target_size)
        return mask





def rmse(y_true, y_pred, mask):
    """
    Parameters:
        -y_true: tensor (batch_size, seq_len, 1)
        -y_pred: tensor (batch_size, seq_len, 1)
        -mask: tensor (batch_size, seq_len, features)
    Returns:
        -rmse: torch.float
    """
    mask = mask[:, :, 0].unsqueeze(-1).view(-1) # (batch_size*seq_len)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_true = y_true[mask.bool()]
    y_pred = y_pred[mask.bool()]
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred, mask):
    """
    Parameters:
        -y_true: tensor (batch_size, seq_len, 1)
        -y_pred: tensor (batch_size, seq_len, 1)
        -mask: tensor (batch_size, seq_len, features)
    Returns:
        -mae: torch.float
    """
    mask = mask[:, :, 0].unsqueeze(-1).view(-1) # (batch_size*seq_len)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_true = y_true[mask.bool()]
    y_pred = y_pred[mask.bool()]
    return torch.mean(torch.abs(y_true - y_pred))

def r_squared(y_true, y_pred, mask):
    """
    Parameters:
        -y_true: tensor (batch_size, seq_len, 1)
        -y_pred: tensor (batch_size, seq_len, 1)
        -mask: tensor (batch_size, seq_len, features)
    Returns:
        -r_squared: torch.float
    """
    mask = mask[:, :, 0].unsqueeze(-1).view(-1) # (batch_size*seq_len)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_true = y_true[mask.bool()]
    y_pred = y_pred[mask.bool()]
    return 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)

def rmse_(y_true, y_pred, mask):
    """
    Parameters:
        -y_true: tensor (batch_size, seq_len, 1)
        -y_pred: tensor (batch_size, seq_len, 1)
        -mask: tensor (batch_size, seq_len, features)
    Returns:
        -rmse: torch.float
    """
    """mask = mask[:, :, 0].unsqueeze(-1).view(-1) # (batch_size*seq_len)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)"""
    if mask is not None:
        y_true = y_true[mask.bool()]
        y_pred = y_pred[mask.bool()]
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def mae_(y_true, y_pred, mask):
    """
    Parameters:
        -y_true: tensor (batch_size, seq_len, 1)
        -y_pred: tensor (batch_size, seq_len, 1)
        -mask: tensor (batch_size, seq_len, features)
    Returns:
        -mae: torch.float
    """
    """mask = mask[:, :, 0].unsqueeze(-1).view(-1) # (batch_size*seq_len)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)"""
    if mask is not None:
        y_true = y_true[mask.bool()]
        y_pred = y_pred[mask.bool()]
    return torch.mean(torch.abs(y_true - y_pred))

def r_squared_(y_true, y_pred, mask):
    """
    Parameters:
        -y_true: tensor (batch_size, seq_len, 1)
        -y_pred: tensor (batch_size, seq_len, 1)
        -mask: tensor (batch_size, seq_len, features)
    Returns:
        -r_squared: torch.float
    """
    """mask = mask[:, :, 0].unsqueeze(-1).view(-1) # (batch_size*seq_len)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)"""
    if mask is not None:
        y_true = y_true[mask.bool()]
        y_pred = y_pred[mask.bool()]
    return 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)
