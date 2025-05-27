import math
from regex import D
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PositionalEmbedding, Transformer, TransformerLayer
from models import MLP_0

class Generator(nn.Module):
    def __init__(self, feature_size, d_model, d_ff, n_heads, n_blocks, dropout):
        super(Generator, self).__init__()

        # Transformer 模块
        self.d_model = d_model
        self.positional_embedding = PositionalEmbedding(d_model, max_len=1000)
        self.block1 = nn.ModuleList(TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks))
        self.block2 = nn.ModuleList(TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(2 * n_blocks))
        self.Transformer1 = Transformer(d_model, n_heads, d_ff, n_blocks, dropout)


        self.layer_norm = nn.LayerNorm(feature_size)
        self.MLP_0 = MLP_0(feature_size, d_model, dropout)

        self.histExtractor = nn.Sequential(     # Historical Feature Extractor
            nn.Linear(1, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 2 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model))   
        
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model//2, 1))
        
        

    def forward(self, features, y_true, mask):
        """
        Parameters:
            features: 输入的特征数据 (batch_size, seq_len, feature_size)
            y: 输入的利差数据 (batch_size, seq_len, 1)
            mask: 输入的mask数据 (batch_size, seq_len)
        Returns:
            spread_predict: 预测的利差 (batch_size, seq_len)
        """
        features = features[:, :, 1:].clone() # features.drop(columns=['债券代码']).values
        # stock_features = features[:, :, 1:13]   # 证券因子
        # macro_features = features[:, :, 13:23]   # 宏观因子
        # firm_features = features[:, :, 23:]   # 企业因子
        
        featuresExtractor = self.MLP_0(features, mask)
        hist_y = self.histExtractor(y_true) * math.sqrt(self.d_model)
        x = self.Transformer1(featuresExtractor, featuresExtractor, hist_y, mask=mask, mask_now=True)
        out = self.linear_out(x)
        
        return out  # shape(batch, seq_len, 1)

