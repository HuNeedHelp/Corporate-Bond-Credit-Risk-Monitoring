import torch 
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Discriminator, self).__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.dropout = dropout
        self.NeuralNetwork = nn.Sequential(
            nn.Linear(1, d_model), nn.Softmax(),
            nn.Linear(d_model, 2*d_model), nn.Softmax(),
            nn.Linear(2*d_model, 2), nn.Softmax(),
        )


    def forward(self, spread, mask):
        """
        Parameters:
            -真实或预测的spread: 输入数据 (batch_size, seq_len, 1)
            -mask: (batch_size, seq_len, features)
        Returns:
            prediction (batch_size*seq_len, 2)
            ignore_index
        """
        pred = self.NeuralNetwork(spread).view(-1, 2)     # (batch_size*seq_len, 2)
        if mask is not None:
            ignore_index = mask[:, :, 0].view(-1)
            pred = pred[ignore_index.bool()]   # 使用布尔索引
        else:
            ignore_index = None
        return pred


