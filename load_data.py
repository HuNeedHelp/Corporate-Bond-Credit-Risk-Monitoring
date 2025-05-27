from numpy import float32
import pandas as pd
import os
import numpy as np  
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_mask(data, mask_size):
    """
    Params: data (DataFrame), mask_size (int)
    Returns: mask (Tensor) 告诉哪里是padding的. 1表示有数据, 0表示padding
    """
    mask = torch.ones((data.shape[-2], data.shape[-1]))
    mask[-mask_size:] = 0   # mask_size是需要填充的行数
    return mask


def truncated_fill(data, padding_size):
    """
    Params: data (DataFrame), padding_size (int)
    Returns: DataFrame, mask_size (int)
    """
    if len(data) < padding_size:
        mask_size = padding_size - len(data)
        padding = pd.DataFrame(np.zeros(shape=(mask_size, data.shape[-1])), columns=data.columns)
        data = pd.concat([data, padding], axis=0)
        mask = create_mask(data, mask_size)
        return data, mask
    else:
        data = data[:padding_size]
        return data, create_mask(data, mask_size=0)


class BondDataset(Dataset):
    """
    Parameters: data (DataFrame), seq_len_max (int)
    Returns: Dataset -> (features, y, mask_size)
    """
    def __init__(self, data, seq_len_max=None):
        self.data = data
        self.seq_len_max = seq_len_max
        self.bond_codes = data['债券代码'].unique()

        # 如果没有指定最大长度，则计算所有债券序列的最大时间跨度
        if seq_len_max is None:
            self.seq_len_max = max(data.groupby('债券代码').size())

    def __len__(self):
        return len(self.bond_codes)

    def __getitem__(self, idx):
        bond_code = self.bond_codes[idx]
        bond_data = self.data[self.data['债券代码']==bond_code]
        bond_data, mask = truncated_fill(bond_data, self.seq_len_max)
        features = bond_data.drop(columns=['交易日期', 'stkcd']).values # drop'债券代码'这一步移动到generator中
        y = bond_data['spread'].values
        # 转化为tensor
        features = torch.tensor(features, dtype=torch.float32)  # shape: (seq_len, features)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) # shape: (seq_len, 1)
        # print(features.shape)
        return features, y, mask


def dataloader(data_path, batch_size, padding_size=None, shuffle=True):
    """
    Parameters: data_path (str), batch_size (int), padding_size (int)
    Returns: dataLoader
    """
    data = pd.read_csv(data_path)#.iloc[:100]
    dataloader = DataLoader(BondDataset(data, seq_len_max=padding_size), batch_size, shuffle)
    return dataloader


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__)) # Change working directory to current file
    data_path = r"data\train_data.csv"
    train_dataloader = dataloader(data_path, batch_size=32, shuffle=False)
    for batch in train_dataloader:
        print(batch)
        break
    