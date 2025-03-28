# Corporate-Bond-Default-Risk-Monitoring-A-Deep-Learning-Based-Time-Series-Forecasting-Approach

本代码库实现了一个基于深度学习的企业债券信用风险监测系统，采用了注意力机制和对抗训练的端到端预测框架(DR-AIM)。

## 文件说明

- **transformers.py**: 定义了Transformer模型的各个组件，包括位置编码、缩放点积注意力、多头注意力、前馈网络和Transformer层。

- **models.py**: 包含了多层感知机（MLP）模型和一些用于评估模型性能的指标函数，如均方根误差（RMSE）、平均绝对误差（MAE）和拟合优度（R²）。

- **generator.py**: 实现了生成器模型，使用Transformer和MLP模块来生成债券利差的预测。

- **discriminator.py**: 实现了判别器模型，用于区分真实和生成的债券利差数据。

- **load_data.py**: 提供了数据加载和预处理的功能，包括创建数据掩码和填充数据以适应模型输入的长度。

- **main_sweep.py**: 主程序文件，负责训练和评估生成器和判别器模型。使用WandB进行超参数搜索和结果记录。

## 环境配置

使用`environment.yaml`文件来配置运行环境。确保已安装Anaconda或Miniconda，然后运行以下命令来创建环境：

```bash
conda env create -f 
conda activate DR-AIM
```

## 运行代码
在配置好环境后，可以使用以下命令来运行主程序：
```bash
PYTHONWARNINGS="ignore" nohup python main_sweep.py > log/runing_result.log 2>&1 &
```

**本文数据来源**：

姜富伟, 柴百霖, 林奕皓. 深度学习与企业债券信用风险[DS/OL]. V2. Science Data Bank, 2024[2025-03-23]. https://cstr.cn/31253.11.sciencedb.j00214.00053. CSTR:31253.11.sciencedb.j00214.00053.
,.