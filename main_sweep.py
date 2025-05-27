import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from matplotlib.lines import lineStyles
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import load_data
from generator import Generator
from discriminator import Discriminator
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from models import rmse_, mae_, r_squared_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
os.environ['PYTHONWARNINGS'] = 'ignore'

# 训练和验证函数
def train_and_evaluate():
    # 初始化
    wandb.init(project='Spread-predict')
    config = wandb.config
    default_params = {
    'feature_size': config.feature_size,
    'dropout': config.dropout,
    'd_model': config.d_model,
    'n_blocks': config.n_blocks,
    'd_ff': config.d_ff,
    'n_heads': config.n_heads
    }

    # 加载数据
    data_path_dict = {
        'train': 'data/train_data.csv',
        'valid': 'data/valid_data.csv',
        'test': 'data/test_data.csv'
    }
    train_dataloader = load_data.dataloader(data_path_dict['train'], batch_size=config.batch_size, shuffle=True, padding_size=96) 
    valid_dataloader = load_data.dataloader(data_path_dict['valid'], batch_size=config.batch_size, shuffle=False, padding_size=33)
    test_dataloader = load_data.dataloader(data_path_dict['test'], batch_size=config.batch_size, shuffle=False, padding_size=33)
    warm_up_dataloader_from_train = load_data.dataloader(data_path_dict['train'], batch_size=config.batch_size, shuffle=False, padding_size=96)  # 用于预热, 不计入评价指标, shuffle=False
    warm_up_dataloader_from_valid = load_data.dataloader(data_path_dict['valid'], batch_size=config.batch_size, shuffle=False, padding_size=33)  # 用于预热, 不计入评价指标, shuffle=False
    print(f"Train dataloader batch size (total number of bonds) is {len(train_dataloader)}")


    # 初始化
    generator = Generator(**default_params).to(device)
    discriminator = Discriminator(default_params['d_model'], default_params['d_ff'], default_params['dropout']).to(device)
    best_rmse = float('inf')    # 用于保存最好的模型
    
    # 训练阶段
    generator_loss_train, discriminator_loss_train = [], []     # 用于计算整个训练过程中，每个epoch的生成器，判别器的平均损失, 列表长度为epoch
    rmse_train_record, mae_train_record, r_squared_train_record = [], [], [] # 用于记录整个训练过程中，每个epoch的RMSE
    # 验证阶段
    generator_loss_valid, discriminator_loss_valid = [], []     # 用于计算整个验证过程中，每个epoch的生成器，判别器的平均损失, 列表长度为epoch
    rmse_valid_record, mae_valid_record, r_squared_valid_record = [], [], [] # 用于记录整个验证过程中，每个epoch的RMSE
    
    # 定义损失函数
    ce_loss = nn.CrossEntropyLoss()

    # 根据 Sweep 选择优化器
    if config.optimizer == "adam":
        g_optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        g_optimizer = optim.SGD(generator.parameters(), lr=config.learning_rate)
        d_optimizer = optim.SGD(discriminator.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")





    # 开始训练
    # 训练循环
    for epoch in range(config.epochs):
        generator.train()
        discriminator.train()

        g_total_loss, d_total_loss = 0.0, 0.0   # 用于计算每个epoch的生成器和判别器的总损失
        all_true, all_pred, all_mask = [], [], []
        total_batches = 0  # 用于计算训练时每个epoch的平均RMSE

        # 遍历训练数据
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            features, y, mask = batch # shape: (batch, seq_len, features)
            features, y, mask = features.to(device), y.to(device), mask.to(device)    

            # Step 1: 训练判别器
            d_optimizer.zero_grad()
            # 判别器预测
            real_preds = discriminator(y, mask=mask)   # shape: (batch_size, seq_len, 2), logits
            # 判别器的损失 (真实数据 -> 1, 假数据 -> 0)
            real_labels = torch.ones(real_preds.size(0), dtype=torch.long).to(device)    # shape: (batch_size), logits
            d_loss_real = ce_loss(real_preds, real_labels)

            with torch.no_grad():
                y_hat = generator(features, y, mask)    # y用于实现自回归预测
            fake_preds = discriminator(y_hat, mask=mask)   # shape: (batch_size*seq_len, 2), logits
            fake_labels = torch.zeros(fake_preds.size(0), dtype=torch.long).to(device)   # shape: (batch_size), logits
            d_loss_fake = ce_loss(fake_preds, fake_labels)
            
            # 更新判别器参数
            d_loss = d_loss_real + d_loss_fake
            # if epoch < 2:   # 只更新两轮判别器, 判别器会不会记住答案?
            d_loss.backward()
            d_optimizer.step()

            # Step 2: 训练生成器
            g_optimizer.zero_grad()
            # 重新生成假数据
            y_hat = generator(features, y, mask)   # shape: (batch_size, seq_len, 1)
            # 生成器希望判别器将假数据判为真实 (1)
            fake_preds = discriminator(y_hat, mask=mask)
            g_loss = ce_loss(fake_preds, real_labels)  # 假数据目标是被判为真实
            # 计算RMSE
            rmse_loss = rmse_(y.view(-1), y_hat.view(-1), mask[:, :, 0].view(-1))
            # 更新生成器参数
            (g_loss + config.lambda_ * rmse_loss).backward()  # 生成器的损失是判别器的损失和RMSE的和
            # rmse_loss.backward()
            g_optimizer.step()

            # Step 3: 记录
            d_total_loss += d_loss.cpu().item()
            g_total_loss += g_loss.cpu().item()
            all_true.append(y.view(-1))
            all_pred.append(y_hat.view(-1))
            all_mask.append(mask[:, :, 0].view(-1))
            total_batches += 1

        # ----------------------------------------------------------------------------------
        # 一个epoch结束
        # 记录训练结果(用于画图)
        generator_loss_train.append(g_total_loss / total_batches)   # 列表长度为EPOCHS， 值为每个epoch的平均损失
        discriminator_loss_train.append(d_total_loss / total_batches)
        all_true = torch.cat(all_true)
        all_pred = torch.cat(all_pred)
        all_mask = torch.cat(all_mask)
        rmse_train_record.append(rmse_(all_true, all_pred, all_mask).cpu().item())
        mae_train_record.append(mae_(all_true, all_pred, all_mask).cpu().item())
        r_squared_train_record.append(r_squared_(all_true, all_pred, all_mask).cpu().item())

        # 开始验证
        # 每个epoch，验证一次
        avg_generator_loss_valid, avg_discriminator_loss_valid, rmse_valid, mae, r_sqaured, pred_result = validate(generator, discriminator, warm_up_dataloader_from_train, valid_dataloader)
        # 记录验证结果(用于画图)
        generator_loss_valid.append(avg_generator_loss_valid)   # 列表长度为EPOCHS， 值为每个epoch的平均损失
        discriminator_loss_valid.append(avg_discriminator_loss_valid)
        rmse_valid_record.append(rmse_valid)    # 列表长度为EPOCHS， 值为每个epoch的平均RMSE
        mae_valid_record.append(mae)
        r_squared_valid_record.append(r_sqaured)

        # wandb记录
        wandb.log({
            "generator_loss_train": g_total_loss / total_batches,
            "discriminator_loss_train": d_total_loss / total_batches,
            "rmse_train": round(rmse_train_record[-1], 4),
            "generator_loss_valid": avg_generator_loss_valid,
            "discriminator_loss_valid": avg_discriminator_loss_valid,
            "rmse_valid": round(rmse_valid_record[-1], 4),
            "mae_valid": round(mae_valid_record[-1], 4),
            "r_squared_valid": round(r_squared_valid_record[-1], 4),
        })

        # 打印一个epoch的信息
        print(
            f"Epoch {epoch + 1}/{config.batch_size}, "
            "Train ->"
            f"D Loss: {discriminator_loss_train[-1]:.4f}, "
            f"G Loss: {generator_loss_train[-1]:.4f}, "
            f"MAE_train: {mae_train_record[-1]:.4f}, "
            f"RMSE_train: {rmse_train_record[-1]: 4f}, "
            f"R^2_train: {r_squared_train_record[-1]: 4f}"
        )
        print(
            f"Epoch {epoch + 1}/{config.batch_size}, "
            "Validation ->"
            f"Discriminator Loss: {discriminator_loss_valid[-1]:.4f}, "
            f"Generator Loss: {generator_loss_valid[-1]:.4f}, "
            f"MAE_valid: {mae_valid_record[-1]:.4f}, "
            f"RMSE_valid: {rmse_valid_record[-1]: 4f}, "
            f"R^2_valid: {r_squared_valid_record[-1]: 4f}"
        )


        # 保存最好的模型
        if rmse_valid < best_rmse:
            best_rmse = rmse_valid
            torch.save(generator.state_dict(), f'log/generator.pth')
            torch.save(discriminator.state_dict(), f'log/discriminator.pth')
            pred_result.to_csv(f'log/valid_data_pred_result.csv', index=False)  # 保存预测结果的csv文件
        torch.cuda.empty_cache()
    

        # 画图RMSE: 每个epoch画一次
        plt.figure(figsize=(10, 5))
        current_epoch = len(rmse_train_record)  #从零开始
        plt.plot(range(1, current_epoch+1), rmse_train_record, label="Train", color='deepskyblue')
        plt.plot(range(1, current_epoch+1), rmse_valid_record, label="Valid", color='crimson')
        plt.xlabel('Epoch')
        plt.ylabel('Average RMSE')
        # plt.title('Generator vs Discriminator Losses')
        plt.legend()
        plt.savefig(f'log/AVG_RMSE.png')
        plt.show()



    # 训练结束--------------------------------------------------------------------------------
    # 画图Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.epochs+1), generator_loss_train, label="Generator Loss (Train)", color='blue')
    plt.plot(range(1, config.epochs+1), discriminator_loss_train, label="Discriminator Loss (Train)", color='turquoise')     # 训练时的loss
    plt.plot(range(1, config.epochs+1), generator_loss_valid, label="Generator Loss (Valid)", color='orange', ls='--')
    plt.plot(range(1, config.epochs+1), discriminator_loss_valid, label="Discriminator Loss (Valid)", color='crimson', ls='--')     # 验证的loss
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator Losses')
    plt.legend()
    plt.savefig(f'log/GAN_loss.png')
    plt.show()
    # 开始测试
    # 更新pred_result csv表格
    test(generator, discriminator, warm_up_dataloader_from_train, warm_up_dataloader_from_valid, test_dataloader, pred_result)

    




# 验证函数
def validate(generator, discriminator, warm_up_dataloader_from_train, valid_dataloader):
    """
    Parameters: 
        - generator
        - discriminator
        - warm_up_dataloader_from_train: 用于预热 warm up, 预测结果不计入评价指标
        - valid_dataloader
    Returns:
        - avg_generator_loss_valid, avg_discriminator_loss_valid, avg_rmse_valid
        - pred_result: DataFrame, 保存验证集上的预测
    """
    generator.eval()
    discriminator.eval()
    # 定义损失函数
    ce_loss = nn.CrossEntropyLoss()
    # 初始化
    g_total_loss, d_total_loss = 0.0, 0.0   # 用于计算验证阶段生成器和判别器的总损失
    all_true, all_pred, all_mask = [], [], []
    total_batches = 0  # 用于计算训练时每个epoch的平均RMSE

    # 初始化DataFrame保存验证集上的预测
    pred_result = pd.DataFrame(columns=['债券代码', 'y', 'y_pred', 'split'])
    with torch.no_grad():
        for train_batch, valid_batch in zip(warm_up_dataloader_from_train, valid_dataloader):
            # 需注意要确保两个来自同一个证券
            # 合并训练集和验证集, 训练集部分用于warm up, 不计入评价指标
            features_train, y_train, mask_train = train_batch # shape: (batch, seq_len, features)
            features_train, y_train, mask_train = features_train.to(device), y_train.to(device), mask_train.to(device)
            features_valid, y_valid, mask_valid = valid_batch
            features_valid, y_valid, mask_valid = features_valid.to(device), y_valid.to(device), mask_valid.to(device)
            features, y, mask = torch.cat([features_train, features_valid], dim=1), torch.cat([y_train, y_valid], dim=1), torch.cat([mask_train, mask_valid], dim=1)
            mask_for_loss = mask.clone()
            mask_for_loss[:, :mask_train.size(1), :] = 0    # 只计算验证集的损失, 所以让训练集的mask为0

            # Step 1: 判别器预测
            real_preds = discriminator(y, mask_for_loss)   # shape: (batch_size, seq_len, 2), logits
            # 判别器的损失 (真实数据 -> 1, 假数据 -> 0)
            real_labels = torch.ones(real_preds.size(0), dtype=torch.long).to(device)    # shape: (batch_size), logits
            d_loss_real = ce_loss(real_preds, real_labels)

            y_hat = generator(features, y, mask)
            fake_preds = discriminator(y_hat, mask_for_loss)   # shape: (batch_size*seq_len, 2), logits
            fake_labels = torch.zeros(fake_preds.size(0), dtype=torch.long).to(device)   # shape: (batch_size), logits
            d_loss_fake = ce_loss(fake_preds, fake_labels)
            
            # 更新判别器参数
            d_loss = d_loss_real + d_loss_fake

            # Step 2: 生成器预测
            y_hat = generator(features, y, mask)   # shape: (batch_size, seq_len, 1)
            # 生成器希望判别器将假数据判为真实 (1)
            fake_preds = discriminator(y_hat, mask_for_loss)
            g_loss = ce_loss(fake_preds, real_labels)  # 假数据目标是被判为真实
            # 计算RMSE
            # rmse_loss = rmse(y, y_hat, mask_for_loss)

            # Step 3: 记录
            d_total_loss += d_loss.cpu().item()
            g_total_loss += g_loss.cpu().item()
            all_true.append(y.view(-1))
            all_pred.append(y_hat.view(-1))
            all_mask.append(mask_for_loss[:, :, 0].view(-1))
            """epoch_rmse += rmse_loss.cpu().item()
            epoch_mae += mae(y, y_hat, mask_for_loss).cpu().item()
            epoch_r_squared += r_squared(y, y_hat, mask_for_loss).cpu().item()"""
            total_batches += 1

            """保存预测结果至DataFrame"""
            stkcd = features[:, :, 0].reshape(-1)[mask[:, :, 0].view(-1).bool()].cpu().numpy() # stock code 证券代码
            y = y.reshape(-1)[mask[:, :, 0].view(-1).bool()].cpu().numpy()      # [mask_for_loss[:, :, 0].view(-1).bool()]用于选取非padding的数据
            y_hat = y_hat.reshape(-1)[mask[:, :, 0].view(-1).bool()].cpu().numpy()
            split = np.array((['train'] * mask_train.size(1) + ['valid'] * mask_valid.size(1)) * mask_train.size(0))[mask[:, :, 0].view(-1).cpu().numpy() == 1]
            pred_result = pd.concat([pred_result, pd.DataFrame({'债券代码': stkcd, 'y': y, 'y_pred': y_hat, 'split': split})], axis=0)

        all_true = torch.cat(all_true)
        all_pred = torch.cat(all_pred)
        all_mask = torch.cat(all_mask)
        rmse = rmse_(all_true, all_pred, all_mask).cpu().item()
        mae = mae_(all_true, all_pred, all_mask).cpu().item()
        r_squared = r_squared_(all_true, all_pred, all_mask).cpu().item()
        return g_total_loss/total_batches, d_total_loss/total_batches, rmse, mae, r_squared, pred_result
        #return g_total_loss/total_batches, d_total_loss/total_batches, epoch_rmse/total_batches, epoch_mae/total_batches, epoch_r_squared/total_batches, pred_result


def test(generator, discriminator, warm_up_dataloader_from_train, warm_up_dataloader_from_valid, test_dataloader, pred_result):
    """
    Parameters: 
        - generator
        - discriminator
        - warm_up_dataloader_from_train: 用于预热 warm up, 预测结果不计入评价指标
        - warm_up_dataloader_from_valid: 用于预热 warm up, 预测结果不计入评价指标
        - valid_dataloader
        - pred_result: DataFrame, 保存验证集上的预测
    Returns:
        不return, 只打印和保存结果
    """
    generator.load_state_dict(torch.load('log/generator.pth', map_location=device))
    discriminator.load_state_dict(torch.load('log/discriminator.pth', map_location=device))
    generator.eval()
    discriminator.eval()
    # 定义损失函数
    ce_loss = nn.CrossEntropyLoss()
    # 初始化
    g_total_loss, d_total_loss = 0.0, 0.0   # 用于计算验证阶段生成器和判别器的总损失
    all_true, all_pred, all_mask = [], [], []
    total_batches = 0  # 用于计算训练时每个epoch的平均RMSE

    # 初始化DataFrame保存验证集上的预测
    pred_result = pd.DataFrame(columns=['债券代码', 'y', 'y_pred', 'split'])
    with torch.no_grad():
        for train_batch, valid_batch, test_batch in zip(warm_up_dataloader_from_train, warm_up_dataloader_from_valid, test_dataloader):
            # 需注意要确保两个来自同一个证券
            # 合并训练集, 验证集和测试集, 训练集,和验证集部分用于warm up, 不计入评价指标计算
            features_train, y_train, mask_train = train_batch # shape: (batch, seq_len, features)
            features_train, y_train, mask_train = features_train.to(device), y_train.to(device), mask_train.to(device)
            features_valid, y_valid, mask_valid = valid_batch
            features_valid, y_valid, mask_valid = features_valid.to(device), y_valid.to(device), mask_valid.to(device)
            features_test, y_test, mask_test = test_batch
            features_test, y_test, mask_test = features_test.to(device), y_test.to(device), mask_test.to(device)
            features, y, mask = torch.cat([features_train, features_valid, features_test], dim=1), torch.cat([y_train, y_valid, y_test], dim=1), torch.cat([mask_train, mask_valid, mask_test], dim=1)
            mask_for_loss = mask.clone()
            mask_for_loss[:, :(mask_train.size(1)+mask_valid.size(1)), :] = 0    # 只计算测试集的损失, 所以让warmup部分的mask为0

            # Step 1: 判别器预测
            real_preds = discriminator(y, mask_for_loss)   # shape: (batch_size, seq_len, 2), logits
            # 判别器的损失 (真实数据 -> 1, 假数据 -> 0)
            real_labels = torch.ones(real_preds.size(0), dtype=torch.long).to(device)    # shape: (batch_size), logits
            d_loss_real = ce_loss(real_preds, real_labels)

            y_hat = generator(features, y, mask)
            fake_preds = discriminator(y_hat, mask_for_loss)   # shape: (batch_size*seq_len, 2), logits
            fake_labels = torch.zeros(fake_preds.size(0), dtype=torch.long).to(device)   # shape: (batch_size), logits
            d_loss_fake = ce_loss(fake_preds, fake_labels)
            
            # 更新判别器参数
            d_loss = d_loss_real + d_loss_fake

            # Step 2: 生成器预测
            y_hat = generator(features, y, mask)   # shape: (batch_size, seq_len, 1)
            # 生成器希望判别器将假数据判为真实 (1)
            fake_preds = discriminator(y_hat, mask_for_loss)
            g_loss = ce_loss(fake_preds, real_labels)  # 假数据目标是被判为真实
            # 计算RMSE
            # rmse_loss = rmse(y, y_hat, mask_for_loss)

            # Step 3: 记录
            d_total_loss += d_loss.cpu().item()
            g_total_loss += g_loss.cpu().item()
            all_true.append(y.view(-1))
            all_pred.append(y_hat.view(-1))
            all_mask.append(mask_for_loss[:, :, 0].view(-1))
            """epoch_rmse += rmse_loss.cpu().item()
            epoch_mae += mae(y, y_hat, mask_for_loss).cpu().item()
            epoch_r_squared += r_squared(y, y_hat, mask_for_loss).cpu().item()"""
            total_batches += 1

            """保存预测结果至DataFrame"""
            stkcd = features[:, :, 0].reshape(-1)[mask[:, :, 0].view(-1).bool()].cpu().numpy() # stock code 证券代码
            y = y.reshape(-1)[mask[:, :, 0].view(-1).bool()].cpu().numpy()      # [mask_for_loss[:, :, 0].view(-1).bool()]用于选取非padding的数据
            y_hat = y_hat.reshape(-1)[mask[:, :, 0].view(-1).bool()].cpu().numpy()
            split = np.array((['train'] * mask_train.size(1) + ['valid'] * mask_valid.size(1) + ['test'] * mask_test.size(1)) * mask_train.size(0))[mask[:, :, 0].view(-1).cpu().numpy() == 1] # 这里[mask[:, :, 0].view(-1).cpu().numpy() == 1]是bool索引
            pred_result = pd.concat([pred_result, pd.DataFrame({'债券代码': stkcd, 'y': y, 'y_pred': y_hat, 'split': split})], axis=0)

        all_true = torch.cat(all_true)
        all_pred = torch.cat(all_pred)
        all_mask = torch.cat(all_mask)
        rmse_test = rmse_(all_true, all_pred, all_mask).cpu().item()
        mae = mae_(all_true, all_pred, all_mask).cpu().item()
        r_squared = r_squared_(all_true, all_pred, all_mask).cpu().item()

        # 保存预测结果至csv文件
        pred_result.to_csv(f'log/test_data_pred_result.csv', index=False)

        # 打印测试集结果
        print(
            "Test ->"
            f"Discriminator Loss: {d_total_loss/total_batches:.4f}, "
            f"Generator Loss: {g_total_loss/total_batches:.4f}, "
            f"MAE_test: {mae:.4f}, "
            f"RMSE_test: {rmse_test:.4f}, "
            f"R^2_test: {r_squared:.4f}"
        )

        # wandb记录结果
        wandb.log({
            "generator_loss_test": g_total_loss / total_batches,
            "discriminator_loss_test": d_total_loss / total_batches,
            "rmse_test": round(rmse_test, 4),
            "mae_test": round(mae, 4),
            "r_squared_test": round(r_squared, 4),
        })




# 开始训练
# train_and_evaluate()

# 使用wandb训练

# 1: Define objective/training function --> train_and_evaluate()
# 2: Define the search space
sweep_config = {
    "method": "bayes",  # 搜索策略：grid, random, bayes
    "metric": {
        "goal": "minimize",
        "name": "rmse_valid"  # 优化的目标指标, 验证阶段
    },
    "parameters": {
        "feature_size": {
            "value": 38     ##################################################
        },
        "learning_rate": {
            "min": 4e-5,
            "max": 1e-3  # 搜索范围
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        
        "dropout": {
            "values": [0.1, 0.2, 0.3]
        },
        "d_model": {
            "values": [128, 256, 512]
        },
        "n_blocks": {
            "values": [6, 8, 10]
        },
        "d_ff": {
            "values": [512, 1024, 2048]
        },
        "n_heads": {
            "values": [8, 16, 32]
        },
        "epochs":{
            "values": [30, 40, 50]
        },
        "optimizer": {
            "values": ["adam", "sgd"]
        },
        "lambda_": {
            "min": 0.05,
            "max": 0.2  # 搜索范围
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="Spread-predict")
wandb.agent("huneedhelp/Spread-predict/p1dyblqh", function=train_and_evaluate, count=50)



