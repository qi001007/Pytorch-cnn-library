import copy
import os
import glob
import time
import json
import yaml
import torch
import socket
import argparse
import importlib

from pathlib import Path
from torchvision import transforms
from torch.amp import autocast, GradScaler
from pynput import keyboard
from tensorboardX import SummaryWriter

import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd

from my_bar import simple_bar
from data_video_set import VideoDataset
from model import C3D_VGG11


# -------------------- 训练函数 --------------------
def train_val_data_process():
    # 加载数据集
    data_ = VideoDataset(dataset_path='data', images_path='test', clip_len=16)
    # 分割训练集和测试集
    train_data, val_data = Data.random_split(data_,
                                             lengths=[round(0.8 * len(data_)), round(0.2 * len(data_))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=16,
                                       shuffle=True,
                                       num_workers=4)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=4)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs, log_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # 指定训练设备
    print(device)
    # 创建优化器（梯度下降法的优化方法：SGD,Adam,A,B,C）
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # 定义学习率的更新策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 创建损失函数（均方误差损失函数：回归,交叉熵损失函数Cross-Entropy Loss：分类）
    criterion = nn.CrossEntropyLoss()

    # 将模型放到训练设备当中
    model = model.to(device)
    criterion.to(device)

    # 日志记录
    log_dir = os.path.join(log_dir, time.strftime('%b%d_%H-%M-%S', time.localtime()) + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    best_model_wts = copy.deepcopy(model.state_dict())      # 复制当前模型参数
    scaler = GradScaler()

    # 初始化参数
    # 开始模型的训练
    train_val_loaders = {'train': train_dataloader, 'val': val_dataloader}  # 将验证集和训练集以字典的形式报存
    # 最高精确度
    best_acc = 0.0
    # 训练集损失函数列表，集准确度列表
    # 验证集损失函数列表，集准确度列表
    All = {'train': {'loss_all': [], 'acc_all': []},
           'val': {'loss_all': [], 'acc_all': []}}
    # 最终信息表
    train_process = pd.DataFrame(columns=["epoch", "train_loss_all", "train_acc_all", "val_loss_all", "val_acc_all"])
    # 当前时间
    since = time.time()
    date_str_start = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    os.makedirs(f'./model_wts/{date_str_start}model', exist_ok=True)

    for i, epoch in simple_bar(num_epochs, flag=kill_epoch, is_open_bar=False):
        print("-"*50)
        print(f"Epoch {epoch+1}/{num_epochs}")

        # 初始化每一轮的参数
        # 训练集损失值，集准确度，样本数量
        # 验证集损失值，集准确度，样本数量
        metrics = {
            'train': {'loss': 0.0, 'corrects': 0, 'num': 0},
            'val': {'loss': 0.0, 'corrects': 0, 'num': 0}
        }

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for step, (b_x, b_y) in simple_bar(train_val_loaders[phase], flag=kill_epoch,
                                               desc=f"{phase}_dataloader_running", step_size=2):
                if kill_epoch:
                    print("\n[q] 被按下，立即终止训练。")
                    # raise KeyboardInterrupt
                    break
                # 将特征放入到训练设备中
                # 把requires_grad = True加在输入上，一般只在要实现“对抗样本 / 梯度攻击”或“可学习输入”时才需要；正常前向传播不需要
                # 迁移到GPU时优先用non_blocking = True可让拷贝与计算并行，加速训练
                b_x = b_x.to(device, non_blocking=True).requires_grad_(True)
                # 将标签放入到训练设备中
                b_y = b_y.to(device, non_blocking=True)

                # 把原来的 forward + backward + step 三步换成 “autocast 包前向 + scaler 包反向”
                # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                # 前向：只在 autocast 上下文里做
                with autocast(str(device)):
                    if phase == "train":
                        with torch.enable_grad():
                            output = model(b_x)
                    else:
                        with torch.no_grad():
                            output = model(b_x)
                    # 计算softmax的输出概率
                    probs = nn.Softmax(dim=1)(output)
                    # 计算最大概率值的标签
                    pre_lab = torch.max(probs, 1)[1]
                    # 计算每一个batch的损失值
                    loss = criterion(output, b_y.long())

                # 将梯度初始化为0
                optimizer.zero_grad()
                if phase == "train":
                    # 反向：先放大损失，再反向
                    scaler.scale(loss).backward()
                    # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
                    # 更新：scaler负责把梯度缩回去并step
                    scaler.step(optimizer)
                    # 更新缩放因子
                    scaler.update()
                # 对损失函数进行累加
                metrics[phase]['loss'] += loss.item() * b_x.size(0)
                # 如果预测正确，则准确度corrects加1
                metrics[phase]['corrects'] += torch.sum(pre_lab == b_y.data).item()
                # 当前用于训练/验证的样本数量
                metrics[phase]['num'] += b_x.size(0)

            # 计算并保存每一次迭代的损失值和准确率
            if metrics[phase]['num'] > 0:
                # 计算并保存训练集/验证集的loss值
                All[phase]['loss_all'].append(metrics[phase]['loss'] / metrics[phase]['num'])
                # 计算并保存训练集/验证集的准确率
                All[phase]['acc_all'].append(np.double(metrics[phase]['corrects']) / metrics[phase]['num'])

            if phase == "train":
                writer.add_scalar(tag='data/train_loss_epoch', scalar_value=All[phase]['loss_all'][-1],
                                  global_step=epoch)
                writer.add_scalar(tag='data/train_acc_epoch', scalar_value=All[phase]['acc_all'][-1],
                                  global_step=epoch)
            else:
                writer.add_scalar(tag='data/val_loss_epoch', scalar_value=All[phase]['loss_all'][-1],
                                  global_step=epoch)
                writer.add_scalar(tag='data/val_acc_epoch', scalar_value=All[phase]['acc_all'][-1],
                                  global_step=epoch)

        # 跳出条件
        if kill_epoch:
            print("\n[q] 被按下，立即终止此轮。")
            # raise KeyboardInterrupt
            break

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, All['train']['loss_all'][-1],
                                                              All['train']['acc_all'][-1]))
        # [-1] 是 Python 列表索引语法，表示 “最后一个元素”
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, All['val']['loss_all'][-1],
                                                          All['val']['acc_all'][-1]))

        # 寻找最高准确度的权重
        if All['val']['acc_all'][-1] > best_acc:
            # 保存当前最高准确度
            best_acc = All['val']['acc_all'][-1]
            # 保存模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 学习率的更新
        scheduler.step()
        # 每轮训练的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

        # 选择每轮最优参数
        if epoch+1 == num_epochs:
            # 如果训练执行到最后循环，将最后的最优参数保存，删掉best_model_epoch_break
            date_str_end = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
            torch.save(best_model_wts,
                       f'./model_wts/'
                       f'{date_str_start}model/best_model_{date_str_end}.pth')
            last_ckpt = (f'./model_wts/'
                         f'{date_str_start}model/best_model_{num_epochs-1}epoch.pth')
            if os.path.exists(last_ckpt):
                os.remove(last_ckpt)
        else:
            torch.save(best_model_wts,
                       f'./model_wts/'
                       f'{date_str_start}model/best_model_{epoch+1}epoch.pth')
            old_ckpt = (f'./model_wts/'
                        f'{date_str_start}model/best_model_{epoch}epoch.pth')
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

        # 将关键数据存入pandas的DtaFrame中，方便之后画图调用
        # len(all['train']['loss_all'])保证长度一致
        train_process = pd.DataFrame({"epoch": range(len(All['train']['loss_all'])),
                                      "train_loss_all": All['train']['loss_all'],
                                      "train_acc_all": All['train']['acc_all'],
                                      "val_loss_all": All['val']['loss_all'][:len(All['train']['loss_all'])],
                                      "val_acc_all": All['val']['acc_all'][:len(All['train']['acc_all'])]})
    writer.close()

    return train_process, date_str_start


def matplot_acc_loss(train_process, date_str_start):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.savefig(f'./model_wts/{date_str_start}model/loss_acc.jpg')
    plt.show()


# -------------------- 热键退出 --------------------
def on_press(key):
    global kill_epoch
    try:
        if key.char and key.char.lower() == 'q':
            # 只改标志，不 raise
            kill_epoch = True
            # 给自己发 SIGINT，等价于 Ctrl-C
            # signal.raise_signal(signal.SIGINT)
    except AttributeError:
        pass


def _gentle_shutdown(loader):
    """温和关闭 DataLoader 的多进程 worker，不抛异常、不报警告。"""
    itr = getattr(loader, '_iterator', None)          # 拿到迭代器
    if itr:
        shutdown = getattr(itr, '_shutdown_workers', None)
        if callable(shutdown):                        # 如果方法存在就调
            shutdown()


# -------------------- Windows 安全入口 --------------------
def main():
    global kill_epoch, Log_dir
    # 退出标志位
    kill_epoch = False
    # 启动后台监听线程（非阻塞）
    listener = keyboard.Listener(on_press=on_press, daemon=True)
    listener.start()
    # 将模型实例化
    Model = C3D_VGG11(num_classes=101, pretrained=True)
    # 数据分批
    Train_dataloader, Val_dataloader = train_val_data_process()
    # 参数初始化
    Train_process = pd.DataFrame(
        columns=["epoch", "train_loss_all", "train_acc_all", "val_loss_all", "val_acc_all"])
    Date_str_start = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    try:
        # 开始训练
        Train_process, Date_str_start = train_model_process(Model, Train_dataloader, Val_dataloader, num_epochs=10, log_dir=Log_dir)
        # 绘制图形
        matplot_acc_loss(Train_process, Date_str_start)
    except KeyboardInterrupt:
        print("\n[停止] 被按下，已停止训练。")
        # 绘制图形
        matplot_acc_loss(Train_process, Date_str_start)
        print(Train_process, Date_str_start)
    finally:
        _gentle_shutdown(Train_dataloader)
        _gentle_shutdown(Val_dataloader)
        listener.stop()


if __name__ == "__main__":
    kill_epoch = False
    Log_dir = 'log'
    main()
