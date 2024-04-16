# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code
@Time    : 2023/10/7 8:28
@Author  : Rao Zhi
@File    : random_seeds_set.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm
@ref: javis
"""
import random
import torch
import numpy as np
import time
import os

# from sklearn.metrics import f1_score
from torch import nn
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


# 计算F1score
# def calc_f1(y_true, y_pre, threshold=0.5):
#     y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
#     y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
#     return f1_score(y_true, y_pre)


# 打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return "{:.0f}m{:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60)


def left_time(since, done_epoch):
    time_elapsed = (time.time() - since) * done_epoch
    return "{:.0f}m{:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# 多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction="none")
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()


def seed_torch(seed=1029):
    """
    参考:https://blog.csdn.net/john_bh/article/details/107731443
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # To disable hash randomization and make the experiment reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def newline(p1, p2):
    """
    Draws a line between two points

    Arguments:
        p1 {list} -- coordinate of the first point
        p2 {list} -- coordinate of the second point

    Returns:
        mlines.Line2D -- the drawn line
    """
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax], linewidth=1, linestyle="--")
    ax.add_line(l)
    return l


def bland_altman(data1, data2):
    """
    Computes mean +- 1.96 sd

    Arguments:
        data1 {array} -- series 1
        data2 {array} -- series 2
    """

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, alpha=0.1, s=4)
    plt.axhline(md, color="gray", linestyle="--")
    plt.axhline(md + 1.96 * sd, color="gray", linestyle="--")
    plt.axhline(md - 1.96 * sd, color="gray", linestyle="--")
    plt.ylim(ymin=-75, ymax=75)
    plt.xlabel("Avg. of Target and Estimated Value (mmHg)", fontsize=14)
    plt.ylabel("Error in Prediction (mmHg)", fontsize=14)
    print(md + 1.96 * sd, md - 1.96 * sd)


def if_exit_pth(directory_list):
    """
    load trained model
    judge if have .pth in a folder
    """
    for filename in directory_list:
        if filename.endswith(".pth"):
            return True
    return False
