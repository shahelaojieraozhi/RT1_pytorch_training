# -*- coding: utf-8 -*-
"""
@Project ：emotion 
@Time    : 2024/2/29 22:33
@Author  : Rao Zhi
@File    : rt1_loss.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@描述     :
@detail   ：
@infer    :
"""

import torch
import torch.nn as nn


class SparseCategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SparseCategoricalCrossEntropyLoss, self).__init__()
        # 使用 PyTorch 的交叉熵损失
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        bs, fs, _, _ = logits.size()  # bs:batch_size, fs:frames size
        true_bs = bs * fs
        logits = logits.reshape(true_bs, 11, 256)
        # PyTorch's cross entropy loss requires the label to be a long integer
        labels = labels.reshape(true_bs, 11).long()

        loss_value = []
        for bs_idx in range(len(labels)):
            # c = logits[bs_idx]  # [11, 256]
            # d = labels[bs_idx]  # [11,]
            loss = self.loss_function(logits[bs_idx], labels[bs_idx])
            loss_value.append(loss)
            # print("Loss:", loss.item())

        return sum(loss_value)


if __name__ == '__main__':
    logits = torch.randn((2, 6, 11, 256))
    labels = torch.randint(0, 256, (2, 6, 11))
    loss_F = SparseCategoricalCrossEntropyLoss()
    loss_sum = loss_F(logits, labels)
    print(loss_sum)
