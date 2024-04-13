# -*- coding: utf-8 -*-
"""
@Project ：robotic-transformer-pytorch-main 
@Time    : 2024/3/2 16:16
@Author  : Rao Zhi
@File    : RT1_Dataset.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@描述     :
@detail   ：
@infer    :
"""

import os
import numpy as np
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class RT1Dataset(Dataset):
    """
    self.temp_data = [[video, instru, action], [video, instru, action], ....]

    """

    def __init__(self, mode):
        super(RT1Dataset, self).__init__()

        self.input_dir = "/mnt/VLMART/rt1_data_prepro"

        if mode == "train":
            self.train_idx_table = pd.read_csv(filepath_or_buffer="train_idx_427618.csv", header=None)
            self.size = 427618
            self.start_point = self.train_idx_table.iloc[:, 0]
            self.start_sample_name = self.train_idx_table.iloc[:, 1]

        else:
            self.val_idx_table = pd.read_csv("val_idx_107274.csv", header=None)
            self.size = 107274
            self.start_point = self.val_idx_table.iloc[:, 0]
            self.start_sample_name = self.val_idx_table.iloc[:, 1]

    def __getitem__(self, index):

        data_idx = sum(1 for r in self.start_point if index >= r) - 1
        data_margin = self.start_point[data_idx]

        start_sample_name = self.start_sample_name[data_idx]
        self.data = torch.load(os.path.join(self.input_dir, start_sample_name))

        video, order, action = self.data[index - data_margin]
        video = torch.tensor(video, dtype=torch.float32)
        return video, order, action.squeeze()

    def __len__(self):
        return self.size


if __name__ == "__main__":
    mod = "train"
    data = RT1Dataset(mode=mod)
    datasize = len(data)
    print(datasize)

    # data_ = RT1Dataset(mode=mod)
    # a = data_[83600]
    # print(a)
    # print(datasize)
    # for sample in data:
    #     vid, order, act = sample

    #     print(act)
    #     print(order[0])

    #     trans_vid = torch.reshape(vid, (6, 3, 224, 224))

    #     images = trans_vid.numpy()
    #     fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    #     for i in range(2):
    #         for j in range(3):
    #             image = np.transpose(images[i * 3 + j], (1, 2, 0))
    #             axes[i, j].imshow(image)
    #             axes[i, j].set_title(f'Image {i * 3 + j + 1}')  # 添加标题
    #             axes[i, j].axis('off')

    #     plt.subplots_adjust(wspace=0.2, hspace=0.2)  # 调整子图之间的间距
    #     plt.show()
