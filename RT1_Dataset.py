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
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class RT1Dataset(Dataset):
    """
    self.temp_data = [[video, instru, action], [video, instru, action], ....]

    """

    def __init__(self, mode):
        super(RT1Dataset, self).__init__()
        self.input_dir = "/mnt/VLMART/pick_after_prepro"
        self.input_dir_list = os.listdir(self.input_dir)

        sample_size = int(0.8 * len(self.input_dir_list))
        train_sample_list = random.sample(self.input_dir_list, sample_size)
        other_sample_list = [item for item in self.input_dir_list if item not in train_sample_list]
        self.size = 0

        self.data = []

        if mode == "train":
            for sample in train_sample_list:
                self.temp_data = torch.load(os.path.join(self.input_dir, sample))
                self.size += len(self.temp_data)
                self.data += self.temp_data
        else:
            for sample in other_sample_list:
                self.temp_data = torch.load(os.path.join(self.input_dir, sample))
                self.size += len(self.temp_data)
                self.data += self.temp_data

    def __getitem__(self, index):
        video, order, action = self.data[index]
        video = torch.tensor(video, dtype=torch.float32)
        return video, order, action.squeeze()

    def __len__(self):
        return self.size


if __name__ == '__main__':
    data = RT1Dataset(mode='test')
    datasize = len(data)
    print(datasize)
    for sample in data:
        vid, order, act = sample

        print(act)
        print(order[0])

        trans_vid = torch.reshape(vid, (6, 3, 224, 224))

        images = trans_vid.numpy()
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))

        for i in range(2):
            for j in range(3):
                image = np.transpose(images[i * 3 + j], (1, 2, 0))
                axes[i, j].imshow(image)
                axes[i, j].set_title(f'Image {i * 3 + j + 1}')  # 添加标题
                axes[i, j].axis('off')

        plt.subplots_adjust(wspace=0.2, hspace=0.2)  # 调整子图之间的间距
        plt.show()
