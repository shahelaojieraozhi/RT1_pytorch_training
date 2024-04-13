import os
import numpy as np
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


input_dir = "/mnt/VLMART/rt1_data_prepro"
input_dir_list = os.listdir(input_dir)

sample_size = int(0.8 * len(input_dir_list))
train_sample_list = random.sample(input_dir_list, sample_size)
other_sample_list = [item for item in input_dir_list if item not in train_sample_list]


for mode in ["val", "train"]:

    size = 0
    idx = 0
    with open(f"{mode}_idx.csv", "w") as file:
        sample_list = train_sample_list if mode == "train" else other_sample_list
        sample_list = tqdm(sample_list, file=sys.stdout)
        for sample in sample_list:
            # print(sample)
            temp_data = torch.load(os.path.join(input_dir, sample))
            size += len(temp_data)

            file.write(f"{idx},")
            file.write(f"{sample}\n")

            for single_temp_idx in range(len(temp_data)):
                idx += 1

    with_idx_filename = f"{mode}_idx_{idx}.csv"
    os.rename(f"{mode}_idx.csv", with_idx_filename)

    print(size)

