# -*- coding: utf-8 -*-
"""
@Project ：robotic-transformer-pytorch-main 
@Time    : 2024/3/2 16:39
@Author  : Rao Zhi
@File    : DataPreprocess.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@描述     :
@detail   ：
@infer    :
"""
import os
import cv2
import sys
import torch
import numpy as np
from tqdm import tqdm
from action_tokenizer_detokenize import action_tokenize


def rt1_data_prepro(ori_data, input_pth):
    input_data = []

    for single_sample in ori_data:
        video, action, instru = (
            single_sample["image"],
            single_sample["action"],
            single_sample["instruction"].tolist(),
        )
        instru = [x.decode("utf-8") for x in instru]
        resized_video = np.array([cv2.resize(img, (224, 224)) for img in video])
        transposed_video = torch.tensor(np.transpose(resized_video, (3, 0, 1, 2)))

        world_vector = torch.unsqueeze(
            torch.tensor(action["world_vector"], dtype=torch.float32), dim=0
        )
        rotation_delta = torch.unsqueeze(
            torch.tensor(action["rotation_delta"], dtype=torch.float32), dim=0
        )  # roll, pitch, yaw
        gripper_closedness_action = torch.unsqueeze(
            torch.tensor(action["gripper_closedness_action"], dtype=torch.int32), dim=0
        )  # opening of the gripper
        base_displacement_vector = torch.unsqueeze(
            torch.tensor(action["base_displacement_vector"], dtype=torch.float32), dim=0
        )  # x y
        base_displacement_vertical_rotation = torch.unsqueeze(
            torch.tensor(
                action["base_displacement_vertical_rotation"], dtype=torch.float32
            ),
            dim=0,
        )  # yaw
        base_movement = torch.cat(
            [base_displacement_vector, base_displacement_vertical_rotation], dim=-1
        )

        terminate_episode = torch.unsqueeze(
            torch.tensor(action["terminate_episode"], dtype=torch.int32), dim=0
        )

        action_order = [
            world_vector,
            rotation_delta,
            gripper_closedness_action,
            base_movement,
            terminate_episode,
        ]
        """
        world_vector[1, 6 ,3]  rotation_delta[1, 6 ,3]  gripper_closedness_action[1, 6 ,1]
        base_movement[1, 6 ,3]  terminate_episode[1, 6, 3]
        """

        final_action_tokens = action_tokenize(action_order, vocab_size)
        input_data.append([transposed_video, instru, final_action_tokens])

    torch.save(input_data, input_pth)


if __name__ == "__main__":
    import time

    vocab_size = 256
    data_root_dir = "/mnt/VLMART/pick"
    output_root_dir = "/mnt/VLMART/pick_after_prepro"
    for sample in tqdm(os.listdir(data_root_dir), file=sys.stdout):
        # print(sample)
        # time.sleep(5)
        sub_dataset = torch.load(os.path.join(data_root_dir, sample))
        input_data_path = os.path.join(output_root_dir, sample)
        print(input_data_path)
        rt1_data_prepro(sub_dataset, input_data_path)
