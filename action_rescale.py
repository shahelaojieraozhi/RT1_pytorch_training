# -*- coding: utf-8 -*-
"""
@Project ：emotion 
@Time    : 2024/2/28 14:26
@Author  : Rao Zhi
@File    : action_rescale.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@描述     :
@detail   ：
@infer    :
"""

import torch
import torchvision.transforms.functional as ttf


def resize(image):
    target_height, target_width = 256, 320

    # 获取输入图像的高度和宽度
    input_height, input_width = image.shape[-2], image.shape[-1]

    # 计算需要添加的垂直和水平填充量
    pad_vertical = max(0, target_height - input_height)
    pad_horizontal = max(0, target_width - input_width)

    # 计算填充的上下左右数量
    pad_top = pad_vertical // 2
    pad_bottom = pad_vertical - pad_top
    pad_left = pad_horizontal // 2
    pad_right = pad_horizontal - pad_left

    # 使用pad函数进行填充
    padded_image = ttf.pad(image, (pad_left, pad_top, pad_right, pad_bottom), 0)

    # 使用resize函数进行调整大小
    resized_image = ttf.resize(padded_image, (target_height, target_width))

    # 将图像类型转换为uint8
    resized_image = resized_image.to(torch.uint8)

    return resized_image


def rescale_action_with_bound(actions, low, high, safety_margin=0, post_scaling_max=1.0, post_scaling_min=-1.0):
    resc_actions = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min

    # 裁剪操作
    return torch.clamp(resc_actions, post_scaling_min + safety_margin, post_scaling_max - safety_margin)


def rescale_action(action):
    """Rescales action."""

    action['world_vector'] = rescale_action_with_bound(
        action['world_vector'],
        low=-0.05,
        high=0.05,
        safety_margin=0.01,
        post_scaling_max=1.75,
        post_scaling_min=-1.75,
    )
    action['rotation_delta'] = rescale_action_with_bound(
        action['rotation_delta'],
        low=-0.25,
        high=0.25,
        safety_margin=0.01,
        post_scaling_max=1.4,
        post_scaling_min=-1.4,
    )

    return action


def to_model_action(from_step):
    """Convert dataset action to model action. This function is specific for the Bridge dataset."""

    ori_model_action = {'world_vector': torch.Tensor(from_step['action']['world_vector'].numpy()),
                        'terminate_episode': torch.Tensor(from_step['action']['terminate_episode'].numpy()),
                        'rotation_delta': torch.Tensor(from_step['action']['rotation_delta'].numpy()),
                        'gripper_closedness_action': torch.Tensor(from_step['action']['gripper_closedness_action'].numpy())}

    return rescale_action(ori_model_action)


if __name__ == '__main__':
    sub_dataset = torch.load("G:/zhi_backup/RT/RT1_data_preprocessed_ex.h5")
    sub_dataset_shape = len(sub_dataset)
    single_sample = sub_dataset[0]
    model_action = to_model_action(single_sample)
    print(model_action)

    action_order = ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']

