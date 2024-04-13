# -*- coding: utf-8 -*-
"""
@Project ：emotion 
@Time    : 2024/2/28 16:37
@Author  : Rao Zhi
@File    : action_tokenizer_detokenize.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@描述     :
@detail   ：
@infer    :
"""
import torch

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple action tokenizer used with Robotics Transformer 1.

As an example, if an action is:
terminate = [0, 1]
world_vector = [0.9, 0.8, -0.3]
rotation_delta = [-0.1, 0.2, .6]
gripper_closedness = 0.9

Then we build a sequence of tokens of length 8 [one for each dimension].
The int32 type action dimensions are already assumed discrete and tokenized,
the float dimensions are bucketed according to the specs min and max. Each
dimension has 'vocab_size' buckets.

然后我们构建一个长度为8的标记序列[每个维度一个]。
int32类型的操作维度已经假定为离散的和标记化的，
float dimensions 根据规格 min 和 max 进行 bucketed 。每一个维度有'vocab_size'个buckets。

Currently, this tokenizer assumes one action spec and it is highly recommended
to specify the 'action_order', eg [terminate, world_vector, rotation_delta,
gripper_closedness]. Since after tokenization you lose that information, this
will be useful for debugging. Actions may also be subselected for prediction,
since not all actions are needed in the action_order.

目前，这个 tokenizer 假设一个操作规范，强烈推荐使用它
要指定'action_order'，例如[terminate, world_vector, rotation_delta，gripper_closedness]。
因为在 tokenization 之后你失去了这些信息，这将有用的调试。动作也可以被子选择用于预测，
因为不是所有的动作都需要在action_order中。

"""


def action_tokenize(action_ord, vocab_size):
    """Tokenizes an action."""
    action_tokens = []
    for spec in action_ord:
        c = spec.dtype
        if spec.dtype == torch.int32:
            # Int32 actions are already assumed to be tokens, assume it is smaller
            # than the vocab size, so all we need to do is pad zeros.

            # tf.debugging.assert_equal(1, tf.reduce_sum(a, axis=-1))   #

            # extract the token [batch, 1]
            token = torch.argmax(spec, dim=-1, keepdim=False).to(torch.int32)
            # token = tf.argmax(spec, axis=-1, output_type=torch.int32)

            # Add a seq dimension [batch, 1]
            token = torch.unsqueeze(token, dim=-1)
        else:
            # q = torch.min(spec)
            # w = torch.max(spec)
            a = torch.clamp(spec, torch.min(spec), torch.max(spec))

            if torch.min(spec) == torch.max(spec):
                token = a
            # Normalize the action [batch, actions_size]
            else:
                token = (a - torch.min(spec)) / (torch.max(spec) - torch.min(spec))

            # Bucket and discretize the action to vocab_size, [batch, actions_size]
            # token = tf.cast(token * (self._vocab_size - 1), tf.int32)

            token = (token * (vocab_size - 1)).to(torch.int32)

        action_tokens.append(token)
    # Append all actions, [batch, all_actions_size]
    action_tokens = torch.cat(action_tokens, dim=2)

    return action_tokens


if __name__ == '__main__':
    vocab_size = 256
    sub_dataset = torch.load("G:/zhi_backup/RT/RT1_data_preprocessed_ex.h5")
    sub_dataset_shape = len(sub_dataset)
    single_sample = sub_dataset[154]
    action = single_sample['action']

    world_vector = torch.unsqueeze(torch.tensor(action['world_vector'].numpy(), dtype=torch.float32)[0], dim=0)
    rotation_delta = torch.unsqueeze(torch.tensor(action['rotation_delta'].numpy(), dtype=torch.float32)[0],
                                     dim=0)  # roll, pitch, yaw
    gripper_closedness_action = torch.unsqueeze(torch.tensor(action['gripper_closedness_action'].numpy(),
                                                             dtype=torch.int32)[0], dim=0)  # opening of the gripper
    base_displacement_vector = torch.unsqueeze(
        torch.tensor(action['base_displacement_vector'].numpy(), dtype=torch.float32)[0], dim=0)  # x y
    base_displacement_vertical_rotation = torch.unsqueeze(
        torch.tensor(action['base_displacement_vertical_rotation'].numpy(),
                     dtype=torch.float32)[0], dim=0)  # yaw
    base_movement = torch.cat([base_displacement_vector, base_displacement_vertical_rotation], dim=-1)

    terminate_episode = torch.unsqueeze(torch.tensor(action['terminate_episode'].numpy(), dtype=torch.int32)[0], dim=0)

    action_order = [world_vector, rotation_delta, gripper_closedness_action, base_movement, terminate_episode]
    final_action_tokens = action_tokenize(action_order, vocab_size)
    print(final_action_tokens)
