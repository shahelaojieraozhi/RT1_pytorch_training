# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/12 9:00
@Author  : Rao Zhi
@File    : train.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import random
import shutil
import time
import argparse
import utils
import sys
import datetime
import warnings
import torch
import numpy as np
from torch import optim

# import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# from RT1_Dataset import RT1Dataset
from RT1_Dataset_whole import RT1Dataset
from torch.utils.tensorboard import SummaryWriter
from robotic_transformer_pytorch import MaxViT, RT1
from loss_func import SparseCategoricalCrossEntropyLoss

from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
import math

warnings.filterwarnings("ignore")


def main(args):

    # 初始化各进程环境
    init_distributed_mode(args=args)

    "load param"
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers

    model_save_dir = f"save/{args.model}_{args.describe}"
    os.makedirs(model_save_dir, exist_ok=True)

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print("Global learning rate is:", args.lr)
        print(
            'Start Tensorboard with "tensorboard --logdir=save", view at http://localhost:6006/'
        )

        tb_writer = SummaryWriter(model_save_dir)
        print("Using {} dataloader workers every process".format(nw))

        print()
        print()
        # print("loading data...")

    """load data"""
    train_data_set = RT1Dataset(mode="train")
    val_data_set = RT1Dataset(mode="val")

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_sampler=train_batch_sampler,
        pin_memory=True,
        num_workers=nw,
        # shuffle=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data_set,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=nw,
        # shuffle=False,
    )

    "load model"
    vit = MaxViT(
        num_classes=1000,
        dim_conv_stem=64,
        dim=96,
        dim_head=32,
        depth=(2, 2, 5, 2),
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
    )

    model = RT1(
        vit=vit, num_actions=11, depth=6, heads=8, dim_head=64, cond_drop_prob=0.2
    )

    # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

        # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_save_dir_lis = os.listdir(model_save_dir)
    if utils.if_exit_pth(model_save_dir_lis):
        # if it's not empty folder

        pth_file_lis = [x for x in model_save_dir_lis if x.split(".")[-1] == "pth"]
        epoch_lis = [int(y.split(".")[0].split("-")[-1]) for y in pth_file_lis]
        last_epoch = max(epoch_lis)
        args.start_epoch = last_epoch + 1

        # 加载模型状态
        model.load_state_dict(
            torch.load(
                model_save_dir + "/model-{}.pth".format(last_epoch), map_location="cpu"
            )
        )

        # 加载优化器状态
        optimizer.load_state_dict(
            torch.load(
                model_save_dir + "/optimizer-{}.pth".format(last_epoch),
                map_location="cpu",
            )
        )

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.gpu,
        find_unused_parameters=True,
    )

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf)
        + args.lrf
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()

        val_mean_loss = evaluate(model=model, data_loader=val_loader, device=device)

        if rank == 0:
            print("[epoch {}] val_mean_loss: {}".format(epoch, round(val_mean_loss, 3)))
            tags = ["loss", "val_mean_loss", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], val_mean_loss, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(
                model.module.state_dict(),
                model_save_dir + "/model-{}.pth".format(epoch),
            )

            torch.save(
                optimizer.state_dict(),
                model_save_dir + "/optimizer-{}.pth".format(epoch),
            )
            if epoch >= 3:
                # reserve the last three model
                os.remove(model_save_dir + "/model-{}.pth".format(epoch - 3))
                os.remove(model_save_dir + "/optimizer-{}.pth".format(epoch - 3))


    cleanup()

    print(datetime.datetime.now())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model describe
    parser.add_argument("-t", "--model", type=str, default="rt1", help="model type")
    parser.add_argument(
        "-d",
        "--describe",
        type=str,
        default="test",
        help="describe for this model",
    )

    parser.add_argument(
        "-wd", "--weight_decay", type=int, default=1e-3, help="weight_decay"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lrf", type=float, default=0.1)

    """multi-gpus"""
    # 是否启用SyncBatchNorm
    parser.add_argument("--syncBN", type=bool, default=True)
    parser.add_argument("--weights", type=str, default="", help="initial weights path")
    parser.add_argument("--freeze-layers", type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument(
        "--device", default="cuda", help="device id (i.e. 0 or 0,1 or cpu)"
    )
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument(
        "--world-size", default=4, type=int, help="number of distributed processes"
    )

    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()
    # print(f"args: {vars(args)}")

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # if errors occur, kill all processes and release the GPUs memory
    try:
        main(args)
    except:
        cleanup()

# env: rt
# tensorboard --logdir=save --port=6007
# ctrl+shift+P 输入 tensorboard
