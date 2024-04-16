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
from torch.utils.data import DataLoader
from RT1_Dataset import RT1Dataset
from torch.utils.tensorboard import SummaryWriter
from robotic_transformer_pytorch import MaxViT, RT1
from loss_func import SparseCategoricalCrossEntropyLoss

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


def loss_calculate(pre, label):
    loss_object = SparseCategoricalCrossEntropyLoss()
    return loss_object(pre, label)


def seed_torch(seed=666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # To prohibit hash randomization and make the experiment replicable
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, "current_w.pth")
    best_w = os.path.join(model_save_dir, "best_w.pth")
    torch.save(state, current_w)
    if is_best:
        shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, train_dataloader, opt):
    model.train()

    loss_meter, it_count = 0, 0
    for vid, order, act in train_dataloader:
        vid = vid.to(device)
        order = list(order[0])
        act_hat = model(vid, order).cpu()
        optimizer.zero_grad()

        loss = loss_calculate(act_hat, act)

        loss_meter += loss.item()

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()

        it_count += 1
        if it_count != 0 and it_count % opt.show_interval == 0:
            print("%d, loss: %.3e" % (it_count, loss.item()))
            # print("%d, whole loss: %.3e, sbp loss: %.3e, dbp loss: %.3e" % (
            #     it_count, loss.item(), loss_sbp.item(), loss_dbp.item()))

    return loss_meter / it_count


# def val_epoch(model, val_dataloader, opt):
#     model.eval()
#     loss_meter, loss_sbp_meter, loss_dbp_meter, it_count = 0, 0, 0, 0
#     with torch.no_grad():
#         for (ppg, sbp, dbp) in val_dataloader:
#             ppg = ppg.to(device)
#             # ppg = ppg.unsqueeze(dim=0)
#             # ppg = torch.transpose(ppg, 1, 0)
#             ppg = use_derivative(ppg) if opt.using_derivative else ppg
#             bp_hat = model(ppg).cpu()
#             sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
#
#             loss_sbp = loss_calculate(sbp_hat, sbp, opt)
#             loss_dbp = loss_calculate(dbp_hat, dbp, opt)
#
#             loss = loss_dbp + loss_sbp
#             loss_meter += loss.item()
#             loss_sbp_meter += loss_sbp.item()
#             loss_dbp_meter += loss_dbp.item()
#
#             it_count += 1
#
#     return loss_meter / it_count, loss_sbp_meter / it_count, loss_dbp_meter / it_count


def train(opt):
    "load param"
    best_loss = opt.best_loss
    lr = opt.lr
    start_epoch = opt.start_epoch
    stage = opt.stage
    step = opt.decay_step
    weight_decay = opt.weight_decay

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

    model = model.to(device)

    model_save_dir = f'save/{opt.model}_{opt.describe}_{time.strftime("%Y%m%d%H")}'
    os.makedirs(model_save_dir, exist_ok=True)

    """load data"""
    print("loading data...")
    train_data = RT1Dataset(mode="test")
    # val_data = PPG2BPDataset('val')

    train_loader = DataLoader(
        train_data, batch_size=opt.batch, shuffle=True, num_workers=0
    )
    # val_loader = DataLoader(val_data, batch_size=opt.batch, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    states = []

    for epoch in range(start_epoch, opt.n_epochs):
        since = time.time()

        train_loss = train_epoch(model, optimizer, train_loader, opt)
        # val_loss, val_sbp_loss, val_dbp_loss = val_epoch(model, val_loader, opt)

        # print('#epoch: %02d stage: %d train_loss: %.3e val_loss: %0.3e time: %s'
        #       % (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)), end='\n')
        # print('#train_sbp_loss: %.3e train_dbp_loss: %0.3e val_sbp_loss: %.3e val_dbp_loss: %.3e\n'
        #       % (train_sbp_loss, train_dbp_loss, val_sbp_loss, val_dbp_loss))

        print(
            "#epoch: %02d stage: %d train_loss: %.3e time: %s"
            % (epoch, stage, train_loss, utils.print_time_cost(since)),
            end="\n",
        )
        # print('#train_sbp_loss: %.3e train_dbp_loss: %0.3e\n' % (train_sbp_loss, train_dbp_loss))

        # Determine approximate time left
        epoch_done = opt.n_epochs - epoch

        # Print log
        sys.stdout.write("\rETA(left time): %s" % utils.left_time(since, epoch_done))

        writer = SummaryWriter(model_save_dir)
        writer.add_scalar("train_loss", train_loss, epoch)
        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('train_sbp_loss', train_sbp_loss, epoch)
        # # writer.add_scalar('val_sbp_loss', val_sbp_loss, epoch)
        # writer.add_scalar('train_dbp_loss', train_dbp_loss, epoch)
        # writer.add_scalar('val_dbp_loss', val_dbp_loss, epoch)
        writer.close()

        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "lr": lr,
            "stage": stage,
        }
        states.append(state)

        # save_ckpt(state, best_loss > val_loss, model_save_dir)
        # best_loss = min(best_loss, val_loss)

        if epoch in step:
            stage += 1
            lr /= 10

            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

    # torch.save(states, f'./save/resnet18_1D_states.pth')
    print(datetime.datetime.now())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--model", type=str, default="rt1_tf", help="model type")
    parser.add_argument(
        "-d",
        "--describe",
        type=str,
        default="100-episode",
        help="describe for this model",
    )
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=50, help="number of epochs of training"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=8, help="batch size of training"
    )
    parser.add_argument("-bl", "--best_loss", type=int, default=1e3, help="best_loss")
    parser.add_argument("-lr", "--lr", type=int, default=1e-3, help="learning rate")
    parser.add_argument("-se", "--start_epoch", type=int, default=1, help="start_epoch")
    parser.add_argument("-st", "--stage", type=int, default=1, help="stage")
    parser.add_argument(
        "-ds",
        "--decay_step",
        type=list,
        default=[100],
        help="decay step list of learning rate",
    )
    parser.add_argument(
        "-wd", "--weight_decay", type=int, default=1e-3, help="weight_decay"
    )
    parser.add_argument(
        "--show_interval", type=int, default=1, help="how long to show the loss value"
    )
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    train(args)

# tensorboard --logdir=cnn_202305061217 --port=6007
