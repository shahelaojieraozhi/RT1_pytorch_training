import sys

from tqdm import tqdm
import torch
from loss_func import SparseCategoricalCrossEntropyLoss

from multi_train_utils.distributed_utils import reduce_value, is_main_process


def loss_calculate(pre, label):
    loss_object = SparseCategoricalCrossEntropyLoss()
    return loss_object(pre, label)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (vid, order, act) in enumerate(data_loader):
        vid = vid.to(device)
        order = list(order[0])
        act_hat = model(vid.to(device), order)
        loss = loss_calculate(act_hat, act.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    val_mean_loss = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (vid, order, act) in enumerate(data_loader):
        vid = vid.to(device)
        order = list(order[0])
        act_hat = model(vid.to(device), order)
        loss = loss_calculate(act_hat, act.to(device))

        loss = reduce_value(loss, average=True)
        val_mean_loss = (val_mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return val_mean_loss.item()






