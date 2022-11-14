from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
import glob
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
import torch.optim as optim
import config
import evaluate
from dataset import heartDataset

from model.proposed import UNet, RecombinationBlock
from model.unet3d import UNet3D
from model.r2unet3d import R2UNet3D
from model.r2attunet3d import R2AttUNet3D
from model.resunet3d import ResUNet3D

from tqdm import tqdm
import metrics
import logger
from collections import OrderedDict
import inference


def to_one_hot_3d(tensor, n_classes=2):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def val(model, val_loader):
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            target = to_one_hot_3d(target.long())
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)
            data = data.transpose(2, 4)
            target = target.transpose(2, 4)
            output = model(data)
            loss = metrics.DiceMeanLoss()(output, target)
            dice0 = metrics.dice(output, target, 0)
            dice1 = metrics.dice(output, target, 1)

            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)

    val_loss /= len(val_loader)
    val_dice0 /= len(val_loader)
    val_dice1 /= len(val_loader)

    return OrderedDict({'Val Loss': val_loss, 'Val dice0': val_dice0,
                        'Val dice1': val_dice1})


def train(model, train_loader, optimizer):
    print("=======Epoch:{}=======".format(epoch))
    model.train()
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    for idx, (data, target) in enumerate(train_loader):
        target = to_one_hot_3d(target.long())
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        data = data.transpose(2, 4)
        target = target.transpose(2, 4)
        output = model(data)
        optimizer.zero_grad()

        # loss = metric_p1.WeightDiceLoss()(output, target)
        loss = metrics.DiceMeanLoss()(output, target)
        # loss = metrics.cross_entropy_3D()(output,target)
        loss.backward()
        optimizer.step()

        train_loss += float(loss)
        train_dice0 += float(metrics.dice(output, target, 0))
        train_dice1 += float(metrics.dice(output, target, 1))
    train_loss /= len(train_loader)
    train_dice0 /= len(train_loader)
    train_dice1 /= len(train_loader)
    print({'Train Loss': train_loss, 'Train dice0': train_dice0,
           'Train dice1': train_dice1})
    return OrderedDict({'Train Loss': train_loss, 'Train dice0': train_dice0,
                        'Train dice1': train_dice1})


if __name__ == '__main__':
    args = config.args
    root = os.getcwd()
    if not os.path.exists(os.path.join('output')):
        os.mkdir(os.path.join('output'))
    if not os.path.exists(os.path.join('output', args.net)):
        os.mkdir(os.path.join('output', args.net))
    save_path = os.path.join("output", args.net,
                             "Adam_DiceMeanLoss_lr_{}_bs_{}".format(str(args.lr).replace(".", ""),
                                                                    args.batch_size))
    print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    # data info
    train_set = heartDataset(args.crop_size, mode="train")
    test_set = heartDataset(args.crop_size, mode="test")

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=6, shuffle=True)
    val_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, num_workers=6, shuffle=True)

    # model info
    if args.net == "proposed":
        model = UNet(1, [16, 32, 48, 64, 96], 2, net_mode='3d', conv_block=RecombinationBlock).to(device)
    elif args.net == 'unet':
        model = UNet3D(in_channels=1, out_channels=2).to(device)
    elif args.net == 'resunet':
        model = ResUNet3D(in_channels=1, out_channels=2).to(device)
    elif args.net == 'r2unet':
        model = R2UNet3D(in_channels=1, out_channels=1, num_class=2).to(device)
    elif args.net == 'r2attunet':
        model = R2AttUNet3D(in_channels=1, out_channels=1, num_class=2).to(device)
    # optimizer infor
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0001)
    #     optimizer = optim.SGD(model.parameters(),lr=args.lr)
    #     optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.8)
    log = logger.Logger(save_path)

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer)
        val_log = val(model, val_loader)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val dice1'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, "best_model.pth"))
            best[0] = epoch
            best[1] = val_log['Val dice1']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
    if not os.path.exists(os.path.join(save_path, 'seg_result')): os.mkdir(os.path.join(save_path, 'seg_result'))
    inference.inference(args, save_path, device, model)
    if not os.path.exists(os.path.join(save_path, '3d_result')): os.mkdir(os.path.join(save_path, '3d_result'))
    evaluate.evaluate(args, save_path)
