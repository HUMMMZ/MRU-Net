import os
import time
import datetime

import torch
from src import UNet, NNet, RNNet
from src import swin_tiny_patch4_window7_224 as create_model
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_data_set import DriveDataset
import transforms as T
import numpy as np


class SegmentationPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=[0.248, 0.248, 0.248], std=[0.108, 0.108, 0.108]):
        # min_size = int(0.5 * base_size)
        # max_size = int(1.2 * base_size)

        self.transforms = T.Compose([
            # T.RandomResize(min_size, max_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, img, target):
        img, target = self.transforms(img, target)
        return img, target


class SegmentationPresetEval:
    def __init__(self, mean=[0.248, 0.248, 0.248], std=[0.108, 0.108, 0.108]):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, img, target):
        img, target = self.transforms(img, target)
        return img, target


def get_transform(train, mean=[0.248, 0.248, 0.248], std=[0.108, 0.108, 0.108]):
    # base_size = 780
    crop_size = 512

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = RNNet(in_channels=3, num_classes=num_classes, base_c=64)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    mean = [0.248, 0.248, 0.248]
    std = [0.108, 0.108, 0.108]

    # 用来保存训练以及验证过程中信息
    logs_path = "save_weights/two_loss/TN3K_RNNet_54.txt"
    results_file = logs_path.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std),
                                 flag="train")

    test_dataset = DriveDataset(args.data_path,
                                train=False,
                                transforms=get_transform(train=False, mean=mean, std=std),
                                flag="test")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True)

    # create_model --> Unet
    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.eval:
        # 测试模式
        confmat_test, dice_test = evaluate(model, test_loader, device=device, num_classes=num_classes)

        train_info = f"dice test: {dice_test:.3f}\n" \
                     f"confmat test:\n {confmat_test}\n"
        print(train_info)
    else:
        # 训练模式
        best_dice = 0.
        min_loss = 100.
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

            confmat_test, dice_test = evaluate(model, test_loader, device=device, num_classes=num_classes)
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice test: {dice_test:.3f}\n" \
                         f"confmat test: {confmat_test}\n"
            print(train_info)
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                f.write(train_info + "\n\n")

            if args.save_best is True:
                if best_dice < dice_test:
                    best_dice = dice_test

            if mean_loss < min_loss:
                min_loss = mean_loss
                args.save_best = True
            else:
                args.save_best = False

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            if args.save_best is True:
                torch.save(save_file, "save_weights/cross/TN3K_RNnet_54.pth")
            else:
                torch.save(save_file, "save_weights/cross/model_{}.pth".format(epoch))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="Data root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    parser.add_argument('--eval', default=False, type=bool, help='eval mode')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
