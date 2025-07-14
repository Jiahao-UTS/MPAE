import argparse

from Config import cfg, update_config

from MPAE import MPAE
from Dataloader import CelebA_Dataset, CUB_Dataset, PartImageNet_Dataset, PartImageNet_Seg_Dataset
from tools import validate_CUB, validate_PartImage, validate_CelebA
from backbone import Vgg19


import torch


import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Mask Part AutoEncoder')
    parser.add_argument('--Dataset', type=str, default='CUB')
    parser.add_argument('--model', type=str, default='./model/CUB_K16.pth')
    parser.add_argument('--Num_Part', type=int, default=17)
    parser.add_argument('--Epoch', type=int, default=70)
    parser.add_argument('--LR_step', type=list, default=[50, 60])
    args = parser.parse_args()

    return args


def main_function():
    args = parse_args()
    update_config(cfg, args)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.DATASET.DATASET == 'CUB':
        model = MPAE(cfg.MODEL.NUM_PART, cfg.MODEL.OUT_DIM, 4.0,
                     cfg.MODEL.IMG_SIZE, 8, cfg.MODEL.RATIO, cfg)
    elif cfg.DATASET.DATASET == 'PartImage':
        model = MPAE(cfg.MODEL.NUM_PART, cfg.MODEL.OUT_DIM, 8.0,
                     cfg.MODEL.IMG_SIZE, 64, cfg.MODEL.RATIO, cfg)
    elif cfg.DATASET.DATASET == 'PartImage_Seg':
        model = MPAE(cfg.MODEL.NUM_PART, cfg.MODEL.OUT_DIM, 8.0,
                     cfg.MODEL.IMG_SIZE, 64, cfg.MODEL.RATIO, cfg)
    elif cfg.DATASET.DATASET == 'CelebA':
        model = MPAE(cfg.MODEL.NUM_PART, cfg.MODEL.OUT_DIM, 8.0,
                     cfg.MODEL.IMG_SIZE, 8, cfg.MODEL.RATIO, cfg)
    else:
        raise NotImplementedError

    vgg = Vgg19()
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    vgg = torch.nn.DataParallel(vgg, device_ids=cfg.GPUS).cuda()

    checkpoint_file = args.model
    checkpoint = torch.load(checkpoint_file)
    model.module.load_state_dict(checkpoint)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if cfg.DATASET.DATASET == 'CelebA':
        valid_dataset = CelebA_Dataset(
            cfg, cfg.CELE.ROOT,  False,
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif cfg.DATASET.DATASET == 'CUB':
        valid_dataset = CUB_Dataset(
            cfg, cfg.CUB.ROOT,  False,
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif cfg.DATASET.DATASET == 'PartImage':
        valid_dataset = PartImageNet_Dataset(
            cfg, cfg.PartImageNet.ROOT,  False,
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif cfg.DATASET.DATASET == 'PartImage_Seg':
        valid_dataset = PartImageNet_Seg_Dataset(
            cfg, cfg.PartImageNet.Seg_ROOT,  False,
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise NotImplementedError


    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    if cfg.DATASET.DATASET == 'CUB':
        perf_indicator = validate_CUB(
            cfg, valid_loader, model, vgg
        )
    elif cfg.DATASET.DATASET == 'PartImage' or cfg.DATASET.DATASET == 'PartImage_Seg':
        perf_indicator = validate_PartImage(
            cfg, valid_loader, model, vgg
        )
    elif cfg.DATASET.DATASET == 'CelebA':
        perf_indicator = validate_CelebA(
            cfg, valid_loader, model, vgg
        )
    else:
        raise NotImplementedError

    print("The NMI on the entire dataset is {}".format(perf_indicator))





if __name__ == '__main__':
    main_function()
