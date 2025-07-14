import argparse

from Config import cfg
from Config import update_config

from MPAE import MPAE
from Dataloader import CelebA_Dataset, CUB_Dataset, PartImageNet_Dataset, PartImageNet_Seg_Dataset

import torch.nn.functional as F

import torch
import cv2
import numpy as np

import torchvision.transforms as transforms

color_manual = np.array([[255, 0, 0],
                  [0, 255, 0],
                  [0, 0, 255],
                  [255, 0, 255],
                  [0, 255, 255],
                  [255, 255, 0],
                  [128, 128, 0],
                  [128, 0, 128],
                  [0, 128, 128],
                  [64, 128, 64],
                  [64, 192, 128],
                  [0, 192, 64],
                  [64, 0, 0],
                  [128, 192, 128],
                  [0, 64, 192],
                  [64, 64, 128]])
color_random = np.load('Color_board.npz')["Color"]


def parse_args():

    parser = argparse.ArgumentParser(description='Visualize part discovery results')
    parser.add_argument('--Dataset', type=str, default='PartImage_Seg')
    parser.add_argument('--model', type=str, default='./model/PartImage_S_K50.pth')
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

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

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
            cfg, cfg.CUB.ROOT, False,
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
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = args.model
    checkpoint = torch.load(checkpoint_file)
    model.module.load_state_dict(checkpoint)

    model.eval()

    if cfg.MODEL.NUM_PART > 17:
        color = color_random
    else:
        color = color_manual

    with torch.no_grad():
        for i, meta in enumerate(valid_loader):
            input = meta['Img'].cuda()

            score_map = model(input)

            score_map = score_map.cpu().numpy()[0]
            score_map = score_map.transpose(1, 2, 0)
            score_map = np.argmax(score_map, axis=2)

            image = F.interpolate(meta['Img'], size=(128, 128)).cpu().numpy().transpose(0, 2, 3, 1)
            image = ((image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255.0

            image = image[0].copy()

            for idx in range(cfg.MODEL.NUM_PART-1):
                image[score_map == idx] = image[score_map == idx] * 0.5 + 0.5 * color[idx]

            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('test', image)
            cv2.waitKey(0)

if __name__ == '__main__':
    main_function()

