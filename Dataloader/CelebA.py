import copy

import cv2
import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


class CelebA_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root

        self.Fraction = cfg.CELE.FRACTION
        self.Translation_Factor = cfg.CELE.TRANSLATION
        self.Rotation_Factor = cfg.CELE.ROTATION
        self.Scale_Factor = cfg.CELE.SCALE
        self.Flip = cfg.CELE.FLIP

        if not is_train:
            aug_Transform = [
                transforms.Resize((self.Image_size, self.Image_size), antialias=True),
                ]
            self.Transform = transforms.Compose(aug_Transform + transform)
        else:
            aug_Transform = [
                transforms.Resize(self.Image_size, antialias=True),
                transforms.RandomHorizontalFlip(p=self.Flip),
                transforms.ColorJitter(),
                transforms.RandomAffine(degrees=self.Rotation_Factor, translate=(self.Translation_Factor, self.Translation_Factor),
                                        scale=(1.0 - self.Scale_Factor, 1.0 + self.Scale_Factor)),
                transforms.RandomCrop(self.Image_size)
                ]
            self.Transform = transforms.Compose(aug_Transform + transform)

        # 获取标注文件路径
        if is_train:
            self.annotation_file = os.path.join(root, 'cele_train_lm.txt')
        else:
            self.annotation_file = os.path.join(root, 'MAFL_test_lm.txt')

        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_info = temp_info.split(',')

            temp_name = os.path.join(self.root, 'img_celeba', temp_info[0])
            points = np.array([float(temp_info[1]), float(temp_info[2]), float(temp_info[3]), float(temp_info[4]),
                               float(temp_info[5]), float(temp_info[6]), float(temp_info[7]), float(temp_info[8]),
                               float(temp_info[9]), float(temp_info[10])])
            points = points.reshape((5, 2))

            Data_base.append({'Img': temp_name,
                              'points': points})

        return Data_base

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        points = db_slic['points']

        # 读取图片
        Img = cv2.imread(Img_path)
        Img_shape = Img.shape


        Img = cv2.resize(Img, (self.Image_size, self.Image_size))
        points[:, 0] = points[:, 0] / Img_shape[1] * self.Image_size
        points[:, 1] = points[:, 1] / Img_shape[0] * self.Image_size

        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if self.is_train == True:
            input = Image.fromarray(Img)
            if self.Transform is not None:
                input = self.Transform(input)
                Small_input = F.interpolate(input.unsqueeze(0), size=(128, 128), mode='bilinear')
                Small_input = Small_input.squeeze(0)

            meta = {'Img': input,
                    'Small_input': Small_input,
                    'Img_path': Img_path}

            return meta

        else:
            input = Image.fromarray(Img)

            if self.Transform is not None:
                input = self.Transform(input)
                Small_input = F.interpolate(input.unsqueeze(0), size=(128, 128), mode='bilinear')
                Small_input = Small_input.squeeze(0)

            meta = {
                'Img': input,
                'points': points,
                'Small_input': Small_input,
                'Img_path': Img_path,
            }

            return meta