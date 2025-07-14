import copy
import cv2
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

class CUB_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.Mask_ratio = cfg.MODEL.RATIO
        self.is_train = is_train
        self.root = root

        self.Fraction = cfg.CUB.FRACTION
        self.Translation_Factor = cfg.CUB.TRANSLATION
        self.Rotation_Factor = cfg.CUB.ROTATION
        self.Scale_Factor = cfg.CUB.SCALE
        self.Flip = cfg.CUB.FLIP

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


        if is_train:
            self.Image_path = os.path.join(root, 'train_list.txt')
        else:
            self.Image_path = os.path.join(root, 'test_list.txt')
        self.box_path = os.path.join(root, 'bounding_boxes.txt')
        self.landmark_path = os.path.join(root, "parts", "part_locs.txt")
        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.Image_path) as f:
            img_list = f.read().splitlines()
            f.close()

        landmark_dict = {}

        box_dict = {}

        with open(self.landmark_path) as f:
            landmark_list = f.read().splitlines()
            f.close()

        for temp_landmark in landmark_list:
            temp_landmark = temp_landmark.split(' ')
            if temp_landmark[0] not in landmark_dict.keys():
                landmark_dict[temp_landmark[0]] = np.zeros((15, 3), dtype=np.float)
            landmark_dict[temp_landmark[0]][int(temp_landmark[1])-1, :] = \
                np.array([float(temp_landmark[2]), float(temp_landmark[3]), float(temp_landmark[4])], dtype=np.float)

        with open(self.box_path) as f:
            box_list = f.read().splitlines()
            f.close()

        for temp_box in box_list:
            temp_box = temp_box.split(' ')

            box_dict[temp_box[0]] = np.array([float(temp_box[1]), float(temp_box[2]),
                                              float(temp_box[3]), float(temp_box[4])], dtype=np.float)

        for temp_path in img_list:
            temp_path = temp_path.split(' ')
            label = int(temp_path[1].split(".")[0]) - 1
            temp_image_path = os.path.join(self.root, 'images', temp_path[1])
            temp_image_number = temp_path[0]
            Data_base.append({'Img': temp_image_path,
                              'label': label,
                              'box': box_dict[temp_image_number],
                              'landmark': landmark_dict[temp_image_number].copy(),
                              'number': int(temp_image_number)})

        return Data_base

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        landmark = db_slic['landmark']
        box = db_slic['box']
        label = db_slic['label']

        Img = cv2.imread(Img_path)
        Img_shape = Img.shape

        landmark[:,:2] = (landmark[:,:2] * self.Image_size) / np.array([[Img_shape[1], Img_shape[0]]], dtype=np.float)
        box[:2] = (box[:2] * self.Image_size) / np.array([[Img_shape[1], Img_shape[0]]], dtype=np.float)
        box[2:4] = (box[2:4] * self.Image_size) / np.array([[Img_shape[1], Img_shape[0]]], dtype=np.float)

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
                Small_input = F.interpolate(input.unsqueeze(0), size=(256, 256), mode='bilinear')
                Small_input = Small_input.squeeze(0)

            meta = {'Img': input,
                    'Small_input': Small_input,
                    'landmark': landmark,
                    'Img_path': Img_path}

            return meta

        else:
            input = Image.fromarray(Img)

            if self.Transform is not None:
                input = self.Transform(input)
                Small_input = F.interpolate(input.unsqueeze(0), size=(256, 256), mode='bilinear')
                Small_input = Small_input.squeeze(0)

            meta = {
                'Img': input,
                'Small_input': Small_input,
                'label': label,
                'box': box,
                'Img_path': Img_path,
                'landmark': landmark,
            }

            return meta