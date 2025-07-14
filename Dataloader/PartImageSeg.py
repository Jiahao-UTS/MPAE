import copy
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
from collections import defaultdict
from PIL import Image
import torch
import torchvision.transforms as transforms

from Dataloader import IMAGENET2012_CLASSES
from torch.utils.data import Dataset

class PartImageNet_Seg_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.Mask_ratio = cfg.MODEL.RATIO
        self.is_train = is_train
        self.root = root

        self.Fraction = cfg.PartImageNet.FRACTION
        self.Translation_Factor = cfg.PartImageNet.TRANSLATION
        self.Rotation_Factor = cfg.PartImageNet.ROTATION
        self.Scale_Factor = cfg.PartImageNet.SCALE
        self.Flip = cfg.PartImageNet.FLIP

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
            self.anno_path = os.path.join(root, 'annotations', 'train', 'train.json')
        else:
            self.anno_path = os.path.join(root, 'annotations', 'test', 'test.json')
        self.database = self.get_file_information()

    def _preprocess_annotations(self):
        json_dict = copy.deepcopy(self.coco.dataset)
        for ann in json_dict['annotations']:
            if ann["area"] == 0 or ann["iscrowd"] == 1:
                continue
            for poly_num, seg in enumerate(ann['segmentation']):
                if len(seg) == 4:
                    x1, y1, w, h = ann['bbox']
                    x2 = x1 + w
                    y2 = y1 + h
                    seg_poly = [x1, y1, x1, y2, x2, y2, x2, y1]
                    ann['segmentation'][poly_num] = seg_poly
        self.coco.dataset = copy.deepcopy(json_dict)
        self.coco.createIndex()

    def get_file_information(self):
        # 文件列表
        self.coco = COCO(self.anno_path)
        self._preprocess_annotations()
        self.image_ids = [img_dict['id'] for img_dict in self.coco.imgs.values()]
        # Number of key-points in the dataset (Ground truth parts)
        self.num_kps = len(self.coco.cats)
        # Coarse-grained classes in the dataset
        self.super_categories = list(dict.fromkeys([self.coco.cats[cat]['supercategory'] for cat in self.coco.cats]))
        self.super_categories.sort()
        self.img_id_to_label = {}
        self.image_id_to_name = {}
        self.img_id_to_supercat = {}

        self.class_names = []

        for img_dict in self.coco.imgs.values():
            img_name = img_dict['file_name'].split('\\')[-1]
            # img_name = os.path.basename(img_dict['file_name'])
            class_name_wordnet = img_name.split('_')[0]
            self.class_names.append(IMAGENET2012_CLASSES[class_name_wordnet])

        self.class_names = list(dict.fromkeys(self.class_names))
        self.class_names.sort()
        self.num_classes = len(self.class_names)
        self.class_names_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.class_idx_to_names = {idx: class_name for idx, class_name in enumerate(self.class_names)}

        filtered_img_iterator = 0
        self.filtered_img_id_to_orig_img_id = {}
        self.img_ids_filtered = []
        # Number of instances per class
        self.per_class_count = defaultdict(int)
        for image_id in self.image_ids:
            annIds = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            img_name = self.coco.loadImgs(image_id)[0]['file_name']

            if anns:
                cats = [ann['category_id'] for ann in anns if ann['area'] > 0]
                supercat_img = list(dict.fromkeys([self.coco.cats[cat]['supercategory'] for cat in cats]))[0]
                # class_name_wordnet = os.path.basename(img_name).split('_')[0]
                class_name_wordnet = img_dict['file_name'].split('\\')[-1].split('_')[0]
                class_idx = self.class_names_to_idx[IMAGENET2012_CLASSES[class_name_wordnet]]
                if self.is_train:
                    self.image_id_to_name[filtered_img_iterator] = os.path.join(self.root, 'images', "train",
                                                                                img_name)
                else:
                    self.image_id_to_name[filtered_img_iterator] = os.path.join(self.root, 'images', "test",
                                                                                img_name)
                self.img_ids_filtered.append(filtered_img_iterator)
                self.img_id_to_label[filtered_img_iterator] = class_idx
                self.filtered_img_id_to_orig_img_id[filtered_img_iterator] = image_id
                self.img_id_to_supercat[filtered_img_iterator] = supercat_img
                self.per_class_count[self.class_idx_to_names[class_idx]] += 1
                filtered_img_iterator += 1
        # For top-K loss (class distribution)
        self.cls_num_list = [self.per_class_count[self.class_idx_to_names[idx]] for idx in range(self.num_classes)]

    def __len__(self):
        return len(self.img_ids_filtered)

    def getmasks(self, img_id):
        coco = self.coco
        original_img_id = self.filtered_img_id_to_orig_img_id[img_id]
        anns = coco.imgToAnns[original_img_id]
        img = coco.imgs[original_img_id]
        mask_tensor = torch.zeros(size=(self.num_kps, img['height'], img['width']))
        for i, ann in enumerate(anns):
            if ann["area"] == 0 or ann["iscrowd"] == 1:
                continue
            cat = ann['category_id']
            mask = torch.as_tensor(coco.annToMask(ann), dtype=torch.float32)
            mask_tensor[cat] += mask
        return mask_tensor

    def __getitem__(self, idx):
        img_id = self.img_ids_filtered[idx]
        image_path = self.image_id_to_name[img_id]
        image_path = image_path.replace('\\', '/')
        mask = self.getmasks(img_id)

        Img = cv2.imread(image_path)
        Img_shape = Img.shape

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
                    'Img_path': image_path}
            return meta

        else:
            mask_gt_background = torch.full(size=(1, mask.shape[-2], mask.shape[-1]), fill_value=0.1)
            mask = torch.cat((mask, mask_gt_background), dim=0)
            mask = mask.permute(1, 2, 0).numpy()
            mask = np.argmax(mask, axis=-1)
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask[np.newaxis, :, :]).float()

            input = Image.fromarray(Img)

            if self.Transform is not None:
                input = self.Transform(input)
                Small_input = F.interpolate(input.unsqueeze(0), size=(128, 128), mode='bilinear')
                Small_input = Small_input.squeeze(0)

            meta = {
                'Img': input,
                'Small_input': Small_input,
                'mask': mask,
                'Img_path': image_path,
            }

            return meta