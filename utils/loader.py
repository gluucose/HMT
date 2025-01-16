import torch
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import monai.transforms as T
from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa

from utils.util import get_surv_array


class Loader:
    def __init__(self, root, cfg, samples, mode='train'):
        self.root = root
        self.cfg = cfg
        self.samples = samples
        self.mode = mode

        self.transforms_train = T.Compose([
            T.Resize((112, 112, 112)),
            T.ToTensor(),
        ])
        self.transforms_eval = T.Compose([
            T.CenterSpatialCrop(roi_size=112),
            T.Resize((112, 112, 112)),
            T.ToTensor()
        ])

        self.set = MyDataset(root=self.root,
                             tabular=self.cfg.tabular,
                             samples=self.samples,
                             intervals=self.cfg.intervals,
                             mode=mode,
                             transform=self.transforms_train if mode == 'train' else self.transforms_eval,
                             seed=self.cfg.seed)

    def __call__(self):
        is_train = self.mode == 'train'
        return DataLoader(self.set,
                          batch_size=self.cfg.batch_size if is_train else self.cfg.batch_size_eval,
                          shuffle=is_train,
                          num_workers=self.cfg.num_workers,
                          pin_memory=self.cfg.pin_memory,
                          drop_last=is_train)


class MyDataset(Dataset):
    def __init__(self, root, tabular, samples, intervals, mode='train', transform=None, seed=0):
        self.root = root
        self.tabular = tabular
        self.samples = samples
        self.intervals = intervals
        self.mode = mode
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt, ct = torch.zeros(0), torch.zeros(0)
        image_list = []
        
        try:
            pt = nib.load(f'{self.root}/images_preprocessed/{self.samples[idx]}__PT.nii.gz').get_fdata()
            pt = pt[np.newaxis, np.newaxis, ...]
            image_list.append(pt)
        except FileNotFoundError as e:
            pass
        
        ct = nib.load(f'{self.root}/images_preprocessed/{self.samples[idx]}__CT.nii.gz').get_fdata()
        ct = ct[np.newaxis, np.newaxis, ...]
        image_list.append(ct)
        
        seg = nib.load(f'{self.root}/images_preprocessed/{self.samples[idx]}__Seg.nii.gz').get_fdata()
        seg = seg[np.newaxis, np.newaxis, ...]
        seg = np.where(seg == 1, 1, 0)
        image_list.append(seg)
        
        if self.mode == 'train':
            image_list = self.augmentation(image_list)
        if self.transform:
            image_list = [self.transform(np.squeeze(image, axis=0)).float() for image in image_list]
        if len(image_list) == 3:
            pt, ct, seg = image_list
        else:
            ct, seg = image_list
        
        df = pd.read_csv(f'{self.root}/{self.tabular}')
        line = df[df.iloc[:, 0] == self.samples[idx]]
        tabular = torch.from_numpy(line.iloc[:, 1:-2].values).float().squeeze(0)
        time = line.iloc[:, -1:].values
        event = line.iloc[:, -2:-1].values
        surv_array = torch.from_numpy(get_surv_array(time, event, self.intervals)).float().squeeze(0)
        time = torch.from_numpy(time).float().squeeze(0)
        event = torch.from_numpy(event).float().squeeze(0)
        
        return pt, ct, seg, tabular, time, event, surv_array, idx
    
    def augmentation(self, image_list):
        aug_seq = iaa.Sequential([
            iaa.Affine(translate_percent={"x": [-0.1, 0.1], "y": [0, 0]},
                       scale={"x": (0.9, 1.1), "y": (1.0, 1.0)},
                       shear=(-10, 10),
                       rotate=(-10, 10)),
            iaa.CenterCropToFixedSize(width=112, height=None)
            ], random_order=False)
        
        n = len(image_list)
        
        """pre-process data shape"""
        for i in range(n):
            image_list[i] = image_list[i][:, 0, :, :, :]
        
        """flip/translate in x axls, rotate along z axls"""
        images = np.concatenate(image_list, axis=-1)
        images_aug = np.array(aug_seq(images=images))
        
        for i in range(n):
            image_list[i] = images_aug[..., int(images_aug.shape[3]/n)*i:int(images_aug.shape[3]/n)*(i+1)]
            image_list[i] = np.transpose(image_list[i], (0, 3, 1, 2))
        """translate in z axls, rotate along y axls"""
        images = np.concatenate(image_list, axis=-1)
        images_aug = np.array(aug_seq(images=images))
        
        for i in range(n):
            image_list[i] = images_aug[..., int(images_aug.shape[3]/n)*i:int(images_aug.shape[3]/n)*(i+1)]
            image_list[i] = np.transpose(image_list[i], (0, 3, 1, 2))
        """translate in y axls, rotate along x axls"""
        images = np.concatenate(image_list, axis=-1)
        images_aug = np.array(aug_seq(images=images))
        
        """recover axls"""
        for i in range(n):
            image_list[i] = images_aug[..., int(images_aug.shape[3]/n)*i:int(images_aug.shape[3]/n)*(i+1)]
            image_list[i] = np.transpose(image_list[i], (0, 3, 1, 2))
        
        """reset Seg mask to 1/0"""
        for i in range(image_list[-1].shape[0]):
            _, image_list[-1][i] = cv2.threshold(image_list[-1][i], 0.2, 1, cv2.THRESH_BINARY)
        
        """post-process data shape"""
        for i in range(n):
            image_list[i] = image_list[i][..., np.newaxis].transpose((0, 4, 1, 2, 3))
        
        return image_list
