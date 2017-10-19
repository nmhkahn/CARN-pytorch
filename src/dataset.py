import os
import glob
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps

def random_crop_pair(hr, size, scale):
    size = size*scale
    w, h = hr.size
    lr_size = int(size/scale)

    x1 = random.randint(0, w-size)
    y1 = random.randint(0, h-size)

    crop_hr = hr.crop((x1, y1, x1+size, y1+size))
    crop_lr = crop_hr.resize((lr_size, lr_size), Image.BICUBIC)

    return crop_hr, crop_lr


def random_flip_pair(im1, im2):
    if random.random() < 0.5:
        im1 = ImageOps.flip(im1)
        im2 = ImageOps.flip(im2)

    if random.random() < 0.5:
        im1 = ImageOps.mirror(im1)
        im2 = ImageOps.mirror(im2)

    return im1, im2


def random_rotate_pair(im1, im2):
    angle = random.choice([0, 90, 180, 270])
    im1 = im1.rotate(angle)
    im2 = im2.rotate(angle)

    return im1, im2


class SRDataset(data.Dataset):
    def __init__(self, path, scale, patch_size):
        super(SRDataset, self).__init__()

        self.scale = scale
        self.patch_size = patch_size
        
        # average size of DIV2K train data
        self.mean_LR_size = (int(1500/self.scale), int(2000/self.scale))
        self.patch_per_im = int(self.mean_LR_size[0]/patch_size * \
                                self.mean_LR_size[1]/patch_size)

        self.hr_list = glob.glob(os.path.join(path, "HR/*.png"))
        self.lr_list = glob.glob(os.path.join(path, "LR_X{}/*.png".format(scale)))

        self.hr_list.sort()
        self.lr_list.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        index = index % len(self.hr_list)
        hr = Image.open(self.hr_list[index])
        
        hr, lr = random_crop_pair(hr, self.patch_size, self.scale)
        hr, lr = random_flip_pair(hr, lr)
        hr, lr = random_rotate_pair(hr, lr)

        return self.transform(hr), self.transform(lr)

    def __len__(self):
        return len(self.hr_list) * self.patch_per_im
        

# TODO: need self ensemble method
class TestDataset(data.Dataset):
    def __init__(self, dirname, scale, self_ensemble=False):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale = scale
        self.hr_list = glob.glob(os.path.join(dirname, "HR/*.png"))
        self.lr_list = glob.glob(os.path.join(dirname, "LR_X{}/*.png".format(scale)))

        self.hr_list.sort()
        self.lr_list.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr_list[index])
        lr = Image.open(self.lr_list[index])

        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        filename = self.hr_list[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr_list)
