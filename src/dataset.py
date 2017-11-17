import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size]
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize]

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle).copy()
    im2 = np.rot90(im2, angle).copy()

    return im1, im2


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        self.scale = scale

        h5f = h5py.File(path, "r")
        self.hr = [v[:] for v in h5f["HR"].values()]
        self.lr = [v[:] for v in h5f["x{}".format(scale)].values()]
        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr, lr = self.hr[index], self.lr[index]
        hr, lr = random_crop(hr, lr, self.size, self.scale)
        hr, lr = random_flip_and_rotate(hr, lr)

        return self.transform(hr), self.transform(lr)

    def __len__(self):
        return len(self.hr)
        

class TestDataset(data.Dataset):
    def __init__(self, dirname, scale, self_ensemble=False):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale = scale
        
        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join(dirname, "HR/*.png"))
            self.lr = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
        else:
            all_files = glob.glob(os.path.join(dirname, "image_SRF_{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]

        self.hr.sort()
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])

        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
