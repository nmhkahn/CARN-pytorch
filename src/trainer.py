import os
import time
import math
import numpy as np
import scipy.misc as misc
import skimage.color as color
import skimage.measure as measure
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import SRDataset

class Trainer():
    def __init__(self, config):
        if config.model in ["mdrn"]:
            from model.mdrn import MDRN
            self.refiner = MDRN(config.scale)
        elif config.model in ["base"]:
            from model.base import MDRN
            self.refiner = MDRN(config.scale)
        elif config.model in ["mdrn_v2"]:
            from model.mdrn_v2 import MDRN
            self.refiner = MDRN(config.scale)

        self.loss_fn = nn.L1Loss()
        self.opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            config.lr)

        self.train_data = SRDataset(config.train_data_path, 
                                    scale=config.scale, 
                                    patch_size=config.patch_size)
        self.test_data  = SRDataset(config.test_data_path, 
                                    scale=config.scale, 
                                    patch_size=-1, train=False)

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config.batch_size,
                                       num_workers=4,
                                       shuffle=True, drop_last=True)
        self.test_loader  = DataLoader(self.test_data,
                                       batch_size=1,
                                       num_workers=1,
                                       shuffle=False)
        
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.opt, 
            [config.decay], gamma=0.1)

        if config.cuda:
            self.refiner = self.refiner.cuda()
            self.loss_fn = self.loss_fn.cuda()
        
        self.start_epoch = 0 
        self.config = config
        
        if config.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

    def fit(self):
        config = self.config
        num_steps_per_epoch = len(self.train_loader)
        
        if config.num_gpu > 0:
            refiner = nn.DataParallel(self.refiner, 
                                      device_ids=range(config.num_gpu))
        
        for epoch in range(self.start_epoch, config.max_epoch):
            t1 = time.time()
            for step, inputs in enumerate(self.train_loader):
                hr = Variable(inputs[0], requires_grad=False)
                lr = Variable(inputs[1], requires_grad=False)

                if config.cuda:
                    hr, lr = hr.cuda(), lr.cuda()

                sr = refiner(lr)
                loss = self.loss_fn(sr, hr)
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.lr_scheduler.step()

            t2 = time.time()
            remain_epoch = config.max_epoch - epoch
            eta = (t2-t1)*remain_epoch/3600
            
            if config.verbose and (epoch+1)%100 == 0:
                psnr = self.evaluate(True, epoch+1)
                print("[{}/{}] PSNR: {:.3f} ETA:{:.1f} hours".
                    format(epoch+1, config.max_epoch, psnr, eta))
        
                self.save(config.ckpt_dir, config.ckpt_name, epoch)
        
        if config.verbose:
            psnr = self.evaluate(True, epoch+1)
            print("[Final] PSNR: {:.3f}".format(psnr))

    def evaluate(self, save=True, msg=""):
        config = self.config
        mean_psnr = 0
        for step, inputs in enumerate(self.test_loader):
            hr = Variable(inputs[0], volatile=True)
            lr = Variable(inputs[1], volatile=True)

            if config.cuda:
                hr, lr = hr.cuda(), lr.cuda()

            sr = self.refiner(lr)
                      
            if save:
                torchvision.utils.save_image(hr.data,
                    "sample/{}_{}_hr.png".format(msg, step))
                torchvision.utils.save_image(sr.data,
                    "sample/{}_{}_sr.png".format(msg, step))
                
            hr = hr.data.cpu().numpy()
            sr = sr.data.cpu().numpy()
            lr = lr.data.cpu().numpy()
            
            for im1, im2 in zip(hr, sr):
                im1 = np.transpose(im1, (1, 2, 0))
                im2 = np.transpose(im2, (1, 2, 0))
            
                im1 = (np.clip(im1, 0, 1)*255).astype(np.uint8)
                im2 = (np.clip(im2, 0, 1)*255).astype(np.uint8)
                
                im2 = misc.imresize(im2, im1.shape[:-1], "bicubic")

                # remove borderline for benchmark
                im1 = im1[config.scale:-config.scale, config.scale:-config.scale]
                im2 = im2[config.scale:-config.scale, config.scale:-config.scale]

                # calculate PSNR with luminance only
                im1 = color.rgb2ycbcr(im1)[:, :, 0] / 255
                im2 = color.rgb2ycbcr(im2)[:, :, 0] / 255

                mean_psnr += measure.compare_psnr(im1, im2)

        return mean_psnr / len(self.test_data)

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        self.start_epoch = int(path.split(".")[0].split("_")[-1])
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name, epoch):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, epoch+1))
        torch.save(self.refiner.state_dict(), save_path)
