import os
import time
import numpy as np
import skimage.measure as measure
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import SRDataset, TestDataset

def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr


def ssim(im1, im2):
    ssim = measure.compare_ssim(im1, im2, 
                                K1=0.01, K2=0.03,
                                gaussian_weights=True, 
                                sigma=1.5,
                                use_sample_covariance=False,
                                multichannel=True)

    return ssim


class Trainer():
    def __init__(self, config):
        if config.model in ["mdrn"]:
            from model.mdrn import MDRN
            self.refiner = MDRN(config.scale)
        elif config.model in ["base"]:
            from model.base import MDRN
            self.refiner = MDRN(config.scale)
        elif config.model in ["mdrn_multi"]:
            from model.mdrn_multi import MDRN
            self.refiner = MDRN(config.scale)
        elif config.model in ["mdrn_multi_v2"]:
            from model.mdrn_multi_v2 import MDRN
            self.refiner = MDRN(config.scale)

        self.loss_fn = nn.L1Loss()
        self.opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            config.lr)

        self.train_data = SRDataset(config.train_data_path, 
                                    scale=config.scale, 
                                    patch_size=config.patch_size)
        self.test_data  = TestDataset(config.test_data_path, 
                                      scale=config.scale)

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config.batch_size,
                                       num_workers=4,
                                       shuffle=True, drop_last=True)
        
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
            refiner = nn.DataParallel(self.refiner, device_ids=range(config.num_gpu))
        
        for epoch in range(self.start_epoch, config.max_epoch):
            t1 = time.time()
            psnr, ssim = self.evaluate()
            print("{:.4f}, {:.4f}".format(psnr, ssim))
			
            lr = self.decay_learning_rate(epoch)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr

            for step, inputs in enumerate(self.train_loader):
                hr = Variable(inputs[0], requires_grad=False)
                lr = Variable(inputs[1], requires_grad=False)

                if config.cuda:
                    hr, lr = hr.cuda(), lr.cuda()

                sr = refiner(lr)
                loss = self.loss_fn(sr, hr)
                
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), config.clip) 
                self.opt.step()

            t2 = time.time()
            remain_epoch = config.max_epoch - epoch
            eta = (t2-t1)*remain_epoch/3600
            
            if config.verbose:
                psnr, ssim = self.evaluate()
                print("[{}/{}] PSNR: {:.4f} SSIM: {:.4f} ETA:{:.1f} hours".
                    format(epoch+1, config.max_epoch, psnr, ssim, eta))
        
                self.save(config.ckpt_dir, config.ckpt_name, epoch)
        
        if config.verbose:
            psnr, ssim = self.evaluate()
            print("[Final] PSNR: {:.4f} SSIM: {:.4f}".format(psnr, ssim))

    def evaluate(self):
        config = self.config
        mean_psnr, mean_ssim = 0, 0

        for step, (hr, lr, name) in enumerate(self.test_data):
            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + config.shave, w_half + config.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = Variable(lr_patch, volatile=True).cuda()
            
            sr = self.refiner(lr_patch).data
            
            h, h_half, h_chop = h*config.scale, h_half*config.scale, h_chop*config.scale
            w, w_half, w_chop = w*config.scale, w_half*config.scale, w_chop*config.scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).cuda()
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

            # evaluate PSNR and SSIM
            bnd = 6+config.scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]

            mean_psnr += psnr(im1, im2) / len(self.test_data)
            mean_ssim += ssim(im1, im2) / len(self.test_data)

        return mean_psnr, mean_ssim

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.start_epoch = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.start_epoch = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name, epoch):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, epoch+1))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self, epoch):
        lr = self.config.lr * (0.1 ** (epoch // self.config.step))
        return lr
