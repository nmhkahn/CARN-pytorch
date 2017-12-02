import os
import random
import time
import numpy as np
import skimage.measure as measure
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset

class Trainer():
    def __init__(self, model, cfg):
        self.refiner = model()
        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.SGD(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr, weight_decay=cfg.wd, momentum=0.9)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.test_data  = [TestDataset(cfg.test_data_dir, scale=i) for i in range(2, 5)]

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        self.test_loader  = [DataLoader(data,
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=False) for data in self.test_data]
        
        self.refiner = self.refiner.cuda()
        self.loss_fn = self.loss_fn.cuda()
        
        self.cfg = cfg
        self.step = 0 
        
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))
        
        t1 = time.time()
        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:
                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]
                
                hr = Variable(hr, requires_grad=False)
                lr = Variable(lr, requires_grad=False)
                hr, lr = hr.cuda(), lr.cuda()
                sr = refiner(lr, scale)
                loss = self.loss_fn(sr, hr)
                
                self.optim.zero_grad()
                loss.backward()
                clip_value = cfg.clip_theta / learning_rate
                nn.utils.clip_grad_norm(self.refiner.parameters(), clip_value)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate
                
                self.step += 1
                if self.step % 1000 == 0:
                    if cfg.verbose:
                        t2 = time.time()
                        remain_step = cfg.max_steps - self.step
                        eta = (t2-t1)*remain_step/1000/3600
                        
                        if cfg.scale > 0:
                            psnr = self.evaluate(scale=cfg.scale)
                            print("[{}K/{}K] {:.2f} ETA: {:.1f} hours".
                                  format(int(self.step/1000), int(cfg.max_steps/1000), psnr, eta))
                        else:    
                            psnr = [self.evaluate(scale=i) for i in range(2, 5)]
                            print("[{}K/{}K] {:.2f} {:.2f} {:.2f} ETA: {:.1f} hours".
                                  format(int(self.step/1000), int(cfg.max_steps/1000), 
                                         psnr[0], psnr[1], psnr[2], eta))
                            
                        t1 = time.time()
        
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break

    def evaluate(self, scale=2):
        cfg = self.cfg
        mean_psnr = 0
        
        test_loader = self.test_loader[scale-2]
        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)
            name = inputs[2]

            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = Variable(lr_patch, volatile=True).cuda()
            
            # run refine process in here!
            sr = self.refiner(lr_patch, scale).data
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).cuda()
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

            # evaluate PSNR
            bnd = 6+scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(self.test_data[0])

        return mean_psnr

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr


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
