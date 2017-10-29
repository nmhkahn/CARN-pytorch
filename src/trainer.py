import os
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
        self.refiner = model(cfg.scale)
        self.loss_fn = nn.MSELoss()
        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.test_data  = TestDataset(cfg.test_data_dir, 
                                      scale=cfg.scale)

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        self.test_loader  = DataLoader(self.test_data,
                                       batch_size=1,
                                       num_workers=1,
                                       shuffle=False)
        
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
        while True:
            for inputs in self.train_loader:
                hr = Variable(inputs[0], requires_grad=False)
                lr = Variable(inputs[1], requires_grad=False)

                hr, lr = hr.cuda(), lr.cuda()
                sr = refiner(lr)
                loss = self.loss_fn(sr, hr)
                
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                lr = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = lr
                
                self.step += 1
                if self.step % 1000 == 0:
                    t2 = time.time()
                    remain_step = cfg.max_steps - self.step
                    eta = (t2-t1)*remain_step/1000/3600
                    
                    if cfg.verbose:
                        psnr, ssim = self.evaluate()
                        print("[{}K/{}K] PSNR:{:.2f} SSIM:{:.4f} ETA:{:.1f} hours".
                            format(int(self.step/1000), int(cfg.max_steps/1000), psnr, ssim, eta))
        
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)
                    t1 = time.time()

            if self.step > cfg.max_steps: break
        
        self.save(cfg.ckpt_dir, cfg.ckpt_name)
        if cfg.verbose:
            psnr, ssim = self.evaluate()
            print("[Final] PSNR: {:.2f} SSIM: {:.4f}".format(psnr, ssim))

    def evaluate(self):
        cfg = self.cfg
        mean_psnr, mean_ssim = 0, 0

        for step, inputs in enumerate(self.test_loader):
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
            
            sr = self.refiner(lr_patch).data
            
            h, h_half, h_chop = h*cfg.scale, h_half*cfg.scale, h_chop*cfg.scale
            w, w_half, w_chop = w*cfg.scale, w_half*cfg.scale, w_chop*cfg.scale
            
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
            bnd = 6+cfg.scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]

            mean_psnr += psnr(im1, im2) / len(self.test_data)
            mean_ssim += ssim(im1, im2) / len(self.test_data)

        return mean_psnr, mean_ssim

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
