import torch
import torch.nn as nn
from model import ops

class MDRN(nn.Module):
    def __init__(self, scale):
        super(MDRN, self).__init__()
        n_dims = 64
        n_branch_dims = [32, 32]
        n_btn_dims = 48
        scale = scale

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, n_dims, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

        self.block1 = nn.Sequential(
            *[ops.BtnResBlock(n_dims, n_btn_dims, dilation=1) for _ in range(16)]
        )
        self.block2 = nn.Sequential(
            *[ops.BtnMDResBlock(n_dims, n_branch_dims) for _ in range(18)],
            *[ops.BasicBlock(n_dims, dilation=1) for _ in range(2)]
        )
        self.block3 = nn.Sequential(
            *[ops.BtnResBlock(n_dims, n_btn_dims, dilation=1) for _ in range(16)],
            *[ops.BasicBlock(n_dims, dilation=1) for _ in range(2)],
        )

        self.combine = nn.Sequential(
            nn.Conv2d(n_dims*3, n_dims, 1, 1, 0),
            *[ops.ResBlock(n_dims, dilation=1) for _ in range(3)],
            nn.Conv2d(n_dims, n_dims, 3, 1, 1)
        )

        self.upsample = ops.UpsampleBlock(n_dims, scale)
        self.recon = nn.Sequential(
            *[ops.ResBlock(n_dims, dilation=1) for _ in range(2)],
            nn.Conv2d(n_dims, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.act(self.entry(x))

        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)

        mid = torch.cat([b1, b2, b3], dim=1)
        mid = self.combine(mid)
        mid += x

        up = self.upsample(mid)
        out = self.recon(up)
        out = self.add_mean(out)

        return out
