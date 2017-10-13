import torch.nn as nn
from model import ops

class MDRN(nn.Module):
    def __init__(self, scale):
        super(MDRN, self).__init__()
        n_dims = 64
        n_btn_dims = 48
        scale = scale

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, n_dims, 3, 1, 1)
        self.blocks = nn.Sequential(
            *[ops.BtnResBlock(n_dims, n_btn_dims, dilation=1) for _ in range(12)],
            *[ops.BasicBlock(n_dims, dilation=1)],
            *[ops.BtnMDResBlock(n_dims, [32, 16, 16], [24, 16, 16]) for _ in range(12)],
            *[ops.BasicBlock(n_dims, dilation=1)],
            *[ops.BtnResBlock(n_dims, n_btn_dims, dilation=1) for _ in range(12)],
            *[ops.BasicBlock(n_dims, dilation=1) for _ in range(2)],
            nn.Conv2d(n_dims, n_dims, 3, 1, 1),
        )
        self.upsample = ops.UpsampleBlock(n_dims, scale)
        self.exit = nn.Conv2d(n_dims, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)

        mid = self.blocks(x)
        mid += x

        up = self.upsample(mid)
        out = self.exit(up)
        out = self.add_mean(out)

        return out
