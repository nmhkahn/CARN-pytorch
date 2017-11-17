import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()
        
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.relu = nn.ReLU()
        self.entry = nn.Conv2d(3, 64, 3, 1, 1, bias=False)

        self.block1 = nn.Sequential(
            *[ops.ResidualBlock(64, 64, dilation=1, act=self.relu) for _ in range(4)]
        )
        self.block2 = nn.Sequential(
            *[ops.ResidualBlock(64, 64, dilation=2, act=self.relu) for _ in range(2)],
            ops.BasicBlock(64, 64, dilation=1, act=self.relu)
        )
        self.block3 = nn.Sequential(
            *[ops.ResidualBlock(64, 64, dilation=4, act=self.relu) for _ in range(2)],
            ops.BasicBlock(64, 64, dilation=1, act=self.relu)
        )

        self.combine = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            self.relu
        )
        self.upsample = ops.UpsampleBlock(64, scale)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        
        out = self.combine(b1 + b2 + b3) + x

        out = self.upsample(out)
        out = self.exit(out)
        
        out = self.add_mean(out)

        return out
