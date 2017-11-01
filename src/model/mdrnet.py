import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.relu = nn.ReLU()
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(
            *[ops.ResidualBlock(64, 64, dilation=1, act=self.relu) for _ in range(4)],
            *[ops.MDResidualBlock(64, 64, dilation=[2, 4], act=self.relu) for _ in range(4)],
            *[ops.BasicBlock(64, 64, dilation=1, act=self.relu) for _ in range(2)]
        )
        self.upsample = ops.UpsampleBlock(64, scale)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        out = self.blocks(x)
        out += x

        out = self.upsample(out)
        out = self.exit(out)
        
        out = self.add_mean(out)

        return out
