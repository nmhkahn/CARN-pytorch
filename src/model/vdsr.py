import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.up = nn.Upsample(scale_factor=scale)
        self.relu = nn.ReLU()
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(
            *[ops.BasicBlock(64, act=self.relu) for _ in range(18)]
        )
        self.upsample = ops.UpsampleBlock(64, scale)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.up(x)
        x = self.sub_mean(x)
        
        out = self.relu(self.entry(x))
        out = self.blocks(out)
        out = self.exit(out)

        out += x
        out = self.add_mean(out)

        return out
