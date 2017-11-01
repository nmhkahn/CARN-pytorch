import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()
        nFeat = 64

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.relu = nn.ReLU()
        
        self.entry = nn.Conv2d(3, nFeat, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(
            *[ops.EDSRBlock(nFeat, act=self.relu) for _ in range(16)],
            nn.Conv2d(nFeat, nFeat, 3, 1, 1, bias=False)
        )
        self.upsample = ops.UpsampleBlock(nFeat, scale)
        self.exit = nn.Conv2d(nFeat, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        out = self.blocks(x)
        out += x

        out = self.upsample(out)
        out = self.exit(out)
        
        out = self.add_mean(out)

        return out
