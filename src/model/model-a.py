import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.relu = nn.ReLU()
        self.entry = nn.Conv2d(3, 96, 3, 1, 1)

        self.body = nn.Sequential(
            *[ops.ResidualBlock(96, 64, 96, group=16) for _ in range(18)],
            ops.BasicBlock(96, 96, dilation=1, act=self.relu)
        )
        
        self.upsamplex2 = ops.UpsampleBlock(96, 2, reduce=True)
        self.exit = nn.Sequential(
            nn.Conv2d(96, 3, 3, 1, 1)
        )
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        out = self.body(x) 
        out = self.upsamplex2(out)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
