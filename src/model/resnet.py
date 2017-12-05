import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.relu = nn.ReLU()
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.body = nn.Sequential(
            *[ops.ResidualBlock(64, 48, 64, group=1) for _ in range(18)],
            ops.BasicBlock(64, 64, dilation=1, act=self.relu)
        )
        
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale, reduce=False)
        
        self.exit = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1)
        )
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        out = self.body(x) 
        out = self.upsample(out, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
