import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.relu = nn.ReLU()
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(
            *[ops.ResidualBlock(64, 64, act=self.relu) for _ in range(9)]
        )

        #self.upsamplex2 = ops.UpsampleBlock(64, 2)
        #self.upsamplex3 = ops.UpsampleBlock(64, 3)
        self.upsamplex4 = ops.UpsampleBlock(64, 4, reduce=False)

        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
        
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        out = self.blocks(x)
        out += x

        if scale == 2:
            out = self.upsamplex2(out)
        elif scale == 3:
            out = self.upsamplex3(out)
        elif scale == 4:
            out = self.upsamplex4(out)
        
        out = self.exit(out)
        out = self.add_mean(out)

        return out
