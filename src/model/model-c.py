import torch
import torch.nn as nn
import model.ops as ops
import torch.nn.init as init
from torch.autograd import Variable

def init_weights(modules):
    for m in modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode="fan_out")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.relu = nn.ReLU()
        self.entry = nn.Conv2d(3, 96, 3, 1, 1)

        self.layers = nn.Sequential(
            *[ops.MDRBlock(96, 64, 96, 4, dilation=[1, 1, 1, 1], act=self.relu) for _ in range(6)],
            *[ops.MDRBlock(96, 64, 96, 4, dilation=[2, 2, 2, 2], act=self.relu) for _ in range(6)],
            *[ops.MDRBlock(96, 64, 96, 4, dilation=[4, 4, 4, 4], act=self.relu) for _ in range(6)],
            ops.BasicBlock(96, 96, dilation=1, act=self.relu)
        )
        
        self.upsamplex2 = ops.UpsampleBlock(96, 2, reduce=True)
        self.exit = nn.Sequential(
            nn.Conv2d(96, 3, 3, 1, 1)
        )
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        out = self.layers(x) 
        out = self.upsamplex2(out)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
