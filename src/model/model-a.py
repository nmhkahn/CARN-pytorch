import torch
import torch.nn as nn
import model.ops as ops
import torch.nn.init as init
from torch.autograd import Variable

def init_weights(modules):
    for m in modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode="fan_out")


class Block(nn.Module):
    def __init__(self, 
                 in_channels, reduce_channels, out_channels,
                 num_groups,
                 act=nn.ReLU()):
        super(Block, self).__init__()

        branch_channels = int(reduce_channels/4)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1, dilation=1, groups=num_groups),
            act,
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 2, dilation=2, groups=num_groups),
            act,
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 3, dilation=3, groups=num_groups),
            act,
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 4, dilation=4, groups=num_groups),
            act,
        )
        
        self.combine = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 1, 1, 0),
            act
        )

        self.act = act
        init_weights(self.modules)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat((b1, b2, b3, b4), dim=1)
        out = self.combine(out)
        out += x
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.relu = nn.ReLU()
        self.entry = nn.Conv2d(3, 96, 3, 1, 1)

        self.layers = nn.Sequential(
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
        
        out = self.layers(x) 
        out = self.upsamplex2(out)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
