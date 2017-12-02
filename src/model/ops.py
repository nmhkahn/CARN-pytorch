import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

def init_weights(modules):
    for m in modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode="fan_out")


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, dilation=1, 
                 act=nn.ReLU()):
        super(BasicBlock, self).__init__()

        # assume input.shape == output.shape
        pad = dilation

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, 1, pad, dilation=dilation, bias=False),
            act
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, reduce_channels, out_channels,
                 dilation=1, group=1,
                 act=nn.ReLU()):
        super(ResidualBlock, self).__init__()

        # assume input.shape == output.shape
        pad = dilation

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, 1, 1, 0),
            act,
            nn.Conv2d(reduce_channels, reduce_channels, 3, 1, pad, groups=group, dilation=dilation),
            act,
            nn.Conv2d(reduce_channels, out_channels, 1, 1, 0),
        )

        if not in_channels == out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self.act = act
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        if getattr(self, "identity", None):
            x = self.identity(x)
        out = self.act(out + x)
        return out
       
 
class MDRBlock(nn.Module):
    def __init__(self, 
                 in_channels, reduce_channels, out_channels,
                 num_groups, dilation,
                 act=nn.ReLU()):
        super(MDRBlock, self).__init__()

        branch_channels = int(reduce_channels/4)

        pad = [d for d in dilation]
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, pad[0], dilation=dilation[0], groups=num_groups),
            act,
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, pad[1], dilation=dilation[1], groups=num_groups),
            act,
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, pad[2], dilation=dilation[2], groups=num_groups),
            act,
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0),
            act,
            nn.Conv2d(branch_channels, branch_channels, 3, 1, pad[3], dilation=dilation[3], groups=num_groups),
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


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, reduce=True, act=nn.ReLU()):
        super(UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                if reduce:
                    modules += [nn.Conv2d(n_channels, 4*n_channels, 1, 1, 0), act]
                else:
                    modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1), act]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            if reduce:
                modules += [nn.Conv2d(n_channels, 9*n_channels, 1, 1, 0), act]
            else:
                modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1), act]
            modules += [nn.PixelShuffle(3)]
        self.body = nn.Sequential(*modules)

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
