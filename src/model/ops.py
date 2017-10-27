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
                 n_dims, 
                 dilation=1, 
                 act=nn.LeakyReLU(0.2, True)):
        super(BasicBlock, self).__init__()

        # assume input.shape == output.shape
        pad = dilation

        self.body = nn.Sequential(
            nn.Conv2d(n_dims, n_dims, 3, 1, pad, dilation=dilation, bias=False),
            act
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 n_dims, 
                 dilation=1, 
                 act=nn.LeakyReLU(0.2, True)):
        super(ResidualBlock, self).__init__()

        # assume input.shape == output.shape
        pad = dilation

        self.body = nn.Sequential(
            act,
            nn.Conv2d(n_dims, n_dims, 3, 1, pad, dilation=dilation, bias=False),
            act,
            nn.Conv2d(n_dims, n_dims, 3, 1, pad, dilation=dilation, bias=False),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x) + x
        return out

 
class UpsampleBlock(nn.Module):
    def __init__(self, n_dims, scale, act=False):
        super(UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_dims, 4*n_dims, 3, 1, 1)]
                modules += [nn.PixelShuffle(2)]
                if act:
                    modules += [act]
        elif scale == 3:
            modules += [nn.Conv2d(n_dims, 9*n_dims, 3, 1, 1)]
            modules += [nn.PixelShuffle(3)]
            if act:
                modules += [act]
        self.body = nn.Sequential(*modules)

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
