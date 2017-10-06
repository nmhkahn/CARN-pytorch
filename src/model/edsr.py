from model import common

import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, scale):
        super(EDSR, self).__init__()
        nResBlock = 32
        nFeat = 256
        scale = scale

        self.subMean = common.MeanShift((0.4488, 0.4371, 0.4040))

        # Head convolution for feature extracting
        self.headConv = common.conv3x3(3, nFeat)

        # Main branch
        modules = [common.ResBlock(nFeat) for _ in range(nResBlock)]
        modules.append(common.conv3x3(nFeat, nFeat))
        self.body = nn.Sequential(*modules)

        # Upsampler
        self.upsample = common.Upsampler(scale, nFeat)

        # Tail convolution for reconstruction
        self.tailConv = common.conv3x3(nFeat, 3)

        self.addMean = common.MeanShift((0.4488, 0.4371, 0.4040))

    def forward(self, x):
        x = self.subMean(x)
        x = self.headConv(x)

        res = self.body(x)
        res += x

        us = self.upsample(res)
        output = self.tailConv(us)
        output = self.addMean(output)

        return output
