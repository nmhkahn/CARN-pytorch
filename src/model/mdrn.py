from model import common

import torch.nn as nn

class MDRN(nn.Module):
    def __init__(self, scale):
        super(MDRN, self).__init__()
        nResBlock = 33
        nFeat = 256
        nBTFeat = 192
        scale = scale

        self.subMean = common.MeanShift((0.4488, 0.4371, 0.4040))

        # Head convolution for feature extracting
        self.headConv = common.conv3x3(3, nFeat)

        # Main branch
        modules = []
        modules += [common.BTResBlock(nFeat, nBTFeat, dilation=1) for _ in range(10)]
        modules += [common.BasicModule(256) for _ in range(2)]
        modules += [common.BTResBlock(nFeat, nBTFeat, dilation=2) for _ in range(5)]
        modules += [common.BasicModule(256) for _ in range(2)]
        modules += [common.BTResBlock(nFeat, nBTFeat, dilation=2) for _ in range(5)]
        modules += [common.BasicModule(256) for _ in range(2)]
        modules += [common.BTResBlock(nFeat, nBTFeat, dilation=1) for _ in range(10)]
        modules += [common.BasicModule(256) for _ in range(4)]
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
