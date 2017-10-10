import torch
import torch.nn as nn
import torch.nn.init as init

from model import common

class Block_A(nn.Module):
    def __init__(self, n_dims, n_bt_dims):
        super(Block_A, self).__init__()
        
        self.conv1 = nn.Conv2d(n_dims, n_bt_dims, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(n_bt_dims)
        self.conv2 = nn.Conv2d(n_bt_dims, n_bt_dims, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(n_bt_dims)
        self.conv3 = nn.Conv2d(n_bt_dims, n_dims, 3, 1, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(n_dims)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
    
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = torch.add(out, identity)

        return self.act(out)


class Block_B(nn.Module):
    def __init__(self, n_dims, n_b_dims, n_b_bt_dims):
        super(Block_B, self).__init__()
        
        # num_pad and num_dilation
        np, nd = [1, 2, 3], [1, 2, 3]
        
        self.conv1_1 = nn.Conv2d(n_dims, n_b_bt_dims[0], 1, 1, 0, bias=False)
        self.bn1_1   = nn.BatchNorm2d(n_b_bt_dims[0])
        self.conv1_2 = nn.Conv2d(n_b_bt_dims[0], n_b_bt_dims[0], 3, 1, np[0], dilation=nd[0], bias=False)
        self.bn1_2   = nn.BatchNorm2d(n_b_bt_dims[0])
        self.conv1_3 = nn.Conv2d(n_b_bt_dims[0], n_b_dims[0], 3, 1, np[0], dilation=nd[0], bias=False)
        self.bn1_3   = nn.BatchNorm2d(n_b_dims[0])
        
        self.conv2_1 = nn.Conv2d(n_dims, n_b_bt_dims[1], 1, 1, 0, bias=False)
        self.bn2_1   = nn.BatchNorm2d(n_b_bt_dims[1])
        self.conv2_2 = nn.Conv2d(n_b_bt_dims[1], n_b_bt_dims[1], 3, 1, np[1], dilation=nd[1], bias=False)
        self.bn2_2   = nn.BatchNorm2d(n_b_bt_dims[1])
        self.conv2_3 = nn.Conv2d(n_b_bt_dims[1], n_b_dims[1], 3, 1, np[1], dilation=nd[1], bias=False)
        self.bn2_3   = nn.BatchNorm2d(n_b_dims[1])

        self.conv3_1 = nn.Conv2d(n_dims, n_b_bt_dims[2], 1, 1, 0, bias=False)
        self.bn3_1   = nn.BatchNorm2d(n_b_bt_dims[2])
        self.conv3_2 = nn.Conv2d(n_b_bt_dims[2], n_b_bt_dims[2], 3, 1, np[2], dilation=nd[2], bias=False)
        self.bn3_2   = nn.BatchNorm2d(n_b_bt_dims[2])
        self.conv3_3 = nn.Conv2d(n_b_bt_dims[2], n_b_dims[2], 3, 1, np[2], dilation=nd[2], bias=False)
        self.bn3_3   = nn.BatchNorm2d(n_b_dims[2])
        
        self.conv4 = nn.Conv2d(n_dims, n_dims, 1, 1, 0, bias=False)
        self.bn4   = nn.BatchNorm2d(n_dims)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        identity = x

        branch1 = self.act(self.bn1_1(self.conv1_1(x)))
        branch1 = self.act(self.bn1_2(self.conv1_2(branch1)))
        branch1 = self.act(self.bn1_3(self.conv1_3(branch1)))

        branch2 = self.act(self.bn2_1(self.conv2_1(x)))
        branch2 = self.act(self.bn2_2(self.conv2_2(branch2)))
        branch2 = self.act(self.bn2_3(self.conv2_3(branch2)))

        branch3 = self.act(self.bn3_1(self.conv3_1(x)))
        branch3 = self.act(self.bn3_2(self.conv3_2(branch3)))
        branch3 = self.act(self.bn3_3(self.conv3_3(branch3)))

        out = torch.cat((branch1, branch2, branch3), dim=1)
        out = self.bn4(self.conv4(out))
        out = torch.add(out, identity)

        return self.act(out)


class MDRN(nn.Module):
    def __init__(self, scale):
        super(MDRN, self).__init__()

        n_dims = 128
        n_bt_dims = 96
        n_b_dims = [32, 64, 32]
        n_b_bt_dims = [24, 48, 24]
        
        self.sub_mean = common.MeanShift((0.4488, 0.4371, 0.4040))
        
        self.entry = nn.Sequential(
            nn.Conv2d(3, n_bt_dims, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_bt_dims, n_bt_dims, 3, 1, 1, bias=False),
            nn.BatchNorm2d(n_bt_dims),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.blocks = nn.Sequential(
            nn.Conv2d(n_bt_dims, n_dims, 3, 1, 1, bias=False),
            nn.BatchNorm2d(n_dims),
            nn.LeakyReLU(0.2, inplace=True),

            Block_A(n_dims, n_bt_dims),
            *[Block_B(n_dims, n_b_dims, n_b_bt_dims) for _ in range(12)],
            Block_A(n_dims, n_bt_dims),
        )
        
        self.exit = nn.Sequential(
            nn.Conv2d(n_dims, n_bt_dims, 3, 1, 1, bias=False),
            nn.BatchNorm2d(n_bt_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_bt_dims, n_bt_dims, 3, 1, 1, bias=False),
            nn.BatchNorm2d(n_bt_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_bt_dims, n_bt_dims, 3, 1, 1, bias=False)
        )

        self.upsample = common.Upsampler(scale, n_bt_dims)

        # Tail convolution for reconstruction
        self.last = nn.Conv2d(n_bt_dims, 3, 3, 1, 1, bias=True)

        self.add_mean = common.MeanShift((0.4488, 0.4371, 0.4040))

    def forward(self, x):
        x = self.sub_mean(x)
        residual = self.entry(x)

        out = self.blocks(residual)
        out = self.exit(out)
        out = torch.add(out, residual)

        out = self.upsample(out)
        out = self.last(out)
        out = self.add_mean(out)

        return out
