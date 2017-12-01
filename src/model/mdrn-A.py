import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.relu = nn.ReLU()
        self.entry = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        
        self.stem = nn.Sequential(
            ops.BasicBlock(64, 64, dilation=1, act=self.relu),
            ops.BasicBlock(64, 64, dilation=1, act=self.relu)
        )

        self.block1 = nn.Sequential(
            *[ops.MDRBlockA(64, 32, 64, dilation=[2, 4], act=self.relu) for _ in range(4)],
            ops.BasicBlock(64, 64, dilation=1, act=self.relu)
        )
        self.block2 = nn.Sequential(
            *[ops.MDRBlockA(64, 32, 64, dilation=[2, 4], act=self.relu) for _ in range(4)],
            ops.BasicBlock(64, 64, dilation=1, act=self.relu)
        )
        self.block3 = nn.Sequential(
            ops.BasicBlock(64, 64, dilation=1, act=self.relu),
            ops.BasicBlock(64, 64, dilation=1, act=self.relu)
        )
        
        self.combine = nn.Sequential(
            ops.BasicBlock(64, 64, dilation=1, act=self.relu),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            self.relu
        )

        self.upsamplex2 = ops.UpsampleBlock(64, 2)
        self.upsamplex3 = ops.UpsampleBlock(64, 3)
        self.upsamplex4 = ops.UpsampleBlock(64, 4)

        self.exit = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1)
        )
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        o1 = self.stem(x)
        o2 = self.block1(o1)
        o3 = self.block2(o2)
        o4 = self.block3(o3)

        out = self.combine(o1+o2+o3+o4) + x

        if scale == 2:
            out = self.upsamplex2(out)
        elif scale == 3:
            out = self.upsamplex3(out)
        elif scale == 4:
            out = self.upsamplex4(out)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
