import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.relu = nn.ReLU()
        self.entry = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        
        self.stem = nn.Sequential(
            *[ops.ResidualBlock(64, 64, dilation=1, act=self.relu) for _ in range(2)]
        )

        # call each block 4 times
        self.block1 = nn.Sequential(
            *[ops.MDRBlockB(64, 32, 64, dilation=[2, 4], act=self.relu) for _ in range(2)],
            # ops.BasicBlock(64, 64, dilation=1, act=self.relu) # is it necessary?
        )
        self.block2 = nn.Sequential(
            *[ops.MDRBlockB(64, 32, 64, dilation=[2, 4], act=self.relu) for _ in range(2)],
        )
        self.block3 = nn.Sequential(
            *[ops.MDRBlockB(64, 32, 64, dilation=[2, 4], act=self.relu) for _ in range(2)],
        )
        self.block4 = nn.Sequential(
            *[ops.BasicBlock(64, 64, dilation=1, act=self.relu) for _ in range(2)]
        )
        
        self.combine = nn.Sequential(
            ops.BasicBlock(64, 64, dilation=1, act=self.relu),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False), self.relu
        )
        self.upsample = ops.UpsampleBlock(64, scale)
        
        """
        # call this block 3 times
        self.up_block1 = nn.Sequential(
            *[ops.ResidualBlock(64, 64, dilation=2, act=self.relu) for _ in range(2)]
        )
        """

        self.exit = nn.Sequential(
            ops.BasicBlock(64, 64, dilation=1, act=self.relu),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        
        stem = self.stem(x)
        
        call = 3

        b1 = stem
        for _ in range(call):
            b1 = self.block1(b1)

        b2 = b1
        for _ in range(call):
            b2 = self.block2(b2)

        b3 = b2
        for _ in range(call):
            b3 = self.block3(b3)

        b4 = b3
        for _ in range(call):
            b4 = self.block4(b4)

        up = self.combine(b1+b2+b3+b4) + x
        up = self.upsample(up)
        
        """
        up_b1 = up
        for _ in range(call):
            up_b1 = self.up_block1(up_b1)

        out = self.exit(up_b1)
        """

        out = self.exit(up)
        out = self.add_mean(out)

        return out
