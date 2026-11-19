import torch
from torch import nn
from .my_module import DSConv
from .conv import Conv


class Dami_Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, ratio=0.25):
        super().__init__()

        assert 0.0 < ratio < 1.0

        # semantic branch channels
        self.c1 = int(c2 * ratio)

        # detail branch channels
        self.c2 = c2 - self.c1

        # channel adjust
        self.cv1 = Conv(c1, c2, k=1, s=1)

        # branch1 (main semantic branch)
        self.branch1 = DSConv(self.c1, self.c1, k=3, s=2)

        # branch2 (cheap detail branch)
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # feature fusion
        self.fuse = Conv(c2, c2, k=1, s=1, act=False)

        # shortcut
        self.shortcut = nn.Sequential(
            Conv(c1, c2, k=1, s=2, act=False)
        )

        # activation
        self.act = nn.SiLU()

    def forward(self, x):

        # residual
        identity = self.shortcut(x)

        # channel adjust
        x = self.cv1(x)

        # unequal split
        x1, x2 = torch.split(x, [self.c1, self.c2], dim=1)

        # semantic branch
        y1 = self.branch1(x1)

        # detail branch
        y2 = self.branch2(x2)

        # concat
        y = torch.cat((y1, y2), dim=1)

        # fuse
        y = self.fuse(y)

        # residual add
        y = y + identity

        # activation
        y = self.act(y)

        return y