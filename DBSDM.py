import torch
from torch import nn
from .my_module import *
from .conv import Conv

def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()

    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)

    return x

class Dami_Conv2(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.c = c2 // 2
        self.branch1 = DSConv(self.c, self.c, k=3, s=2)
        self.branch2_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = Conv(c2, c2, k=1, s=1)
        self.shortcut = Conv(c1, c2, k=1, s=2)
        self.act = nn.SiLU()
    def forward(self, x):
        identity = self.shortcut(x)
        x = channel_shuffle(x, 2)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        y1 = self.branch1(x1)
        y2 = self.branch2_pool(x2)
        y = torch.cat((y1, y2), dim=1)
        y = self.fuse(y)
        y = y + identity
        y = self.act(y)
        return y
