"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module
from torch.nn.modules.utils import _pair
from timm.layers import DropBlock2d


__all__ = ["Conv_SplAt"]


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=nn.BatchNorm2d,
                 dropblock_prob=0.0, block_size=7, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            # self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
            #                    groups=groups*radix, bias=bias, **kwargs)
            #

            # SplAt 的“主卷积”换成 深度可分离卷积
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels * radix,
                    kernel_size,
                    stride,
                    padding,
                    groups=in_channels,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels * radix),
                nn.SiLU(),

                nn.Conv2d(
                    in_channels * radix,
                    channels * radix,
                    kernel_size=1,
                    bias=False
                )
            )


        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)

        # 原版
        # self.relu = ReLU(inplace=True)
        self.act = nn.SiLU()
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2d(dropblock_prob, block_size)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.act(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.act(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x



class Conv_SplAt(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, d=1):
        super().__init__()
        self.conv = SplAtConv2d(c1, c2, k, s, autopad(k, p, d), dropblock_prob=0.151, block_size=7)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)  # 1 先做卷积（这里是 SplAtConv2d）
        x = self.bn(x)  # 2再做 BatchNorm 归一化
        x = self.act(x)  # 3最后做激活函数（SiLU）
        return x

# padding=1  norm_layer=nn.BatchNorm2d,