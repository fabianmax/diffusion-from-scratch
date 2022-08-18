from inspect import isfunction
from torch import nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)