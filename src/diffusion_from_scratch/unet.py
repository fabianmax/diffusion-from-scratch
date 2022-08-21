import torch
from torch import nn
from functools import partial

from src.diffusion_from_scratch.utils import default, exists
from src.diffusion_from_scratch.blocks import ConvNextBlock, ResNetBlock
from src.diffusion_from_scratch.embeddings import SinusoidalPositionEmbeddings
from src.diffusion_from_scratch.attention import LinearAttention, Attention


def upsample(dim):
    """ Alias for nn.ConvTranspose2d.
    """
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def downsample(dim):
    """ Alias for nn.Conv2d.
    """
    return nn.Conv2d(dim, dim, 4, 2, 1)


class PreNorm(nn.Module):
    """ Applies group normalization BEFORE a function (e.g. attention layer).
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    """Residual module, just adds the input to the output of the given function.

    Args:
        fn (nn.Module): Function where input is added to output

    Returns:
        torch.Tensor: Output of function
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class UNet(nn.Module):
    """ U-Net implementation as in https://arxiv.org/abs/1505.04597

    Network takes as input a batch of noisy images and noise lvls and should predict the noise added to the input.
    Input: noisy images of shape (batch_size, num_channels, height, width) AND
           batch of noise levels of shape (batch_size, 1)
    Output: Noise prediction of shape (batch_size, num_channels, height, width)

    Args:
        dim (int): Dimension of input and output
        init_dim (int): Dimension of initial convolution
        out_dim (int): Dimension of output convolution
        dim_mults (tumple of int): Multiplicative factors for dimension of each
        channels (int): Number of channels in input and output
        with_time_emb (bool): Whether to use time embeddings
        resnet_block_groups (int): Number of groups for resnet blocks
        use_convnext (bool): Whether to use ConvNextBlock or ResNetBlock
        convnext_mult (int): Multiplicative factor for ConvNextBlock

    Returns:
        torch.Tensor: Output of model
    """
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResNetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):

        # Conv on noisy images
        x = self.init_conv(x)
        # Positional embedding of noise levels
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # Encoder: downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck: ResNet/ConvNext + attention
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Encoder: upsampling
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # Final convolution
        out = self.final_conv(x)

        return out
