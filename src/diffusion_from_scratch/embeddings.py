import math
import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """ Position embeddings as in Vaswani et al. 2017.

    Helps to keep track of the time (therefore noice lvl) since parameters as shared across time steps.
    Takes tensor of shape [batch_size, 1] as input and returns tensor of shape [batch_size, dim].

    Args:
        dim (int): Dimension of the embeddings

    Returns:
        torch.Tensor: Position embeddings
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
