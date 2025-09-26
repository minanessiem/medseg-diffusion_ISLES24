import math
import torch
import torch.nn as nn

class TimeSinusoidalPE(nn.Module):
    def __init__(self, dim, out_dim):
        """
        Args:
            dim (int): The dimension of the sinusoidal embedding's space.
            out_dim (int): The dimension of the output after projection.

        Attributes:
            projector (nn.Sequential): Projection layers to map the encoding to the output dimension.
            sin_cos_emb (Tensor): Precomputed sinusoidal embeddings.
        """
        super().__init__()

        half_dim = dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000) / (half_dim - 1)))
        emb = torch.cat((emb.sin(), emb.cos())).view(1, -1)
        self.register_buffer('sin_cos_emb', emb)

        self.projector = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor representing time or a one-dimensional sequence, shape [batch_size].

        Returns:
            Tensor: Projected sinusoidal encoding, shape [batch_size, out_dim].
        """
        x = x.unsqueeze(-1) * self.sin_cos_emb

        return self.projector(x)