"""
Time conditioning components for DiffSwinTr.

Contains timestep embedding and modulation utilities adapted from
Facebook Research's DiT (Diffusion Transformer) implementation.

References:
    - DiT: https://github.com/facebookresearch/DiT
    - GLIDE: https://github.com/openai/glide-text2im
"""

import math
import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply adaptive layer normalization modulation.
    
    This is the core operation of AdaLN-Zero conditioning, where learned
    scale and shift parameters modify normalized features.
    
    Args:
        x: Input tensor [B, N, C] or [B, H, W, C]
        shift: Shift parameter [B, C]
        scale: Scale parameter [B, C]
        
    Returns:
        Modulated tensor with same shape as input
        
    Note:
        The operation is: x * (1 + scale) + shift
        Scale is centered at 0 (not 1) so (1 + scale) gives multiplicative factor
    """
    if x.dim() == 3:
        # [B, N, C] format (standard transformer)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif x.dim() == 4:
        # [B, H, W, C] format (Swin transformer NHWC)
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    
    Uses sinusoidal positional encoding (same as Transformer) followed by
    a 2-layer MLP to project to the hidden dimension.
    
    This is copied directly from Facebook's DiT implementation to ensure
    compatibility with proven diffusion model conditioning.
    
    Args:
        hidden_size: Output embedding dimension
        frequency_embedding_size: Dimension of sinusoidal encoding (before MLP)
        
    Example:
        >>> embedder = TimestepEmbedder(hidden_size=256)
        >>> t = torch.tensor([0, 100, 500, 999])
        >>> emb = embedder(t)  # [4, 256]
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, 
        dim: int, 
        max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        This matches the implementation in GLIDE and other diffusion models.
        
        Args:
            t: 1-D Tensor of N timestep indices, one per batch element.
               May be fractional.
            dim: Dimension of the output embedding.
            max_period: Controls the minimum frequency of the embeddings.
            
        Returns:
            Tensor of shape [N, dim] containing positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * 
            torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], 
                dim=-1
            )
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.
        
        Args:
            t: Timestep tensor [B] with integer or float values
            
        Returns:
            Timestep embeddings [B, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
