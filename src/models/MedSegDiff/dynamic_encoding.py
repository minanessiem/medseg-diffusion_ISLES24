import torch
import torch.nn as nn

from .unet_util import LayerNorm, ResnetBlock

class AttentionLikeMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        """
        Args:
            dim (int): The number of features in the input tensors, typically the channel dimension.

        Attributes:
            layer_norm_i (LayerNorm): Layer normalization for the image feature tensor.
            layer_norm_x (LayerNorm): Layer normalization for the input segmentation mask tensor.
        """
        self.layer_norm_i = LayerNorm(dim, bias=True)
        self.layer_norm_x = LayerNorm(dim, bias=True)

    def forward(self, x, image):
        """
        Args:
            x (Tensor): The noisy segmentation mask at step t and layer l of the U-Net's encoder with shape [batch_size, channels, height, width].
            image (Tensor): The embedding of the image combined with feature maps from a previous fusion layer with shape [batch_size, channels, height, width].

        Returns:
            Tensor: The refined segmentation mask, having the same shape as the input tensors with shape [batch_size, channels, height, width].
        """
        normed_x = self.layer_norm_x(x)
        normed_i = self.layer_norm_i(image)

        return (normed_x * normed_i) * normed_i

class FFParser(nn.Module):
    def __init__(self, feature_map_size, dim):
        """
        Args:
            feature_map_size (int): The size of the spatial dimensions (height and width) of the feature maps.
            dim (int): The number of channels in the feature maps.

        Attributes:
            ff_parser_attn_map (Parameter): A learnable tensor that serves as the attentive map in the Fourier space. Initialized to ones.
        """
        super().__init__()
        self.ff_parser_attn_map = torch.nn.Parameter(torch.ones(dim, feature_map_size, feature_map_size))

    def forward(self, x):
        """
        Args:
            x (Tensor): The input feature map tensor with shape [batch_size, channels, height, width].

        Returns:
            Tensor: The processed feature map, having the same shape as the input, with high-frequency components modulated.
        """
        dtype = x.dtype

        x = torch.fft.fft2(x) * self.ff_parser_attn_map

        x = torch.fft.ifft2(x).real.type(dtype)
        return x

class DynamicFusionLayer(nn.Module):
    def __init__(self, feature_map_size, dim):
        """
        Args:
            feature_map_size (int): The size of the spatial dimensions (height and width) of the feature maps.
            dim (int): The number of channels in the feature maps.
        Attributes:
            ff_parser (FFParser): Module to modulate high-frequency components in the feature map.
            attention (AttentionLikeMechanism): Module to apply an attention-like mechanism for feature enhancement.
            block (ResnetBlock): A ResNet block for additional feature processing and refinement.
        """
        super().__init__()
        self.ff_parser = FFParser(feature_map_size, dim)
        self.attention = AttentionLikeMechanism(dim)
        self.block = ResnetBlock(dim, dim)

    def forward(self, x, image):
        """
        Args:
            x (Tensor): The input segmentation map tensor with shape [batch_size, channels, height, width].
            image (Tensor): The prior image embedding tensor with shape [batch_size, channels, height, width].

        Returns:
            Tensor: The output of the Dynamic Fusion Layer, representing the enhanced and refined feature map.
        """
        x = self.ff_parser(x)

        attention_out = self.attention(x, image)

        return self.block(attention_out)
