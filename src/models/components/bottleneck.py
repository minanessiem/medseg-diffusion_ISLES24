import torch.nn as nn

from unet_util import ResnetBlock, Transformer

class BottleNeck(nn.Module):
    def __init__(self, dim, time_embedding_dim, transformer_layers, groups=8, **attention_configs):
        """
        Args:
            dim (int): The number of channels in the feature maps.
            time_embedding_dim (int): The dimension of the time embedding.
            groups (int, optional): The number of groups for grouped convolutions in ResnetBlocks. Defaults to 8.
            **transformer_configs: Additional configurations for the Transformer layer.

        Attributes:
            conv_mask (ResnetBlock): ResNet block for processing the mask.
            conv_img (ResnetBlock): ResNet block for processing the image.
            transformer_fuser (Transformer): Transformer layer for feature fusion.
            conv_fuser (ResnetBlock): ResNet block for processing the fused features.
        """
        super().__init__()

        self.conv_mask = ResnetBlock(dim, dim, groups=groups, time_embedding_dim=time_embedding_dim)
        self.conv_img = ResnetBlock(dim, dim, groups=groups, time_embedding_dim=time_embedding_dim)

        self.transformer_fuser = Transformer(dim, depth=transformer_layers, **attention_configs)

        self.conv_fuser = ResnetBlock(dim, dim, groups=groups, time_embedding_dim=time_embedding_dim)

    def forward(self, x, image, time_embedding):
        """
        Args:
            x (Tensor): The input mask tensor with shape [batch_size, channels, height, width].
            image (Tensor): The input image tensor with shape [batch_size, channels, height, width].
            time_embedding (Tensor): The time embedding tensor with shape [batch_size, time_embedding_dim].

        Returns:
            Tensor: The output tensor representing fused and refined features, with shape [batch_size, channels, height, width].
        """
        x = self.conv_mask(x, time_embedding)
        image = self.conv_img(image, time_embedding)

        x = x + image

        x = self.transformer_fuser(x)

        x = self.conv_fuser(x, time_embedding)
        return x