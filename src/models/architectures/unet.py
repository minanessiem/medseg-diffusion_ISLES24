import copy
import torch
import torch.nn as nn

from unet_util import Attention, LinearAttention, Downsample, Upsample, Residual, ResnetBlock, InitWeights_He
from src.models.components.dynamic_encoding import DynamicFusionLayer
from src.models.components.time_encoding import TimeSinusoidalPE
from src.models.components.bottleneck import BottleNeck
from src.utils.general import device_grad_decorator

class Unet(nn.Module):
    def __init__(
        self,
        cfg,
        weight_initializer=InitWeights_He,
        **kwargs,
    ):
        """
        Args:
            cfg (DictConfig): Hydra configuration object with model parameters.
            weight_initializer (Class): A class to initialize the weights of the layers.
            **kwargs: Additional keyword arguments (for compatibility).
        """
        super().__init__()

        # Basic model configuration and parameters
        self.num_layers = cfg.model.num_layers
        self.image_size = cfg.model.image_size
        self.skip_connect_image_feature_maps = cfg.model.skip_connect_image_feature_maps
        self.weight_initializer = weight_initializer
        self.first_conv_channels = cfg.model.first_conv_channels
        self.image_channels = cfg.model.image_channels
        self.mask_channels = cfg.model.mask_channels
        self.output_channels = cfg.model.output_channels
        self.time_embedding_dim = cfg.model.time_embedding_dim
        self.linear_attn_layers = cfg.model.linear_attn_layers or [i < (self.num_layers - 1) for i in range(self.num_layers)] # [T, T, T, F]

        self.mask_initial_conv = nn.Conv2d(self.mask_channels, self.first_conv_channels, 7, padding=3)
        self.img_initial_conv = nn.Conv2d(self.image_channels, self.first_conv_channels, 7, padding=3)


        self.time_embedding_layer = TimeSinusoidalPE(self.first_conv_channels, self.time_embedding_dim)

        attention_heads = cfg.model.att_heads
        attention_head_dim = cfg.model.att_head_dim
        bottleneck_transformer_num_layers = cfg.model.bottleneck_transformer_layers

        channels_num_factors = [int(2 ** i) for i in range(len(self.linear_attn_layers))]
        hidden_channels =  [self.first_conv_channels, *map(lambda m: self.first_conv_channels * m, channels_num_factors)]
        in_out = list(zip(hidden_channels[:-1], hidden_channels[1:]))

        num_resolutions = len(in_out)

        self.dynamic_fusion_layers = nn.ModuleList([])
        self.mask_encoders = nn.ModuleList([])
        self.image_encoders = nn.ModuleList([])


        # This loop constructs each encoder layer with two ResnetBlocks, a Residual Attention module (full or linear),
        # and concludes with a Downsample layer, except for the last layer which uses a Conv2D layer.
        current_feature_map_size = self.image_size
        for i, ((dim_in, dim_out), use_linear) in enumerate(zip(in_out, self.linear_attn_layers)):
            is_last_layer = i == (num_resolutions - 1)

            first_resnet_block = ResnetBlock(dim_in, dim_in, time_embedding_dim=self.time_embedding_dim)
            second_resnet_block = ResnetBlock(dim_in, dim_in, time_embedding_dim=self.time_embedding_dim)

            attention_class = LinearAttention if use_linear else Attention
            attention_configs = dict(
                dim_head=attention_head_dim,
                heads=attention_heads,
            )
            attention = attention_class(dim_in, **attention_configs)

            dynamic_fusion_layer = DynamicFusionLayer(current_feature_map_size, dim_in)
            last_conv_layer = nn.Conv2d(dim_in, dim_out, 3, padding=1) if is_last_layer else Downsample(dim_in, dim_out)

            current_feature_map_size = current_feature_map_size // 2 if not is_last_layer else current_feature_map_size

            self.dynamic_fusion_layers.append(dynamic_fusion_layer)
            self.mask_encoders.append(nn.ModuleList([
                first_resnet_block,
                second_resnet_block,
                Residual(attention),
                last_conv_layer,
            ]))


        self.image_encoders = copy.deepcopy(self.mask_encoders)

        self.bottleneck = BottleNeck(
            hidden_channels[-1],
            self.time_embedding_dim,
            transformer_layers=bottleneck_transformer_num_layers,
            **dict(
                dim_head=attention_head_dim,
                heads=attention_heads,
            ),
        )

        self.decoders = nn.ModuleList([])
        for i, ((dim_in, dim_out), use_linear) in enumerate(zip(reversed(in_out), reversed(self.linear_attn_layers))):
            is_last_layer = i == (num_resolutions - 1)

            skip_connect_dim = dim_in * (2 if self.skip_connect_image_feature_maps else 1)

            first_resnet_block = ResnetBlock(dim_out + skip_connect_dim, dim_out, time_embedding_dim=self.time_embedding_dim)
            second_resnet_block = ResnetBlock(dim_out + skip_connect_dim, dim_out, time_embedding_dim=self.time_embedding_dim)

            attention_class = LinearAttention if use_linear else Attention
            attention_configs = dict(
                dim_head=attention_head_dim,
                heads=attention_heads,
            )
            attention = attention_class(dim_out, **attention_configs)

            last_conv_layer = nn.Conv2d(dim_out, dim_in, 3, padding=1) if is_last_layer else Upsample(dim_out, dim_in)

            self.decoders.append(nn.ModuleList([
                first_resnet_block,
                second_resnet_block,
                Residual(attention),
                last_conv_layer,
            ]))


        self.final_res_block = ResnetBlock(self.first_conv_channels * 2, self.first_conv_channels, time_embedding_dim=self.time_embedding_dim)

        self.final_conv = nn.Conv2d(self.first_conv_channels, self.output_channels, 1)

        # Apply the weight initializer if provided
        if self.weight_initializer is not None:
            self.apply(self.weight_initializer)

    @device_grad_decorator(device=None)
    def forward(self, x, time, image):
        """
        Args:
            x (torch.Tensor): The input mask tensor.
            time (torch.Tensor): The tensor representing time embeddings.
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output tensor after segmentation.
        """
        dtype, skip_connect_image = x.dtype, self.skip_connect_image_feature_maps

        x = self.mask_initial_conv(x)
        image_emb = self.img_initial_conv(image)

        x_clone = x.clone()

        t = self.time_embedding_layer(time)
        skip_connections = []

        # Encoding path: iterate through each layer in mask and image encoders
        for mask_encoder, image_encoder, dynamic_fusion_layer in zip(self.mask_encoders, self.image_encoders, self.dynamic_fusion_layers):
            block_1, block_2, attention, downsample = mask_encoder
            image_block_1, image_block_2, image_attention, image_downsample = image_encoder

            x = block_1(x, t)
            image_emb = image_block_1(image_emb, t)

            skip_connections.append([x, image_emb] if skip_connect_image else [x])

            x = block_2(x, t)
            image_emb = image_block_2(image_emb, t)

            x = attention(x)
            image_emb = image_attention(image_emb)

            image_emb = dynamic_fusion_layer(x, image_emb)

            skip_connections.append([x, image_emb] if skip_connect_image else [x])

            x = downsample(x)
            image_emb = image_downsample(image_emb)

        # Processing through the bottleneck
        x = self.bottleneck(x, image_emb, t)

        # Decoding path: processing through decoders
        for block_1, block_2, attention, upsample in self.decoders:
            x = torch.cat((x, *skip_connections.pop()), dim=1)

            x = block_1(x, t)

            x = torch.cat((x, *skip_connections.pop()), dim=1)

            x = block_2(x, t)
            x = attention(x)
            x = upsample(x)

        x = torch.cat((x, x_clone), dim = 1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)