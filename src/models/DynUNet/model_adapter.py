"""
MONAI DynUNet adapter for discriminative segmentation.

Wraps MONAI's DynUNet for use with the existing training pipeline while
exposing the interface attributes expected by Diffusion adapters.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig

from monai.networks.nets import DynUNet


class DynUNetAdapter(nn.Module):
    """
    Wrapper for MONAI DynUNet for discriminative segmentation (2D or 3D).

    Key differences from diffusion model adapters:
    - NO timestep conditioning
    - NO mask concatenation
    - Direct segmentation prediction from modalities

    The adapter applies sigmoid to DynUNet outputs so downstream discriminative
    losses can keep using the current `apply_sigmoid: false` defaults.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        model_cfg = cfg.model

        spatial_dims_cfg = getattr(model_cfg, "spatial_dims", None)
        if spatial_dims_cfg is None and hasattr(cfg, "data_mode"):
            spatial_dims_cfg = getattr(cfg.data_mode, "dim", None)

        self.spatial_dims = self._parse_spatial_dims(spatial_dims_cfg)
        self.image_size = int(model_cfg.image_size)
        self.expanded_img_size = self._resolve_img_size(model_cfg.image_size, self.spatial_dims)

        # Attributes required by Diffusion base class
        self.image_channels = int(model_cfg.image_channels)
        self.mask_channels = int(model_cfg.out_channels)
        self.output_channels = int(model_cfg.out_channels)

        kernel_size = self._parse_stage_sequence(model_cfg.kernel_size, "kernel_size")
        strides = self._parse_stage_sequence(model_cfg.strides, "strides")
        upsample_kernel_size = self._parse_stage_sequence(
            model_cfg.upsample_kernel_size, "upsample_kernel_size"
        )

        filters = None
        if hasattr(model_cfg, "filters") and model_cfg.filters is not None:
            filters = self._parse_int_list(model_cfg.filters, "filters")

        self.deep_supervision = bool(model_cfg.get("deep_supervision", False))
        self.deep_supr_num = int(model_cfg.get("deep_supr_num", 1))
        self.res_block = bool(model_cfg.get("res_block", False))
        self.trans_bias = bool(model_cfg.get("trans_bias", False))

        self.model = DynUNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.image_channels,
            out_channels=self.output_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            dropout=model_cfg.get("dropout", None),
            norm_name=model_cfg.get("norm_name", "INSTANCE"),
            act_name=model_cfg.get("act_name", "leakyrelu"),
            deep_supervision=self.deep_supervision,
            deep_supr_num=self.deep_supr_num,
            res_block=self.res_block,
            trans_bias=self.trans_bias,
        )

        self._print_init_info(kernel_size, strides, upsample_kernel_size, filters)

    def _parse_spatial_dims(self, value) -> int:
        """Parse supported 2D/3D spatial dims from config token."""
        if value is None:
            raise ValueError(
                "Missing spatial dims configuration. Expected cfg.model.spatial_dims "
                "or cfg.data_mode.dim."
            )

        token = str(value).strip().lower()
        if token.endswith("d"):
            token = token[:-1]
        if token not in {"2", "3"}:
            raise ValueError(
                f"Invalid spatial dims value '{value}'. Expected one of: 2, 3, '2d', '3d'."
            )
        return int(token)

    def _resolve_img_size(self, value, spatial_dims: int) -> tuple:
        """Expand scalar image_size to a tuple by spatial dims."""
        if isinstance(value, (list, tuple, ListConfig)):
            raise ValueError(
                "DynUNetAdapter expects model.image_size to be a scalar integer. "
                "List/tuple image_size is not supported."
            )

        image_size = int(value)
        if image_size <= 0:
            raise ValueError(f"model.image_size must be > 0, got {image_size}")
        return tuple([image_size] * spatial_dims)

    def _parse_stage_sequence(self, value, field_name: str) -> tuple:
        """
        Parse stage-wise sequence for DynUNet.

        Supports:
        - [3, 3, 3, 3]
        - [[3,3,3], [3,3,3], ...]
        """
        if not isinstance(value, (list, tuple, ListConfig)):
            raise ValueError(
                f"Expected {field_name} to be a sequence, got {type(value)}: {value}"
            )

        parsed = []
        for item in value:
            if isinstance(item, (list, tuple, ListConfig)):
                parsed.append(tuple(int(x) for x in item))
            else:
                parsed.append(int(item))
        return tuple(parsed)

    def _parse_int_list(self, value, field_name: str) -> list:
        """Parse int list (supports ListConfig/list/tuple)."""
        if isinstance(value, (list, tuple, ListConfig)):
            return [int(x) for x in value]
        raise ValueError(
            f"Expected {field_name} to be a list/tuple/ListConfig, got {type(value)}: {value}"
        )

    def _print_init_info(self, kernel_size, strides, upsample_kernel_size, filters):
        """Print initialization information."""
        print("[DynUNetAdapter] Initialized:")
        print(f"  Spatial dims: {self.spatial_dims}")
        print(f"  Image size (base): {self.image_size}")
        print(f"  Image size (expanded): {self.expanded_img_size}")
        print(f"  Input channels: {self.image_channels}")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Kernel size: {kernel_size}")
        print(f"  Strides: {strides}")
        print(f"  Upsample kernel size: {upsample_kernel_size}")
        print(f"  Filters: {filters if filters is not None else 'MONAI default'}")
        print(f"  Deep supervision: {self.deep_supervision}")
        print(f"  Deep supr num: {self.deep_supr_num}")
        print(f"  Residual blocks: {self.res_block}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for discriminative segmentation.

        Args:
            x: Input modalities [B, image_channels, *spatial_shape]

        Returns:
            - Eval / no deep supervision: [B, out_channels, *spatial_shape]
            - Train + deep supervision: [B, S, out_channels, *spatial_shape]
            Values are mapped to [0, 1] via sigmoid.
        """
        logits = self.model(x)
        return torch.sigmoid(logits)
