import os
import math
from typing import Dict, List, Optional

from collections import defaultdict

from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

import torch
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.models.architectures.unet_util import unnormalize_to_zero_to_one
from omegaconf import DictConfig


class BaseOutputFormat:
    """Abstract base class for output formats."""

    def writekvs(self, kvs: Dict[str, float], step: int) -> None:
        raise NotImplementedError


class ConsoleOutput(BaseOutputFormat):
    """Write key-value pairs as a formatted table to stdout."""

    def __init__(self, table_format: str = "simple") -> None:
        self.table_format = table_format

    def writekvs(self, kvs: Dict[str, float], step: int) -> None:  # noqa: D401
        if not kvs:
            return
        rows = [(k, f"{v:.6g}") for k, v in sorted(kvs.items())]
        print("\n===== Logger dump (step", step, ") =====")
        print(tabulate(rows, headers=["key", "value"], tablefmt=self.table_format))
        print("================================\n")


class TensorboardOutput(BaseOutputFormat):
    """Write scalars to TensorBoard."""

    def __init__(self, log_dir: str, writer: Optional[SummaryWriter] = None) -> None:
        self.writer = writer or SummaryWriter(log_dir=log_dir)

    def writekvs(self, kvs: Dict[str, float], step: int) -> None:
        for k, v in kvs.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.close()


class Logger:
    """Aggregates scalars and periodically dumps them to multiple outputs.

    This class is intentionally lightweight; it performs no disk I/O except via
    the chosen OutputFormats (console, tensorboard, etc.).
    """

    def __init__(
        self,
        log_dir: str,
        enabled_outputs: Optional[List[str]] = None,
        log_interval: int = 100,
        table_format: str = "simple",
        writer: Optional[SummaryWriter] = None,
        cfg: Optional[DictConfig] = None,  # For logging configs like font_size
    ) -> None:
        self.cfg = cfg
        self.log_interval = log_interval
        enabled_outputs = enabled_outputs or ["console"]

        self.output_formats: List[BaseOutputFormat] = []
        for out in enabled_outputs:
            if out == "console":
                self.output_formats.append(ConsoleOutput(table_format))
            elif out == "tensorboard":
                self.output_formats.append(TensorboardOutput(log_dir, writer))
            else:
                raise ValueError(f"Unknown logging output '{out}'")

        # mean accumulation buffers
        self.name2val: Dict[str, float] = defaultdict(float)
        self.name2cnt: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public accumulation methods
    # ------------------------------------------------------------------
    def logkv(self, key: str, val: float) -> None:
        """Log an instantaneous value (no mean aggregation)."""
        self.name2val[key] = val
        # ensure it is dumped even with mean logic; cnt=1 marks as non-mean
        self.name2cnt[key] = 1

    def logkv_mean(self, key: str, val: float) -> None:
        """Accumulate a value toward its running mean."""
        old = self.name2val.get(key, 0.0)
        cnt = self.name2cnt.get(key, 0)
        new_val = (old * cnt + float(val)) / (cnt + 1)
        self.name2val[key] = new_val
        self.name2cnt[key] = cnt + 1

    # ------------------------------------------------------------------
    # Diffusion-specific helper
    # ------------------------------------------------------------------
    def logkv_loss_quartiles(
        self,
        diffusion,  # has .num_timesteps
        ts,  # torch.Tensor[int] shape [batch]
        losses: Dict[str, "torch.Tensor"],  # key -> per-sample tensor same shape
    ) -> None:
        import torch  # local import to avoid torch in logger namespace for tools w/o torch

        num_steps = diffusion.num_timesteps
        if num_steps == 0:
            return

        for key, value_tensor in losses.items():
            if value_tensor.numel() == 0:
                continue
            # ensure 1-D same length as ts
            vals = value_tensor.detach().flatten()
            for sub_t, sub_loss in zip(ts.detach().cpu().tolist(), vals.cpu().tolist()):
                quartile = min(3, int(4 * sub_t / num_steps))
                self.logkv_mean(f"{key}_q{quartile}", sub_loss)

    # ------------------------------------------------------------------
    def dumpkvs(self, step: int) -> None:
        kvs = {k: float(v) for k, v in self.name2val.items()}
        for fmt in self.output_formats:
            fmt.writekvs(kvs, step)
        # reset means after dump
        self.name2val.clear()
        self.name2cnt.clear()

    def clear_accumulators(self) -> None:
        """Explicitly clear the accumulation buffers for manual resets."""
        self.name2val.clear()
        self.name2cnt.clear()

    def close(self) -> None:
        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutput):
                fmt.close()

    def log_image_grid(self, tag: str, images: List[torch.Tensor], step: int, metrics: Optional[Dict[int, float]] = None, grid_layout: str = 'horizontal', labels: Optional[List[str]] = None, per_sample_ncol: Optional[int] = None) -> None:
        if not any(isinstance(fmt, TensorboardOutput) for fmt in self.output_formats):
            return  # Skip if TensorBoard not enabled

        # Preprocess: Handle MetaTensor, split multi-channel, normalize (keep 1-channel for grayscale)
        processed_images = []
        label_index = 0  # For assigning labels after splitting
        for img in images:
            if hasattr(img, 'as_tensor'):  # MONAI MetaTensor
                img = img.as_tensor()
            img = unnormalize_to_zero_to_one(img).detach().cpu()
            if img.dim() == 3 and img.shape[0] > 1:  # Multi-channel: split into list of 1-channel
                for c in range(img.shape[0]):
                    processed_images.append(img[c:c+1])
                    label_index += 1
            else:
                processed_images.append(img)  # Keep as-is (1-channel)
                label_index += 1

        # Add labels to each processed image if enabled and provided
        if labels and len(labels) == len(processed_images):
            labeled_images = []
            for idx, p_img in enumerate(processed_images):
                # Convert to PIL (expand to 3 channels temporarily for drawing)
                pil_img = Image.fromarray((p_img.repeat(3,1,1).permute(1,2,0).numpy() * 255).astype('uint8'))
                draw = ImageDraw.Draw(pil_img)
                font_size = self.cfg.label_font_size if self.cfg and hasattr(self.cfg, 'label_font_size') else 14
                
                # Try multiple common truetype fonts for scalability
                font_paths = ["arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "freesansbold.ttf"]
                font = None
                for path in font_paths:
                    if os.path.exists(path):
                        font = ImageFont.truetype(path, size=font_size)
                        break
                if font is None:
                    print("Warning: No truetype font found; using default (size may not apply). Install fonts-dejavu for better results.")
                    font = ImageFont.load_default()
                
                text = labels[idx]
                bbox = draw.textbbox((0, 0), text, font=font)
                text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                # Position: lower_right with offset
                position = (pil_img.width - text_size[0] - 10, pil_img.height - text_size[1] - 10)
                draw.text(position, text, fill=(255, 0, 0), font=font)
                # Convert back to tensor, keeping RGB for color preservation
                labeled_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1) / 255.0  # 3-channel RGB
                labeled_images.append(labeled_tensor)
            processed_images = labeled_images

        # Create grid
        nrow = len(processed_images) if grid_layout == 'horizontal' else 1
        if grid_layout == 'horizontal' and per_sample_ncol:
            nrow = per_sample_ncol  # Group into rows of per_sample_ncol images
        grid = make_grid(processed_images, nrow=nrow, normalize=True)

        # Add text overlays if metrics provided
        if metrics:
            array = ((grid.repeat(3,1,1) if grid.shape[0]==1 else grid[:3].permute(1, 2, 0)).numpy() * 255).astype('uint8')
            pil_grid = Image.fromarray(array)
            draw = ImageDraw.Draw(pil_grid)
            font = ImageFont.load_default()
            for i, metric in metrics.items():
                draw.text((10, 10 + i*20), f"Loss: {metric:.4f}", fill=(255, 0, 0), font=font)
            grid = torch.from_numpy(np.array(pil_grid)).permute(2, 0, 1) / 255.0

        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutput):
                fmt.writer.add_image(tag, grid, step)

    def print_config(self, config_yaml: str, step: int = 0) -> None:
        """Pretty print the configuration YAML string to console and TensorBoard."""
        print("\n===== Resolved Configuration =====")
        print(config_yaml)
        print("================================\n")
        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutput):
                fmt.writer.add_text("Config", config_yaml.replace("\n", "  \n"), step)