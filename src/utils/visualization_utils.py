"""
Visualization helpers for preparing 2D TensorBoard panels.

This module intentionally converts 3D tensors to 2D display panels before they
reach logger sinks.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def _resolve_visualization_mode(logging_cfg) -> Tuple[str, bool]:
    mode = str(
        OmegaConf.select(logging_cfg, "visualization.mode", default="triplanar")
        or "triplanar"
    ).strip().lower()
    use_mip = bool(
        OmegaConf.select(logging_cfg, "visualization.use_mip", default=False)
    )
    if mode != "triplanar":
        raise ValueError(
            "Unsupported logging.visualization.mode. "
            f"Expected 'triplanar', got '{mode}'."
        )
    return mode, use_mip


def _resize_2d(slice_2d: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if slice_2d.dim() != 2:
        raise ValueError(
            f"Expected 2D tensor for resize. Got shape={tuple(slice_2d.shape)}."
        )
    resized = F.interpolate(
        slice_2d.unsqueeze(0).unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def _project_triplanar(volume_hwd: torch.Tensor, use_mip: bool) -> torch.Tensor:
    if volume_hwd.dim() != 3:
        raise ValueError(
            f"Expected 3D spatial tensor [H,W,D]. Got shape={tuple(volume_hwd.shape)}."
        )

    height, width, depth = volume_hwd.shape

    if use_mip:
        axial = volume_hwd.max(dim=2).values  # [H, W]
        coronal = volume_hwd.max(dim=1).values  # [H, D]
        sagittal = volume_hwd.max(dim=0).values  # [W, D]
    else:
        center_d = depth // 2
        center_w = width // 2
        center_h = height // 2
        axial = volume_hwd[:, :, center_d]  # [H, W]
        coronal = volume_hwd[:, center_w, :]  # [H, D]
        sagittal = volume_hwd[center_h, :, :]  # [W, D]

    # Display convention: rotate each plane 90 degrees counterclockwise so that
    # triplanar panels match the expected "north-up" orientation in TensorBoard.
    axial = torch.rot90(axial, k=1, dims=(0, 1))
    coronal = torch.rot90(coronal, k=1, dims=(0, 1))
    sagittal = torch.rot90(sagittal, k=1, dims=(0, 1))

    target_hw = tuple(int(v) for v in axial.shape)  # [W, H] after rotation
    coronal = _resize_2d(coronal.float(), target_hw=target_hw)
    sagittal = _resize_2d(sagittal.float(), target_hw=target_hw)
    axial = axial.float()

    return torch.cat([axial, coronal, sagittal], dim=1)  # [H, 3W]


def _normalize_panel_to_minus_one_one(panel_chw: torch.Tensor) -> torch.Tensor:
    if panel_chw.dim() != 3:
        raise ValueError(
            f"Expected panel tensor with shape [C,H,W]. Got {tuple(panel_chw.shape)}."
        )
    panel = panel_chw.detach().float().cpu()
    min_val = panel.min()
    max_val = panel.max()
    if max_val > min_val:
        panel = (panel - min_val) / (max_val - min_val + 1e-8)
    else:
        panel = torch.zeros_like(panel)
    return (panel * 2.0) - 1.0


def _to_single_channel_panel_for_logging(
    tensor: torch.Tensor,
    logging_cfg,
) -> torch.Tensor:
    """
    Convert [1,H,W] or [1,H,W,D] tensor to display panel [1,H,W*{1|3}].
    """
    _, use_mip = _resolve_visualization_mode(logging_cfg)

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Expected single-channel 2D tensor [1,H,W]. "
                f"Got shape={tuple(tensor.shape)}."
            )
        return _normalize_panel_to_minus_one_one(tensor)

    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Expected single-channel 3D tensor [1,H,W,D]. "
                f"Got shape={tuple(tensor.shape)}."
            )
        projected = _project_triplanar(tensor[0], use_mip=use_mip)  # [H, 3W]
        return _normalize_panel_to_minus_one_one(projected.unsqueeze(0))

    raise ValueError(
        "Expected tensor with shape [1,H,W] or [1,H,W,D]. "
        f"Got shape={tuple(tensor.shape)}."
    )


def prepare_discriminative_tensor_panel(
    tensor: torch.Tensor,
    logging_cfg,
) -> torch.Tensor:
    """
    Prepare one discriminative tensor (input/pred/target) as a loggable panel.
    """
    return _to_single_channel_panel_for_logging(
        tensor=tensor,
        logging_cfg=logging_cfg,
    )


def prepare_discriminative_sample_panels(
    sample_img: torch.Tensor,
    sample_pred: torch.Tensor,
    sample_mask: torch.Tensor,
    modality_names: Sequence[str],
    logging_cfg,
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Prepare discriminative logging panels for both 2D and 3D inputs.
    """
    if sample_img.dim() not in {3, 4}:
        raise ValueError(
            "Expected sample_img with shape [C,H,W] or [C,H,W,D]. "
            f"Got shape={tuple(sample_img.shape)}."
        )

    num_modalities = int(sample_img.shape[0])
    if len(modality_names) == num_modalities:
        resolved_modality_names = list(modality_names)
    else:
        resolved_modality_names = [f"Ch{i}" for i in range(num_modalities)]

    panels: List[torch.Tensor] = []
    labels: List[str] = []

    for modality_idx in range(num_modalities):
        modality_panel = prepare_discriminative_tensor_panel(
            tensor=sample_img[modality_idx : modality_idx + 1],
            logging_cfg=logging_cfg,
        )
        panels.append(modality_panel)
        labels.append(f"Modality: {resolved_modality_names[modality_idx]}")

    pred_panel = prepare_discriminative_tensor_panel(
        tensor=sample_pred,
        logging_cfg=logging_cfg,
    )
    target_panel = prepare_discriminative_tensor_panel(
        tensor=sample_mask,
        logging_cfg=logging_cfg,
    )
    panels.append(pred_panel)
    labels.append("Prediction")
    panels.append(target_panel)
    labels.append("Target")
    return panels, labels
