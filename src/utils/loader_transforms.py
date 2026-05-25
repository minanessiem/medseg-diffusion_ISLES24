"""
Reusable data-dict transforms for dataset loader pipelines.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

import torch


class ProcessModalitiesTransform:
    """
    Materialize processed modality tensors in a data dict.

    The concrete modality parsing and processing logic is supplied by callbacks
    so dataset-specific loaders can reuse this transform without changing
    modality contracts.
    """

    def __init__(
        self,
        modalities: Sequence[str],
        *,
        resolve_base_modality: Callable[[str], str],
        process_modality: Callable[[str, torch.Tensor], torch.Tensor],
        processed_prefix: str = "processed_",
    ):
        self.modalities = tuple(str(modality) for modality in modalities)
        self.resolve_base_modality = resolve_base_modality
        self.process_modality = process_modality
        self.processed_prefix = processed_prefix

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        output = dict(data)
        for modality_config in self.modalities:
            base_modality = self.resolve_base_modality(modality_config)
            if base_modality not in output:
                raise KeyError(
                    f"Missing modality key '{base_modality}' during preprocessing."
                )
            raw_tensor = output[base_modality]
            if not isinstance(raw_tensor, torch.Tensor):
                raw_tensor = torch.as_tensor(raw_tensor)
            output[f"{self.processed_prefix}{modality_config}"] = self.process_modality(
                modality_config,
                raw_tensor,
            )
        return output


class MergeProcessedChannelsTransform:
    """
    Merge processed modality tensors into a single `image` tensor and clean keys.
    """

    def __init__(
        self,
        modalities: Sequence[str],
        *,
        resolve_base_modality: Callable[[str], str],
        processed_prefix: str = "processed_",
    ):
        self.modalities = tuple(str(modality) for modality in modalities)
        self.resolve_base_modality = resolve_base_modality
        self.processed_prefix = processed_prefix
        self.base_modalities = tuple(
            sorted(set(self.resolve_base_modality(modality) for modality in self.modalities))
        )

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        output = dict(data)
        processed_channels = [
            output[f"{self.processed_prefix}{modality}"] for modality in self.modalities
        ]
        output["image"] = torch.cat(processed_channels, dim=0)

        cleanup_keys = [
            *[f"{self.processed_prefix}{modality}" for modality in self.modalities],
            *self.base_modalities,
        ]
        for key in cleanup_keys:
            output.pop(key, None)
            output.pop(f"{key}_meta_dict", None)
        return output
