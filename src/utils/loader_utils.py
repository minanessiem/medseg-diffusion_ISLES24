"""
Shared loader helper utilities for dataset adapters.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch


class LoaderDataUtils:
    """Static helpers shared across loader-stack implementations."""

    @staticmethod
    def as_int_tuple(
        values: Sequence[Any],
        *,
        expected_len: int,
        field_name: str,
    ) -> Tuple[int, ...]:
        if len(values) != expected_len:
            raise ValueError(
                f"{field_name} must have exactly {expected_len} entries, got {len(values)}."
            )
        return tuple(int(value) for value in values)

    @staticmethod
    def as_float_tuple(
        values: Sequence[Any],
        *,
        expected_len: int,
        field_name: str,
    ) -> Tuple[float, ...]:
        if len(values) != expected_len:
            raise ValueError(
                f"{field_name} must have exactly {expected_len} entries, got {len(values)}."
            )
        return tuple(float(value) for value in values)

    @staticmethod
    def resolve_case_input_paths(
        filedict: Mapping[str, Any],
        *,
        base_modalities: Sequence[str],
        include_label: bool = True,
    ) -> Dict[str, str]:
        keys = list(base_modalities)
        if include_label:
            keys.append("label")

        case_input: Dict[str, str] = {}
        for key in keys:
            value = filedict.get(key)
            if not value:
                continue
            filepath = value[0] if isinstance(value, list) else value
            case_input[key] = str(filepath)
        return case_input

    @staticmethod
    def compute_tensor_data_stats(raw_tensor: torch.Tensor) -> Dict[str, float]:
        raw_np = raw_tensor.detach().cpu().numpy()
        finite_vals = raw_np[np.isfinite(raw_np)]
        if finite_vals.size == 0:
            return {"min_val": 0.0, "max_val": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "min_val": float(np.min(finite_vals)),
            "max_val": float(np.max(finite_vals)),
            "mean": float(np.mean(finite_vals)),
            "std": float(np.std(finite_vals)),
        }

    @staticmethod
    def as_sample_list(samples: Any) -> list[Mapping[str, Any]]:
        if isinstance(samples, Mapping):
            return [samples]
        if isinstance(samples, list):
            return samples
        if isinstance(samples, tuple):
            return list(samples)
        raise ValueError(
            "Expected patch pipeline output to be mapping or list/tuple of mappings, "
            f"got: {type(samples).__name__}."
        )

    @staticmethod
    def is_non_empty(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return len(value.strip()) > 0
        return True

    @staticmethod
    def resolve_path_value(basedir: str, value: object) -> object:
        """
        Resolve relative path values against a base directory.
        """
        if isinstance(value, str) and LoaderDataUtils.is_non_empty(value):
            if os.path.isabs(value):
                return value
            return os.path.join(basedir, value)
        if isinstance(value, list):
            return [LoaderDataUtils.resolve_path_value(basedir, item) for item in value]
        return value

    @staticmethod
    def normalize_case_id(case_id: object, *, field_name: str = "caseID") -> str:
        if isinstance(case_id, list):
            if len(case_id) == 0:
                raise ValueError(f"Record has empty list for '{field_name}'.")
            case_id = case_id[0]
        if not LoaderDataUtils.is_non_empty(case_id):
            raise ValueError(f"Record requires non-empty '{field_name}'.")
        return str(case_id)

    @staticmethod
    def get_base_modality_key(modality_token: str) -> str:
        token = str(modality_token).strip()
        if len(token) == 0:
            raise ValueError("Modality token cannot be empty.")
        return token.split("_", 1)[0]

    @staticmethod
    def parse_modality_token(modality_token: str) -> Tuple[str, str]:
        token = str(modality_token).strip()
        if "_" not in token:
            raise ValueError(
                "Modality token must use '<modality_key>_<preprocessing_key>' format. "
                f"Got '{modality_token}'."
            )
        base_modality, preprocessing_key = token.split("_", 1)
        if not base_modality or not preprocessing_key:
            raise ValueError(
                "Modality token must define both base modality and preprocessing key. "
                f"Got '{modality_token}'."
            )
        return base_modality, preprocessing_key
