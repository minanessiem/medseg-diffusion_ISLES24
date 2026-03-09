"""Streaming IO producer for diffusion/custom model probability predictions."""

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from scripts.evaluation.contracts import SliceSample
from scripts.evaluation.mask_builder import build_ground_truth_mask
from scripts.evaluation.provenance import parse_diffusion_slice_identity
from src.utils.ensemble import mean_ensemble, soft_staple

BatchType = Any
SampleFn = Callable[[Any, Tensor], Tensor]
CaseDef = Dict[str, object]


def iter_diffusion_slice_samples(
    model: Any,
    dataloader: Iterable[BatchType],
    device: str,
    ensemble_num_samples: int = 1,
    ensemble_method: str = "single",
    staple_max_iters: int = 5,
    staple_tolerance: float = 0.02,
    sample_fn: Optional[SampleFn] = None,
    show_progress: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[SliceSample]:
    """
    Backward-compatible single-case streaming API.
    """
    if ensemble_num_samples <= 0:
        raise ValueError(f"ensemble_num_samples must be > 0, got {ensemble_num_samples}")
    if ensemble_method not in {"single", "mean", "soft_staple"}:
        raise ValueError(
            "ensemble_method must be one of {'single', 'mean', 'soft_staple'}."
        )
    if ensemble_method != "single" and ensemble_num_samples < 2:
        raise ValueError(
            "Ensemble methods 'mean' and 'soft_staple' require ensemble_num_samples >= 2."
        )
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    case = {
        "key": "single_case",
        "method": ensemble_method,
        "num_samples": ensemble_num_samples,
    }
    for _, sample in iter_diffusion_case_slice_samples(
        model=model,
        dataloader=dataloader,
        device=device,
        analysis_cases=[case],
        max_requested_size=ensemble_num_samples,
        staple_max_iters=staple_max_iters,
        staple_tolerance=staple_tolerance,
        sample_fn=sample_fn,
        show_progress=show_progress,
        max_samples=max_samples,
    ):
        yield sample


def iter_diffusion_case_slice_samples(
    model: Any,
    dataloader: Iterable[BatchType],
    device: str,
    analysis_cases: Sequence[CaseDef],
    max_requested_size: int,
    staple_max_iters: int = 5,
    staple_tolerance: float = 0.02,
    sample_fn: Optional[SampleFn] = None,
    show_progress: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[Tuple[str, SliceSample]]:
    """
    Yield `(case_key, SliceSample)` pairs for multiple ensemble analysis cases.

    Sampling is done once per batch up to `max_requested_size`, then each case
    derives its merged prediction from that same sample stack.
    """
    if max_requested_size <= 0:
        raise ValueError("max_requested_size must be > 0.")
    if not analysis_cases:
        raise ValueError("analysis_cases must not be empty.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")
    _validate_cases(analysis_cases, max_requested_size)

    infer = sample_fn or _default_sample_fn
    total_batches = _safe_len(dataloader)
    if show_progress:
        total_str = str(total_batches) if total_batches is not None else "unknown"
        print(
            f"  Starting evaluation inference loop: "
            f"analysis_cases={len(analysis_cases)}, max_samples/input={max_requested_size}, "
            f"batches={total_str}"
        )

    batch_iterable = _wrap_with_progress(dataloader, total_batches, show_progress)
    yielded_base_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(batch_iterable):
            image, mask, paths, metas = _unpack_batch(batch)
            image = image.to(device)

            sample_stack = _generate_sample_stack(
                model=model,
                image=image,
                infer=infer,
                num_samples=max_requested_size,
            )
            predictions_by_case = _merge_predictions_by_case(
                sample_stack=sample_stack,
                analysis_cases=analysis_cases,
                staple_max_iters=staple_max_iters,
                staple_tolerance=staple_tolerance,
            )

            batch_size = sample_stack.shape[1]
            for item_idx in range(batch_size):
                if max_samples is not None and yielded_base_samples >= max_samples:
                    return

                case_id, slice_id, volume_id, slice_index = _resolve_sample_identity(
                    paths, batch_idx, item_idx
                )
                gt_mask = build_ground_truth_mask(mask[item_idx].cpu())

                for case in analysis_cases:
                    case_key = str(case["key"])
                    case_method = str(case["method"])
                    case_num_samples = int(case["num_samples"])
                    pred = predictions_by_case[case_key][item_idx].cpu()
                    sample_meta: Dict[str, object] = {
                        "source": "diffusion_probability",
                        "batch_index": batch_idx,
                        "item_index": item_idx,
                        "ensemble_num_samples": case_num_samples,
                        "ensemble_method": case_method,
                        "case_key": case_key,
                    }
                    extra_meta = _resolve_item_meta(metas, item_idx)
                    if extra_meta is not None:
                        sample_meta.update(extra_meta)
                    yield case_key, SliceSample(
                        case_id=case_id,
                        slice_id=slice_id,
                        volume_id=volume_id,
                        slice_index=slice_index,
                        prediction_prob=pred,
                        ground_truth_mask=gt_mask,
                        metadata=sample_meta,
                    )
                yielded_base_samples += 1


def _generate_sample_stack(
    model: Any,
    image: Tensor,
    infer: SampleFn,
    num_samples: int,
) -> Tensor:
    samples: List[Tensor] = []
    for _ in range(num_samples):
        pred = _normalize_probability_tensor(infer(model, image))
        samples.append(pred)
    return torch.stack(samples, dim=0)  # [N, B, C, H, W]


def _merge_predictions_by_case(
    sample_stack: Tensor,
    analysis_cases: Sequence[CaseDef],
    staple_max_iters: int,
    staple_tolerance: float,
) -> Dict[str, Tensor]:
    merged: Dict[str, Tensor] = {}
    for case in analysis_cases:
        case_key = str(case["key"])
        case_method = str(case["method"])
        case_num_samples = int(case["num_samples"])
        case_samples = sample_stack[:case_num_samples]

        if case_method == "single":
            merged[case_key] = case_samples[0].clamp(0, 1)
        elif case_method == "mean":
            merged[case_key] = mean_ensemble(case_samples).clamp(0, 1)
        elif case_method == "soft_staple":
            merged[case_key] = soft_staple(
                case_samples,
                max_iters=staple_max_iters,
                tolerance=staple_tolerance,
            ).clamp(0, 1)
        else:
            raise ValueError(f"Unsupported case method: {case_method}")
    return merged


def _validate_cases(analysis_cases: Sequence[CaseDef], max_requested_size: int) -> None:
    for case in analysis_cases:
        if "key" not in case or "method" not in case or "num_samples" not in case:
            raise ValueError("Each case must contain key, method, and num_samples.")
        method = str(case["method"])
        num_samples = int(case["num_samples"])
        if method not in {"single", "mean", "soft_staple"}:
            raise ValueError(f"Unsupported case method: {method}")
        if num_samples <= 0:
            raise ValueError(f"Case num_samples must be > 0, got {num_samples}.")
        if num_samples > max_requested_size:
            raise ValueError(
                f"Case num_samples {num_samples} exceeds max_requested_size {max_requested_size}."
            )


def _default_sample_fn(model: Any, image: Tensor) -> Tensor:
    """
    Default inference behavior:
    - uses model.sample(image) when available (diffusion-style)
    - otherwise falls back to model(image)
    """
    if hasattr(model, "sample") and callable(model.sample):
        return model.sample(image, disable_tqdm=True)
    return model(image)


def _normalize_probability_tensor(prediction: Tensor) -> Tensor:
    """
    Normalize predictions to [0, 1] probabilities.

    If values already appear within [0, 1], clamp small overflows.
    Otherwise apply sigmoid (logit-style output).
    """
    pred = prediction.detach().float()
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    return pred.clamp(0, 1)


def _unpack_batch(
    batch: BatchType,
) -> Tuple[Tensor, Tensor, Optional[Sequence[str]], Optional[Sequence[Dict[str, object]]]]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(
            "Batch must be tuple/list like (image, mask, [path])."
        )
    image = batch[0]
    mask = batch[1]
    paths = batch[2] if len(batch) > 2 else None
    metas = batch[3] if len(batch) > 3 else None
    return image, mask, paths, metas


def _resolve_item_meta(
    metas: Optional[object],
    item_idx: int,
) -> Optional[Dict[str, object]]:
    if metas is None:
        return None
    if isinstance(metas, dict):
        item_meta: Dict[str, object] = {}
        for key, value in metas.items():
            try:
                if isinstance(value, (list, tuple)):
                    item_meta[str(key)] = value[item_idx]
                else:
                    item_meta[str(key)] = value
            except Exception:
                continue
        return item_meta
    if item_idx >= len(metas):
        return None
    item_meta = metas[item_idx]
    if not isinstance(item_meta, dict):
        return None
    return dict(item_meta)


def _resolve_sample_identity(
    paths: Optional[Sequence[str]],
    batch_idx: int,
    item_idx: int,
) -> Tuple[str, str, Optional[str], Optional[int]]:
    if paths is not None and item_idx < len(paths):
        raw = str(paths[item_idx])
        volume_id, slice_index = parse_diffusion_slice_identity(raw)
        case_id = volume_id
        slice_id = f"{volume_id}_slice{slice_index}"
        return case_id, slice_id, volume_id, slice_index
    fallback = f"batch{batch_idx}_item{item_idx}"
    return fallback, fallback, None, None


def _safe_len(iterable: Iterable[Any]) -> Optional[int]:
    """Return len(iterable) when available, else None."""
    try:
        return len(iterable)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _wrap_with_progress(
    dataloader: Iterable[BatchType],
    total_batches: Optional[int],
    show_progress: bool,
) -> Iterable[BatchType]:
    """Wrap iterable with tqdm progress bar when enabled."""
    if not show_progress:
        return dataloader
    return tqdm(
        dataloader,
        total=total_batches,
        desc="Evaluating validation batches",
        leave=True,
    )

