"""
Generic deep-supervision utilities for discriminative segmentation losses.

This module centralizes:
1. Model-output parsing into canonical head tensors.
2. Supervision-schedule validation and resolution.
3. Generic per-term/per-head loss aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf

from .segmentation_losses import (
    BCELoss,
    DiceFocalLoss,
    DiceLoss,
    FocalLoss,
    GeneralizedDiceLoss,
    HausdorffDTLoss,
    TverskyLoss,
)


LossFactory = Callable[[Mapping[str, Any]], nn.Module]


@dataclass(frozen=True)
class DiscriminativeTermSpec:
    """Normalized discriminative loss-term definition."""

    loss_key: str
    input_domain: str
    weight: float
    params: Dict[str, Any]
    supervision: Dict[str, Any]


@dataclass(frozen=True)
class SupervisionPlan:
    """Resolved head/weight supervision plan for one term."""

    heads: Tuple[int, ...]
    weights: Tuple[float, ...]


@dataclass
class DiscriminativeDeepSupervisionResult:
    """Result payload returned by deep-supervision loss aggregation."""

    total_loss: torch.Tensor
    loss_components: Dict[str, float]
    final_prediction: torch.Tensor
    head_predictions: Dict[int, torch.Tensor]


def _require_param(params: Mapping[str, Any], key: str, loss_key: str) -> Any:
    """Return a required loss parameter or raise a config error."""

    if key not in params:
        raise ValueError(f"Loss term '{loss_key}' requires params.{key}.")
    return params[key]


def _build_dice_loss(params: Mapping[str, Any]) -> nn.Module:
    return DiceLoss(
        smooth=float(_require_param(params, "smooth", "DiceLoss")),
        apply_sigmoid=bool(_require_param(params, "apply_sigmoid", "DiceLoss")),
    )


def _build_bce_loss(params: Mapping[str, Any]) -> nn.Module:
    return BCELoss(
        pos_weight=_require_param(params, "pos_weight", "BCELoss"),
        apply_sigmoid=bool(_require_param(params, "apply_sigmoid", "BCELoss")),
    )


def _build_hausdorff_dt_loss(params: Mapping[str, Any]) -> nn.Module:
    return HausdorffDTLoss(
        alpha=float(_require_param(params, "alpha", "HausdorffDTLoss")),
        include_background=bool(
            _require_param(params, "include_background", "HausdorffDTLoss")
        ),
        to_onehot_y=bool(_require_param(params, "to_onehot_y", "HausdorffDTLoss")),
        sigmoid=bool(_require_param(params, "sigmoid", "HausdorffDTLoss")),
        softmax=bool(_require_param(params, "softmax", "HausdorffDTLoss")),
        reduction=str(_require_param(params, "reduction", "HausdorffDTLoss")),
        batch=bool(_require_param(params, "batch", "HausdorffDTLoss")),
    )


def _build_tversky_loss(params: Mapping[str, Any]) -> nn.Module:
    return TverskyLoss(
        include_background=bool(_require_param(params, "include_background", "TverskyLoss")),
        to_onehot_y=bool(_require_param(params, "to_onehot_y", "TverskyLoss")),
        sigmoid=bool(_require_param(params, "sigmoid", "TverskyLoss")),
        softmax=bool(_require_param(params, "softmax", "TverskyLoss")),
        alpha=float(_require_param(params, "alpha", "TverskyLoss")),
        beta=float(_require_param(params, "beta", "TverskyLoss")),
        reduction=str(_require_param(params, "reduction", "TverskyLoss")),
        smooth_nr=float(_require_param(params, "smooth_nr", "TverskyLoss")),
        smooth_dr=float(_require_param(params, "smooth_dr", "TverskyLoss")),
        batch=bool(_require_param(params, "batch", "TverskyLoss")),
        soft_label=bool(_require_param(params, "soft_label", "TverskyLoss")),
    )


def _build_focal_loss(params: Mapping[str, Any]) -> nn.Module:
    return FocalLoss(
        include_background=bool(_require_param(params, "include_background", "FocalLoss")),
        to_onehot_y=bool(_require_param(params, "to_onehot_y", "FocalLoss")),
        gamma=float(_require_param(params, "gamma", "FocalLoss")),
        alpha=_require_param(params, "alpha", "FocalLoss"),
        weight=_require_param(params, "weight", "FocalLoss"),
        reduction=str(_require_param(params, "reduction", "FocalLoss")),
        use_softmax=bool(_require_param(params, "use_softmax", "FocalLoss")),
    )


def _build_dice_focal_loss(params: Mapping[str, Any]) -> nn.Module:
    return DiceFocalLoss(
        include_background=bool(
            _require_param(params, "include_background", "DiceFocalLoss")
        ),
        to_onehot_y=bool(_require_param(params, "to_onehot_y", "DiceFocalLoss")),
        sigmoid=bool(_require_param(params, "sigmoid", "DiceFocalLoss")),
        softmax=bool(_require_param(params, "softmax", "DiceFocalLoss")),
        squared_pred=bool(_require_param(params, "squared_pred", "DiceFocalLoss")),
        jaccard=bool(_require_param(params, "jaccard", "DiceFocalLoss")),
        reduction=str(_require_param(params, "reduction", "DiceFocalLoss")),
        smooth_nr=float(_require_param(params, "smooth_nr", "DiceFocalLoss")),
        smooth_dr=float(_require_param(params, "smooth_dr", "DiceFocalLoss")),
        batch=bool(_require_param(params, "batch", "DiceFocalLoss")),
        gamma=float(_require_param(params, "gamma", "DiceFocalLoss")),
        weight=_require_param(params, "weight", "DiceFocalLoss"),
        lambda_dice=float(_require_param(params, "lambda_dice", "DiceFocalLoss")),
        lambda_focal=float(_require_param(params, "lambda_focal", "DiceFocalLoss")),
        alpha=_require_param(params, "alpha", "DiceFocalLoss"),
    )


def _build_generalized_dice_loss(params: Mapping[str, Any]) -> nn.Module:
    return GeneralizedDiceLoss(
        include_background=bool(
            _require_param(params, "include_background", "GeneralizedDiceLoss")
        ),
        to_onehot_y=bool(_require_param(params, "to_onehot_y", "GeneralizedDiceLoss")),
        sigmoid=bool(_require_param(params, "sigmoid", "GeneralizedDiceLoss")),
        softmax=bool(_require_param(params, "softmax", "GeneralizedDiceLoss")),
        w_type=str(_require_param(params, "w_type", "GeneralizedDiceLoss")),
        reduction=str(_require_param(params, "reduction", "GeneralizedDiceLoss")),
        smooth_nr=float(_require_param(params, "smooth_nr", "GeneralizedDiceLoss")),
        smooth_dr=float(_require_param(params, "smooth_dr", "GeneralizedDiceLoss")),
        batch=bool(_require_param(params, "batch", "GeneralizedDiceLoss")),
        soft_label=bool(_require_param(params, "soft_label", "GeneralizedDiceLoss")),
    )


DISCRIMINATIVE_LOSS_FACTORIES: Dict[str, LossFactory] = {
    "DiceLoss": _build_dice_loss,
    "BCELoss": _build_bce_loss,
    "HausdorffDTLoss": _build_hausdorff_dt_loss,
    "TverskyLoss": _build_tversky_loss,
    "FocalLoss": _build_focal_loss,
    "DiceFocalLoss": _build_dice_focal_loss,
    "GeneralizedDiceLoss": _build_generalized_dice_loss,
}


def _to_python_config(cfg: Any) -> Any:
    """Convert OmegaConf containers to native Python containers."""

    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _ensure_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    value = _to_python_config(value)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {field_name} to be a mapping, got {type(value)}")
    return value


def _ensure_list(value: Any, field_name: str) -> List[Any]:
    value = _to_python_config(value)
    if not isinstance(value, list):
        raise ValueError(f"Expected {field_name} to be a list, got {type(value)}")
    return value


def normalize_discriminative_head_outputs(
    model_output: Any,
    head_parser: str = "auto",
) -> Dict[int, torch.Tensor]:
    """
    Normalize model output to a canonical head-indexed dictionary.

    Supported parsers:
    - "single": model_output is one tensor [B, C, ...] -> {0: tensor}
    - "stacked": model_output is [B, S, C, ...] -> {i: tensor[:, i, ...]}
    - "list": model_output is list/tuple[Tensor] -> {i: tensor_i}
    - "auto": infer parser from model_output type/shape
    """

    parser = str(head_parser).strip().lower()
    if parser not in {"auto", "single", "stacked", "list"}:
        raise ValueError(
            f"Unsupported head_parser='{head_parser}'. "
            "Expected one of: auto, single, stacked, list."
        )

    if parser == "auto":
        if isinstance(model_output, (list, tuple)):
            parser = "list"
        elif torch.is_tensor(model_output) and model_output.dim() >= 5:
            # [B,S,C,H,W] (2D) or [B,S,C,H,W,D] (3D)
            parser = "stacked"
        else:
            parser = "single"

    if parser == "single":
        if not torch.is_tensor(model_output):
            raise ValueError(
                "head_parser='single' requires model_output to be a torch.Tensor."
            )
        return {0: model_output}

    if parser == "stacked":
        if not torch.is_tensor(model_output):
            raise ValueError(
                "head_parser='stacked' requires model_output to be a torch.Tensor."
            )
        if model_output.dim() < 5:
            raise ValueError(
                "head_parser='stacked' expects shape [B,S,C,...] "
                f"with dim >= 5, got shape {tuple(model_output.shape)}."
            )
        num_heads = int(model_output.shape[1])
        if num_heads < 1:
            raise ValueError("Parsed stacked model_output with zero heads.")
        return {i: model_output[:, i, ...] for i in range(num_heads)}

    # parser == "list"
    if not isinstance(model_output, (list, tuple)):
        raise ValueError(
            "head_parser='list' requires model_output to be list/tuple of tensors."
        )
    if len(model_output) == 0:
        raise ValueError("head_parser='list' received an empty output sequence.")

    heads: Dict[int, torch.Tensor] = {}
    for i, head_tensor in enumerate(model_output):
        if not torch.is_tensor(head_tensor):
            raise ValueError(
                f"head_parser='list' expects tensor elements. "
                f"Element {i} is {type(head_tensor)}."
            )
        heads[i] = head_tensor
    return heads


def resolve_discriminative_terms(
    discriminative_cfg: Mapping[str, Any],
) -> List[DiscriminativeTermSpec]:
    """Resolve explicit discriminative loss-term definitions."""

    cfg = _ensure_mapping(discriminative_cfg, "discriminative")
    legacy_fields = [field for field in ("dice", "bce") if field in cfg]
    if legacy_fields:
        raise ValueError(
            "Legacy discriminative loss fields are not supported. "
            f"Remove {legacy_fields} and define loss.discriminative.terms explicitly."
        )

    raw_terms = cfg.get("terms", None)
    if raw_terms is None:
        raise ValueError(
            "loss.discriminative.terms is required. "
            "Legacy discriminative.dice/discriminative.bce fields are not supported."
        )

    terms: List[DiscriminativeTermSpec] = []
    seen_loss_keys = set()
    for idx, raw_term in enumerate(_ensure_list(raw_terms, "discriminative.terms")):
        term = _ensure_mapping(raw_term, f"discriminative.terms[{idx}]")
        required_fields = {"loss", "input_domain", "weight", "params", "supervision"}
        unsupported_fields = sorted(set(term.keys()) - required_fields)
        if unsupported_fields:
            raise ValueError(
                f"discriminative.terms[{idx}] has unsupported field(s): "
                f"{unsupported_fields}. Supported fields: {sorted(required_fields)}."
            )

        missing_fields = [field for field in required_fields if field not in term]
        if missing_fields:
            raise ValueError(
                f"discriminative.terms[{idx}] is missing required field(s): "
                f"{missing_fields}."
            )

        loss_key = str(term["loss"]).strip()
        if not loss_key:
            raise ValueError(f"discriminative.terms[{idx}].loss must not be empty.")
        if loss_key in seen_loss_keys:
            raise ValueError(
                f"Duplicate discriminative loss term '{loss_key}' is not supported. "
                "Add alias support before configuring repeated loss classes."
            )
        seen_loss_keys.add(loss_key)

        input_domain = str(term["input_domain"]).strip().lower()
        if input_domain not in {"logits", "probabilities"}:
            raise ValueError(
                f"discriminative.terms[{idx}].input_domain must be one of "
                f"['logits', 'probabilities'], got '{input_domain}'."
            )

        weight = float(term["weight"])
        if weight <= 0.0:
            raise ValueError(
                f"discriminative.terms[{idx}].weight must be > 0 because term "
                f"presence means the loss is active. Got {weight}."
            )

        terms.append(
            DiscriminativeTermSpec(
                loss_key=loss_key,
                input_domain=input_domain,
                weight=weight,
                params=_ensure_mapping(term["params"], f"{loss_key}.params"),
                supervision=_ensure_mapping(
                    term["supervision"], f"{loss_key}.supervision"
                ),
            )
        )

    if not terms:
        raise ValueError(
            "loss.discriminative.terms must contain at least one explicit loss term."
        )
    return terms


def _parse_weighted_plan(
    plan_cfg: Mapping[str, Any],
    available_heads: Mapping[int, torch.Tensor],
) -> SupervisionPlan:
    heads_raw = _ensure_list(plan_cfg.get("heads", None), "supervision.heads")
    weights_raw = _ensure_list(plan_cfg.get("weights", None), "supervision.weights")

    heads = tuple(int(h) for h in heads_raw)
    weights = tuple(float(w) for w in weights_raw)

    if len(heads) == 0:
        raise ValueError("Supervision plan must include at least one head index.")
    if len(heads) != len(weights):
        raise ValueError(
            "Supervision plan mismatch: number of heads and weights must match."
        )
    if any(w < 0.0 for w in weights):
        raise ValueError("Supervision weights must be non-negative.")
    if all(w == 0.0 for w in weights):
        raise ValueError("Supervision weights must not all be zero.")

    for head_idx in heads:
        if head_idx not in available_heads:
            raise ValueError(
                f"Supervision schedule references unavailable head {head_idx}. "
                f"Available heads: {sorted(available_heads.keys())}"
            )

    return SupervisionPlan(heads=heads, weights=weights)


def _parse_all_heads_plan(
    plan_cfg: Mapping[str, Any],
    available_heads: Mapping[int, torch.Tensor],
    final_head: int,
) -> SupervisionPlan:
    """
    Build a supervision plan that spans all available heads.

    Head order is always [final_head, remaining heads in ascending order].
    """

    if final_head not in available_heads:
        raise ValueError(
            f"Configured final_head={final_head} is unavailable. "
            f"Available heads: {sorted(available_heads.keys())}"
        )

    ordered_heads = [final_head] + sorted(
        h for h in available_heads.keys() if h != final_head
    )
    if len(ordered_heads) == 0:
        raise ValueError("No available heads to supervise.")

    weighting = str(plan_cfg.get("weighting", "geometric")).strip().lower()
    if weighting not in {"geometric", "uniform", "custom"}:
        raise ValueError(
            f"Unsupported all_heads weighting='{weighting}'. "
            "Expected one of: geometric, uniform, custom."
        )

    if weighting == "uniform":
        weights = [1.0 for _ in ordered_heads]
    elif weighting == "geometric":
        decay = float(plan_cfg.get("decay", 0.5))
        if decay <= 0.0:
            raise ValueError(
                f"all_heads geometric weighting requires decay > 0. Got {decay}."
            )
        weights = [float(decay**i) for i in range(len(ordered_heads))]
    else:  # custom
        raw_weights = _ensure_list(plan_cfg.get("weights", None), "supervision.weights")
        weights = [float(w) for w in raw_weights]
        if len(weights) != len(ordered_heads):
            raise ValueError(
                "all_heads custom weighting requires one weight per available head. "
                f"Got {len(weights)} weight(s) for {len(ordered_heads)} head(s)."
            )

    if any(w < 0.0 for w in weights):
        raise ValueError("Supervision weights must be non-negative.")
    if all(w == 0.0 for w in weights):
        raise ValueError("Supervision weights must not all be zero.")

    normalize = bool(plan_cfg.get("normalize", False))
    if normalize:
        weight_sum = float(sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("Cannot normalize all_heads weights with non-positive sum.")
        weights = [w / weight_sum for w in weights]

    return SupervisionPlan(
        heads=tuple(int(h) for h in ordered_heads),
        weights=tuple(float(w) for w in weights),
    )


def resolve_discriminative_supervision_plan(
    term: DiscriminativeTermSpec,
    deep_supervision_cfg: Mapping[str, Any],
    available_heads: Mapping[int, torch.Tensor],
    final_head: int,
    deep_supervision_enabled: bool,
) -> SupervisionPlan:
    """
    Resolve and validate one term's supervision plan.

    Modes:
    - final_only
    - weighted_heads
    - all_heads
    - inherit_default
    """

    if final_head not in available_heads:
        raise ValueError(
            f"Configured final_head={final_head} is unavailable. "
            f"Available heads: {sorted(available_heads.keys())}"
        )

    if not deep_supervision_enabled:
        return SupervisionPlan(heads=(final_head,), weights=(1.0,))

    supervision_cfg = _ensure_mapping(term.supervision, f"{term.loss_key}.supervision")
    mode = str(supervision_cfg.get("mode", "inherit_default")).strip().lower()

    if mode == "final_only":
        return SupervisionPlan(heads=(final_head,), weights=(1.0,))

    if mode == "weighted_heads":
        return _parse_weighted_plan(supervision_cfg, available_heads)

    if mode == "all_heads":
        return _parse_all_heads_plan(
            plan_cfg=supervision_cfg,
            available_heads=available_heads,
            final_head=final_head,
        )

    if mode == "inherit_default":
        default_cfg = _ensure_mapping(
            deep_supervision_cfg.get("default_supervision", None),
            "discriminative.deep_supervision.default_supervision",
        )
        default_mode = str(default_cfg.get("mode", "weighted_heads")).strip().lower()
        if default_mode == "final_only":
            return SupervisionPlan(heads=(final_head,), weights=(1.0,))
        if default_mode == "all_heads":
            return _parse_all_heads_plan(
                plan_cfg=default_cfg,
                available_heads=available_heads,
                final_head=final_head,
            )
        if default_mode != "weighted_heads":
            raise ValueError(
                f"Unsupported default supervision mode '{default_mode}'. "
                "Expected final_only, weighted_heads, or all_heads."
            )
        return _parse_weighted_plan(default_cfg, available_heads)

    raise ValueError(
        f"Unsupported supervision mode '{mode}' for term '{term.loss_key}'. "
        "Expected one of: final_only, weighted_heads, all_heads, inherit_default."
    )


def _build_probability_heads(
    logits_heads: Mapping[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    Convert logit heads to probability heads for binary segmentation.

    TODO: This sigmoid conversion is intentionally scoped to the current
    single-channel binary discriminative tasks. Future multi-class support
    should introduce an explicit activation policy rather than assuming sigmoid.
    """

    return {head_idx: torch.sigmoid(head) for head_idx, head in logits_heads.items()}


def _select_term_heads(
    *,
    input_domain: str,
    logits_heads: Mapping[int, torch.Tensor],
    probability_heads: Mapping[int, torch.Tensor],
) -> Mapping[int, torch.Tensor]:
    if input_domain == "logits":
        return logits_heads
    if input_domain == "probabilities":
        return probability_heads
    raise ValueError(f"Unsupported input_domain='{input_domain}'.")


def compute_discriminative_deep_supervision_loss(
    model_output: Any,
    target: torch.Tensor,
    discriminative_cfg: Mapping[str, Any],
    loss_factories: Optional[Mapping[str, LossFactory]] = None,
) -> DiscriminativeDeepSupervisionResult:
    """
    Compute generic discriminative loss with optional deep supervision.

    Returns scalar total loss plus flat loss_components for logging.
    """

    cfg = _ensure_mapping(discriminative_cfg, "discriminative")
    deep_cfg = _ensure_mapping(cfg.get("deep_supervision", {}), "discriminative.deep_supervision")
    deep_enabled = bool(deep_cfg.get("enabled", False))
    final_head = int(deep_cfg.get("final_head", 0))
    head_parser = str(deep_cfg.get("head_parser", "auto"))
    effective_head_parser = _resolve_effective_head_parser(
        head_parser=head_parser,
        model_output=model_output,
        target=target,
    )

    logits_heads = normalize_discriminative_head_outputs(
        model_output=model_output,
        head_parser=effective_head_parser,
    )

    if final_head not in logits_heads:
        raise ValueError(
            f"Configured final_head={final_head} is unavailable. "
            f"Available heads: {sorted(logits_heads.keys())}"
        )

    if not deep_enabled and len(logits_heads) > 1:
        raise ValueError(
            "Model returned multi-head output while deep_supervision.enabled=false. "
            "Either disable model deep supervision or enable "
            "loss.discriminative.deep_supervision."
        )

    terms = resolve_discriminative_terms(cfg)

    factories = dict(DISCRIMINATIVE_LOSS_FACTORIES)
    if loss_factories is not None:
        factories.update({str(k): v for k, v in loss_factories.items()})

    probability_heads = _build_probability_heads(logits_heads)

    total_loss = target.new_tensor(0.0)
    loss_components: Dict[str, float] = {}

    for term in terms:
        loss_key = term.loss_key
        if loss_key not in factories:
            raise ValueError(
                f"Unknown discriminative loss '{term.loss_key}'. "
                f"Available loss keys: {sorted(factories.keys())}"
            )

        term_loss_fn = factories[loss_key](term.params)
        plan = resolve_discriminative_supervision_plan(
            term=term,
            deep_supervision_cfg=deep_cfg,
            available_heads=logits_heads,
            final_head=final_head,
            deep_supervision_enabled=deep_enabled,
        )
        term_heads = _select_term_heads(
            input_domain=term.input_domain,
            logits_heads=logits_heads,
            probability_heads=probability_heads,
        )

        weighted_head_sum = target.new_tensor(0.0)
        for head_idx, head_weight in zip(plan.heads, plan.weights):
            pred_head = term_heads[head_idx]
            if tuple(pred_head.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for term '{term.loss_key}' head{head_idx}: "
                    f"pred.shape={tuple(pred_head.shape)} vs target.shape={tuple(target.shape)}"
                )

            head_loss = term_loss_fn(pred_head, target)
            if head_loss.dim() != 0:
                head_loss = head_loss.mean()

            loss_components[f"{term.loss_key}_head{head_idx}"] = float(
                head_loss.detach().item()
            )
            weighted_head_sum = weighted_head_sum + (float(head_weight) * head_loss)

        term_total = float(term.weight) * weighted_head_sum
        total_loss = total_loss + term_total
        loss_components[term.loss_key] = float(term_total.detach().item())

    loss_components["total"] = float(total_loss.detach().item())

    return DiscriminativeDeepSupervisionResult(
        total_loss=total_loss,
        loss_components=loss_components,
        final_prediction=probability_heads[final_head],
        head_predictions=probability_heads,
    )


def _resolve_effective_head_parser(
    head_parser: str,
    model_output: Any,
    target: torch.Tensor,
) -> str:
    """
    Resolve parser mode with target-aware auto disambiguation.

    For auto mode:
    - list/tuple output -> list
    - tensor with dim == target.dim() + 1 -> stacked
    - tensor with dim == target.dim() -> single
    - fallback -> single
    """

    parser = str(head_parser).strip().lower()

    if isinstance(model_output, (list, tuple)):
        return "list"

    if torch.is_tensor(model_output):
        if parser == "stacked":
            # MONAI DynUNet returns stacked heads only in train mode.
            # In eval mode, even with deep_supervision=True, output is single-head.
            if model_output.dim() == target.dim():
                return "single"
            return "stacked"

        if parser == "single":
            return "single"

        if parser == "list":
            return "list"

        # auto mode
        if model_output.dim() == target.dim() + 1:
            return "stacked"
        if model_output.dim() == target.dim():
            return "single"
        return "single"

    # Non-tensor fallback
    if parser in {"single", "stacked", "auto"}:
        return "single"
    return parser
