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

from .segmentation_losses import BCELoss, DiceLoss


LossFactory = Callable[[Mapping[str, Any]], nn.Module]


@dataclass(frozen=True)
class DiscriminativeTermSpec:
    """Normalized discriminative loss-term definition."""

    name: str
    enabled: bool
    loss_key: str
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


def build_default_discriminative_loss_factories() -> Dict[str, LossFactory]:
    """Default loss-factory registry for discriminative terms."""

    return {
        "dice": lambda params: DiceLoss(
            smooth=float(params.get("smooth", 1.0e-5)),
            apply_sigmoid=bool(params.get("apply_sigmoid", False)),
        ),
        "bce": lambda params: BCELoss(
            pos_weight=params.get("pos_weight", None),
            apply_sigmoid=bool(params.get("apply_sigmoid", False)),
        ),
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


def _build_legacy_terms(discriminative_cfg: Dict[str, Any]) -> List[DiscriminativeTermSpec]:
    """
    Build terms list from legacy discriminative.dice / discriminative.bce fields.

    This preserves backward compatibility for existing non-DS configurations.
    """

    terms: List[DiscriminativeTermSpec] = []

    dice_cfg = _ensure_mapping(discriminative_cfg.get("dice", {}), "discriminative.dice")
    if bool(dice_cfg.get("enabled", False)):
        terms.append(
            DiscriminativeTermSpec(
                name="dice",
                enabled=True,
                loss_key="dice",
                weight=float(dice_cfg.get("weight", 1.0)),
                params={
                    "smooth": float(dice_cfg.get("smooth", 1.0e-5)),
                    "apply_sigmoid": bool(dice_cfg.get("apply_sigmoid", False)),
                },
                supervision={"mode": "final_only"},
            )
        )

    bce_cfg = _ensure_mapping(discriminative_cfg.get("bce", {}), "discriminative.bce")
    if bool(bce_cfg.get("enabled", False)):
        terms.append(
            DiscriminativeTermSpec(
                name="bce",
                enabled=True,
                loss_key="bce",
                weight=float(bce_cfg.get("weight", 1.0)),
                params={
                    "pos_weight": bce_cfg.get("pos_weight", None),
                    "apply_sigmoid": bool(bce_cfg.get("apply_sigmoid", False)),
                },
                supervision={"mode": "final_only"},
            )
        )

    return terms


def resolve_discriminative_terms(
    discriminative_cfg: Mapping[str, Any],
) -> List[DiscriminativeTermSpec]:
    """Resolve discriminative terms from new schema or legacy fields."""

    cfg = _ensure_mapping(discriminative_cfg, "discriminative")
    raw_terms = cfg.get("terms", None)

    if raw_terms is None:
        terms = _build_legacy_terms(cfg)
    else:
        terms = []
        for idx, raw_term in enumerate(_ensure_list(raw_terms, "discriminative.terms")):
            term = _ensure_mapping(raw_term, f"discriminative.terms[{idx}]")
            name = str(term.get("name", f"term{idx}")).strip()
            if not name:
                raise ValueError(f"discriminative.terms[{idx}] has empty 'name'.")

            terms.append(
                DiscriminativeTermSpec(
                    name=name,
                    enabled=bool(term.get("enabled", True)),
                    loss_key=str(term.get("loss", name)).strip().lower(),
                    weight=float(term.get("weight", 1.0)),
                    params=_ensure_mapping(term.get("params", {}), f"{name}.params"),
                    supervision=_ensure_mapping(
                        term.get("supervision", {}), f"{name}.supervision"
                    ),
                )
            )

    enabled_terms = [t for t in terms if t.enabled and t.weight > 0.0]
    if not enabled_terms:
        raise ValueError(
            "No enabled discriminative terms were resolved. "
            "Enable at least one term (or legacy dice/bce)."
        )
    return enabled_terms


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

    supervision_cfg = _ensure_mapping(term.supervision, f"{term.name}.supervision")
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
        f"Unsupported supervision mode '{mode}' for term '{term.name}'. "
        "Expected one of: final_only, weighted_heads, all_heads, inherit_default."
    )


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

    head_predictions = normalize_discriminative_head_outputs(
        model_output=model_output,
        head_parser=effective_head_parser,
    )

    if final_head not in head_predictions:
        raise ValueError(
            f"Configured final_head={final_head} is unavailable. "
            f"Available heads: {sorted(head_predictions.keys())}"
        )

    if not deep_enabled and len(head_predictions) > 1:
        raise ValueError(
            "Model returned multi-head output while deep_supervision.enabled=false. "
            "Either disable model deep supervision or enable "
            "loss.discriminative.deep_supervision."
        )

    terms = resolve_discriminative_terms(cfg)

    factories = build_default_discriminative_loss_factories()
    if loss_factories is not None:
        factories.update({str(k).lower(): v for k, v in loss_factories.items()})

    total_loss = target.new_tensor(0.0)
    loss_components: Dict[str, float] = {}

    for term in terms:
        loss_key = term.loss_key.lower()
        if loss_key not in factories:
            raise ValueError(
                f"Unknown loss key '{term.loss_key}' for term '{term.name}'. "
                f"Available loss keys: {sorted(factories.keys())}"
            )

        term_loss_fn = factories[loss_key](term.params)
        plan = resolve_discriminative_supervision_plan(
            term=term,
            deep_supervision_cfg=deep_cfg,
            available_heads=head_predictions,
            final_head=final_head,
            deep_supervision_enabled=deep_enabled,
        )

        weighted_head_sum = target.new_tensor(0.0)
        for head_idx, head_weight in zip(plan.heads, plan.weights):
            pred_head = head_predictions[head_idx]
            if tuple(pred_head.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for term '{term.name}' head{head_idx}: "
                    f"pred.shape={tuple(pred_head.shape)} vs target.shape={tuple(target.shape)}"
                )

            head_loss = term_loss_fn(pred_head, target)
            if head_loss.dim() != 0:
                head_loss = head_loss.mean()

            loss_components[f"{term.name}_head{head_idx}"] = float(
                head_loss.detach().item()
            )
            weighted_head_sum = weighted_head_sum + (float(head_weight) * head_loss)

        term_total = float(term.weight) * weighted_head_sum
        total_loss = total_loss + term_total
        loss_components[term.name] = float(term_total.detach().item())

    loss_components["total"] = float(total_loss.detach().item())

    return DiscriminativeDeepSupervisionResult(
        total_loss=total_loss,
        loss_components=loss_components,
        final_prediction=head_predictions[final_head],
        head_predictions=head_predictions,
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
