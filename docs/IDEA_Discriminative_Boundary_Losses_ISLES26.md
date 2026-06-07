# Idea: Discriminative Boundary-Aware Losses for ISLES26

**Status:** Idea / next experiment design  
**Context:** ISLES26 discriminative DynUNet/SwinUNETR experiments  
**Motivation:** Reduce visually smoothed lesion boundaries and improve surface fidelity.

## Overview

ISLES26 discriminative models are currently good at localizing lesions and
recovering approximate lesion extent, but direct 3D patch predictions can look
like smoothed probability blobs around the ground-truth mask. The labels remain
voxel-jagged while predictions have softer, rounder borders. This points to a
learned output/loss behavior rather than a sliding-window or TensorBoard artifact.

The new discriminative loss contract makes this easier to explore:

- DynUNet and SwinUNETR adapters can expose logits.
- Each discriminative loss term can choose `input_domain: logits` or
  `input_domain: probabilities`.
- Metrics/logging still consume probabilities.

This enables clean experimentation with both probability-domain overlap/surface
losses and logits-native classification losses.

## Working Hypothesis

The current Dice+BCE objective optimizes voxel overlap and foreground/background
classification, but it does not strongly penalize boundary distance or contour
sharpness. The model can therefore learn a well-localized soft probability field
whose thresholded mask has acceptable Dice while still looking rounded or
averaged at lesion borders.

Boundary-fidelity losses should be added as auxiliary terms, not as replacements
for overlap losses.

## Candidate Loss Families

### Dice / BCE Baselines

Current baseline behavior should remain available:

```yaml
terms:
  - loss: DiceLoss
    input_domain: probabilities
    weight: 1.0
    params:
      smooth: 1.0e-5
      apply_sigmoid: false

  - loss: BCELoss
    input_domain: probabilities
    weight: 1.0
    params:
      pos_weight: null
      apply_sigmoid: false
```

Now that logits are available, a stronger baseline is:

```yaml
terms:
  - loss: DiceLoss
    input_domain: probabilities
    weight: 1.0
    params:
      smooth: 1.0e-5
      apply_sigmoid: false

  - loss: BCELoss
    input_domain: logits
    weight: 1.0
    params:
      pos_weight: null
      apply_sigmoid: true
```

This keeps Dice probability-based while using numerically stable
`BCEWithLogitsLoss` behavior for BCE.

### MONAI HausdorffDTLoss

MONAI 1.3.0 provides `HausdorffDTLoss`, based on:

Karimi et al., "Reducing the Hausdorff Distance in Medical Image Segmentation
with Convolutional Neural Networks."

Why it is relevant:

- It directly targets distance-transform boundary errors.
- It is better aligned with the observed problem than pure Dice or BCE.
- It can be used on probability-domain predictions.

Recommended first use:

```yaml
terms:
  - loss: DiceLoss
    input_domain: probabilities
    weight: 1.0

  - loss: BCELoss
    input_domain: logits
    weight: 1.0

  - loss: HausdorffDTLoss
    input_domain: probabilities
    weight: 0.05
```

Suggested ablation weights:

- `0.025`
- `0.05`
- `0.1`
- `0.2`

For DynUNet deep supervision, start with `HausdorffDTLoss` on the final head only.
Distance-transform losses on lower-resolution/deep-supervised heads may be
expensive or noisy.

### MONAI TverskyLoss

MONAI 1.3.0 provides `TverskyLoss`.

Why it is relevant:

- Controls false-positive vs false-negative trade-off.
- Useful if boundary smoothing corresponds to systematic oversegmentation or
  undersegmentation.
- Not a direct boundary loss.

Suggested settings:

- Dice-like control: `alpha=0.5`, `beta=0.5`
- Penalize false positives / bloated predictions: `alpha=0.7`, `beta=0.3`
- Penalize false negatives / eroded predictions: `alpha=0.3`, `beta=0.7`

Recommended experiment:

- Replace Dice with Tversky while keeping BCE.
- Do not stack Dice and Tversky at first unless there is a clear reason.

### MONAI FocalLoss / DiceFocalLoss

MONAI 1.3.0 provides `FocalLoss` and `DiceFocalLoss`. These are now more viable
because discriminative model logits can be routed to logits-native losses.

Why it is relevant:

- Focal loss downweights easy voxels and emphasizes hard examples.
- Boundary voxels are often hard examples.
- It may help if boundaries are uncertain because easy interior/background
  voxels dominate the gradient.

Suggested first use:

```yaml
terms:
  - loss: DiceFocalLoss
    input_domain: logits
    weight: 1.0
    params:
      gamma: 2.0
      lambda_dice: 1.0
      lambda_focal: 0.5
```

Potential caveat:

- Focal is not inherently a surface-distance loss.
- It may improve hard voxel classification without directly improving contour
  geometry.

### Custom Boundary-Band BCEWithLogits

This is not a MONAI 1.3.0 built-in, but it may be the most targeted practical
loss for the visual issue.

Idea:

- Build a narrow boundary band around the ground-truth mask.
- Compute BCEWithLogits everywhere.
- Upweight voxels in/near the boundary band.

Why it is relevant:

- Directly increases gradient pressure where prediction smoothing is observed.
- Keeps global foreground/background learning stable.
- Cheaper and easier to reason about than Hausdorff-style objectives.

Possible configuration:

```yaml
terms:
  - loss: DiceLoss
    input_domain: probabilities
    weight: 1.0

  - loss: BoundaryBandBCELoss
    input_domain: logits
    weight: 1.0
    params:
      boundary_weight: 3.0
      band_width_voxels: 1
```

Suggested ablations:

- `boundary_weight: 2.0`, `3.0`, `5.0`
- `band_width_voxels: 1`, `2`

Implementation should use tensor operations where possible, e.g. max-pooling
dilation/erosion, rather than CPU morphology.

### GeneralizedDiceLoss

Useful for class imbalance, but lower priority for this specific symptom.

ISLES26 random-patch training already emphasizes foreground-bearing patches, and
the current issue seems more like boundary fidelity than class imbalance alone.

### GeneralizedWassersteinDiceLoss

Low priority for binary lesion/background segmentation. It is more natural for
multi-class problems where class distances have semantic meaning.

### MaskedDiceLoss

Potentially useful only if paired with a boundary/ROI mask. On its own, it does
not solve the boundary-smoothing issue.

## Recommended Experiment Order

### Step 1: Establish Logits Baseline

Run current Dice+BCE behavior exactly as migrated:

- `DiceLoss` on probabilities
- `BCELoss` on probabilities

Then run a logits-BCE baseline:

- `DiceLoss` on probabilities
- `BCELoss` on logits with `apply_sigmoid: true`

Goal:

- Confirm the new loss-domain routing does not harm baseline behavior.
- Determine whether logits-native BCE changes calibration or boundary sharpness.

### Step 2: Add HausdorffDTLoss

Run:

- `DiceLoss` probability-domain
- `BCELoss` logits-domain
- `HausdorffDTLoss` probability-domain, final head only

Weights:

- `0.05`
- `0.1`
- possibly `0.2`

Primary metrics:

- Dice
- surface Dice
- HD95
- predicted/GT volume ratio

Qualitative check:

- Direct patch output visualizations, not sliding-window output.

### Step 3: Boundary-Band BCE

Implement and test custom `BoundaryBandBCELoss`.

Run:

- `DiceLoss` probability-domain
- `BoundaryBandBCELoss` logits-domain

Optionally add ordinary `BCELoss` if boundary-only BCE destabilizes interiors.

### Step 4: Tversky / Focal Ablations

Use these to diagnose whether boundary smoothing is tied to FP/FN imbalance or
hard-voxel underweighting.

Suggested configs:

- `TverskyLoss + BCELoss`
- `DiceFocalLoss`
- `DiceLoss + FocalLoss`

## Deep Supervision Policy

For DynUNet:

- Keep Dice and BCE on all heads using the existing geometric schedule.
- Start boundary/surface losses on final head only.
- Only expand boundary losses to all heads if final-head-only results are
  promising and stable.

Reason:

- Lower-resolution auxiliary heads are not ideal targets for boundary precision.
- Boundary/surface losses can be more expensive.
- Coarse heads may encourage overly smooth boundary priors if over-weighted.

## Evaluation Notes

Do not judge these losses only by Dice.

Track:

- `dice_3d`
- `surface_dice_monai_3d`
- `hd95_3d`
- predicted volume
- ground-truth volume
- predicted/GT volume ratio
- visual direct 3D patch outputs

Boundary-aware losses may slightly reduce Dice while improving surface fidelity.
The useful target is a Pareto improvement or acceptable Dice trade-off with
better surface metrics and visibly sharper masks.

## Implementation Notes

Likely files for adding losses:

- `src/losses/segmentation_losses.py`
- `src/losses/discriminative_deep_supervision.py`
- `configs/loss/discriminative_*.yaml`
- new loss configs under `configs/loss/`

Future loss registry entries should avoid silent defaults. Required parameters
should be explicit in YAML and validated in code.

For MONAI losses, wrap only what is needed and expose a stable local config
contract. Do not leak every MONAI constructor argument into configs unless it is
actually useful for experiments.

## Open Questions

- Should final model selection include a surface metric, or only report surface
  metrics alongside Dice?
- How much Dice can be traded for visibly sharper and lower-HD95 masks?
- Are smoothed boundaries mainly oversegmented, undersegmented, or just soft at
  the threshold?
- Does `T1_RAW` benefit from different boundary losses than normalized variants?
- Should boundary-aware losses be paired with raw-specific augmentation presets?

## Suggested Next PRD

Create a focused implementation PRD for:

1. Adding MONAI `HausdorffDTLoss`, `TverskyLoss`, and `DiceFocalLoss` wrappers.
2. Adding a custom `BoundaryBandBCELoss`.
3. Creating ISLES26 DynUNet loss ablation configs.
4. Defining a 100K-step screening protocol with Dice, surface Dice, HD95, and
   direct patch visualization review.
