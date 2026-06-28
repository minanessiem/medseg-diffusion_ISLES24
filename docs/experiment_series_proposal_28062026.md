# ISLES26 Experiment Series Proposal - 2026-06-28

## Overview

This document captures the next planned ISLES26 experiment series after the
loss-function, threshold-calibration, ROI-size, positive-crop-weighting, and
DynUNet topology sweeps.

The current high-water mark remains approximately **0.633 3D Dice** from a
DynUNet 128^3 random-patch configuration with DiceFocal deep supervision,
T1_RAW input, and spatial-only 3D augmentation. Recent experiments have been
useful diagnostically, but none has meaningfully changed the performance
ceiling:

- Loss changes improved or degraded performance modestly, but did not create a
  clear new regime.
- Global threshold tuning produced negligible deployable gain.
- Per-case oracle thresholding exposed only about two Dice points of theoretical
  upper-bound gain.
- Smaller ROIs hurt performance, suggesting that 128^3 context is important.
- Increasing `RandCropByPosNegLabeld` positive weighting did not help.
- DynUNet remains slightly stronger than SwinUNETR in the tested settings.

The next series should therefore move away from single scalar knobs and toward
structured model/input hypotheses: capacity, context integration, architecture
family, case sampling, and cascaded/two-stage designs.

## Experiment Set 1: DynUNet Topology Sweep

Status: **started and completed**

Goal: test whether the current DynUNet ceiling is limited by width, depth, or
simple late-stage kernel-size changes while keeping the known-strong training
setup fixed.

Fixed setup:

- Model family: DynUNet
- ROI: `128^3`
- Loss: `discriminative_dicefocal_deepsupervision`
- Deep supervision: enabled, `deep_supr_num=2`
- Input: `T1_RAW`
- Augmentation: `spatial_only_3d`
- Random patch sampling: `pos=1`, `neg=1`
- Training: 100K steps, BF16 AMP

Results:

| Run Name | Filters | Kernel Size | 3D Dice | Best Step | Status |
| --- | --- | --- | ---: | ---: | --- |
| `dynunet_128_3d_k3-3-3-3_f24-48-96-192_b3_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-06-26_20-37-26` | `[24, 48, 96, 192]` | `[3, 3, 3, 3]` | 0.625343 | 60000 | complete |
| `dynunet_128_3d_k3-3-3-3_f32-64-128-256_b3_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-06-26_20-37-24` | `[32, 64, 128, 256]` | `[3, 3, 3, 3]` | 0.621244 | 70000 | complete |
| `dynunet_128_3d_k3-3-3-3_f48-96-192-384_b3_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-06-26_20-37-28` | `[48, 96, 192, 384]` | `[3, 3, 3, 3]` | 0.620516 | 75000 | complete |
| `dynunet_128_3d_k3-3-5-5_f32-64-128-256_b3_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-06-26_20-37-39` | `[32, 64, 128, 256]` | `[3, 3, 5, 5]` | 0.617800 | 40000 | complete |
| `dynunet_128_3d_k3-3-3-3-3_f24-48-96-192-320_b3_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-06-26_20-37-30` | `[24, 48, 96, 192, 320]` | `[3, 3, 3, 3, 3]` | 0.605973 | 50000 | complete |
| `dynunet_128_3d_k3-3-3-3-3_f32-64-128-256-320_b3_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-06-26_20-37-33` | `[32, 64, 128, 256, 320]` | `[3, 3, 3, 3, 3]` | 0.600104 | 45000 | complete |
| `dynunet_128_3d_k3-3-3-5_f32-64-128-256_b3_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-06-26_20-37-35` | `[32, 64, 128, 256]` | `[3, 3, 3, 5]` | 0.590550 | 40000 | complete |

Key takeaways:

- The topology sweep did **not** break the current high-water mark of roughly
  0.633 Dice.
- Narrower DynUNet width performed best within this sweep, but still below the
  best known model.
- Wider and deeper variants did not help and may have made optimization harder.
- Naive larger late-stage kernels did not help.
- The next experiments should not simply continue widening/deepening the same
  DynUNet shape.

## Experiment Set 2: DynUNet Context / Receptive Field Sweep

Status: **planned**

Goal: improve global or semi-global context integration without relying on
plain width/depth increases that already failed in set 1.

Motivation:

- The ROI x positive-weighting experiment showed that 128^3 context is useful.
- The topology sweep showed that simply adding another downsampling stage or
  larger late kernels does not automatically improve performance.
- A more targeted context experiment should distinguish "more parameters" from
  "better context pathway."

Candidate experiments:

1. **Low-resolution context channel**
   - Provide a downsampled full-volume or larger-field context representation
     alongside the local 128^3 patch.
   - Hypothesis: the model may need broader anatomical context but still require
     128^3 local detail.

2. **Two-resolution DynUNet input**
   - Concatenate local high-resolution T1 crop with a matched low-resolution
     context crop.
   - This preserves the current training loop more cleanly than a full cascade.

3. **Dilated bottleneck or context block**
   - Add a lightweight dilated convolution block near the bottleneck rather than
     changing the whole encoder/decoder topology.
   - Hypothesis: receptive field can increase without the optimization penalty
     seen in deeper DynUNet variants.

4. **Context-aware validation/export review**
   - Compare failure cases where the model prediction is anatomically close but
     spatially incomplete against cases where lesions are entirely invisible.
   - This determines whether model context is a plausible bottleneck.

Recommended first implementation:

- Add a minimal context-block DynUNet variant or adapter option before building
  a full two-branch model.
- Keep all training/runtime settings identical to the current best DynUNet.

## Experiment Set 3: Architecture Family Sweep

Status: **planned**

Goal: test whether another 3D medical segmentation architecture is better
matched to ISLES26 than DynUNet/SwinUNETR under the same training and evaluation
contract.

Motivation:

- DynUNet has shown a small but consistent advantage over SwinUNETR.
- The next architecture sweep should prioritize strong 3D convolutional medical
  segmentation baselines rather than more generic transformer tuning.

Candidate model families:

1. **MONAI SegResNet / SegResNetDS**
   - High-priority challenger.
   - Strong residual 3D segmentation baseline.
   - Likely better aligned with small medical datasets than heavier transformer
     models.

2. **MONAI BasicUNet / BasicUNetPlusPlus**
   - Useful control family.
   - Lower complexity, easier optimization, may establish whether DynUNet
     complexity is actually needed.

3. **VNet-style residual architecture**
   - Potentially useful if volumetric continuity and residual learning matter
     more than encoder depth.

4. **SwinUNETR tuned control**
   - Lower priority because prior results suggest it trails DynUNet.
   - Worth revisiting only if the comparison was underpowered or used a poor
     SwinUNETR configuration.

Recommended grid:

- Keep loss, ROI, augmentation, validation, and training schedule fixed.
- Add one adapter/config at a time, starting with SegResNet.
- Compare against the current DynUNet incumbent under identical evaluation.

## Experiment Set 4: Preprocessing Sweep

Status: **intentionally dropped**

Reason:

- T1 preprocessing variants have already been tried or explored sufficiently in
  prior experiments.
- This set should not be reintroduced unless a later failure analysis points
  specifically to intensity normalization/site-shift as the dominant error mode.

## Experiment Set 5: Case Sampling, Not Crop Sampling

Status: **planned**

Goal: change which cases are emphasized during training, not merely where
patches are sampled inside a selected case.

Motivation:

- `RandCropByPosNegLabeld` positive weighting changes crop-center selection
  inside a case.
- It does not guarantee that tiny-lesion, rare, difficult, or previously failed
  cases are seen more often.
- The positive-weighting sweep suggests that naive within-case foreground
  oversampling is not enough.

Candidate sampling strategies:

1. **Lesion-volume-stratified case sampling**
   - Bin cases by ground-truth lesion volume.
   - Oversample small-lesion or low-Dice bins.
   - Hypothesis: low-volume lesions may be underrepresented in effective
     gradient signal even with positive crops.

2. **Hard-case replay**
   - Use prior validation/evaluation results to identify hard training cases.
   - Increase their sampling probability in subsequent runs.

3. **Balanced batch construction**
   - Enforce batches containing a mix of small, medium, and large lesion cases.
   - This reduces the chance that training dynamics are dominated by easier
     medium/large lesions.

4. **False-negative-oriented sampling**
   - If failure analysis shows missed lesions dominate, oversample cases or
     lesion regions associated with high false-negative burden.

Implementation requirements:

- Add case-level sampling weights or a custom sampler in the loader stack.
- Preserve the current patch policy initially: ROI 128, `pos=1`, `neg=1`.
- Track sampling policy explicitly in Hydra config and run names.

## Experiment Set 6: Cascaded / Two-Stage Models

Status: **planned, higher implementation cost**

Goal: split the segmentation problem into detection/context and refinement
stages if single-stage patch training continues to plateau.

Motivation:

- Current single-stage DynUNet may be trying to solve localization, lesion
  detection, and boundary refinement in one pass.
- If failure analysis shows missed lesions or poor localization, a two-stage
  system may be more appropriate.

Candidate designs:

1. **Coarse-to-fine cascade**
   - Stage 1: low-resolution full-volume model predicts coarse lesion
     probability or candidate regions.
   - Stage 2: high-resolution 128^3 patch model refines candidate regions.

2. **High-recall detector plus segmenter**
   - Stage 1 optimized for lesion recall.
   - Stage 2 optimized for precision/boundaries.

3. **Context prior plus local refinement**
   - Stage 1 produces a soft context prior.
   - Stage 2 consumes T1 crop plus prior crop as additional input.

4. **Candidate connected-component refinement**
   - Use connected components from a high-recall model to drive local
     refinement patches.

Risks:

- More complex training and evaluation pipeline.
- More opportunities for leakage or inconsistent preprocessing if not carefully
  config-driven.
- Requires robust artifact handoff between stages.

Recommended entry point:

- Do not start here until case-level failure analysis confirms that missed
  detection or localization, rather than boundary refinement alone, is the
  dominant error mode.

## Recommended Next Order

1. Complete reporting and analysis for Experiment Set 1.
2. Run a case-level error-budget report on the best current models.
3. Start Experiment Set 3 with a SegResNet adapter/config if implementation
   cost is low.
4. In parallel, design the lowest-risk version of Experiment Set 2, preferably
   a small context-block or low-resolution context input rather than a full
   cascade.
5. Implement Experiment Set 5 only after the error-budget report confirms that
   lesion-volume or hard-case imbalance is a major driver.
6. Reserve Experiment Set 6 for when the failure profile justifies the added
   complexity.

## Decision Principle

Future experiments should be tied to a diagnosed failure mode. If a proposed run
does not answer a specific question about capacity, context, architecture
family, case imbalance, or detection/refinement decomposition, it should not be
queued merely because it is easy to run.
