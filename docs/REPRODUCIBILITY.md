# Reproducibility contract

This release separates values written explicitly in the paper from fixed implementation choices needed to execute the method. No release default was selected by fitting the reported test table.

## Paper-specified values

| Item | Value |
|---|---:|
| Base detector | Faster R-CNN R50-FPN |
| Input resize | short side 800; maximum side 1333 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| FPN levels | P2--P5 |
| Embedding dimension | 256 |
| Corridor points | 64 per direction |
| Maximum length ratio | 3.0 |
| Maximum width ratio | 1.0 |
| `beta_cons` | 1.0 |
| `beta_dom` | 1.0 |
| `beta_dec` | 0.05 |
| `lambda_neg` | 1.0 |
| `lambda_att` | 0.5 |
| Dominance margin | 0.2 |
| Epsilon | 0.01 |
| Hardware/reporting | one A100 80GB; mean and std over three seeds |

## Fixed release defaults

| Item omitted numerically from paper | Release value | Reason |
|---|---:|---|
| `beta_wake` | 1.0 | Unit weighting consistent with the two other primary auxiliary losses |
| Seeds | 0, 1, 2 | Fixed public triplet |
| Master split seed | 2026 | Stable paper-year seed |
| Split ratio | 70/15/15 | Train/validation/test, grouped by sequence and stratified by source |
| Epochs | 24 | Standard two-stage 2x-length release schedule |
| Effective batch size | 16 | Standard R50-FPN optimization scale |
| Images per A100 step | 4 | Fits the auxiliary branch; accumulation 4 gives effective batch 16 |
| Learning rate | 0.02 | Standard batch-16 Faster R-CNN scale |
| Warmup | 1000 iterations from factor 0.001 | Explicit stable detector warmup |
| LR milestones | epochs 16 and 22 | Fixed 24-epoch multistep schedule |
| LR decay | 0.1 | Standard multistep decay |
| Directional bias initial values | 1.0, 1.0 | Symmetric neutral initialization; both are learned through softplus |
| Corridor sampling grid | 8 x 8 | Exactly 64 deterministic points per direction |
| FPN assignment | canonical scale 224 to P4, clamped to P2--P5 | Standard scale-aware proposal mapping |
| Two corridor response maps | direction-gate-weighted probability map | Matches the descriptor's learned bow--stern selection |
| Response-to-RoI resize | bilinear, `align_corners=False` | Fully specifies the attenuation operator |
| Hard-negative proposal weight | 2.0 when at least 10% of sampled points hit wake-like/clutter masks | Makes “especially hard negatives” executable |
| Horizontal flip | 0.5 | Only spatial training augmentation; applied jointly to all masks and boxes |
| FP score threshold | 0.50 | Fixed diagnostic operating point |
| FP vessel-match IoU | 0.50 | Standard match threshold |
| Wake-like box coverage | 0.25 | Requires substantial, not incidental, wake-like overlap |

Changing any row above creates a new experiment and should be reported.

## Run manifest

Every training run writes `run_manifest.json` containing:

- the fully resolved configuration;
- package and CUDA versions;
- visible GPU information;
- command line and working directory;
- Git commit when available;
- seed and deterministic-algorithm setting;
- start time.

Checkpoints contain model, optimizer, scheduler, AMP scaler, epoch, best metric, resolved config, and Python/NumPy/PyTorch random states. Resume restores all of them.

## Determinism

`experiment.deterministic: true` enables deterministic PyTorch algorithms and disables cuDNN benchmarking. Some CUDA operators may be unavailable in deterministic mode on a different PyTorch/CUDA pair; the pinned environment is therefore part of the contract.

## Three-seed reporting

`scripts/run_three_seeds.py` trains seeds sequentially and writes a summary only from completed run metrics. It does not fabricate missing seeds. Means and sample standard deviations are computed from raw JSON results.

## Data integrity

The release cannot redistribute or checksum private Curated-Wake files. For an internal reproduction, archive the three split JSON SHA-256 hashes alongside the run manifest. The trainer records these hashes automatically.
