# HullWake

Official method-oriented implementation of **Hull First, Wake Second: Wake-Reliance Suppression for Robust Maritime Vessel Detection**.

HullWake keeps vessel localization tied to proposal-centered hull evidence while using directional wake context only as bounded auxiliary evidence. The release contains the model, losses, annotation schema, deterministic split builder, training and evaluation code, ablation configurations, tests, and the exact release defaults needed to remove implementation ambiguity.

> **Data policy.** No image, annotation, derived frame, pretrained checkpoint, or dataset sample is included. This repository provides only code, schemas, configurations, parameters, and links to the original public data sources. Curated-Wake contains additional author-created labels and must be assembled separately by an authorized user.

## Method

For every Faster R-CNN proposal, HullWake:

1. obtains a hull descriptor from the standard RoIAlign feature;
2. predicts hull orientation and samples two proposal-anchored candidate wake corridors to handle bow--stern ambiguity;
3. aggregates 64 points per direction with directional attention;
4. injects the selected wake descriptor through a sigmoid-bounded residual gate while retaining hull-only box regression; and
5. applies wake-response supervision, wake-attenuated consistency, wake-only confidence suppression, and hull--wake decorrelation.

The main model is implemented directly on TorchVision Faster R-CNN R50-FPN. It does not require Detectron2 or a private codebase.

## Repository layout

```text
HullWake/
├── configs/                 # Main, 4070, and component-ablation recipes
├── docs/                    # Dataset schema, metrics, and reproduction contract
├── hullwake/                # Installable Python package
│   ├── data/                # Extended COCO loader and transforms
│   ├── modeling/            # Detector, RoI head, wake extractor, losses
│   ├── evaluation/          # COCO and wake-oriented metrics
│   ├── engine/              # Training/evaluation loop
│   └── utils/               # Seeding, checkpoints, logging
├── scripts/                 # Train, evaluate, infer, and three-seed runners
├── tools/                   # Annotation validation and leakage-safe split creation
└── tests/                   # Geometry, loss, schema, metric, and model smoke tests
```

## Installation

The paper release targets Python 3.10, PyTorch 2.2.2, TorchVision 0.17.2, and CUDA 12.1.

```bash
conda create -n hullwake python=3.10 -y
conda activate hullwake
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-deps
pip install -r requirements-base.txt
python -m hullwake.doctor
pytest -q
```

For a CPU-only validation environment, install the matching CPU wheels instead:

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
pip install -r requirements-base.txt -r requirements-dev.txt
pytest -q
```

## Data sources

Download source data from the respective maintainers and comply with each source's license and terms:

- [Ships/Vessels in Aerial Images](https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images)
- [Singapore Maritime Dataset (SMD)](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset)
- [SeaDronesSee downloads](https://seadronessee.cs.uni-tuebingen.de/dataset)

This code expects an extended COCO file. Each vessel annotation adds `wake_attribute` and `wake_segmentation`; image-level `regions` store wake-like negatives and water clutter. The full contract and a zero-data example are in [docs/ANNOTATION_FORMAT.md](docs/ANNOTATION_FORMAT.md).

Recommended private data layout (all paths are ignored by Git):

```text
data/curated_wake/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── curated_wake_train.json
    ├── curated_wake_val.json
    └── curated_wake_test.json
```

Validate a completed master annotation file and create deterministic, source-stratified, sequence-safe splits:

```bash
python tools/validate_annotations.py \
  --annotations /path/to/curated_wake_master.json \
  --images /path/to/all_images \
  --require-files --require-wake-masks

python tools/create_splits.py \
  --annotations /path/to/curated_wake_master.json \
  --output-dir /path/to/data/curated_wake/annotations \
  --seed 2026 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

The split tool groups frames by `sequence_id`, so adjacent frames from SMD or SeaDronesSee cannot leak across splits.

## Training

Edit only the six data paths in `configs/hullwake_r50_fpn.yaml`, then run:

```bash
python scripts/train.py --config configs/hullwake_r50_fpn.yaml
```

Resume exactly, including optimizer, scheduler, scaler, epoch, and random states:

```bash
python scripts/train.py \
  --config configs/hullwake_r50_fpn.yaml \
  --resume outputs/hullwake_r50_fpn/seed_0/checkpoint_last.pt
```

Run the paper's three-seed protocol sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_three_seeds.py \
  --config configs/hullwake_r50_fpn.yaml --seeds 0 1 2
```

For the RTX 4070, use the memory-adjusted configuration. It preserves the effective batch size through gradient accumulation:

```bash
python scripts/train.py --config configs/hullwake_r50_fpn_4070.yaml
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/hullwake_r50_fpn.yaml \
  --checkpoint outputs/hullwake_r50_fpn/seed_0/checkpoint_best.pt \
  --split test \
  --save-predictions outputs/test_predictions.json
```

The evaluator reports standard COCO AP/AP50/AP75 plus:

- `AP_NoWake`: COCO AP on weak/no-wake vessels, with other wake groups ignored;
- `FP_WakeLike`: unmatched detections whose boxes substantially cover a wake-like-negative mask;
- `WG_AP`: minimum AP across clear-wake and weak/no-wake groups;
- `Delta_wake`: mean matched-detection confidence drop after internal wake attenuation.

See [docs/METRICS.md](docs/METRICS.md) for thresholds and matching rules. Ambiguous vessels contribute to ordinary AP but are ignored by group-wise AP.

## Parameters

Parameters explicitly stated in the paper are preserved in the main config:

| Parameter | Value |
|---|---:|
| Points per direction | 64 |
| Maximum corridor length ratio | 3.0 |
| Maximum corridor width ratio | 1.0 |
| Embedding dimension | 256 |
| FPN levels | P2--P5 |
| `beta_cons` | 1.0 |
| `beta_dom` | 1.0 |
| `beta_dec` | 0.05 |
| `lambda_neg` | 1.0 |
| `lambda_att` | 0.5 |
| Dominance margin | 0.2 |
| Numerical epsilon | 0.01 |
| Resize | short side 800, long side at most 1333 |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 |

The paper does not numerically specify `beta_wake`, learning rate, batch size, epoch count, LR milestones, directional-bias initialization, exact three seeds, split seed, or diagnostic thresholds. Their fixed **release defaults** are listed and justified in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), recorded in every run manifest, and never inferred from reported test results.

## Ablations

All table components are represented as config-only switches:

```bash
python scripts/train.py --config configs/ablations/base_detector.yaml
python scripts/train.py --config configs/ablations/naive_wake_fusion.yaml
python scripts/train.py --config configs/ablations/wake_supervision.yaml
python scripts/train.py --config configs/ablations/consistency.yaml
python scripts/train.py --config configs/ablations/dominance.yaml
python scripts/train.py --config configs/ablations/decorrelation.yaml
```

## ICIG 2026 Poster Presentation

**Paper:**  
*Hull First, Wake Second: Wake-Reliance Suppression for Robust Maritime Vessel Detection*

**Resources:**  
- [1-Minute Video Presentation](assets/icig2026/HullWake_ICIG2026_1min_Presentation.MOV)  

## Citation

```bibtex
@inproceedings{wang2026hullwake,
  title     = {Hull First, Wake Second: Wake-Reliance Suppression for Robust Maritime Vessel Detection},
  author    = {Wang, Yefan and Wang, Xingyu and Zhu, Ruibiao and Wu, Yusen},
  year      = {2026}
}
```

## License

Code is released under the MIT License. Dataset licenses remain with the original data providers. The additional Curated-Wake annotations are not included in this package.
