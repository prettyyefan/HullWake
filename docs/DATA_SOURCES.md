# Data acquisition notes

This repository does not download data automatically because the three providers use different licenses, access flows, and directory formats. Automatic scraping would make license acceptance and version identity ambiguous.

## Original sources

| Source | Maintainer page | Used role |
|---|---|---|
| Ships/Vessels in Aerial Images | https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images | Aerial vessel boxes and selected images |
| Singapore Maritime Dataset | https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset | On-shore/on-board maritime video frames and object annotations |
| SeaDronesSee | https://seadronessee.cs.uni-tuebingen.de/dataset | Cross-view maritime detection frames and annotations |

## Curation boundary

The paper's approximately 10,000-image Curated-Wake set is not the union of every downloaded source file. More than 3,000 images are selected from each source, existing vessel/hull boxes are retained when suitable, missing hull boxes are added, and the diagnostic annotations described in `ANNOTATION_FORMAT.md` are created.

The code intentionally does not encode a private filename list. A legal local copy of the curated master JSON is the source of truth. `tools/create_splits.py` converts that master file into deterministic split files without copying images.

## Frame leakage

For video sources, every image record must contain a `sequence_id`. All frames with the same `sequence_id` remain together during splitting. For independent aerial images, use a stable per-scene or per-original-image identifier; augmentations or crops of one original image must share it.
