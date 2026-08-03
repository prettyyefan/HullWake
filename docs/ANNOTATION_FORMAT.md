# Curated-Wake annotation contract

HullWake uses a strict extension of COCO detection JSON. Standard COCO keys remain valid; extra fields carry wake diagnostics. No annotation file or image from Curated-Wake is distributed with this code.

## Top-level keys

| Key | Type | Required | Meaning |
|---|---|---:|---|
| `images` | list | yes | COCO image records plus source/sequence metadata |
| `annotations` | list | yes | Hull/vessel instances |
| `categories` | list | yes | Contiguous foreground categories; the paper setting has one `vessel` category |
| `regions` | list | yes | Non-vessel wake-like-negative and water-clutter masks |
| `info`, `licenses` | object/list | no | Standard COCO metadata |

## Image record

```json
{
  "id": 1,
  "file_name": "relative/path/frame_000001.jpg",
  "width": 1920,
  "height": 1080,
  "source": "smd",
  "sequence_id": "smd_onboard_01"
}
```

`source` must be one of `smd`, `seadronessee`, or `ships_vessels` in the paper setting. `sequence_id` is mandatory for video-derived frames and must be stable across adjacent frames. The split tool keeps a sequence in exactly one split.

## Vessel annotation

```json
{
  "id": 101,
  "image_id": 1,
  "category_id": 1,
  "bbox": [620.0, 410.0, 88.0, 31.0],
  "area": 2018.0,
  "iscrowd": 0,
  "segmentation": [[620.0, 420.0, 640.0, 410.0, 708.0, 418.0, 700.0, 441.0]],
  "wake_attribute": "clear_wake",
  "wake_segmentation": [[610.0, 421.0, 520.0, 392.0, 415.0, 370.0, 510.0, 430.0]]
}
```

The numeric values above illustrate the schema only; they do not describe a real image or released annotation.

Required extension fields:

- `wake_attribute`: exactly one of `clear_wake`, `weak_no_wake`, or `ambiguous`.
- `wake_segmentation`: COCO polygons or compressed/uncompressed RLE for the wake belonging to this vessel. An empty list is valid for a confirmed weak/no-wake vessel.
- `segmentation`: hull mask. This is used for validation and optional analysis; Faster R-CNN training uses `bbox` for localization.

## Diagnostic region

```json
{
  "id": 501,
  "image_id": 1,
  "type": "wake_like_negative",
  "segmentation": [[110.0, 80.0, 240.0, 90.0, 260.0, 120.0, 125.0, 112.0]]
}
```

Allowed `type` values are:

- `wake_like_negative`: elongated or trailing-looking water structure without a vessel hull;
- `water_clutter`: waves, reflection, turbulence, shoreline trace, or other confusing water-surface texture.

An image may contain zero or more regions of each type. Empty types are represented by absence of records, not a fake all-zero polygon.

## Mask conventions

- Polygon coordinates are absolute image-space `(x, y)` pairs.
- RLE follows the COCO convention and uses the image's full `height` and `width`.
- Masks must not extend outside the image after clipping.
- A vessel's `segmentation` should cover the visible hull, not its wake.
- A vessel's `wake_segmentation` should cover the visible wake, not the hull.
- Wake-like-negative and water-clutter regions must not encode a true vessel hull.

## Category mapping

TorchVision reserves label `0` for background. Dataset category IDs are mapped to contiguous labels `1..K` in ascending category-ID order and mapped back during evaluation. The paper's class-agnostic vessel setup is:

```json
{"categories": [{"id": 1, "name": "vessel"}]}
```

## Validation

Run:

```bash
python tools/validate_annotations.py \
  --annotations /path/to/annotations.json \
  --images /path/to/images \
  --require-files --require-wake-masks
```

The validator checks IDs, dimensions, file paths, boxes, category references, attributes, polygon/RLE syntax, sequence metadata, and mask availability. It never rewrites user annotations.
