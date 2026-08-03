# Evaluation protocol

All thresholds are stored under `evaluation` in the YAML configuration and copied into each run manifest.

## Standard detection metrics

`AP`, `AP50`, and `AP75` use the official `pycocotools` COCO evaluator with IoU thresholds 0.50:0.05:0.95 and at most 100 detections per image. The JSON result also reports `AP50_95` as an explicit alias of COCO `AP`, avoiding ambiguous AP naming across toolkits.

## AP_NoWake

The evaluator creates an in-memory COCO view in which weak/no-wake instances are regular ground truth and all clear-wake or ambiguous vessel instances are marked ignored. Predictions matching an ignored vessel are not counted as false positives. `AP_NoWake` is COCO AP over IoU 0.50:0.05:0.95 on this view.

## WG_AP

Group AP is computed independently for:

- `clear_wake`;
- `weak_no_wake`.

Ambiguous instances are ignored. Worst-group AP is

\[
\operatorname{WG\text{-}AP}=\min_{g\in\{\text{clear},\text{weak/no}\}}\operatorname{AP}_g.
\]

If a requested group has no valid ground-truth instance, the evaluator returns `null` for that group and refuses to report a misleading worst-group value.

## FP_WakeLike

For each prediction with score at least `0.50`:

1. discard it from this diagnostic if it matches any vessel ground truth at IoU at least `0.50`;
2. rasterize the union of `wake_like_negative` regions for the image;
3. compute the fraction of the predicted box covered by that union;
4. count one false positive when coverage is at least `0.25`.

The metric is a count on the fixed test split, so split identity and the three thresholds must accompany any comparison.

## Delta_wake

At inference the same proposal is classified twice: once normally and once after predicted wake-response attenuation with `lambda_att=0.5`. NMS is selected by the normal score, and the corresponding attenuated score is carried to the final detection.

Final detections are greedily matched to ground-truth vessels by descending score at IoU at least `0.50`. For matched detections,

\[
\Delta_{\mathrm{wake}}=\frac{1}{|\mathcal P|}\sum_{i\in\mathcal P}(s_i-\tilde{s}_i).
\]

The signed difference is retained; it is not clipped at zero. A lower mean indicates less confidence dependence on the predicted wake response.

## Source-wise results

The evaluator also reports the same group metrics per source when `source` is available. Image IDs, not filename patterns, define the source subsets.
