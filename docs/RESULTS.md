# Results

This repo has three example runs: a street scene, a citrus orchard, and a
post-harvest sorting clip. The street run is closer to the article's typical
VOS test footage; citrus stress-tests small/dense/occluded objects; the
harvesting run stress-tests something harder still — objects with **no
matching COCO class at all**.

## Example 1: Street scene

### Input clip

- Source: Pexels (portrait/vertical stock video), original resolution
  2160x3840, 30 fps, ~19.5s. See [`data/README.md`](../data/README.md) for
  the license note.
- Scene: a busy street/tram stop with pedestrians, a tram, cars and street
  furniture — a reasonably hard, crowded scene for both detection and
  tracking.
- Processed: first ~15s resized/cropped to 1080x1920, 30 fps, **449 frames**.

### Full metrics

See [`metrics_summary.csv`](../results/street/metrics_summary.csv) for the raw row;
summarized here:

| Metric | YOLO11-seg (A) | Mask R-CNN (B) |
|---|---:|---:|
| Mean inference time | 33.0 ms/frame | 241.6 ms/frame |
| Speed | 30.3 FPS | 4.1 FPS |
| Mean objects detected / frame | 4.0 | 19.6 |
| Distinct track IDs (whole clip) | 78 | 993 |
| Mean temporal stability (IoU, same ID, consecutive frames) | 0.884 | 0.713 |
| Mean A-vs-B mask agreement | 0.513 | 0.513 |

Charts: [`metrics.png`](../results/street/metrics.png) — speed, agreement, temporal
stability, and agreement-over-time for the clip.

Sample frames: [`contact_sheet.png`](../results/street/contact_sheet.png) — three
frames (≈25%, 50%, 75% through the output video) with both models' overlays
visible.

### What the sample frames show

Looking at `contact_sheet.png` (tram/street scene):

- **YOLO11-seg (top row)** cleanly segments the large, nearby, high-confidence
  objects — the tram (labeled `bus` or `train`, depending on the frame — see
  note below), a couple of the closest people, and a car — and gives each a
  stable-looking ID number in the 190-210 range that persists across the
  three sampled frames.
- **Mask R-CNN (bottom row)** segments many more instances per frame:
  the same tram and car, plus a long tail of smaller/farther pedestrians that
  YOLO11-seg doesn't surface at this confidence threshold, and even a
  `clock`/`traffic light`/`dog` in the third frame that YOLO11 misses
  entirely in that frame. Its confidence scores on the extra small detections
  are often lower (0.4-0.6 vs 0.8-1.0 on the obvious objects).
- **Class-label disagreement:** in frame 74/374, YOLO11 calls the tram a
  `bus`; in frame 224 it calls the same object a `train`. Mask R-CNN calls it
  `bus`/`train` too but is more consistent frame-to-frame in this sample.
  This kind of flip is expected — trams are visually ambiguous under COCO's
  vehicle classes (no `tram` class exists), and it's exactly the sort of
  disagreement the "agreement" metric is meant to surface for closer
  inspection, rather than resolve automatically.

### Interpretation

- **Speed vs. recall trade-off, as expected:** YOLO11-seg's ~7.3x speed
  advantage comes with meaningfully lower recall in a dense scene (4.0 vs
  19.6 objects/frame). If the downstream use case needs every small/occluded
  person found (e.g. safety monitoring), Mask R-CNN's extra recall may be
  worth the ~7x slower inference. If it needs to run at interactive/real-time
  rates or track identities reliably, YOLO11-seg + ByteTrack is the clear
  choice here.
- **Tracker quality dominates the ID-count gap.** 993 vs 78 IDs is a ~13x
  difference that is almost certainly driven more by ByteTrack (Kalman-filter
  + IoU + re-ID cascade) vs. this repo's minimal greedy-IoU tracker than by
  any property of the segmenters themselves. A fairer read: pair Mask R-CNN's
  detections with ByteTrack too, and see how much of the gap closes (listed
  as a possible extension).
- **~0.51 agreement is neither high nor low in isolation** — it's most useful
  as a per-frame/per-clip signal for *where to look*. The dip around frame
  ~110-120 and ~330-350 in `metrics.png` (bottom-right subplot) are good
  candidates to inspect manually for crowding or occlusion events.

### Caveats specific to this run

- Only ~15s of a ~19.5s clip was processed (`CFG.MAX_SECONDS = 12` at 30
  fps ≈ 449 frames after the fps cap) — a longer or different clip may shift
  these numbers.
- `CFG.CONF = 0.35` for both models controls this whole recall/precision
  balance; lowering it would likely shrink YOLO11's recall gap (at the cost
  of more false positives) and is an easy first experiment.
- One clip, one scene type (dense urban/pedestrian). Numbers will differ
  substantially on e.g. a single-subject sports clip or a static-camera
  product shot — the notebook is written to process a list of clips in one
  run for exactly this kind of comparison.

## Example 2: Citrus orchard (agriculture)

### Input clip

- Source: Pexels (portrait/vertical stock video), 1080x1920, 60 fps. See
  [`data/README.md`](../data/README.md) for the license note.
- Scene: a panning shot across a citrus tree heavy with fruit — small, dense,
  frequently occluded objects against similarly-colored foliage, and
  continuous camera motion (unlike the mostly-static street scene).
- Processed: **292 frames**.

### Full metrics

See [`metrics_summary.csv`](../results/citrus/metrics_summary.csv) for the
raw row; summarized here:

| Metric | YOLO11-seg (A) | Mask R-CNN (B) |
|---|---:|---:|
| Mean inference time | 65.5 ms/frame | 508.4 ms/frame |
| Speed | 15.3 FPS | 2.0 FPS |
| Mean objects detected / frame | 14.2 | 74.6 |
| Distinct track IDs (whole clip) | 89 | 2,091 |
| Mean temporal stability (IoU, same ID, consecutive frames) | 0.710 | 0.624 |
| Mean A-vs-B mask agreement | 0.468 | 0.468 |

Charts: [`metrics.png`](../results/citrus/metrics.png). Sample frames:
[`contact_sheet.png`](../results/citrus/contact_sheet.png) — three frames
across the clip with both models' overlays visible.

### What the sample frames show

- **YOLO11-seg (top row)** finds the larger, less-occluded oranges near the
  edges of the canopy — 8 to 19 per frame across the three samples — all
  labeled `orange` with moderate confidence (0.4-0.65). It misses most fruit
  clustered deeper in the foliage.
- **Mask R-CNN (bottom row)** finds dramatically more — 75 to 78 per
  frame — including small, partially hidden fruit YOLO11 never surfaces.
  Confidence on these extra detections spans a wide range (0.35 to 0.99).
- **Misclassification:** several fruit are labeled `apple` (purple boxes) by
  Mask R-CNN rather than `orange`, on objects that are visibly oranges.
  YOLO11 does not show this error as often in these samples, but neither
  model was trained on citrus specifically — both are relying on COCO's
  generic `orange`/`apple` classes, which were never designed to
  disambiguate citrus cultivars under orchard lighting, size variation, and
  partial occlusion. This is the clearest actionable finding in this run: a
  COCO-pretrained model is not a reliable species classifier for agricultural
  fruit, however good its detection recall.

### Interpretation

- **The recall gap between models widens on harder, domain-specific scenes.**
  Mask R-CNN found ~5.2x more fruit per frame here (74.6 vs 14.2), a larger
  ratio than the ~5x gap on the street scene — consistent with two-stage
  detectors' region-proposal step giving it an edge on small, dense, occluded
  objects. For a yield-estimation use case, YOLO11-seg here is likely
  **undercounting fruit substantially**.
- **Tracking breaks down further than on the street scene.** Temporal
  stability fell for both models (0.88 → 0.71 for YOLO11-seg/ByteTrack,
  0.71 → 0.62 for Mask R-CNN's IoU tracker) and track-ID counts exploded
  (993 → 2,091 for Mask R-CNN). This is expected: the camera pans
  continuously and the fruit are visually near-identical to their neighbors,
  which is close to a worst case for IoU-based frame-to-frame matching.
  **Track count in this run should not be read as a fruit count** — it
  mostly reflects tracker failure, not the true number of oranges in the
  tree.
- **Agreement (~0.47) is lower than the street scene's ~0.51**, again
  consistent with the two models finding substantially different numbers of
  small objects to begin with.
- **Practical takeaway:** neither model, used off-the-shelf, is suitable for
  production fruit counting or species-accurate yield estimation. See the
  README's [Agriculture / precision-farming extensions](../README.md#agriculture--precision-farming-extensions)
  section for domain-specific datasets and models (CitDet, SAHI, Roboflow
  citrus checkpoints) that would address the recall and misclassification
  issues seen here.

### Caveats specific to this run

- One clip, one tree, one lighting condition — fruit density, occlusion, and
  lighting vary a lot across real orchards; these numbers shouldn't be
  read as general "YOLO vs Mask R-CNN for citrus" conclusions.
- `CFG.CONF = 0.35` (both models) again controls the whole precision/recall
  balance shown here.
- The continuous camera pan is a harder tracking case than a fixed
  surveillance-style camera would be; a static orchard camera (e.g. a
  fixed row-scanning rig) would likely show smaller track-ID inflation.

## Example 3: Post-harvest sorting (out-of-vocabulary objects)

### Input clip

- Source: Pexels (portrait/vertical stock video). See
  [`data/README.md`](../data/README.md) for the license note.
- Scene: a mostly-static, handheld shot of someone sorting onions/shallots
  into wire baskets/crates by hand — none of these object types (onion,
  garlic, wire basket, harvest crate) exist as a COCO class.
- Processed: **449 frames**.

### Full metrics

See [`metrics_summary.csv`](../results/harvesting/metrics_summary.csv) for
the raw row; summarized here:

| Metric | YOLO11-seg (A) | Mask R-CNN (B) |
|---|---:|---:|
| Mean inference time | 27.3 ms/frame | 211.1 ms/frame |
| Speed | 36.7 FPS | 4.7 FPS |
| Mean objects detected / frame | 2.5 | 10.8 |
| Distinct track IDs (whole clip) | 3 | 195 |
| Mean temporal stability (IoU, same ID, consecutive frames) | 0.961 | 0.953 |
| Mean A-vs-B mask agreement | 0.526 | 0.526 |

Charts: [`metrics.png`](../results/harvesting/metrics.png). Sample frames:
[`contact_sheet.png`](../results/harvesting/contact_sheet.png).

### What the sample frames show

- **YOLO11-seg (top row)** finds almost nothing — 2-3 objects/frame across
  the samples, essentially only `person` (the worker), with one stray
  `suitcase` label momentarily placed on the crate (frame 224, 0.52
  confidence). It largely declines to label the onions or containers at all.
- **Mask R-CNN (bottom row)** finds far more (10.8/frame average) by
  substituting the nearest COCO class it has for each unfamiliar object: the
  wire basket becomes `bowl` (0.43-0.65) or `dining table` (0.37-0.49), the
  onions inside become `apple` (0.52-0.65), and single frames add spurious
  `cell phone`, `donut`, and `cup` labels. None of these are correct.
- **This is a different failure mode from citrus.** In the citrus run, both
  models had a real (if imperfect) class to reach for (`orange`) and
  sometimes reached for a neighboring one (`apple`) — a narrow
  classification error. Here, there is **no correct class available at
  all**, and the two models respond differently: YOLO11-seg mostly abstains
  (arguably the more honest behavior for a system feeding downstream
  decisions), while Mask R-CNN forces a confident, wrong label onto the
  scene.

### Interpretation

- **Track-ID inflation here does not mean tracking failure from motion**,
  unlike citrus. Temporal stability is the *highest* of all three example
  runs for both models (0.961/0.953) — the camera is mostly static/handheld,
  not panning. Mask R-CNN's 195 IDs (vs YOLO11's 3) instead reflect its
  tracker faithfully following a *sequence of different wrong labels* over
  time (bowl → suitcase-adjacent → bowl again, etc.) as if each relabeling
  were a new object.
- **Agreement (~0.53) is misleadingly "normal"-looking** — similar in
  magnitude to the street scene's 0.51 — but here it mostly reflects the two
  models occasionally agreeing on *which pixels* belong to the basket, not
  on *what it is*. Mask agreement doesn't capture class correctness at all,
  which is a real limitation of this metric worth flagging if you show this
  clip alongside the others.
- **Practical takeaway:** this is the strongest evidence across the three
  runs that off-the-shelf, COCO-trained detectors are the wrong tool the
  moment a scene's objects fall outside COCO's 80 classes. Fine-tuning
  (or a purpose-built model) isn't optional here the way it might arguably
  be skippable for citrus — without it, one of the two models actively
  fabricates plausible-looking but false labels. See the README's
  [Agriculture / precision-farming extensions](../README.md#agriculture--precision-farming-extensions)
  section, which also notes that for bulk/uniform produce like this,
  density-map counting approaches (e.g. CSRNet, P2PNet) are typically a
  better fit than instance segmentation entirely.

### Caveats specific to this run

- One clip, one crop, one lighting/container setup — a different bulk
  produce or a different camera angle would very likely still fail, but not
  necessarily in the same way (different hallucinated classes).
- `CFG.CONF = 0.35` (both models) again controls the whole precision/recall
  balance shown here; a much lower threshold might reveal more of Mask
  R-CNN's hallucinated classes, not fewer.
- No ground truth exists to quantify "how wrong" the hallucinated labels
  are beyond visual inspection of the contact sheet and video.
