# Results

## Input clip

- Source: Pexels (portrait/vertical stock video), original resolution
  2160x3840, 30 fps, ~19.5s. See [`data/README.md`](../data/README.md) for
  the license note.
- Scene: a busy street/tram stop with pedestrians, a tram, cars and street
  furniture — a reasonably hard, crowded scene for both detection and
  tracking.
- Processed: first ~15s resized/cropped to 1080x1920, 30 fps, **449 frames**.

## Full metrics

See [`metrics_summary.csv`](../results/metrics_summary.csv) for the raw row;
summarized here:

| Metric | YOLO11-seg (A) | Mask R-CNN (B) |
|---|---:|---:|
| Mean inference time | 33.0 ms/frame | 241.6 ms/frame |
| Speed | 30.3 FPS | 4.1 FPS |
| Mean objects detected / frame | 4.0 | 19.6 |
| Distinct track IDs (whole clip) | 78 | 993 |
| Mean temporal stability (IoU, same ID, consecutive frames) | 0.884 | 0.713 |
| Mean A-vs-B mask agreement | 0.513 | 0.513 |

Charts: [`metrics.png`](../results/metrics.png) — speed, agreement, temporal
stability, and agreement-over-time for the clip.

Sample frames: [`contact_sheet.png`](../results/contact_sheet.png) — three
frames (≈25%, 50%, 75% through the output video) with both models' overlays
visible.

## What the sample frames show

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

## Interpretation

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

## Caveats specific to this run

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
