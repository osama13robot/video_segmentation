# Video Object Segmentation: YOLO11-seg vs Mask R-CNN (Shorts-style split screen)

Segment every object in a vertical (9:16) video with **two different computer-vision
libraries** and render a **YouTube-Shorts-sized (1080x1920) split-screen comparison**
video — one model on top, the other on the bottom, same footage, same frame.

Two example domains so far, each showing a very different side of the same two
models — see [Results across domains](#results-across-domains) below:

| | Street scene | Citrus orchard |
|---|---|---|
| | [![Street contact sheet](results/street/contact_sheet.png)](results/street/comparison_output.mp4) | [![Citrus contact sheet](results/citrus/contact_sheet.png)](results/citrus/comparison_output.mp4) |
| Video | [▶ full](results/street/comparison_output.mp4) · [▶ preview](results/street/comparison_output_preview.mp4) | [▶ full](results/citrus/comparison_output.mp4) |

## Why this project

This started from Eddie Smolyansky's article
[*The Basics of Video Object Segmentation*](https://medium.com/@eddiesmo/video-object-segmentation-the-basics-758e77321914),
which frames video object segmentation as **pixel-level object tracking** and
introduces the two metrics the field still uses:

- **Region similarity (J)** — IoU between predicted mask and ground truth
- **Contour accuracy (F)** — F-measure over the mask boundary

The article's classic methods (OSVOS, MaskTrack) are **semi-supervised**: they're
given the correct mask for frame 1 and have to track it forward. This repo instead
runs the **unsupervised** setting — each model finds objects on its own, in every
frame, with no hint — using two modern, off-the-shelf libraries, and compares how
they behave on the same clip. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for
exactly how this maps onto the article's framing, and why the metrics here are
*proxies* for J and F rather than the real thing.

## What it does

```
input video (any size) --> resize/crop to 1080x1920
                              |
              +---------------+---------------+
              |                               |
      Model A: YOLO11-seg              Model B: Mask R-CNN
      (Ultralytics, ByteTrack)         (torchvision, IoU tracker)
              |                               |
        colored masks + IDs             colored masks + IDs
              |                               |
        top half (1080x960)          bottom half (1080x960)
              +---------------+---------------+
                              |
                 stacked 1080x1920 Shorts video
                 + per-frame metrics (CSV/plots)
```

- **Model A — YOLO11-seg** (Ultralytics): single-stage, real-time instance
  segmentation with built-in **ByteTrack** for object IDs.
- **Model B — Mask R-CNN** (torchvision, ResNet-50-FPN-v2): two-stage detector,
  generally slower and often more thorough, paired with a small custom
  IoU-matching tracker (implemented in the notebook, no built-in tracker in
  torchvision).
- Both are trained on **COCO** (80 classes), so the comparison uses the same
  vocabulary.
- Output is piped straight into **ffmpeg** (H.264/yuv420p) — the format YouTube
  and every browser plays natively.

## Results across domains

Full numbers: [`results/street/metrics_summary.csv`](results/street/metrics_summary.csv),
[`results/citrus/metrics_summary.csv`](results/citrus/metrics_summary.csv)

| Metric | Street — YOLO11 | Street — Mask R-CNN | Citrus — YOLO11 | Citrus — Mask R-CNN |
|---|---:|---:|---:|---:|
| Speed | **30.3 FPS** | 4.1 FPS | **15.3 FPS** | 2.0 FPS |
| Avg. objects / frame | 4.0 | **19.6** | 14.2 | **74.6** |
| Distinct tracked IDs | **78** | 993 | **89** | 2,091 |
| Temporal stability | **0.88** | 0.71 | **0.71** | 0.62 |
| A-vs-B agreement | 0.51 | 0.51 | 0.47 | 0.47 |

<p align="center">
  <img src="results/street/metrics.png" width="410" alt="Street metrics charts">
  <img src="results/citrus/metrics.png" width="410" alt="Citrus metrics charts">
</p>

**The pattern holds across both domains, and gets more extreme on citrus:**
Mask R-CNN consistently finds far more objects per frame than YOLO11-seg (5x on
the street clip, **5.2x** in the orchard, where small, clustered, occluded
fruit is an even harder case for a single-stage detector), at a large and
consistent speed cost (~7x slower both times). Tracking degrades on both
models when the scene changes from mostly-static street furniture/pedestrians
to a panning shot over near-identical fruit — temporal stability drops from
0.88/0.71 to 0.71/0.62, and track-ID counts explode (993 → 2,091 for Mask
R-CNN) as the trackers lose and re-acquire near-duplicate objects. See
[`docs/RESULTS.md`](docs/RESULTS.md) for the full write-up of each run,
including a citrus-specific finding: both models, trained only on COCO's
generic `orange`/`apple` classes, **misclassify some oranges as apples** —
a clear sign general-purpose detectors aren't reliable for agricultural
species identification without domain-specific training. See
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md#limitations) for why none of
these numbers say who is "more correct" — there's no ground truth for either
clip.

## Repo structure

```
.
├── notebook/
│   ├── video_segmentation_split_screen.ipynb   # the Kaggle notebook (run this)
│   └── build_notebook.py                        # generates the .ipynb from source (for diffs/PRs)
├── results/
│   ├── street/                                   # example 1: busy tram/street scene
│   │   ├── comparison_output.mp4                 # full split-screen output (1080x1920)
│   │   ├── comparison_output_preview.mp4         # small preview version
│   │   ├── metrics_summary.csv                   # per-video summary metrics
│   │   ├── metrics.png                           # summary charts
│   │   └── contact_sheet.png                     # three sample frames
│   └── citrus/                                   # example 2: citrus orchard (agriculture)
│       ├── comparison_output.mp4
│       ├── metrics_summary.csv
│       ├── metrics.png
│       └── contact_sheet.png
├── data/
│   └── README.md                                 # how to get/point at input footage (not committed, see below)
├── docs/
│   ├── METHODOLOGY.md                            # how this maps to the article; metric definitions & limits
│   ├── RESULTS.md                                # detailed results write-up
│   └── SETUP.md                                  # step-by-step Kaggle setup
├── requirements.txt
├── LICENSE
└── .gitattributes                                # Git LFS config for video/image assets
```

## Quick start

1. Open `notebook/video_segmentation_split_screen.ipynb` in **Kaggle** (GPU +
   Internet on) — see [`docs/SETUP.md`](docs/SETUP.md) for the full checklist.
2. Point it at your own vertical video (Kaggle Dataset upload) or a free stock
   source (Pexels API key, optional). Config lives in the `CFG` class near the
   top of the notebook.
3. Run all cells. Outputs land in `/kaggle/working/outputs/`.

Running locally instead of Kaggle works too — see
[`requirements.txt`](requirements.txt) and [`docs/SETUP.md`](docs/SETUP.md#running-locally).

## Input footage / licensing

The sample videos used for `results/` (street and citrus) came from Pexels.
Pexels/Pixabay/Mixkit footage is **free to use but not public domain** — see
[`data/README.md`](data/README.md) for the license note and why the raw input
clips aren't committed to this repo (only the derived comparison videos are,
which are this project's own output).

## Extending this

- Add **SAM 2** as a third model/panel — it's the modern, semi-supervised
  successor to the article's OSVOS/MaskTrack, seeded with first-frame boxes.
- Run on **DAVIS 2016/2017** (the dataset from the article) to get *real* J
  and F scores against ground truth, instead of the proxy metrics here.
- Add a "disagreement" panel that highlights only the pixels where A and B
  differ.
- Swap `CFG.LAYOUT`/`CFG.FIT` for a side-by-side layout or letterboxed
  (no-crop) framing.

### Agriculture / precision-farming extensions

The citrus run above shows COCO-pretrained models struggling with small,
clustered, occluded fruit and mixing up species (`orange` vs `apple`). Ways
to push this further, roughly cheapest to most involved:

- **SAHI** (Slicing Aided Hyper Inference) — tiles each frame before running
  either detector, then merges results. The standard fix for small/dense
  objects, no retraining required; a natural "Model C" panel.
- **CitDet** — a purpose-built in-orchard citrus detection dataset; fine-tune
  YOLO on it for a fair citrus baseline instead of relying on COCO's generic
  `orange` class.
- **Roboflow Universe** — search "citrus"/"fruit counting" for community
  fine-tuned YOLOv8/v11 checkpoints that drop straight into this repo's
  `Segmenter` interface.
- **PlantCV** — a full plant-phenotyping toolkit (segmentation + trait
  extraction: fruit size, canopy area, leaf health), useful beyond counting.
- **Other crops to try the same way:** grapes via **WGISD** (Wine Grape
  Instance Segmentation Dataset — clusters, not single fruit), wheat via
  **Global Wheat Head Detection** (dense, tiny, textured heads), or
  strawberries/coffee cherries via Roboflow's ripeness-labeled sets.

## Reference

Smolyansky, E. (2017). *The Basics of Video Object Segmentation*.
[medium.com/@eddiesmo](https://medium.com/@eddiesmo/video-object-segmentation-the-basics-758e77321914)

## License

Code in this repo is [MIT-licensed](LICENSE). Model weights (YOLO11, Mask
R-CNN/COCO) and any third-party footage keep their own licenses — see
[`data/README.md`](data/README.md).
