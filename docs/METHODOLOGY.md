# Methodology

## 1. How this maps to the article

[*The Basics of Video Object Segmentation*](https://medium.com/@eddiesmo/video-object-segmentation-the-basics-758e77321914)
(Smolyansky, 2017) splits video object segmentation into two settings:

| Setting | Input | Classic methods (2016-17) |
|---|---|---|
| **Unsupervised** (video saliency) | nothing — the model decides what the "main" object(s) are | — |
| **Semi-supervised** | ground-truth mask of frame 1 | OSVOS, MaskTrack |

The article's two "instant classic" methods — **OSVOS** (fine-tune a
classification net on the first-frame mask, then segment every later frame
independently) and **MaskTrack** (feed the previous frame's predicted mask
back into the network, plus an optical-flow stream) — are both
**semi-supervised**: they get a free hint (frame 1's correct mask) and are
scored on how well they propagate it.

This project instead runs the **unsupervised, multi-instance** case: neither
model is told what to look for or given any ground truth. Each model detects
and segments *every* object it recognizes, in every frame, independently, and
a **tracker** stitches those per-frame detections into consistent identities
over time — playing a similar role to the temporal propagation in
MaskTrack/OSVOS, but built from a modern real-time detector instead of a
frame-by-frame fine-tuned net.

| This project | Article's framing |
|---|---|
| YOLO11-seg detects objects independently each frame | ≈ OSVOS: per-frame segmentation, no temporal signal *within the segmenter* |
| ByteTrack links YOLO11's detections across frames | plays the role MaskTrack's mask-propagation plays, but as a separate tracking-by-detection step rather than a network input |
| Mask R-CNN detects objects independently each frame | also ≈ OSVOS-style: independent per-frame inference |
| Custom IoU tracker links Mask R-CNN's detections | a simpler stand-in for temporal propagation — mask IoU between consecutive frames, no learned motion model |

**Where SAM 2 fits (see the "Extending this" section of the README):** SAM 2 is
the closest modern equivalent to OSVOS/MaskTrack's *semi-supervised* setting —
you give it a prompt (point, box, or mask) on frame 1 and it propagates the
mask forward using a learned memory mechanism. Adding it as a third panel
would let this repo directly compare the semi-supervised and unsupervised
paradigms the article describes, on the same footage.

## 2. Why the metrics here are proxies, not J & F

The article's two metrics require **ground-truth masks**:

- **Region Similarity (J):** IoU between the predicted mask and the ground
  truth mask.
- **Contour Accuracy (F):** F-measure over the boundary contours of predicted
  vs. ground-truth masks.

Free stock video (or any video you film/download yourself) has **no ground
truth** — nobody has hand-annotated the correct mask for every frame. Computing
real J/F requires an annotated benchmark like **DAVIS** (the dataset the
article itself introduces). So this repo reports different, *reference-free*
numbers that only need the two models' outputs:

| Metric | Definition | What it tells you | What it can't tell you |
|---|---|---|---|
| **Agreement** | mean IoU between model A's masks and model B's masks, per frame (Hungarian-style best-match, both directions averaged) | how much the two models overlap | which one (if either) is *correct* — two models can agree and both be wrong |
| **Temporal stability** | IoU of the *same tracked ID's* mask between consecutive frames | how much a single model's own output flickers frame-to-frame | conflates segmentation quality with tracker quality — a good segmenter + bad tracker still scores low |
| **Speed** | wall-clock ms/frame, GPU-synced | end-to-end cost, inference + mask postprocessing | doesn't separate model compute from Python overhead |
| **Track count** | number of distinct IDs ever assigned | rough proxy for ID fragmentation — much higher than the true object count means the tracker is losing and re-finding objects | doesn't distinguish "many real objects" from "one object counted many times" without watching the video |

**Practical consequence, observed in `results/metrics_summary.csv`:** Mask
R-CNN detects ~5x more objects/frame than YOLO11-seg on the sample clip (a
crowded street scene), which is plausibly a real detection-recall difference —
but its 993 distinct track IDs (vs. YOLO11+ByteTrack's 78) mostly reflects the
custom IoU tracker in this repo losing identity in a crowd, not 993 different
people. Don't read "more tracks" as "more objects" without checking the video.

## 3. Fairness of the comparison

Both models:
- run on the **same resized 1080x1920 frame** (no resolution advantage either
  way);
- use the **same confidence threshold** (`CFG.CONF`, default 0.35);
- are restricted to the **same 80 COCO classes** for the class label shown
  (though each network's internal feature extractor and training recipe
  differ, so raw recall/precision per class still isn't identical);
- are timed with `torch.cuda.synchronize()` around inference so GPU work is
  fully counted, not just kernel-launch time.

What is **not** controlled for:
- YOLO11-seg ships a tracker (ByteTrack); Mask R-CNN does not, so this repo
  adds a simple greedy IoU tracker for it. Any tracking-quality gap partly
  reflects ByteTrack vs. a minimal from-scratch tracker, not YOLO11 vs. Mask
  R-CNN as segmenters per se.
- Model sizes differ (`yolo11s-seg` — small — vs. Mask R-CNN ResNet-50-FPN-v2,
  a heavier backbone). A fairer speed comparison would fix compute budget
  (e.g. `yolo11x-seg` vs. Mask R-CNN, or a lighter Mask R-CNN backbone).

## 4. Reproducing on a benchmark with real ground truth

To get true J/F scores instead of the proxies above:

1. Download **DAVIS 2016/2017** (linked from the article).
2. Replace the video-reading cell with DAVIS frame loading + its provided
   masks.
3. Replace `agreement()`/`stability()` in the notebook with:
   - `J = IoU(pred_mask, gt_mask)` per object, per frame.
   - `F` = boundary F-measure (see the DAVIS toolkit for the reference
     implementation — pixel-distance-tolerant boundary matching).
4. Average J and F per sequence, then across sequences, as DAVIS does.

This is listed as a stretch goal in the README's "Extending this" section.
