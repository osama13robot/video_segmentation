# Setup

## Option A: Kaggle (recommended — free GPU)

1. Create a new Kaggle notebook and upload
   `notebook/video_segmentation_split_screen.ipynb` (**File → Upload
   Notebook**), or copy its cells in manually.
2. **Settings → Accelerator → GPU** (T4 or P100).
3. **Settings → Internet → On** (needed to download YOLO11 weights, Mask
   R-CNN/COCO weights, and — if used — Pexels videos).
4. Attach your input video:
   - **Add Input → Upload** your `.mp4` as a Kaggle Dataset, *or*
   - Add a `PEXELS_API_KEY` secret (**Add-ons → Secrets**) to auto-download
     free stock clips — get a free key at <https://www.pexels.com/api/>.
5. In the notebook's `CFG` class and the Step 1 cell, either leave the
   automatic video discovery as-is (it scans `/kaggle/input/**/*.mp4`), or
   point `videos = [...]` at your exact file path.
6. **Run All**. Outputs (split-screen video, `metrics_summary.csv`,
   `metrics.png`, `contact_sheet.png`, `credits.csv` if Pexels was used) land
   in `/kaggle/working/outputs/` — visible under the **Output** tab once the
   run finishes.

## Option B: Running locally

Requires an NVIDIA GPU for reasonable speed (Mask R-CNN in particular is slow
on CPU — expect single-digit FPS to become sub-1-FPS).

```bash
git clone <this-repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then open `notebook/video_segmentation_split_screen.ipynb` in Jupyter:

```bash
pip install jupyter
jupyter notebook notebook/video_segmentation_split_screen.ipynb
```

Adjust these two things for a local (non-Kaggle) environment before running:

- In the setup cell, `WORK = pathlib.Path("/kaggle/working") if ... else
  pathlib.Path("./work")` already falls back to `./work` automatically when
  `/kaggle` doesn't exist — no change needed.
- Point `videos = [...]` (Step 1) at your local `.mp4` path(s) instead of
  relying on the `/kaggle/input` scan.
- `ffmpeg` must be installed and on `PATH` (`apt install ffmpeg` /
  `brew install ffmpeg`).

## Regenerating the notebook from source

`notebook/build_notebook.py` builds the `.ipynb` from plain Python cell
definitions — useful for code review/diffs, since raw `.ipynb` JSON diffs
badly. To regenerate after editing `build_notebook.py`:

```bash
pip install nbformat
python notebook/build_notebook.py
```

This writes `notebook/video_segmentation_split_screen.ipynb`.

## Configuration reference (`CFG` class in the notebook)

| Setting | Default | Meaning |
|---|---|---|
| `OUT_W, OUT_H` | 1080, 1920 | Output canvas size (YouTube Shorts) |
| `LAYOUT` | `"stack"` | `"stack"` = A on top/B below, `"side"` = A left/B right |
| `FIT` | `"cover"` | `"cover"` fills each panel (crops), `"contain"` fits with black bars |
| `CROP_BIAS` | 0.4 | Where to crop when `FIT="cover"` (0=top/left, 1=bottom/right) |
| `MAX_SECONDS` | 12 | Clip length cap, keeps runtime bounded |
| `MAX_FPS` | 30 | Frame-rate cap |
| `CONF` | 0.35 | Detection confidence threshold, both models |
| `ALPHA` | 0.45 | Mask overlay opacity |
| `COLOR_BY` | `"class"` | `"class"`: same colour per object type across both panels; `"instance"`: one colour per tracked ID |
| `YOLO_WEIGHTS` | `yolo11s-seg.pt` | Any Ultralytics `-seg` checkpoint |
| `METRIC_SCALE` | 0.25 | Downscale factor for masks before computing IoU metrics (speed) |
