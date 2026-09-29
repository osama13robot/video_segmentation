# Input data

The raw input video used to produce `results/` is **not committed** to this
repository.

## Why

The sample clip is a stock video from [Pexels](https://www.pexels.com). The
[Pexels License](https://www.pexels.com/license/) allows free use (including
some commercial use) but is **not** public domain or CC0 — redistributing the
raw file itself (as opposed to a derived work like the rendered comparison
video in `results/`) sits in a grayer area than this repo needs to test. So:

- **Committed:** `results/comparison_output.mp4` — this project's own
  derived output (segmentation overlays + layout), which is original work.
- **Not committed:** the raw input clip.

If you re-run this project with Pexels footage via the notebook's built-in
downloader, it also writes a `credits.csv` alongside the outputs recording
each clip's author, source URL, and license — keep that alongside anything
you publish.

## Getting input footage yourself

Pick one:

1. **Free stock, auto-downloaded (Pexels):** get a free key at
   <https://www.pexels.com/api/>, add it as a Kaggle secret
   `PEXELS_API_KEY` (or environment variable locally), and the notebook's
   Step 1 cell downloads a few portrait clips automatically.
2. **Your own footage:** upload an `.mp4` as a Kaggle Dataset (**Add Input →
   Upload**) or drop it in `./data/` locally, then point the notebook's
   `videos = [...]` list at it.
3. **Strictly public-domain footage:** try
   [Wikimedia Commons](https://commons.wikimedia.org) or the
   [Internet Archive](https://archive.org), and paste direct `.mp4` URLs into
   `DIRECT_URLS` in the notebook's Step 1 cell.

For best results with this pipeline, use **vertical/portrait video** — the
notebook center-crops landscape or square input to 9:16, which loses the
sides of the frame.
