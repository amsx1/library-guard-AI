# data/sample/

`sample_video.mp4` is a **synthetic clip generated with OpenCV** (moving
shapes, no real people). It contains no third-party content and no personal
data, so it is safe to commit.

Purpose:

* smoke-testing `scripts/predict.py` and the pipeline end-to-end,
* optional demo input in the Streamlit dashboard,
* CI-friendly (tiny: a few seconds, ~16 KB).

Because the clip contains no real people, the correctly expected output is
"no events" — its job is to prove the pipeline runs, not to showcase
detections. For real behavior examples use your own authorized footage or a
properly licensed dataset (see `docs/datasets.md`).