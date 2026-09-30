# LibraryGuard — Deployment Guide

## Option 1 — Local laptop (recommended for demos)

```bash
git clone https://github.com/<you>/libraryguard-ai.git
cd libraryguard-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open http://localhost:8501. Model weights (~12 MB total) download
automatically on first run.

## Option 2 — Docker

```bash
docker compose up --build
# dashboard on http://localhost:8501
```

`docker-compose.yml` mounts `./outputs` and `./data` so results persist on
the host. **Never** mount personal CCTV footage into a shared/public
container deployment.

## Option 3 — Streamlit Community Cloud

The repository is deployment-ready for Streamlit Community Cloud:

1. Push the repo to your GitHub account.
2. Go to https://share.streamlit.io → **New app**.
3. Select repo `libraryguard-ai`, branch `main`,
   main file path `app/streamlit_app.py`.
4. Deploy. First boot downloads CPU torch + ultralytics (a few minutes).

Notes & limits for Community Cloud:

- CPU-only; expect roughly 2–6 processed FPS depending on clip length.
  Use the **frame skip** slider for long videos.
- Uploaded videos live in the app's ephemeral session storage and are not
  persisted by this codebase. Still, avoid uploading sensitive footage to
  any third-party host — read their data policy first.
- If boot memory is tight, Community Cloud allows a `packages.txt` and
  `.streamlit/config.toml` — both are provided.

## Option 4 — On-prem / RTSP ingestion (future-ready)

The pipeline is source-agnostic (`src/pipeline.py`). To ingest an IP camera:

1. Implement a reader that yields `(frame_index, timestamp, frame)` — e.g.
   wrap `cv2.VideoCapture("rtsp://...")` like `WebcamPipeline` does.
2. Feed it to the same `VideoPipeline.process_frame()` loop.
3. Keep raw streams inside your network; only export the event report.

## Performance guidance

| Hardware | Typical throughput |
|---|---|
| Modern laptop CPU (this repo's demo target) | ~3–8 FPS at 640×360 with frame_skip=2 |
| Laptop GPU (CUDA) | ~15–30 FPS (set `models.*.device: cuda`) |
| Jetson-class edge | works via `device: cuda` + `half_precision: true` |

Tuning levers (all in `configs/config.yaml` / dashboard sidebar):

- `detection.frame_skip` — process every Nth frame (biggest speedup)
- `detection.input_resize` — downscale before inference
- `models.*.name` — `yolov8n` (default) vs `yolov8s/m` for accuracy
- `performance.half_precision` — FP16 on GPU

## Security notes

- No secrets are required to run the app; there is no `.env` needed.
- The CI workflow includes a basic secret scan; keep it enabled.
- Keep `.streamlit/secrets.toml` out of git (already in `.gitignore`) if you
  ever add private resources.

## Operational monitoring

- `outputs/events.json` / `events.csv` per processed video — archive per your
  retention policy.
- CLI reports processing FPS; watch it for regressions on shared hosts.
- For long-running jobs prefer `scripts/predict.py` over the dashboard.
