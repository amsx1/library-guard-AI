# models/ — Model weights (not committed)

Trained/detector weights are **not committed** (size + license hygiene).
Everything here is reproducible or auto-downloaded.

## What lives where

| File | Origin | Size | How to obtain |
|---|---|---|---|
| `yolov8n.pt` | Ultralytics COCO-pretrained detector | ~6 MB | auto-downloaded on first run |
| `yolov8n-pose.pt` | Ultralytics COCO-pretrained pose model | ~7 MB | auto-downloaded on first run |
| `temporal_classifier.pt` | **trained here** via `scripts/train.py` | ~3 MB | run training (below) |
| `history.json` | training history from the run above | tiny | produced by `scripts/train.py` |

## Train the temporal classifier

```bash
# 1. Prepare sequences (see docs/datasets.md)
python scripts/prepare_data.py sequences --manifest data/manifests/splits/train.csv --out data/sequences/train
python scripts/prepare_data.py sequences --manifest data/manifests/splits/val.csv   --out data/sequences/val

# 2. Train (real run; metrics are logged, never fabricated)
python scripts/train.py \
    --train-dir data/sequences/train \
    --val-dir   data/sequences/val \
    --out models/temporal_classifier.pt \
    --epochs 20 --plot

# 3. Evaluate on the held-out test split
python scripts/evaluate.py --checkpoint models/temporal_classifier.pt \
    --data-dir data/sequences/test --out outputs/evaluation --plots
```

Reported metrics live in `outputs/evaluation/` and `models/history.json`.
If no training run has been performed in your environment, those metrics are
**unavailable** — the README states this explicitly rather than inventing
numbers.

## Loading a trained model

```python
import torch
from src.models.temporal import TemporalBehaviorClassifier

payload = torch.load("models/temporal_classifier.pt", map_location="cpu")
cfg = payload["config"]
model = TemporalBehaviorClassifier(
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    num_classes=cfg["num_classes"],
)
model.load_state_dict(payload["model_state"])
```