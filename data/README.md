# data/ — Dataset storage (not committed)

Datasets are **never committed** to this repository (licensing + size +
privacy). See [`docs/datasets.md`](../docs/datasets.md) for the dataset
registry, licenses, and download/preparation commands.

## Expected layout (created locally)

```
data/
├── README.md            ← this file (committed)
├── sample/
│   ├── README.md        ← notes on the bundled synthetic clip
│   └── sample_video.mp4 ← tiny synthetic clip for smoke tests (committed)
├── raw/                 ← downloaded datasets (git-ignored)
│   ├── UCF101/
│   ├── hmdb51/
│   └── roboflow/<project>/
├── manifests/           ← CSV manifests + splits (git-ignored)
│   ├── ucf101.csv
│   └── ucf101_splits/{train,val,test}.csv
└── sequences/           ← .npz feature sequences for training (git-ignored)
    ├── train/
    └── val/
```

## Manifest format

CSV (or JSON) with three required columns:

| column | meaning |
|---|---|
| `path` | file path (relative to `--base-dir`) |
| `label` | `drinking` / `sleeping` / `phone_usage` / `other` |
| `group` | **source video ID** — keeps train/val/test leakage-free |

Optional: `duration`, `split`, `license`, `source_url`.

## Quick commands

```bash
python scripts/prepare_data.py validate --manifest data/manifests/ucf101.csv
python scripts/prepare_data.py split    --manifest data/manifests/ucf101.csv --out data/manifests/splits
python scripts/prepare_data.py extract  --video clip.mp4 --out data/frames
python scripts/prepare_data.py sequences --manifest data/manifests/splits/train.csv --out data/sequences/train
```