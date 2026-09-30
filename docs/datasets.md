# LibraryGuard — Dataset Registry & Preparation

**Policy:** datasets are never committed to this repository. This document is
the registry of datasets considered/used, their licenses, and the exact
commands to prepare them locally. Verify each license yourself before use —
terms can change.

> **Rule of thumb:** if a dataset's license is unclear, do not use it.
> Prefer datasets with explicit research/open licenses.

## Registry

| # | Dataset | URL | Classes used | License | Size (approx.) | How it is used | Preprocessing |
|---|---|---|---|---|---|---|---|
| 1 | **COCO** (detection/pose pretraining) | https://cocodataset.org | person, bottle, cup, wine glass, bowl, cell phone, book, laptop | CC BY 4.0 (images), terms: cocodataset.org | 118k train images | Pretraining source for YOLOv8 detectors used as-is (Ultralytics pretrained weights). Not redistributed. | None (weights from Ultralytics) |
| 2 | **UCF101** | https://www.crcv.ucf.edu/data/UCF101.php | `Drink` (mapped → drinking); other action classes kept as negatives | Non-commercial research/educational use (see site) | 13,320 videos / 101 classes | Training data for the temporal classifier (`scripts/prepare_data.py sequences`); frames → pose/object features → `.npz` sequences | Frame sampling 4 fps, per-person crops via detector, feature extraction, group-by-video split |
| 3 | **HMDB51** | https://serre-lab.clpbrown.edu/hmdb/ | `drink`, `smoke`, `sit`, `eat`-adjacent negatives | Research use (see site) | 6,849 clips / 51 classes | Supplementary temporal-classifier training & evaluation | Same as UCF101 |
| 4 | **Kinetics-700 (subset)** | https://deepmind.com/research/open-source/kinetics (via `k700` mirrors) | `drinking`, `using phone`-adjacent clips (subset only) | CC BY 4.0 (labels) + YouTube source terms; clips must be re-downloaded from source | ~650k clips total; small subset used | Optional augmentation for temporal training | Clip download via official tooling; frame→feature pipeline as above |
| 5 | **Roboflow public projects** (phone usage / sleeping) | https://universe.roboflow.com | project-specific (phone-in-hand, sleeping) | **Per-project license — inspect each one** (many are CC BY 4.0 or MIT) | varies | Optional object-detector fine-tuning for phone/sleeping cues | Export in YOLOv8 format; never re-uploaded |
| 6 | **ActivityNet v1.3** | http://activity-net.org | "drinking" action segments | Research use (registration + terms) | 20k videos | Optional temporal-training clips; event-level eval | Segment clipping, same feature pipeline |

### Notes per dataset

1. **COCO** — Ultralytics ships YOLOv8 weights trained on COCO. This repo
   downloads weights at first use; it does not redistribute COCO images.
2. **UCF101** — action-level labels ("Drink") do not localize *which person*
   drinks; the feature pipeline therefore keeps only clips where a person +
   vessel + hand-to-face evidence co-occur, documented in
   `scripts/prepare_data.py sequences`.
3. **HMDB51** — small and noisy; used as an external check set rather than
   the primary source.
4. **Kinetics** — only classes close to our vocabulary are used; respect
   YouTube's terms when re-downloading clips.
5. **Roboflow** — *do not assume* Roboflow-hosted datasets are open: every
   project page states its license. Record the project + license in a PR if
   you use one.
6. **ActivityNet** — requires agreeing to terms; used only for temporal
   event evaluation when available.

**Redistribution:** none of these datasets (images, videos, frames, labels)
are redistributable in this repository under their terms. Only tiny synthetic
fixtures (generated at test time) live in `tests/`.

## Download & preparation procedures

### 0. Create a local data root (git-ignored)

```bash
mkdir -p data/raw data/manifests data/sequences
```

### 1. UCF101 (primary temporal training source)

```bash
cd data/raw
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar   # official mirror
unrar x UCF101.rar
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip
cd ../..
```

Build a manifest (CSV with `path,label,group` — `group` = source video name,
the leakage-prevention key):

```bash
python scripts/prepare_data.py validate --manifest data/manifests/ucf101.csv --report data/manifests/validation.json
python scripts/prepare_data.py split    --manifest data/manifests/ucf101.csv --out data/manifests/ucf101_splits
```

Extract model-ready feature sequences:

```bash
python scripts/prepare_data.py sequences \
    --manifest data/manifests/ucf101_splits/train.csv \
    --out data/sequences/train --base-dir data/raw
python scripts/prepare_data.py sequences \
    --manifest data/manifests/ucf101_splits/val.csv \
    --out data/sequences/val --base-dir data/raw
```

### 2. HMDB51

```bash
cd data/raw
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hdm51_org.rar
# extract each split rar, then build manifests as above with group=clip stem
```

### 3. Roboflow (optional, per-project license)

Use the project's "Download → YOLOv8" export after accepting its license;
place under `data/raw/roboflow/<project>/` and note the project URL +
license in your manifest's `license` column.

## Data hygiene pipeline

`scripts/prepare_data.py` provides:

| Step | Command | What it does |
|---|---|---|
| Validation | `validate` | missing/empty/corrupt file detection, label legality, exact (MD5) + perceptual (aHash) duplicate detection |
| Splitting | `split` | **group-aware** train/val/test split: all samples from one source video stay in one split (`group` column) |
| Frame extraction | `extract` | uniform temporal sampling with resize |
| Sequence extraction | `sequences` | runs detection/tracking/features → `.npz` for `scripts/train.py` |

### Leakage prevention (important)

Frames/clips from the same source video are highly correlated. Splitting
them randomly leaks information and inflates metrics. `group_split()`:
1. groups rows by the `group` key (source video / subject),
2. assigns whole groups to splits (optionally stratified by dominant label),
3. asserts no group spans two splits (`assert_leakage_free`).

### Duplicate handling

* exact duplicates: MD5 hash → reported, left for the curator to drop,
* near-duplicates: 64-bit average hash, Hamming ≤ 4 flagged.

## Evaluation protocol

* **Clip-level**: `scripts/evaluate.py --checkpoint ... --data-dir data/sequences/test`
  → precision/recall/F1/confusion matrix (per class), saved to
  `outputs/evaluation/`.
* **Event-level**: `scripts/evaluate.py --predictions outputs/events.json --ground-truth data/gt/events.json`
  → temporal-IoU matching (threshold 0.3) with precision/recall/F1 per event type.

Metrics are only ever reported when a real run produced them. See the
"Performance" section of the README for the current status.