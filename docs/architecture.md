# LibraryGuard — System Architecture

This document explains *why* the system is built the way it is. For usage,
see the [README](../README.md).

## 1. Design goal: behavior, not objects

The core design constraint is that **objects are not behaviors**:

| Object cue | What it does NOT prove | What we require instead |
|---|---|---|
| bottle/cup detected | that the person is drinking | vessel near face **+** hand near mouth **+** temporal persistence |
| cell phone detected | that the person is using it | phone held/owned by the person **+** near hands/face **+** persistence |
| motionless person | that the person is sleeping | head-down posture **+** stillness **+** long duration |

Every detector in `src/behavior/` therefore combines *multiple* evidence
sources and the aggregation engine in `src/behavior/engine.py` adds the
temporal dimension (duration, smoothing, hysteresis, cooldowns).

## 2. Architecture options considered

### Option A — detection + pose + temporal rules
* ✅ Fast on CPU, fully interpretable, no training data required
* ❌ Hand-tuned thresholds; upper bound on nuance

### Option B — detection + tracking + temporal neural network
* ✅ Learns subtle temporal patterns
* ❌ Needs labeled sequence datasets; harder to debug; more moving parts

### Option C — dedicated video action-recognition model (SlowFast, VideoMAE, ...)
* ✅ Strong accuracy on benchmarks
* ❌ Heavy compute, large clip windows, weak localization of *which person*,
  datasets rarely match fixed CCTV viewpoints

### Option D — hybrid (chosen)
Combine the strengths:

```
Video ──► Frame extraction
            │
            ▼
   ┌──────────────────┐   YOLOv8n-pose: person boxes + 17 keypoints
   │ Detection+Pose   │   YOLOv8n: bottles, cups, phones, ...
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   ByteTrack-style: Kalman + Hungarian + IoU
   │ Multi-person     │   anonymous IDs (Person #1, #2, ...)
   │ tracking         │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   per-person, per-frame evidence vectors
   │ Feature &        │   (hand-face distance, head tilt, object
   │ evidence scoring │    proximity, movement, ...)
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   EMA smoothing · min duration · hysteresis
   │ Temporal event   │   cooldowns · three-level evidence labels
   │ engine (rules)   │
   └────────┬─────────┘
            │
            ├────────────► Annotated video (boxes, IDs, labels, timestamps)
            └────────────► Event report (JSON / CSV)

   ┌──────────────────┐
   │ Trainable GRU    │   optional learned temporal module over the same
   │ temporal model   │   features (scripts/train.py) — see §4
   └──────────────────┘
```

**Why hybrid wins here**

| Criterion | A | B | C | **D** |
|---|---|---|---|---|
| CPU-laptop demo | ✅ | ⚠️ | ❌ | ✅ |
| No training data needed for demo | ✅ | ❌ | ❌ | ✅ |
| Interpretable decisions | ✅ | ❌ | ❌ | ✅ |
| Path to higher accuracy | ❌ | ✅ | ✅ | ✅ |
| Per-person attribution | ✅ | ✅ | ⚠️ | ✅ |
| Portfolio/reproducibility | ⚠️ | ⚠️ | ❌ | ✅ |

## 3. Component details

### 3.1 Detection (`src/detection/`)
* **Person + pose**: `yolov8n-pose` (COCO keypoints) — one model gives boxes,
  person scores and 17 keypoints.
* **Objects**: `yolov8n` restricted to relevant COCO classes
  (bottle=39, cup=41, wine glass=40, bowl=45, cell phone=67, book=73, laptop=63).
* Both are **pretrained on COCO** (Ultralytics). The repository documents
  this honestly: the shipped demo runs on pretrained detectors; the temporal
  behavior layer is the part engineered in depth here.
* `MultiDetector` runs pose + object models and returns a unified
  `DetectionResult`.

### 3.2 Tracking (`src/tracking/`)
ByteTrack-style two-stage association:

1. Kalman filter predicts each track's box (constant-velocity model over
   `cx, cy, area, aspect`).
2. Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on IoU cost
   matches high-score detections first.
3. Unmatched *tracks* are matched again against low-score detections —
   this keeps IDs through brief occlusions and missed detections.
4. Track lifecycle: `TENTATIVE → CONFIRMED → LOST → REMOVED` with
   `min_hits` / `max_age` from config.

Track IDs are **anonymous and ephemeral** — there is deliberately no
re-identification model. See [privacy.md](privacy.md).

`BoTSORTTracker` adds a Kalman Mahalanobis motion gate; full appearance
features are intentionally not faked and listed as future work.

### 3.3 Behavior evidence (`src/behavior/`)
Each detector implements one method — `score(observation) -> BehaviorScore` —
returning per-frame confidence plus named sub-scores (for explainability):

* **DrinkingDetector** — vessel detected ∩ (hand→face ∪ vessel→face) ∩
  vessel associated with the person. A bottle on a table is hard-gated to a
  low score.
* **SleepingDetector** — head tilt vs. vertical, head-below-shoulder test,
  torso slump, and stillness (from the tracker's centroid movement window).
  Explicitly does **not** claim to analyze eye closure from CCTV.
* **PhoneUsageDetector** — phone detected ∩ owned/held by the person ∩
  hand/face proximity, plus head-tilt-down posture. A phone lying on a desk
  is hard-gated to a low score.

### 3.4 Event engine (`src/behavior/engine.py`)
Turns noisy per-frame scores into events:

| Mechanism | Config key | Purpose |
|---|---|---|
| EMA smoothing | `ema_alpha` | one strong frame can't fire an alert |
| Consecutive frames | `min_consecutive_frames` | flicker rejection |
| Activation threshold | `confidence_threshold` | candidate must exceed it (smoothed) |
| Hysteresis | `deactivate_ratio` (0.75) | no flicker at the boundary |
| Min duration | `min_duration_seconds` | short blips discarded |
| Cooldown | `cooldown_seconds` | no alert spam per person/behavior |
| Peak+mean confidence | — | final confidence is `0.6·peak + 0.4·mean` |
| Evidence labels | — | `DETECTED` / `POSSIBLE` / `HIGH CONFIDENCE` |

Per-behavior defaults live in `configs/config.yaml`
(see [config docs](../configs/config.yaml)).

### 3.5 Pipeline (`src/pipeline.py`)
`VideoPipeline` is **source-agnostic**: it consumes an iterable of
`(index, timestamp, frame)`. `VideoReader` wraps files, `WebcamPipeline`
wraps cameras, and a future RTSP source only needs a new reader — the ML
chain never changes.

## 4. The trainable temporal module (`src/models/temporal.py`)

The hybrid design keeps a *learned* path to higher accuracy:

* `FeatureExtractor` maps each `(person, frame)` observation to a
  17-dim vector — the same evidence the rule engine uses.
* `TemporalBehaviorClassifier` is a bidirectional GRU with attention pooling
  and an MLP head over sequences (default 64 frames), trained with
  `scripts/train.py` on prepared sequence datasets (`.npz` samples produced
  by `scripts/prepare_data.py sequences`).
* Training logs real measured metrics to `models/history.json`; nothing is
  fabricated. See [datasets.md](datasets.md) for the training data path.

This mirrors how production systems evolve: rules that work out of the box,
a learned component that takes over when real data exists.

## 5. Error handling & performance

* Friendly errors (`VideoCaptureError`, `ConfigError`, `DatasetError`)
  instead of raw tracebacks; the Streamlit UI shows `st.error` messages.
* Frame skipping (`detection.frame_skip`), FP16 flag on GPU, batch predict
  API, and detection-then-track (rather than tracking-by-detection loops).
* Inference FPS is measured live and displayed in the UI/CLI.

## 6. Extension points

* **New behavior**: subclass `BehaviorDetector`, add a config section,
  register in `detector_registry()` — nothing else changes
  ([guide](adding_behaviors.md)).
* **New video source**: implement a reader yielding `(index, timestamp, frame)`.
* **Learned detector**: swap the rule score for model inference in the same
  `score()` interface.
