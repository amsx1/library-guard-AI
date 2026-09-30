# LibraryGuard — Adding a New Behavior Detector

The behavior layer is designed for extension: a new behavior is one class +
one config section + one registry entry. Nothing else changes.

## Step 1 — Implement the detector

Create `src/behavior/eating.py` (example):

```python
from ..utils.constants import clamp01
from .base import BehaviorDetector, BehaviorObservation, BehaviorScore


class EatingDetector(BehaviorDetector):
    key = "eating"
    display_name = "Eating"

    def score(self, observation: BehaviorObservation) -> BehaviorScore:
        # Combine multi-source evidence here. Return 0.0 when the behavior is
        # clearly absent -- never guess from a single weak cue.
        ...
        return BehaviorScore(self.key, combined, {"component_a": ..., "component_b": ...})
```

Rules of thumb (learned from the existing detectors):

1. **Score evidence, not objects.** A sandwich on a desk is not eating.
2. **Combine at least two cues** (object + pose + motion) and hard-gate
   impossible configurations to ≤ 0.3.
3. **Keep `components` populated** — the overlay, tests and debugging use it.
4. **Leave temporal logic to the engine** — duration, smoothing, cooldowns
   and hysteresis are already handled centrally.

## Step 2 — Register it

In `src/behavior/engine.py`:

```python
from .eating import EatingDetector

def default_detectors():
    return [DrinkingDetector(), SleepingDetector(), PhoneUsageDetector(), EatingDetector()]

def detector_registry():
    return {
        ...,
        "eating": EatingDetector,
    }
```

## Step 3 — Add configuration

`configs/config.yaml`:

```yaml
behavior:
  eating:
    enabled: true
    min_duration_seconds: 2.0
    confidence_threshold: 0.6
    cooldown_seconds: 5.0
    min_consecutive_frames: 8
```

Mirror the fields in `src/utils/config.py` (`BehaviorConfig`, parsing in
`_parse_behavior_section`) so values are validated like the others.

## Step 4 — Wire config → engine

In `build_engine_from_config()` (same file), add the section mapping and
detector construction.

## Step 5 — UI & reports

- `app/streamlit_app.py`: add to `BEHAVIOR_META` (emoji, label, color).
- Event type strings flow into reports automatically via `event_type`.

## Step 6 — Tests

Add `tests/test_eating.py` following `tests/test_behavior_detectors.py`:

- positive scenario scores high,
- the "object present but behavior absent" scenario scores ≤ 0.35,
- missing cues produce zero/weak confidence.

Run `pytest tests/ -q` — the engine-level tests will pick the new detector up
through the registry.

## Optional — training support

If the behavior should also be learnable:

1. add its label to `_label_index()` in `scripts/prepare_data.py`,
2. extend `EVENT_TYPES` in `src/utils/constants.py`,
3. bump `--num-classes` in `scripts/train.py`.

## Candidate future behaviors

| Behavior | Primary cues | Notes |
|---|---|---|
| Eating | food object near mouth + hand-to-face | very similar to drinking |
| Smoking | hand-to-mouth + smoke/pen-like object | object classes are weak in COCO |
| Unauthorized entry | person track crossing a configured zone | zone config needed |
| Running | high centroid velocity + pose stride | easy |
| Crowding | track count in zone over time | easy |
| Leaving workstation | person track disappears from zone | needs zone config |
