"""
LibraryGuard - Temporal behavior classifier (trainable neural module).

This is the "learned temporal" component of the hybrid architecture:

* :class:`FeatureExtractor` converts per-person, per-frame observations
  (pose geometry, object proximity, motion) into a fixed-length feature
  vector -- the same features the rule engine uses, so both modules share
  one honest representation.
* :class:`TemporalBehaviorClassifier` is a GRU + MLP head that maps a
  sequence of feature vectors to per-behavior logits. It is genuinely
  trainable with :mod:`scripts.train` on prepared sequence datasets
  (e.g. UCF101/HMDB-derived clips mapped to our behavior vocabulary).
* :class:`BehaviorSequenceDataset` loads saved ``.npz`` sequence samples
  produced by ``scripts/prepare_data.py``.

Honesty note: the shipped demo pipeline uses the rule-based engine so the
repository works out of the box without training data. Training this module
is fully supported and its metrics are only ever reported when a real
training run has produced them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a project dependency
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore

from ..behavior.base import BehaviorObservation
from ..utils.constants import (
    EVENT_TYPES,
    KP_LEFT_ELBOW,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_ELBOW,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    clamp01,
)

logger = logging.getLogger(__name__)

# Feature layout (order matters; FeatureExtractor documents each dim)
FEATURE_NAMES: List[str] = [
    "hand_l_to_face",
    "hand_r_to_face",
    "head_tilt_sin",
    "head_tilt_cos",
    "head_below_shoulders",
    "torso_slump",
    "movement",
    "drinking_obj_near_face",
    "drinking_obj_held",
    "phone_near_hand",
    "phone_near_face",
    "phone_held",
    "person_score",
    "wrist_l_height",
    "wrist_r_height",
    "elbow_angle_l",
    "elbow_angle_r",
]
FEATURE_DIM = len(FEATURE_NAMES)


class FeatureExtractor:
    """
    Convert a :class:`BehaviorObservation` into a fixed-length feature vector.

    Deterministic, dependency-free, and shared between the rule engine's
    scoring logic and the trainable temporal classifier.
    """

    feature_names = FEATURE_NAMES

    @staticmethod
    def _angle(a, b, c) -> float:
        """Angle at vertex b in degrees given three (x, y) points."""
        import math

        ab = (a[0] - b[0], a[1] - b[1])
        cb = (c[0] - b[0], c[1] - b[1])
        dot = ab[0] * cb[0] + ab[1] * cb[1]
        norm = math.hypot(*ab) * math.hypot(*cb)
        if norm < 1e-9:
            return 180.0
        cosv = max(-1.0, min(1.0, dot / norm))
        return math.degrees(math.acos(cosv))

    def extract(self, obs: BehaviorObservation) -> np.ndarray:
        import math

        pose = obs.keypoints
        face = None
        wrist_l = wrist_r = elbow_l = elbow_r = shoulder_l = shoulder_r = None
        if pose is not None:
            nose = pose.get_keypoint(KP_NOSE)
            face = (nose.x, nose.y) if nose else None
            wl = pose.get_keypoint(KP_LEFT_WRIST)
            wr = pose.get_keypoint(KP_RIGHT_WRIST)
            el = pose.get_keypoint(KP_LEFT_ELBOW)
            er = pose.get_keypoint(KP_RIGHT_ELBOW)
            sl = pose.get_keypoint(KP_LEFT_SHOULDER)
            sr = pose.get_keypoint(KP_RIGHT_SHOULDER)
            wrist_l = (wl.x, wl.y) if wl else None
            wrist_r = (wr.x, wr.y) if wr else None
            elbow_l = (el.x, el.y) if el else None
            elbow_r = (er.x, er.y) if er else None
            shoulder_l = (sl.x, sl.y) if sl else None
            shoulder_r = (sr.x, sr.y) if sr else None

        def dist(p, q) -> float:
            return math.hypot(p[0] - q[0], p[1] - q[1]) if (p and q) else 1.0

        hand_l_to_face = clamp01(1.0 - dist(wrist_l, face) / 0.3) if (wrist_l and face) else 0.0
        hand_r_to_face = clamp01(1.0 - dist(wrist_r, face) / 0.3) if (wrist_r and face) else 0.0

        head_tilt = 0.0
        head_below = 0.0
        slump = 0.0
        if face and shoulder_l and shoulder_r:
            neck = ((shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2)
            dx = face[0] - neck[0]
            dy = face[1] - neck[1]
            head_tilt = math.atan2(dx, -dy) if dy < 0 else math.atan2(dx, abs(dy) + 1e-6)
            head_below = 1.0 if face[1] > neck[1] else 0.0
            span = abs(shoulder_l[0] - shoulder_r[0]) + 1e-6
            slump = clamp01(abs(dy) / (2.5 * span))

        movement = clamp01(obs.track_movement / 0.1)

        # Object-based features
        drink_near_face = drink_held = phone_hand = phone_face = phone_held = 0.0
        for box, class_name, score in obs.objects:
            center = box.center
            owned = 1.0 if obs.person_box.contains_point(center) else 0.0
            near_face = clamp01(1.0 - dist((center.x, center.y), face) / 0.25) if face else 0.0
            near_hand = clamp01(
                1.0 - min(dist((center.x, center.y), wrist_l), dist((center.x, center.y), wrist_r)) / 0.2
            ) if (wrist_l or wrist_r) else 0.0
            if class_name in ("bottle", "cup", "wine glass", "bowl"):
                drink_near_face = max(drink_near_face, near_face * score)
                drink_held = max(drink_held, owned * max(near_hand, near_face) * score)
            elif class_name == "cell phone":
                phone_hand = max(phone_hand, near_hand * score)
                phone_face = max(phone_face, near_face * score)
                phone_held = max(phone_held, owned * max(near_hand, near_face) * score)

        wrist_l_height = clamp01(1.0 - wrist_l[1]) if wrist_l else 0.0
        wrist_r_height = clamp01(1.0 - wrist_r[1]) if wrist_r else 0.0
        elbow_angle_l = self._angle(shoulder_l, elbow_l, wrist_l) / 180.0 if (shoulder_l and elbow_l and wrist_l) else 1.0
        elbow_angle_r = self._angle(shoulder_r, elbow_r, wrist_r) / 180.0 if (shoulder_r and elbow_r and wrist_r) else 1.0

        features = np.array(
            [
                hand_l_to_face,
                hand_r_to_face,
                math.sin(head_tilt),
                math.cos(head_tilt),
                head_below,
                slump,
                movement,
                drink_near_face,
                drink_held,
                phone_hand,
                phone_face,
                phone_held,
                clamp01(obs.person_score),
                wrist_l_height,
                wrist_r_height,
                elbow_angle_l,
                elbow_angle_r,
            ],
            dtype=np.float32,
        )
        return features


def extract_feature_sequence(
    observations: Sequence[BehaviorObservation],
    track_id: int,
    seq_len: int = 64,
) -> np.ndarray:
    """
    Extract a fixed-length ``(seq_len, FEATURE_DIM)`` array for one track
    from a chronological sequence of observations (zero-padded at the end).
    """
    extractor = FeatureExtractor()
    rows = [extractor.extract(o) for o in observations if o.track_id == track_id]
    if not rows:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
    matrix = np.stack(rows[-seq_len:], axis=0)
    if matrix.shape[0] < seq_len:
        pad = np.zeros((seq_len - matrix.shape[0], FEATURE_DIM), dtype=np.float32)
        matrix = np.concatenate([matrix, pad], axis=0)
    return matrix


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class TemporalBehaviorClassifier(nn.Module):
        """
        GRU-based temporal classifier over per-person feature sequences.

        Input:  (batch, seq_len, FEATURE_DIM)
        Output: (batch, n_classes) raw logits for
                [drinking, sleeping, phone_usage, ...]
        """

        def __init__(
            self,
            input_dim: int = FEATURE_DIM,
            hidden_dim: int = 96,
            num_layers: int = 2,
            num_classes: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            self.num_classes = num_classes

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, F)
            x = self.input_norm(x)
            out, _ = self.gru(x)  # (B, T, 2H)
            weights = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
            context = (out * weights).sum(dim=1)  # (B, 2H)
            return self.head(context)

        @torch.no_grad()
        def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.softmax(self.forward(x), dim=-1)

else:  # pragma: no cover

    class TemporalBehaviorClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TemporalBehaviorClassifier")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class BehaviorSequenceDataset(Dataset):
        """
        Loads ``.npz`` samples produced by ``scripts/prepare_data.py``.

        Each file stores ``features`` (seq_len, FEATURE_DIM) and
        ``label`` (int index into EVENT_TYPES or custom label map).
        A directory of ``*.npz`` files plus an optional ``labels.json``
        defining the class order is expected.
        """

        def __init__(
            self,
            root: Union[str, Path],
            label_map: Optional[Dict[str, int]] = None,
            seq_len: int = 64,
            augment: bool = False,
            seed: int = 42,
        ):
            self.root = Path(root)
            if not self.root.exists():
                raise FileNotFoundError(f"Dataset directory not found: {self.root}")
            self.files = sorted(self.root.glob("*.npz"))
            if not self.files:
                raise FileNotFoundError(f"No .npz samples found in {self.root}")
            self.label_map = label_map or {name: i for i, name in enumerate(EVENT_TYPES)}
            self.seq_len = seq_len
            self.augment = augment
            self.rng = np.random.default_rng(seed)

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, index: int):
            data = np.load(self.files[index])
            features = data["features"].astype(np.float32)
            label = int(data["label"])

            # Fix sequence length.
            if features.shape[0] >= self.seq_len:
                if self.augment:
                    start = int(self.rng.integers(0, features.shape[0] - self.seq_len + 1))
                else:
                    start = features.shape[0] - self.seq_len
                features = features[start:start + self.seq_len]
            else:
                pad = np.zeros((self.seq_len - features.shape[0], features.shape[1]), dtype=np.float32)
                features = np.concatenate([features, pad], axis=0)

            if self.augment:
                features = features + self.rng.normal(0.0, 0.01, features.shape).astype(np.float32)

            return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

else:  # pragma: no cover

    class BehaviorSequenceDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for BehaviorSequenceDataset")


# ---------------------------------------------------------------------------
# Training / evaluation helpers (used by scripts/train.py)
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 96
    num_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 64
    seed: int = 42
    device: str = "auto"
    early_stopping_patience: int = 5


def resolve_torch_device(device: str) -> str:
    if device != "auto" or not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(
    train_dataset,
    val_dataset,
    config: TrainConfig,
    num_classes: int = 3,
    class_weights: Optional[Sequence[float]] = None,
    log_fn=logger.info,
):
    """
    Train the temporal classifier and return ``(model, history)``.

    History is a dict of per-epoch metric lists -- real measured values
    from the run that produced them.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training")

    device = resolve_torch_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = TemporalBehaviorClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=num_classes,
        dropout=config.dropout,
    ).to(device)

    weights = None
    if class_weights is not None:
        weights = torch.tensor(list(class_weights), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_state = None
    patience_left = config.early_stopping_patience

    for epoch in range(1, config.epochs + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * features.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += features.size(0)
        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        running, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                running += loss.item() * features.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += features.size(0)
        val_loss = running / max(total, 1)
        val_acc = correct / max(total, 1)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        log_fn(
            f"Epoch {epoch}/{config.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log_fn(f"Early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history