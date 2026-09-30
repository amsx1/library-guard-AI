"""Tests: feature extraction + temporal model structure (no training data needed)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.temporal import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FeatureExtractor,
    extract_feature_sequence,
)
from tests.conftest import make_box, make_observation, make_pose, object_tuple

torch = pytest.importorskip("torch", reason="PyTorch not installed")


class TestFeatureExtractor:
    def test_feature_dimension(self):
        extractor = FeatureExtractor()
        features = extractor.extract(make_observation())
        assert features.shape == (FEATURE_DIM,)
        assert features.dtype == np.float32
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_features_bounded(self):
        extractor = FeatureExtractor()
        obs = make_observation(
            pose=make_pose(),
            objects=[object_tuple(make_box(0.22, 0.15, 0.30, 0.23), "bottle", 0.9)],
        )
        features = extractor.extract(obs)
        assert np.all(features >= -1.1) and np.all(features <= 2.1)

    def test_hand_to_face_feature_reacts(self):
        extractor = FeatureExtractor()
        near = make_observation(
            pose=make_pose(left_wrist=(0.26, 0.20)),
            objects=[object_tuple(make_box(0.22, 0.15, 0.30, 0.23), "cup", 0.9)],
        )
        far = make_observation(
            pose=make_pose(left_wrist=(0.45, 0.85)),
            objects=[object_tuple(make_box(0.75, 0.80, 0.82, 0.90), "cup", 0.9)],
        )
        feat_near = extractor.extract(near)
        feat_far = extractor.extract(far)
        assert feat_near[0] > feat_far[0]  # hand_l_to_face

    def test_phone_features_react(self):
        extractor = FeatureExtractor()
        with_phone = make_observation(
            objects=[object_tuple(make_box(0.22, 0.15, 0.30, 0.23), "cell phone", 0.9)],
            pose=make_pose(),
        )
        without = make_observation(objects=[])
        feat_a = extractor.extract(with_phone)
        feat_b = extractor.extract(without)
        assert feat_a[11] > feat_b[11]  # phone_held

    def test_extract_sequence_shape_and_padding(self):
        observations = [make_observation(track_id=1, frame_index=i) for i in range(5)]
        seq = extract_feature_sequence(observations, track_id=1, seq_len=16)
        assert seq.shape == (16, FEATURE_DIM)
        assert np.allclose(seq[5:], 0.0)  # zero-padded tail

    def test_extract_sequence_empty_track(self):
        seq = extract_feature_sequence([], track_id=9, seq_len=8)
        assert seq.shape == (8, FEATURE_DIM)
        assert np.allclose(seq, 0.0)


class TestTemporalModel:
    def test_model_forward_shape(self):
        from src.models.temporal import TemporalBehaviorClassifier

        model = TemporalBehaviorClassifier(num_classes=3)
        batch = torch.randn(4, 32, FEATURE_DIM)
        logits = model(batch)
        assert logits.shape == (4, 3)

    def test_model_predict_proba_sums_to_one(self):
        from src.models.temporal import TemporalBehaviorClassifier

        model = TemporalBehaviorClassifier(num_classes=3)
        probs = model.predict_proba(torch.randn(2, 16, FEATURE_DIM))
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-4)

    def test_model_gradients_flow(self):
        from src.models.temporal import TemporalBehaviorClassifier

        model = TemporalBehaviorClassifier(num_classes=3)
        batch = torch.randn(2, 8, FEATURE_DIM)
        loss = model(batch).sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_dataset_loads_npz(self, tmp_path):
        from src.models.temporal import BehaviorSequenceDataset

        for i in range(4):
            np.savez_compressed(
                tmp_path / f"sample{i}.npz",
                features=np.random.default_rng(i).random((20, FEATURE_DIM)).astype(np.float32),
                label=np.int64(i % 3),
            )
        dataset = BehaviorSequenceDataset(tmp_path, seq_len=16)
        assert len(dataset) == 4
        features, label = dataset[0]
        assert features.shape == (16, FEATURE_DIM)
        assert label.dtype == torch.long

    def test_dataset_missing_dir_raises(self, tmp_path):
        from src.models.temporal import BehaviorSequenceDataset

        with pytest.raises(FileNotFoundError):
            BehaviorSequenceDataset(tmp_path / "ghost")

    def test_dataset_empty_dir_raises(self, tmp_path):
        from src.models.temporal import BehaviorSequenceDataset

        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            BehaviorSequenceDataset(tmp_path / "empty")

    def test_training_reduces_loss(self, tmp_path):
        """A tiny real training run must actually optimize."""
        from src.models.temporal import (
            BehaviorSequenceDataset,
            TrainConfig,
            train_model,
        )

        rng = np.random.default_rng(0)
        for split in ("train", "val"):
            d = tmp_path / split
            d.mkdir()
            for i in range(12):
                label = i % 3
                # Class-dependent pattern so the model can learn.
                base = np.zeros((20, FEATURE_DIM), dtype=np.float32)
                base[:, label] = 1.0
                base += rng.normal(0, 0.05, base.shape).astype(np.float32)
                np.savez_compressed(d / f"s{i}.npz", features=base, label=np.int64(label))

        train_ds = BehaviorSequenceDataset(tmp_path / "train", seq_len=16)
        val_ds = BehaviorSequenceDataset(tmp_path / "val", seq_len=16)

        model, history = train_model(
            train_ds,
            val_ds,
            TrainConfig(epochs=6, batch_size=4, hidden_dim=32, num_layers=1, seq_len=16),
            num_classes=3,
            log_fn=lambda *a, **k: None,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]
        assert history["val_acc"][-1] >= 0.8