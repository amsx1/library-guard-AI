"""Tests: dataset validation, duplicate detection, leakage-safe splitting."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.dataset import (
    DatasetError,
    ManifestRow,
    assert_leakage_free,
    group_split,
    hamming_distance,
    average_hash,
    load_manifest,
    save_manifest,
    split_statistics,
    validate_dataset,
)


def rows_for_group(group, labels=("drinking",), n=3):
    return [
        ManifestRow(path=f"{group}_{i}.jpg", label=labels[i % len(labels)], group=group)
        for i in range(n)
    ]


class TestManifest:
    def test_load_csv_manifest(self, tmp_path):
        manifest = tmp_path / "m.csv"
        manifest.write_text(
            "path,label,group\n"
            "a.jpg,drinking,video1\n"
            "b.jpg,sleeping,video1\n",
            encoding="utf-8",
        )
        rows = load_manifest(manifest)
        assert len(rows) == 2
        assert rows[0].group == "video1"
        assert rows[1].label == "sleeping"

    def test_load_json_manifest(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(
            '[{"path": "a.jpg", "label": "phone_usage", "group": "v1", "duration": 2.0}]',
            encoding="utf-8",
        )
        rows = load_manifest(manifest)
        assert rows[0].label == "phone_usage"
        assert rows[0].duration == 2.0

    def test_missing_columns_raises(self, tmp_path):
        manifest = tmp_path / "m.csv"
        manifest.write_text("path,label\na.jpg,drinking\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="missing required columns"):
            load_manifest(manifest)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_manifest(tmp_path / "ghost.csv")

    def test_bad_format_raises(self, tmp_path):
        manifest = tmp_path / "m.txt"
        manifest.write_text("whatever", encoding="utf-8")
        with pytest.raises(DatasetError, match="Unsupported manifest format"):
            load_manifest(manifest)

    def test_save_and_reload_round_trip(self, tmp_path):
        rows = rows_for_group("video1", n=2) + rows_for_group("video2", labels=("sleeping",), n=2)
        for row in rows:
            row.split = "train"
        dest = tmp_path / "out.csv"
        save_manifest(rows, dest)
        loaded = load_manifest(dest)
        assert len(loaded) == len(rows)
        assert {r.group for r in loaded} == {"video1", "video2"}


class TestValidation:
    def test_validate_missing_files(self):
        rows = [ManifestRow(path="ghost1.jpg", label="drinking", group="g1")]
        report = validate_dataset(rows, allowed_labels={"drinking"}, check_duplicates=False)
        assert not report.ok
        assert len(report.missing_files) == 1

    def test_validate_invalid_labels(self, tmp_path):
        img = tmp_path / "a.jpg"
        import cv2

        cv2.imwrite(str(img), np.zeros((10, 10, 3), dtype=np.uint8))
        rows = [ManifestRow(path=str(img), label="fighting", group="g1")]
        report = validate_dataset(rows, allowed_labels={"drinking"}, check_duplicates=False)
        assert len(report.invalid_labels) == 1

    def test_validate_detects_exact_duplicates(self, tmp_path):
        import cv2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[2:8, 2:8] = 200
        for name in ("a.jpg", "b.jpg"):
            cv2.imwrite(str(tmp_path / name), img)
        rows = [
            ManifestRow(path=str(tmp_path / "a.jpg"), label="drinking", group="g1"),
            ManifestRow(path=str(tmp_path / "b.jpg"), label="drinking", group="g1"),
        ]
        report = validate_dataset(rows, allowed_labels={"drinking"}, check_duplicates=True)
        assert len(report.duplicates) == 1

    def test_validate_ok_dataset(self, tmp_path):
        import cv2

        rng = np.random.default_rng(1)
        rows = []
        for i in range(3):
            img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
            path = tmp_path / f"img{i}.jpg"
            cv2.imwrite(str(path), img)
            rows.append(ManifestRow(path=str(path), label="drinking", group=f"g{i}"))
        report = validate_dataset(rows, allowed_labels={"drinking"}, check_duplicates=True)
        assert report.ok
        assert report.valid == 3

    def test_empty_file_flagged(self, tmp_path):
        empty = tmp_path / "empty.jpg"
        empty.write_bytes(b"")
        rows = [ManifestRow(path=str(empty), label="drinking", group="g1")]
        report = validate_dataset(rows, allowed_labels={"drinking"}, check_duplicates=False)
        assert str(empty) in report.empty_files


class TestHashing:
    def test_average_hash_deterministic(self):

        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[8:24, 8:24] = 255
        assert average_hash(img) == average_hash(img.copy())

    def test_hamming_distance(self):
        assert hamming_distance(0b1111, 0b0000) == 4
        assert hamming_distance(0b1010, 0b1010) == 0


class TestSplitting:
    def test_group_split_no_leakage(self):
        rows = []
        for g in range(20):
            rows.extend(rows_for_group(f"video{g}", labels=("drinking", "sleeping"), n=4))
        splits = group_split(rows, train_ratio=0.7, val_ratio=0.15, seed=0)
        assert_leakage_free(splits)

        # Every row assigned exactly once.
        total = sum(len(v) for v in splits.values())
        assert total == len(rows)

        # Groups do not span splits.
        group_to_split = {}
        for name, split_rows in splits.items():
            for row in split_rows:
                group_to_split.setdefault(row.group, set()).add(name)
        for group, names in group_to_split.items():
            assert len(names) == 1, f"group {group} leaked into {names}"

    def test_split_ratios_roughly_respected(self):
        rows = []
        for g in range(40):
            rows.extend(rows_for_group(f"video{g}", n=2))
        splits = group_split(rows, train_ratio=0.7, val_ratio=0.15, seed=1)
        n_train_groups = len({r.group for r in splits["train"]})
        assert 24 <= n_train_groups <= 32  # ~70% of 40 groups

    def test_split_is_deterministic_with_seed(self):
        rows = []
        for g in range(10):
            rows.extend(rows_for_group(f"video{g}", n=2))
        s1 = group_split(rows, seed=7)
        groups1 = {r.group for r in s1["train"]}
        # Reset split labels; rows list is mutated, re-create
        rows2 = []
        for g in range(10):
            rows2.extend(rows_for_group(f"video{g}", n=2))
        s2 = group_split(rows2, seed=7)
        groups2 = {r.group for r in s2["train"]}
        assert groups1 == groups2

    def test_invalid_ratios_raise(self):
        rows = rows_for_group("v1", n=2)
        with pytest.raises(DatasetError):
            group_split(rows, train_ratio=1.5)
        with pytest.raises(DatasetError):
            group_split(rows, train_ratio=0.9, val_ratio=0.2)

    def test_empty_dataset_raises(self):
        with pytest.raises(DatasetError, match="empty"):
            group_split([])

    def test_assert_leakage_free_detects_leak(self):
        rows_a = rows_for_group("shared", n=1)
        rows_b = rows_for_group("shared", n=1)
        with pytest.raises(DatasetError, match="leakage"):
            assert_leakage_free({"train": rows_a, "test": rows_b})

    def test_split_statistics(self):
        rows = []
        for g in range(6):
            rows.extend(rows_for_group(f"v{g}", labels=("drinking", "sleeping"), n=3))
        splits = group_split(rows, seed=3)
        stats = split_statistics(splits)
        assert set(stats) == {"train", "val", "test"}
        assert stats["train"]["samples"] > 0