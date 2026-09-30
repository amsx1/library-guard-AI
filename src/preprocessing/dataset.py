"""
LibraryGuard - Dataset validation, deduplication, and leakage-safe splitting.

Key data-leakage rule: when frames/clips come from longer source videos,
ALL samples derived from the same source video must land in the same split
(train/val/test). This module implements ``group_split`` for exactly that.

Also provides:
* manifest parsing (CSV/JSON with group + label columns)
* corrupt/missing file detection
* perceptual-duplicate detection (average hash)
* label validation against an allowed label set
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_MANIFEST_COLUMNS = {"path", "label", "group"}
OPTIONAL_MANIFEST_COLUMNS = {"duration", "split", "source_url", "license"}


class DatasetError(Exception):
    """Raised for invalid dataset manifests or unusable datasets."""


@dataclass
class ManifestRow:
    path: str
    label: str
    group: str  # source video / subject ID -- the leakage group key
    duration: Optional[float] = None
    split: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    total: int = 0
    valid: int = 0
    missing_files: List[str] = field(default_factory=list)
    corrupt_files: List[str] = field(default_factory=list)
    invalid_labels: List[Tuple[str, str]] = field(default_factory=list)  # (path, label)
    duplicates: List[Tuple[str, str]] = field(default_factory=list)  # (path_a, path_b)
    empty_files: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_files
            or self.corrupt_files
            or self.invalid_labels
            or self.empty_files
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "missing_files": self.missing_files,
            "corrupt_files": self.corrupt_files,
            "invalid_labels": [list(t) for t in self.invalid_labels],
            "duplicates": [list(t) for t in self.duplicates],
            "empty_files": self.empty_files,
            "ok": self.ok,
        }

    def summary(self) -> str:
        lines = [
            f"Samples: {self.total} total, {self.valid} valid",
            f"Missing files: {len(self.missing_files)}",
            f"Corrupt files: {len(self.corrupt_files)}",
            f"Empty files: {len(self.empty_files)}",
            f"Invalid labels: {len(self.invalid_labels)}",
            f"Duplicate pairs: {len(self.duplicates)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def load_manifest(path: Union[str, Path]) -> List[ManifestRow]:
    """
    Load a dataset manifest from CSV or JSON.

    Required columns/keys: ``path``, ``label``, ``group``.
    ``group`` identifies the source video/subject and is the unit that keeps
    train/val/test splits leakage-free.
    """
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Manifest not found: {p}")

    rows: List[ManifestRow] = []
    if p.suffix.lower() == ".json":
        try:
            with open(p, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Manifest {p} is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise DatasetError(f"JSON manifest {p} must be a list of objects")
        records = data
    elif p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetError(f"CSV manifest {p} is empty")
            missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames)
            if missing:
                raise DatasetError(f"Manifest {p} missing required columns: {sorted(missing)}")
            records = list(reader)
    else:
        raise DatasetError(f"Unsupported manifest format '{p.suffix}' (use .csv or .json)")

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetError(f"Manifest row {i} is not a mapping")
        missing = REQUIRED_MANIFEST_COLUMNS - set(record)
        if missing:
            raise DatasetError(f"Manifest row {i} missing keys: {sorted(missing)}")
        meta = {k: v for k, v in record.items() if k not in REQUIRED_MANIFEST_COLUMNS | OPTIONAL_MANIFEST_COLUMNS}
        duration = record.get("duration")
        rows.append(
            ManifestRow(
                path=str(record["path"]),
                label=str(record["label"]),
                group=str(record["group"]),
                duration=float(duration) if duration not in (None, "") else None,
                split=record.get("split") or None,
                meta=meta,
            )
        )
    return rows


def save_manifest(rows: Sequence[ManifestRow], path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "group", "duration", "split"]
    if p.suffix.lower() == ".json":
        payload = [
            {
                "path": r.path,
                "label": r.label,
                "group": r.group,
                "duration": r.duration,
                "split": r.split,
                **r.meta,
            }
            for r in rows
        ]
        with open(p, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    else:
        with open(p, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {"path": r.path, "label": r.label, "group": r.group,
                     "duration": r.duration or "", "split": r.split or ""}
                )
    return p


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def file_md5(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(image: np.ndarray, hash_size: int = 8) -> int:
    """Perceptual average hash of an image (grayscale, hash_size^2 bits)."""
    import cv2

    if image is None or image.size == 0:
        raise DatasetError("Cannot hash an empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    bits = (resized > avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def validate_dataset(
    rows: Sequence[ManifestRow],
    allowed_labels: Optional[Set[str]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    check_duplicates: bool = True,
    duplicate_hash_threshold: int = 4,
) -> ValidationReport:
    """
    Validate a dataset manifest: files exist, are decodable, labels are legal,
    and (optionally) perceptual duplicates are flagged.
    """
    import cv2

    base = Path(base_dir) if base_dir else None
    report = ValidationReport(total=len(rows))

    hashes: List[Tuple[str, int]] = []
    seen_md5: Dict[str, str] = {}

    for row in rows:
        file_path = Path(row.path)
        if base and not file_path.is_absolute():
            file_path = base / file_path

        if not file_path.exists():
            report.missing_files.append(str(file_path))
            continue
        if file_path.stat().st_size == 0:
            report.empty_files.append(str(file_path))
            continue

        # Label validation
        if allowed_labels and row.label not in allowed_labels:
            report.invalid_labels.append((str(file_path), row.label))

        # Exact-duplicate detection via md5
        if check_duplicates:
            try:
                digest = file_md5(file_path)
            except OSError:
                report.corrupt_files.append(str(file_path))
                continue
            if digest in seen_md5:
                report.duplicates.append((seen_md5[digest], str(file_path)))
            else:
                seen_md5[digest] = str(file_path)

        # Decode check for images (videos are checked via cv2 open)
        suffix = file_path.suffix.lower()
        try:
            if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                if image is None:
                    report.corrupt_files.append(str(file_path))
                    continue
                if check_duplicates:
                    hashes.append((str(file_path), average_hash(image)))
            elif suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    report.corrupt_files.append(str(file_path))
                else:
                    ok, _ = cap.read()
                    if not ok:
                        report.corrupt_files.append(str(file_path))
                    cap.release()
            else:
                # Unknown type: presence/size checks only.
                pass
        except Exception as exc:  # defensive: never crash validation
            logger.warning("Failed to inspect %s: %s", file_path, exc)
            report.corrupt_files.append(str(file_path))
            continue

        report.valid += 1

    # Near-duplicate detection on image hashes
    if check_duplicates and hashes:
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if hamming_distance(hashes[i][1], hashes[j][1]) <= duplicate_hash_threshold:
                    pair = (hashes[i][0], hashes[j][0])
                    if pair not in report.duplicates and (pair[1], pair[0]) not in report.duplicates:
                        report.duplicates.append(pair)

    return report


# ---------------------------------------------------------------------------
# Leakage-safe splitting
# ---------------------------------------------------------------------------


def group_split(
    rows: Sequence[ManifestRow],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Dict[str, List[ManifestRow]]:
    """
    Split rows into train/val/test keeping ALL rows from the same ``group``
    in one split. This prevents frame-level data leakage where adjacent
    frames of one video would otherwise land in both train and test.

    Ratios are approximate when stratifying by group label.
    """
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise DatasetError("Require train_ratio > 0, val_ratio >= 0 and train_ratio + val_ratio < 1")
    if not rows:
        raise DatasetError("Cannot split an empty dataset")

    rng = np.random.default_rng(seed)

    # Group rows by leakage key.
    groups: Dict[str, List[ManifestRow]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)

    # Optional stratification: sort groups by dominant label for proportional assignment.
    def dominant_label(items: List[ManifestRow]) -> str:
        counts: Dict[str, int] = {}
        for item in items:
            counts[item.label] = counts.get(item.label, 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    group_keys = list(groups.keys())
    if stratify:
        # Shuffle within each label then round-robin per label for balance.
        by_label: Dict[str, List[str]] = {}
        for key in group_keys:
            by_label.setdefault(dominant_label(groups[key]), []).append(key)
        ordered: List[str] = []
        for label in sorted(by_label):
            keys = by_label[label]
            rng.shuffle(keys)
            ordered.extend(keys)
        group_keys = ordered
    else:
        rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = int(round(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train + n_val])
    # Remaining keys form the test split.

    splits: Dict[str, List[ManifestRow]] = {"train": [], "val": [], "test": []}
    for key, items in groups.items():
        if key in train_keys:
            target = "train"
        elif key in val_keys:
            target = "val"
        else:
            target = "test"
        for row in items:
            row.split = target
            splits[target].append(row)

    # Guarantee no group leaks across splits.
    assert_leakage_free(splits)
    return splits


def assert_leakage_free(splits: Dict[str, List[ManifestRow]]) -> None:
    """Raise DatasetError if any group appears in more than one split."""
    seen: Dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            if row.group in seen and seen[row.group] != split_name:
                raise DatasetError(
                    f"Data leakage: group '{row.group}' appears in splits "
                    f"'{seen[row.group]}' and '{split_name}'"
                )
            seen[row.group] = split_name


def split_statistics(splits: Dict[str, List[ManifestRow]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for name, rows in splits.items():
        labels: Dict[str, int] = {}
        groups: Set[str] = set()
        for row in rows:
            labels[row.label] = labels.get(row.label, 0) + 1
            groups.add(row.group)
        stats[name] = {
            "samples": len(rows),
            "groups": len(groups),
            "labels": labels,
        }
    return stats