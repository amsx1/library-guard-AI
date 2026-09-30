"""
LibraryGuard - Configuration loading and validation.

All tunable thresholds live in ``configs/config.yaml``; this module parses the
file into typed dataclasses and validates values so the rest of the code base
never has to hard-code magic numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Typed configuration sections
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str = "yolov8n"
    confidence_threshold: float = 0.5
    device: str = "auto"
    target_classes: List[int] = field(default_factory=list)


@dataclass
class ModelsConfig:
    person_detector: ModelConfig = field(default_factory=ModelConfig)
    object_detector: ModelConfig = field(default_factory=ModelConfig)
    pose_estimator: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class TrackingConfig:
    tracker: str = "bytetrack"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    max_det_per_frame: int = 100


@dataclass
class DrinkingConfig:
    enabled: bool = True
    min_duration_seconds: float = 1.5
    confidence_threshold: float = 0.6
    cooldown_seconds: float = 5.0
    min_consecutive_frames: int = 5
    hand_to_mouth_distance_threshold: float = 0.15
    object_near_face_threshold: float = 0.2


@dataclass
class SleepingConfig:
    enabled: bool = True
    min_duration_seconds: float = 10.0
    confidence_threshold: float = 0.65
    cooldown_seconds: float = 15.0
    min_consecutive_frames: int = 30
    head_down_angle_threshold: float = 45.0
    low_movement_threshold: float = 0.02
    eye_closed_threshold: float = 0.3


@dataclass
class PhoneConfig:
    enabled: bool = True
    min_duration_seconds: float = 2.0
    confidence_threshold: float = 0.6
    cooldown_seconds: float = 5.0
    min_consecutive_frames: int = 8
    phone_near_hand_threshold: float = 0.15
    phone_near_face_threshold: float = 0.25
    looking_down_angle_threshold: float = 30.0


@dataclass
class BehaviorConfig:
    drinking: DrinkingConfig = field(default_factory=DrinkingConfig)
    sleeping: SleepingConfig = field(default_factory=SleepingConfig)
    phone_usage: PhoneConfig = field(default_factory=PhoneConfig)


@dataclass
class DetectionConfig:
    global_confidence_threshold: float = 0.5
    frame_skip: int = 1
    max_frames: Optional[int] = None
    input_resize: Optional[List[int]] = None


@dataclass
class VideoConfig:
    output_codec: str = "mp4v"
    output_fps: Optional[float] = None
    draw_bboxes: bool = True
    draw_tracking_ids: bool = True
    draw_behavior_labels: bool = True
    draw_confidence: bool = True
    draw_timestamp: bool = True
    font_scale: float = 0.6
    line_thickness: int = 2


@dataclass
class ReportingConfig:
    event_cooldown_seconds: float = 5.0
    min_event_duration_seconds: float = 1.0
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    include_snapshots: bool = False
    snapshot_interval_seconds: float = 1.0


@dataclass
class PerformanceConfig:
    batch_size: int = 1
    half_precision: bool = False
    num_threads: int = 4
    enable_tensorrt: bool = False


@dataclass
class PrivacyConfig:
    blur_faces: bool = True
    blur_license_plates: bool = False
    store_raw_video: bool = False
    anonymize_tracking_ids: bool = True


@dataclass
class PathsConfig:
    models_dir: str = "models"
    data_dir: str = "data"
    output_dir: str = "outputs"
    logs_dir: str = "logs"
    temp_dir: str = "temp"


@dataclass
class AppConfig:
    """Root configuration object."""

    models: ModelsConfig = field(default_factory=ModelsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> Dict[str, Any]:
        def _convert(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        return _convert(self)


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _as_float(section: str, key: str, value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{section}.{key} must be a number, got {value!r}") from exc
    _require(low <= number <= high, f"{section}.{key} must be in [{low}, {high}], got {number}")
    return number


def _as_int(section: str, key: str, value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{section}.{key} must be an integer, got {value!r}") from exc
    _require(low <= number <= high, f"{section}.{key} must be in [{low}, {high}], got {number}")
    return number


def _as_bool(section: str, key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{section}.{key} must be a boolean, got {value!r}")


def _parse_model(section: Dict[str, Any], name: str) -> ModelConfig:
    section = section or {}
    cfg = ModelConfig()
    cfg.name = str(section.get("name", cfg.name))
    cfg.confidence_threshold = _as_float(name, "confidence_threshold", section.get("confidence_threshold", 0.5), 0.0, 1.0)
    cfg.device = str(section.get("device", "auto"))
    _require(cfg.device in {"auto", "cpu", "cuda", "mps"}, f"{name}.device must be auto/cpu/cuda/mps")
    classes = section.get("target_classes", [])
    _require(isinstance(classes, list), f"{name}.target_classes must be a list of ints")
    cfg.target_classes = [int(c) for c in classes]
    return cfg


def _parse_behavior_section(name: str, section: Dict[str, Any]) -> Any:
    section = section or {}
    if name == "drinking":
        cfg = DrinkingConfig()
    elif name == "sleeping":
        cfg = SleepingConfig()
    else:
        cfg = PhoneConfig()

    if "enabled" in section:
        cfg.enabled = _as_bool(name, "enabled", section["enabled"])
    cfg.min_duration_seconds = _as_float(name, "min_duration_seconds", section.get("min_duration_seconds", cfg.min_duration_seconds), 0.1, 3600.0)
    cfg.confidence_threshold = _as_float(name, "confidence_threshold", section.get("confidence_threshold", cfg.confidence_threshold), 0.0, 1.0)
    cfg.cooldown_seconds = _as_float(name, "cooldown_seconds", section.get("cooldown_seconds", cfg.cooldown_seconds), 0.0, 3600.0)
    cfg.min_consecutive_frames = _as_int(name, "min_consecutive_frames", section.get("min_consecutive_frames", cfg.min_consecutive_frames), 1, 100000)

    if name == "drinking":
        cfg.hand_to_mouth_distance_threshold = _as_float(name, "hand_to_mouth_distance_threshold", section.get("hand_to_mouth_distance_threshold", cfg.hand_to_mouth_distance_threshold), 0.01, 1.0)
        cfg.object_near_face_threshold = _as_float(name, "object_near_face_threshold", section.get("object_near_face_threshold", cfg.object_near_face_threshold), 0.01, 1.0)
    elif name == "sleeping":
        cfg.head_down_angle_threshold = _as_float(name, "head_down_angle_threshold", section.get("head_down_angle_threshold", cfg.head_down_angle_threshold), 5.0, 90.0)
        cfg.low_movement_threshold = _as_float(name, "low_movement_threshold", section.get("low_movement_threshold", cfg.low_movement_threshold), 0.0, 1.0)
        cfg.eye_closed_threshold = _as_float(name, "eye_closed_threshold", section.get("eye_closed_threshold", cfg.eye_closed_threshold), 0.0, 1.0)
    else:
        cfg.phone_near_hand_threshold = _as_float(name, "phone_near_hand_threshold", section.get("phone_near_hand_threshold", cfg.phone_near_hand_threshold), 0.01, 1.0)
        cfg.phone_near_face_threshold = _as_float(name, "phone_near_face_threshold", section.get("phone_near_face_threshold", cfg.phone_near_face_threshold), 0.01, 1.0)
        cfg.looking_down_angle_threshold = _as_float(name, "looking_down_angle_threshold", section.get("looking_down_angle_threshold", cfg.looking_down_angle_threshold), 5.0, 90.0)

    return cfg


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load and validate a YAML config file. Falls back to built-in defaults."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        if path is not None:
            raise ConfigError(f"Configuration file not found: {config_path}")
        # No explicit path and no default file: use defaults.
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config {config_path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    raw = copy.deepcopy(raw)
    cfg = AppConfig()

    models = raw.get("models") or {}
    cfg.models = ModelsConfig(
        person_detector=_parse_model(models.get("person_detector"), "models.person_detector"),
        object_detector=_parse_model(models.get("object_detector"), "models.object_detector"),
        pose_estimator=_parse_model(models.get("pose_estimator"), "models.pose_estimator"),
    )

    tracking = raw.get("tracking") or {}
    cfg.tracking.tracker = str(tracking.get("tracker", "bytetrack"))
    _require(cfg.tracking.tracker in {"bytetrack", "botsort", "simple-iou"}, "tracking.tracker must be bytetrack/botsort/simple-iou")
    cfg.tracking.max_age = _as_int("tracking", "max_age", tracking.get("max_age", 30), 1, 100000)
    cfg.tracking.min_hits = _as_int("tracking", "min_hits", tracking.get("min_hits", 3), 1, 100000)
    cfg.tracking.iou_threshold = _as_float("tracking", "iou_threshold", tracking.get("iou_threshold", 0.3), 0.0, 1.0)
    cfg.tracking.max_det_per_frame = _as_int("tracking", "max_det_per_frame", tracking.get("max_det_per_frame", 100), 1, 100000)

    behavior = raw.get("behavior") or {}
    cfg.behavior = BehaviorConfig(
        drinking=_parse_behavior_section("drinking", behavior.get("drinking")),
        sleeping=_parse_behavior_section("sleeping", behavior.get("sleeping")),
        phone_usage=_parse_behavior_section("phone_usage", behavior.get("phone_usage")),
    )

    detection = raw.get("detection") or {}
    cfg.detection.global_confidence_threshold = _as_float("detection", "global_confidence_threshold", detection.get("global_confidence_threshold", 0.5), 0.0, 1.0)
    cfg.detection.frame_skip = _as_int("detection", "frame_skip", detection.get("frame_skip", 1), 1, 1000)
    max_frames = detection.get("max_frames")
    if max_frames is not None:
        max_frames = _as_int("detection", "max_frames", max_frames, 1, 10**9)
    cfg.detection.max_frames = max_frames
    input_resize = detection.get("input_resize")
    if input_resize is not None:
        _require(isinstance(input_resize, list) and len(input_resize) == 2, "detection.input_resize must be [width, height] or null")
        input_resize = [_as_int("detection", "input_resize", v, 16, 8192) for v in input_resize]
    cfg.detection.input_resize = input_resize

    video = raw.get("video") or {}
    cfg.video.output_codec = str(video.get("output_codec", "mp4v"))
    output_fps = video.get("output_fps")
    if output_fps is not None:
        output_fps = _as_float("video", "output_fps", output_fps, 0.1, 240.0)
    cfg.video.output_fps = output_fps
    for key in ("draw_bboxes", "draw_tracking_ids", "draw_behavior_labels", "draw_confidence", "draw_timestamp"):
        if key in video:
            setattr(cfg.video, key, _as_bool("video", key, video[key]))
    cfg.video.font_scale = _as_float("video", "font_scale", video.get("font_scale", 0.6), 0.1, 5.0)
    cfg.video.line_thickness = _as_int("video", "line_thickness", video.get("line_thickness", 2), 1, 20)

    reporting = raw.get("reporting") or {}
    cfg.reporting.event_cooldown_seconds = _as_float("reporting", "event_cooldown_seconds", reporting.get("event_cooldown_seconds", 5.0), 0.0, 3600.0)
    cfg.reporting.min_event_duration_seconds = _as_float("reporting", "min_event_duration_seconds", reporting.get("min_event_duration_seconds", 1.0), 0.0, 3600.0)
    formats = reporting.get("output_formats", ["json", "csv"])
    _require(isinstance(formats, list) and all(f in {"json", "csv"} for f in formats), "reporting.output_formats must be a list containing json and/or csv")
    cfg.reporting.output_formats = list(formats)
    if "include_snapshots" in reporting:
        cfg.reporting.include_snapshots = _as_bool("reporting", "include_snapshots", reporting["include_snapshots"])
    cfg.reporting.snapshot_interval_seconds = _as_float("reporting", "snapshot_interval_seconds", reporting.get("snapshot_interval_seconds", 1.0), 0.1, 3600.0)

    performance = raw.get("performance") or {}
    cfg.performance.batch_size = _as_int("performance", "batch_size", performance.get("batch_size", 1), 1, 64)
    if "half_precision" in performance:
        cfg.performance.half_precision = _as_bool("performance", "half_precision", performance["half_precision"])
    cfg.performance.num_threads = _as_int("performance", "num_threads", performance.get("num_threads", 4), 1, 128)
    if "enable_tensorrt" in performance:
        cfg.performance.enable_tensorrt = _as_bool("performance", "enable_tensorrt", performance["enable_tensorrt"])

    privacy = raw.get("privacy") or {}
    for key in ("blur_faces", "blur_license_plates", "store_raw_video", "anonymize_tracking_ids"):
        if key in privacy:
            setattr(cfg.privacy, key, _as_bool("privacy", key, privacy[key]))

    paths = raw.get("paths") or {}
    for key in ("models_dir", "data_dir", "output_dir", "logs_dir", "temp_dir"):
        if key in paths:
            setattr(cfg.paths, key, str(paths[key]))

    return cfg


def apply_overrides(config: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    """Apply flat dotted overrides such as ``{'behavior.drinking.min_duration_seconds': 2}``."""
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target: Any = config
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ConfigError(f"Unknown config key: {dotted_key}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ConfigError(f"Unknown config key: {dotted_key}")
        setattr(target, leaf, value)
    return config


def resolve_device(device: str) -> str:
    """Resolve a device string (auto/cpu/cuda/mps) to a concrete torch device."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def ensure_directories(config: AppConfig, base: Optional[Union[str, Path]] = None) -> None:
    """Create output/log/temp directories if missing."""
    root = Path(base) if base is not None else Path.cwd()
    for name in (config.paths.output_dir, config.paths.logs_dir, config.paths.temp_dir):
        (root / name).mkdir(parents=True, exist_ok=True)