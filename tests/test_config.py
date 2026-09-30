"""Tests: configuration loading and validation."""

from __future__ import annotations

import pytest

from src.utils.config import (
    AppConfig,
    ConfigError,
    apply_overrides,
    load_config,
)


def test_load_default_config():
    """The repository's shipped config.yaml must load cleanly."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert 0 < config.behavior.drinking.confidence_threshold <= 1
    assert config.behavior.sleeping.min_duration_seconds > config.behavior.drinking.min_duration_seconds


def test_load_config_file(config_file):
    config = load_config(config_file)
    assert config.tracking.tracker == "bytetrack"
    assert config.models.object_detector.target_classes == [39, 41, 67]
    assert config.behavior.phone_usage.min_duration_seconds == 2.0


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_root_type_raises(tmp_path):
    path = tmp_path / "bad2.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_out_of_range_value_raises(config_dict, tmp_path):
    import yaml

    config_dict["behavior"]["drinking"]["confidence_threshold"] = 3.5
    path = tmp_path / "oor.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(path)


def test_non_numeric_value_raises(config_dict, tmp_path):
    import yaml

    config_dict["tracking"]["iou_threshold"] = "high"
    path = tmp_path / "nan.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_device_raises(config_dict, tmp_path):
    import yaml

    config_dict["models"]["person_detector"]["device"] = "tpu"
    path = tmp_path / "dev.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    with pytest.raises(ConfigError, match="device"):
        load_config(path)


def test_invalid_tracker_raises(config_dict, tmp_path):
    import yaml

    config_dict["tracking"]["tracker"] = "magic"
    path = tmp_path / "trk.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    with pytest.raises(ConfigError, match="tracker"):
        load_config(path)


def test_apply_overrides():
    config = load_config()
    apply_overrides(config, {"behavior.drinking.min_duration_seconds": 9.0})
    assert config.behavior.drinking.min_duration_seconds == 9.0


def test_apply_unknown_override_raises():
    config = load_config()
    with pytest.raises(ConfigError, match="Unknown config key"):
        apply_overrides(config, {"behavior.drinking.not_a_key": 1})


def test_to_dict_round_trip():
    config = load_config()
    data = config.to_dict()
    assert data["behavior"]["drinking"]["min_duration_seconds"] == \
        config.behavior.drinking.min_duration_seconds