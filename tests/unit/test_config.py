from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vla_fewshot.config import discover_configs, load_config


ROOT = Path(__file__).resolve().parents[2]


def test_all_tracked_yaml_configs_are_strictly_valid() -> None:
    configs = discover_configs(ROOT / "configs")
    assert configs
    for path in configs:
        load_config(path)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "configs" / "platform" / "gpu_vm.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["unexpected_key"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="unexpected_key"):
        load_config(path)


def test_hard_coded_path_outside_platform_overlay_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        "kind: storage\nschema_version: 1\nunsafe: /mnt/vla/runs\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard-coded platform path"):
        load_config(path)


def test_platform_overlay_may_define_host_specific_defaults() -> None:
    config = load_config(ROOT / "configs" / "platform" / "colab.yaml")
    assert config.storage.data_root_default.startswith("/content/drive/")


def test_env_config_freezes_relative_control_and_identity_transform() -> None:
    config = load_config(ROOT / "configs" / "env.yaml")
    assert config.kind == "env"
    assert config.control_mode == "relative"
    assert config.action_dim == 7
    assert config.orientation.project_transform == "identity"
    assert config.cameras["wrist"].env_raw_key == "image2"
    assert config.cameras["wrist"].policy_key == "observation.images.wrist_image"
