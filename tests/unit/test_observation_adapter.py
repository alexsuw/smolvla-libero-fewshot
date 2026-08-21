from pathlib import Path

import pytest

from vla_fewshot.env.observation_adapter import (
    IDENTITY,
    ROT180,
    apply_canonical_image_keys,
    apply_hwc_transform,
    assert_single_orientation_processor,
    env_pixels_to_policy,
    flatten_libero_robot_state,
    quat_xyzw_to_axis_angle,
)
from vla_fewshot.env.parity import (
    CHOSEN_PROJECT_TRANSFORM,
    frozen_orientation_contract,
    write_parity_bundle,
)


def test_project_adapter_must_not_double_rotate() -> None:
    assert_single_orientation_processor(
        lerobot_libero_processor_rot180=True,
        project_transform=IDENTITY,
    )
    with pytest.raises(ValueError, match="double-transform"):
        assert_single_orientation_processor(
            lerobot_libero_processor_rot180=True,
            project_transform=ROT180,
        )


def test_frozen_transform_is_identity() -> None:
    contract = frozen_orientation_contract()
    assert contract["project_transform"] == IDENTITY
    assert CHOSEN_PROJECT_TRANSFORM == IDENTITY


def test_image2_is_mapped_to_wrist_image() -> None:
    mapped = apply_canonical_image_keys(
        {
            "observation.images.image": "main",
            "observation.images.image2": "wrist",
        }
    )
    assert mapped["observation.images.wrist_image"] == "wrist"
    assert "observation.images.image2" not in mapped
    with pytest.raises(ValueError, match="double alias"):
        apply_canonical_image_keys(
            {
                "observation.images.image": "main",
                "observation.images.image2": "a",
                "observation.images.wrist_image": "b",
            }
        )


def test_env_pixels_require_both_cameras() -> None:
    policy = env_pixels_to_policy({"image": "main", "image2": "wrist"})
    assert set(policy) == {
        "observation.images.image",
        "observation.images.wrist_image",
    }
    with pytest.raises(KeyError, match="image2"):
        env_pixels_to_policy({"image": "main"})


def test_rot180_flips_height_and_width() -> None:
    image = [
        [[1, 0, 0], [2, 0, 0]],
        [[3, 0, 0], [4, 0, 0]],
    ]
    assert apply_hwc_transform(image, ROT180) == [
        [[4, 0, 0], [3, 0, 0]],
        [[2, 0, 0], [1, 0, 0]],
    ]
    assert apply_hwc_transform(image, IDENTITY)[0][0] == [1, 0, 0]


def test_identity_quaternion_is_zero_axis_angle() -> None:
    assert quat_xyzw_to_axis_angle([0.0, 0.0, 0.0, 1.0]) == [0.0, 0.0, 0.0]


def test_flatten_robot_state_is_8d() -> None:
    state = flatten_libero_robot_state(
        {
            "eef": {"pos": [0.1, 0.2, 0.3], "quat": [0.0, 0.0, 0.0, 1.0]},
            "gripper": {"qpos": [0.04, -0.04]},
        }
    )
    assert state == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04]


def test_parity_bundle_writes_candidates(tmp_path: Path) -> None:
    image = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [11, 12, 13]]]
    report = write_parity_bundle(
        output_dir=tmp_path / "parity",
        dataset_main=image,
        dataset_wrist=image,
    )
    assert report["orientation"]["project_transform"] == "identity"
    assert (tmp_path / "parity" / "frames" / "dataset_main_rot180.ppm").exists()
    assert (tmp_path / "parity" / "parity.md").exists()
