import pytest

from vla_fewshot.model.features import (
    LIBERO_INPUT_FEATURES,
    LIBERO_OUTPUT_FEATURES,
    FeatureContractError,
    assert_libero_policy_features,
)


def test_libero_feature_contract_accepts_canonical_keys() -> None:
    report = assert_libero_policy_features(
        input_features=LIBERO_INPUT_FEATURES,
        output_features=LIBERO_OUTPUT_FEATURES,
    )
    assert report["output_features"]["action"]["shape"] == [7]
    assert report["input_features"]["observation.state"]["shape"] == [8]
    assert "observation.images.wrist_image" in report["input_features"]


def test_so100_hub_features_are_rejected() -> None:
    with pytest.raises(FeatureContractError, match="missing LIBERO"):
        assert_libero_policy_features(
            input_features={
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
            },
            output_features={"action": {"type": "ACTION", "shape": [6]}},
        )


def test_leftover_hub_cameras_are_rejected() -> None:
    inputs = dict(LIBERO_INPUT_FEATURES)
    inputs["observation.images.camera1"] = {"type": "VISUAL", "shape": [3, 256, 256]}
    with pytest.raises(FeatureContractError, match="hub visual keys"):
        assert_libero_policy_features(
            input_features=inputs,
            output_features=LIBERO_OUTPUT_FEATURES,
        )


def test_wrong_action_dim_is_rejected() -> None:
    with pytest.raises(FeatureContractError, match="action shape"):
        assert_libero_policy_features(
            input_features=LIBERO_INPUT_FEATURES,
            output_features={"action": {"type": "ACTION", "shape": [6]}},
        )
