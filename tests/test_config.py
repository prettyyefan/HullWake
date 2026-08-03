from pathlib import Path

import pytest

from hullwake.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_main_config_is_valid_and_paper_values_are_locked():
    config = load_config(ROOT / "configs" / "hullwake_r50_fpn.yaml")
    assert config["model"]["corridor_points_per_direction"] == 64
    assert config["model"]["corridor_grid_size"] == 8
    assert config["model"]["length_max_ratio"] == 3.0
    assert config["model"]["width_max_ratio"] == 1.0
    assert config["loss"]["beta_dec"] == 0.05
    assert config["loss"]["lambda_att"] == 0.5


def test_ablation_deep_merge_preserves_base_values():
    config = load_config(ROOT / "configs" / "ablations" / "dominance.yaml")
    assert config["experiment"]["name"] == "ablation_dominance"
    assert config["loss"]["enable_dominance"] is True
    assert config["loss"]["enable_consistency"] is False
    assert config["model"]["embedding_dim"] == 256


def test_invalid_point_grid_is_rejected():
    with pytest.raises(ConfigError, match="corridor_points"):
        load_config(
            ROOT / "configs" / "hullwake_r50_fpn.yaml",
            ["model.corridor_points_per_direction=63"],
        )
