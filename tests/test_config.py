"""Adversarial configuration tests for temporal and path safety."""

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError
import yaml

from src.config.loader import ConfigurationError, load_config, read_yaml
from src.utils.reproducibility import config_hash

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    """Isolated relocatable configuration tree."""
    path = tmp_path / "configs"
    shutil.copytree(PROJECT / "configs", path)
    return path


def edit(directory: Path, filename: str, keys: tuple[str, ...], value: object) -> None:
    """Mutate one YAML field without bypassing loader validation."""
    path = directory / filename
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cursor = raw
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_default_and_single_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_config()
    monkeypatch.chdir(tmp_path)
    assert load_config().project.root == PROJECT
    assert set(config.point_in_time_policy.datasets) == {"retailrocket", "taobao", "yoochoose", "movielens"}
    assert set(load_config(dataset="taobao").point_in_time_policy.datasets) == {"taobao"}
    assert config.dataset("taobao").raw_dir == PROJECT / "data" / "taobao" / "raw"
    assert int(config.splits.taobao.train_end.timestamp()) == 1512172800
    with pytest.raises(ConfigurationError, match="not loaded"):
        config.dataset("unknown")
    with pytest.raises(ConfigurationError, match="unknown dataset"):
        load_config(dataset="../taobao")


def test_portable_hash(configs: Path) -> None:
    assert config_hash(load_config(configs)) == config_hash(load_config())
    before = config_hash(load_config(configs))
    edit(configs, "main_config.yaml", ("runtime", "seed"), 43)
    assert config_hash(load_config(configs)) != before


@pytest.mark.parametrize("filename,keys,value", [
    ("main_config.yaml", ("runtime", "unexpected"), True),
    ("main_config.yaml", ("runtime", "seed"), "42"),
    ("main_config.yaml", ("runtime", "seed"), -1),
    ("main_config.yaml", ("runtime", "chunk_rows"), 0),
    ("main_config.yaml", ("runtime", "training_seeds"), [42, 42]),
    ("main_config.yaml", ("runtime", "training_seeds"), []),
    ("main_config.yaml", ("runtime", "models"), {}),
    ("main_config.yaml", ("runtime", "training", "gnn_fanout"), [-1]),
    ("main_config.yaml", ("runtime", "training", "inner_train_quantiles"), [0.8, 0.5]),
    ("main_config.yaml", ("runtime", "models", "adv3", "attention_heads"), 3),
    ("main_config.yaml", ("runtime", "models", "baseline", "windows_days"), [0]),
    ("main_config.yaml", ("runtime", "models", "adv1", "selected_features"), 900),
    ("main_config.yaml", ("point_in_time_policy", "timestamp_handling", "strict_less_than"), False),
    ("main_config.yaml", ("point_in_time_policy", "synthetic", "negatives", "future_conflict_policy"), "drop_conflicts"),
    ("main_config.yaml", ("point_in_time_policy", "synthetic", "negatives", "difficulty_mix"), [0.5, 0.5, 0.5]),
    ("main_config.yaml", ("point_in_time_policy", "synthetic", "personas", "uncertainty_quantiles"), [0.8, 0.5]),
    ("main_config.yaml", ("point_in_time_policy", "synthetic", "personas", "mixing_probabilities"), [2.0, 0.0, 0.0]),
    ("sequence_config.yaml", ("sequences", "max_history_length"), 51),
    ("sequence_config.yaml", ("sequences", "causal_mask_required"), False),
    ("evaluation_config.yaml", ("evaluation", "candidate_pool_size"), 99),
    ("evaluation_config.yaml", ("evaluation", "primary_targets", "yoochoose"), "buy"),
    ("evaluation_config.yaml", ("evaluation", "repeat_purchase_policy"), {}),
    ("datasets/taobao.yaml", ("dataset", "category_id_availability"), "derived_static"),
    ("datasets/taobao.yaml", ("dataset", "adapter", "header"), True),
    ("datasets/taobao.yaml", ("dataset", "files", "events"), "../other.csv"),
    ("datasets/taobao.yaml", ("dataset", "files"), {}),
    ("datasets/movielens.yaml", ("dataset", "genome_scores_enabled"), True),
    ("splits_config.yaml", ("splits", "taobao", "train_end"), "2017-12-02T00:00:00"),
    ("splits_config.yaml", ("splits", "taobao", "train_end"), "2018-12-02T00:00:00Z"),
    ("splits_config.yaml", ("splits", "movielens", "train_quantile"), 0.95),
])
def test_reject_unsafe_values(configs: Path, filename: str, keys: tuple[str, ...], value: object) -> None:
    edit(configs, filename, keys, value)
    with pytest.raises((ValidationError, ConfigurationError)):
        load_config(configs)


@pytest.mark.parametrize("raw_dir,processed_dir", [("../outside", "data/taobao/processed"), ("data/taobao/raw", "data/taobao/raw"), ("data/other/raw", "data/taobao/processed")])
def test_path_boundaries(configs: Path, raw_dir: str, processed_dir: str) -> None:
    edit(configs, "datasets/taobao.yaml", ("dataset", "raw_dir"), raw_dir)
    edit(configs, "datasets/taobao.yaml", ("dataset", "processed_dir"), processed_dir)
    with pytest.raises(ConfigurationError):
        load_config(configs)


def test_duplicate_and_dangerous_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    for text in ("key: 1\nkey: 2\n", "!!python/object/apply:os.system ['whoami']", "[not,a,mapping]", "key: [broken"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigurationError):
            read_yaml(path)
    with pytest.raises(ConfigurationError, match="cannot load"):
        read_yaml(tmp_path / "absent.yaml")


@pytest.mark.parametrize("filename,keys,value", [
    ("main_config.yaml", ("extra",), {}),
    ("main_config.yaml", ("point_in_time_policy", "evaluation"), {}),
    ("main_config.yaml", ("point_in_time_policy", "datasets"), {}),
    ("sequence_config.yaml", ("extra",), {}),
    ("datasets/taobao.yaml", ("dataset", "name"), "movielens"),
    ("datasets/taobao.yaml", ("dataset", "genres_availability"), "static_assumed"),
])
def test_configuration_ownership(configs: Path, filename: str, keys: tuple[str, ...], value: object) -> None:
    edit(configs, filename, keys, value)
    with pytest.raises((ConfigurationError, ValidationError)):
        load_config(configs)
