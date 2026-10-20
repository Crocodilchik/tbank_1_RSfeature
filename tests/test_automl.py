"""Typed DSL rejection, candidate determinism and a real Adv1 fit/predict run."""

from pathlib import Path
import polars as pl
import pytest

from src.models.adv1_automl.auto_feature_generator import AutoFeatureGenerator
from src.models.baseline.feature_engineering import FeatureSpec
from src.models.baseline.pipeline import run_gbdt
from tests.helpers import model_bundle


@pytest.mark.parametrize("dataset", ["retailrocket", "taobao", "yoochoose", "movielens"])
def test_candidates_are_bounded_reproducible_and_typed(dataset: str) -> None:
    factory = AutoFeatureGenerator(dataset)
    candidates = factory.generate()
    assert 1 <= len(candidates) <= 500
    assert len({s.name for s in candidates}) == len(candidates)
    assert candidates == AutoFeatureGenerator(dataset).generate()
    assert candidates != AutoFeatureGenerator(dataset, seed=43).generate()
    assert not factory.rejected
    assert factory.audit(candidates)["candidate_total"].unique().to_list() == [len(candidates)]
    assert AutoFeatureGenerator(dataset, max_candidate_features=5).generate()


def test_unsafe_dsl_is_rejected() -> None:
    factory = AutoFeatureGenerator("taobao", max_feature_depth=1)
    accepted = factory.validate([FeatureSpec("user", 1, "count"), FeatureSpec("user", 1, "count"),
        FeatureSpec("target", 1, "count"), FeatureSpec("user", 1, "positive_rate"), "SELECT target FROM future"])
    assert len(accepted) == 1 and len(factory.rejected) == 4
    for kwargs in ({"dataset": "bad"}, {"max_candidate_features": 501}, {"windows_days": []}, {"max_feature_depth": 3}, {"windows_days": [0]}):
        with pytest.raises(ValueError):
            AutoFeatureGenerator(**({"dataset": "taobao"} | kwargs))


def test_adv1_end_to_end(tmp_path: Path) -> None:
    config, root = model_bundle(tmp_path)
    factory = AutoFeatureGenerator("retailrocket", max_candidate_features=30)
    result = run_gbdt(config, "retailrocket", processed_root=root, model_name="adv1", specs=factory.generate(), selected_features=5, smoke=True)
    predictions = pl.read_parquet(result / "seed_42/test_predictions.parquet")
    assert predictions.height == 500 and predictions["score"].is_finite().all()
    features = pl.read_parquet(result / "seed_42/model_metadata.parquet")["features"][0]
    assert len(features) <= 5
