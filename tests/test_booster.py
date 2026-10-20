"""Real booster fitting, serialization and forbidden-feature protections."""

from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.models.baseline.booster import BoosterModel, fit_booster, temporal_feature_selection


def training() -> pl.DataFrame:
    """A learnable numeric relation with a train-only categorical field."""
    return pl.DataFrame({"timestamp": list(range(1, 101)), "target": [i % 2 for i in range(100)],
        "x": [float(i % 2) for i in range(100)], "noise": [float(i % 7) for i in range(100)],
        "constant": [1.] * 100, "category_id": [i % 3 for i in range(100)]})


@pytest.mark.parametrize("backend", ["catboost", "lightgbm"])
def test_fitting_and_serialization(tmp_path: Path, backend: str) -> None:
    train = training()
    valid = train.with_columns((pl.col("timestamp") + 100).alias("timestamp"))
    model = fit_booster(train, valid, ["x", "noise", "category_id"], backend=backend, iterations=15, patience=3)
    scores = model.predict(valid)
    assert np.isfinite(scores).all()
    assert scores[train["target"].to_numpy() == 1].mean() > scores[train["target"].to_numpy() == 0].mean()
    model.save(tmp_path)
    assert np.allclose(scores, BoosterModel.load(tmp_path).predict(valid))
    assert model.predict(valid.head(0)).size == 0
    with pytest.raises(ValueError, match="strictly after"):
        model.predict(train)
    assert np.isfinite(model.predict(train, diagnostic_on_train=True)).all()
    softened = fit_booster(train, None, ["x"], backend=backend, iterations=2, soft_targets=np.where(train["target"].to_numpy(), .8, .2))
    assert np.isfinite(softened.predict(valid)).all()


def test_selection_and_errors() -> None:
    train = training()
    assert temporal_feature_selection(train, ["x", "constant"]) == ["x"]
    assert temporal_feature_selection(train, ["constant"]) == ["constant"]
    assert temporal_feature_selection(train, ["x", "noise"], top_k=1) == ["x"]
    for kwargs in ({"backend": "bad"}, {"iterations": 0}, {"soft_targets": [1.1] * 100}, {"soft_targets": [float("nan")] * 100}):
        with pytest.raises(ValueError):
            fit_booster(train, None, ["x"], **kwargs)
    for features in ([], ["x", "x"], ["target"], ["timestamp"], ["future_positive_conflict"]):
        with pytest.raises(ValueError):
            fit_booster(train, None, features, iterations=2)
    with pytest.raises(ValueError):
        fit_booster(train.with_columns(pl.lit(float("nan")).alias("x")), None, ["x"])
    with pytest.raises(ValueError):
        fit_booster(train, train, ["x"])
    with pytest.raises(ValueError):
        fit_booster(train.with_columns(pl.lit(0).alias("target")), None, ["x"])
    with pytest.raises(ValueError):
        temporal_feature_selection(train, ["x", "noise"], top_k=0)
