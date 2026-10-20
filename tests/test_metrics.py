"""Known ranks, tied AUC, pool fairness and temporal calibration invariants."""

import numpy as np
import polars as pl
import pytest

from src.evaluation.metrics import (PlattCalibrator, auc, bind_predictions, compare_models,
    hit_rate_at_k, log_loss, mrr, ndcg_at_k, paired_cluster_bootstrap, per_group_metrics,
    pool_fingerprint, recall_at_k)


def scored() -> pl.DataFrame:
    """Two groups with positive ranks one and two, and a tied AUC pair."""
    return pl.DataFrame({"eval_group_id": [1, 1, 1, 2, 2, 2], "user_id": [10, 10, 10, 20, 20, 20],
        "candidate_item_id": [1, 2, 3, 1, 2, 3], "candidate_label": [1, 0, 0, 0, 1, 0],
        "score": [3., 2., 1., 2., 2., 1.], "probability": [.8, .2, .1, .5, .5, .1]})


def test_known_vector_metrics() -> None:
    labels, scores = [0, 1, 0], [3., 2., 1.]
    assert ndcg_at_k(labels, scores, 1) == 0
    assert ndcg_at_k(labels, scores, 2) == pytest.approx(1 / np.log2(3))
    assert recall_at_k(labels, scores, 2) == hit_rate_at_k(labels, scores, 2) == 1
    assert recall_at_k(labels, scores, 1) == hit_rate_at_k(labels, scores, 1) == 0
    assert mrr(labels, scores) == .5
    assert auc(labels, scores) == .5
    assert auc([1, 0], [1., 1.]) == .5
    assert auc([1, 0], [2., 1.]) == 1
    assert log_loss([1, 0], [.8, .2]) == pytest.approx(-np.log(.8))
    assert np.isfinite(log_loss([1, 0], [0., 1.]))
    assert mrr([1, 0], [1., 1.], [2, 1]) == .5
    assert recall_at_k([1, 1, 0], [3., 2., 1.], 1) == .5


@pytest.mark.parametrize("labels,scores", [([], []), ([2, 0], [1., 2.]), ([0, 1], [1.]), ([1, 0], [np.nan, 1.]), ([[1]], [[1.]])])
def test_invalid_vectors(labels: list, scores: list) -> None:
    with pytest.raises(ValueError):
        auc(labels, scores)


def test_undefined_and_bad_ids() -> None:
    with pytest.raises(ValueError, match="one-class"):
        auc([1, 1], [1., 2.])
    with pytest.raises(ValueError):
        ndcg_at_k([0, 0], [1., 2.], 1)
    with pytest.raises(ValueError):
        recall_at_k([1, 0], [1., 2.], 0)
    with pytest.raises(ValueError, match="IDs"):
        ndcg_at_k([1, 0], [1., 2.], 1, [3, 3])
    with pytest.raises(ValueError, match="probabilities"):
        log_loss([1, 0], [1.1, 0.])


def test_group_metrics_and_locked_pool_comparison() -> None:
    frame = scored()
    groups = per_group_metrics(frame, [1, 2], pool_size=3)
    assert groups["rank"].to_list() == [1, 2]
    assert groups["AUC"].to_list() == [1., .75]
    assert groups["MRR"].to_list() == [1., .5]
    assert groups["Recall@1"].equals(groups["HitRate@1"])
    assert groups["NDCG@2"].to_list() == pytest.approx([1., 1 / np.log2(3)])
    assert groups["LogLoss"].mean() == pytest.approx(log_loss(frame["candidate_label"], frame["probability"]))
    predictions = frame.select("eval_group_id", "candidate_item_id", "score", "probability")
    table = compare_models(frame, {"a": predictions.reverse(), "b": predictions.lazy()}, [1, 2], pool_size=3)
    assert table["MRR"].to_list() == [.75, .75]
    assert table["pool_hash"].n_unique() == 1
    assert pool_fingerprint(frame) == pool_fingerprint(frame.reverse())
    with pytest.raises(ValueError, match="keys"):
        bind_predictions(frame, predictions.head(5))
    with pytest.raises(ValueError, match="keys"):
        bind_predictions(frame, predictions.with_columns((pl.col("candidate_item_id") + 10).alias("candidate_item_id")))
    with pytest.raises(ValueError, match="duplicate"):
        bind_predictions(frame, pl.concat([predictions, predictions]))
    with pytest.raises(ValueError, match="required"):
        bind_predictions(frame, predictions.drop("score"))
    with pytest.raises(ValueError):
        compare_models(frame, {})


@pytest.mark.parametrize("mutation", ["missing", "nan", "null", "probability", "label", "incomplete", "duplicate", "no_positive", "user", "empty"])
def test_group_failures(mutation: str) -> None:
    frame = scored()
    if mutation == "missing":
        frame = frame.drop("score")
    elif mutation == "nan":
        frame = frame.with_columns(pl.lit(float("nan")).alias("score"))
    elif mutation == "null":
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("score"))
    elif mutation == "probability":
        frame = frame.with_columns(pl.lit(1.1).alias("probability"))
    elif mutation == "label":
        frame = frame.with_columns(pl.lit(2).alias("candidate_label"))
    elif mutation == "incomplete":
        frame = frame.head(5)
    elif mutation == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    elif mutation == "no_positive":
        frame = frame.with_columns(pl.lit(0).alias("candidate_label"))
    elif mutation == "user":
        frame = frame.with_row_index("user_id_new").drop("user_id").rename({"user_id_new": "user_id"})
    else:
        frame = frame.head(0)
    with pytest.raises(ValueError):
        per_group_metrics(frame, pool_size=3)


def test_bad_group_settings() -> None:
    for kwargs in ({"ks": []}, {"ks": [1, 1]}, {"ks": [0]}, {"pool_size": 1}):
        with pytest.raises(ValueError):
            per_group_metrics(scored(), **kwargs)


def test_temporal_calibration_and_bootstrap() -> None:
    s = [-2., -1., 0., 1., 2., 3.]
    y = [0, 0, 1, 0, 1, 1]
    calibration = PlattCalibrator.fit(s, y, [100, 100, 101, 101, 102, 102], split="valid", calibration_start_ts=100)
    assert calibration.a >= 0
    probabilities = calibration.predict(s, target_min_ts=103)
    assert np.diff(probabilities).min() >= 0
    assert log_loss(y, probabilities) < log_loss(y, [.5] * 6)
    with pytest.raises(ValueError, match="strictly after"):
        calibration.predict(s, target_min_ts=102)
    assert np.isfinite(calibration.predict(s, target_min_ts=100, diagnostic_on_fit=True)).all()
    for split, start, labels, times in (("test", 100, y, [100] * 6), ("valid", 101, y, [100] * 6), ("valid", 100, [0] * 6, [100] * 6), ("valid", 100, y, [100.] * 6)):
        with pytest.raises(ValueError):
            PlattCalibrator.fit(s, labels, times, split=split, calibration_start_ts=start)
    pairs = pl.DataFrame({"user_id": [1, 1, 2, 3], "a": [.1, .2, .3, .4], "b": [.2, .3, .4, .5]})
    result = paired_cluster_bootstrap(pairs, "a", "b", repetitions=20)
    assert result["difference"] == pytest.approx(.1)
    assert result["ci_lower"] == pytest.approx(.1)
    assert result == paired_cluster_bootstrap(pairs, "a", "b", repetitions=20)
    assert result == paired_cluster_bootstrap(pairs.reverse(), "a", "b", repetitions=20)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(pairs, "a", "b", repetitions=1)


def test_streaming_platt_matches_arrays():
    scores = np.array([-2., -1., 0., 1., 2., 1.5, -.5, 0.5])
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    timestamps = np.arange(10, 18)
    frame = pl.DataFrame({"score": scores, "target": labels, "timestamp": timestamps}).lazy()
    expected = PlattCalibrator.fit(scores, labels, timestamps, split="valid", calibration_start_ts=10)
    actual = PlattCalibrator.fit_frame(frame, split="valid", calibration_start_ts=10)
    assert np.allclose([expected.a, expected.b], [actual.a, actual.b])
    for bad, split in ((frame, "test"), (frame.with_columns(pl.col("timestamp").cast(pl.Float64)), "valid"),
                       (frame.limit(0), "valid"), (frame.with_columns(pl.lit(0).alias("target")), "valid"),
                       (frame.with_columns(pl.lit(float("nan")).alias("score")), "valid")):
        with pytest.raises(ValueError):
            PlattCalibrator.fit_frame(bad, split=split, calibration_start_ts=10)
    with pytest.raises(ValueError):
        PlattCalibrator.fit_frame(frame, split="valid", calibration_start_ts=11)
