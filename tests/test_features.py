"""Independent PIT formulas, timestamp ties and strict metadata availability."""

import polars as pl
import pytest

from src.models.baseline.feature_engineering import FeatureSpec, build_features, feature_columns, manual_specs, reference_features
from src.generator.schemas import ITEMS_SCHEMA


def fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """A target second with simultaneous positives and deliberately future metadata."""
    events = pl.DataFrame({"event_id": [1, 2, 3, 4, 5, 6], "user_id": [1, 2, 1, 1, 1, 1], "item_id": [10, 10, 11, 10, 12, 10],
        "timestamp": [1, 2, 10, 20, 20, 30], "label": [0, 2, 0, 2, 2, 2], "event_type": ["pv", "buy", "pv", "buy", "buy", "buy"], "is_synthetic": [False] * 6})
    queries = pl.DataFrame({"query_group_id": [4, 4, 6], "user_id": [1, 1, 1], "candidate_item_id": [10, 12, 10], "timestamp": [20, 20, 30], "target": [1, 0, 1]})
    metadata = pl.DataFrame([(10, "taobao", "category_id", "1", 1, None, "observed_snapshot", "raw", 1),
        (10, "taobao", "category_id", "2", 20, None, "observed_snapshot", "raw", 20)], schema=ITEMS_SCHEMA, orient="row")
    return events, queries, metadata


def test_causal_counts_te_and_metadata() -> None:
    events, queries, metadata = fixture()
    specs = [FeatureSpec("user", 1, "count"), FeatureSpec("item", 1, "positive_rate"), FeatureSpec("user_item", None, "count"),
        FeatureSpec("user", None, "recency"), FeatureSpec("item", None, "target_encoding"), FeatureSpec("item", 1, "count", "buy")]
    actual = build_features(events.lazy(), queries.lazy(), metadata.lazy(), "taobao", specs, min_history=1).collect().sort(["query_group_id", "candidate_item_id"])
    assert actual["user__1d__any__count"].to_list() == [2., 2., 4.]
    assert actual["item__1d__any__positive_rate"].to_list() == [.5, 0., pytest.approx(2 / 3)]
    assert actual["user_item__all__any__count"].to_list() == [1., 0., 2.]
    assert actual["user__all__any__recency"].to_list() == [10., 10., 10.]
    assert actual["item__1d__buy__count"].to_list() == [1., 0., 2.]
    assert actual["category_id"].to_list() == [1, -1, 2]
    assert actual["item__all__any__target_encoding"][0] == pytest.approx((1 + 10 / 3) / 12)
    # Target and simultaneous labels/types, plus all future rows, cannot alter inputs at T.
    changed = events.with_columns(pl.when(pl.col("timestamp") >= 20).then(pl.lit("pv")).otherwise(pl.col("event_type")).alias("event_type"),
        pl.when(pl.col("timestamp") >= 20).then(pl.lit(0)).otherwise(pl.col("label")).alias("label"))
    modified = build_features(changed.lazy(), queries.filter(pl.col("timestamp") == 20).lazy(), metadata.lazy(), "taobao", specs, min_history=1).collect().sort("candidate_item_id")
    prefix = build_features(events.filter(pl.col("timestamp") < 20).lazy(), queries.filter(pl.col("timestamp") == 20).lazy(), metadata.lazy(), "taobao", specs, min_history=1).collect().sort("candidate_item_id")
    assert modified.select(feature_columns(specs)).equals(prefix.select(feature_columns(specs)))
    assert prefix.select(feature_columns(specs)).equals(actual.head(2).select(feature_columns(specs)))
    reference = reference_features(events.lazy(), queries.lazy(), metadata.lazy(), "taobao", specs, min_history=1).sort(["query_group_id", "candidate_item_id"])
    import numpy as np
    assert np.allclose(reference.select(feature_columns(specs)).to_numpy(), actual.select(feature_columns(specs)).to_numpy())


def test_primitive_validation_and_all_aggregations() -> None:
    events, queries, metadata = fixture()
    specs = [FeatureSpec("user", 1, aggregation) for aggregation in ("sum", "mean", "min", "max", "rating_mean", "recency")]
    result = build_features(events.lazy(), queries.lazy(), metadata.lazy(), "taobao", specs).collect()
    assert result.select(feature_columns(specs)).null_count().sum_horizontal().sum() == 0
    assert len(manual_specs("movielens")) > len(manual_specs("taobao"))
    for spec in (FeatureSpec("target", 1, "count"), FeatureSpec("user", 0, "count"), FeatureSpec("user", 1, "SQL"), FeatureSpec("user", 1, "count", "future"), FeatureSpec("user", None, "max")):
        with pytest.raises(ValueError):
            spec.validate("taobao")
    with pytest.raises(ValueError):
        build_features(events.lazy(), queries.lazy(), metadata.lazy(), "unknown")
    with pytest.raises(ValueError):
        build_features(events.lazy(), queries.lazy(), metadata.lazy(), "taobao", [])
    with pytest.raises(ValueError):
        build_features(events.lazy(), queries.lazy(), metadata.lazy(), "taobao", [specs[0], specs[0]])


def test_movielens_ratings_and_unknown_genres() -> None:
    events, queries, _ = fixture()
    events = events.with_columns(pl.Series("event_type", ["rating_2.0", "rating_4.5", "rating_3.0", "rating_5.0", "rating_4.0", "rating_5.0"]))
    metadata = pl.DataFrame([(10, "movielens", "genres", '["Drama","Action"]', -(2**63), None, "static_assumed", "raw", None)], schema=ITEMS_SCHEMA, orient="row")
    specs = [FeatureSpec("item", None, "rating_mean"), FeatureSpec("category", 1, "count"), FeatureSpec("root_category", None, "recency")]
    actual = build_features(events.lazy(), queries.lazy(), metadata.lazy(), "movielens", specs).collect().sort(["query_group_id", "candidate_item_id"])
    assert actual["genre_count"].to_list() == [2, 0, 2]
    assert actual["category_id"][1] == -1
    assert actual["item__all__any__rating_mean"].to_list() == pytest.approx([3.25, 0., 11.5 / 3])
    reference = reference_features(events.lazy(), queries.lazy(), metadata.lazy(), "movielens", specs).sort(["query_group_id", "candidate_item_id"])
    import numpy as np
    assert np.allclose(actual.select(feature_columns(specs)).to_numpy(), reference.select(feature_columns(specs)).to_numpy())
