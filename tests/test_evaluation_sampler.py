"""Common target groups and dataset-specific repeat/oracle contracts."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config.loader import load_config
from src.generator.evaluation.sampler import CandidateCatalog, EvaluationOracle, _draw_pool, iter_evaluation_pools
from src.generator.splitting.temporal_splitter import select_targets


def facts(dataset: str) -> pl.DataFrame:
    """A prefix catalog, simultaneous positives and future held-out interactions."""
    low = {"retailrocket": "view", "taobao": "pv", "yoochoose": "click", "movielens": "rating_2.5"}[dataset]
    high = {"retailrocket": "transaction", "taobao": "buy", "yoochoose": "click", "movielens": "rating_5.0"}[dataset]
    low_label, high_label = (1, 1) if dataset == "yoochoose" else (0, 2)
    rows = [(i, 1000 + i, i, 10, low_label, low, "train", dataset, False) for i in range(1, 131)]
    rows += [(500, 1, 10, 11, low_label, low, "train", dataset, False),
             (1000, 1, 11, 20, high_label, high, "valid", dataset, False),
             (1001, 1, 12, 30, high_label, high, "valid", dataset, False),
             (1002, 1, 131, 30, high_label, high, "valid", dataset, False),
             (2000, 1, 13, 50, high_label, high, "test", dataset, False)]
    return pl.DataFrame(rows, schema=["event_id", "user_id", "item_id", "timestamp", "label", "event_type", "split", "dataset_source", "is_synthetic"], orient="row")


@pytest.mark.parametrize("dataset", ["retailrocket", "taobao", "yoochoose", "movielens"])
def test_fixed_pools_and_exclusions(dataset: str) -> None:
    settings = load_config().point_in_time_policy.evaluation
    frame = facts(dataset)
    policy = settings.repeat_purchase_policy[dataset]
    selected = select_targets(frame, dataset, "valid", policy).collect()
    assert selected["event_id"].to_list() == [1002]
    first = list(iter_evaluation_pools(frame, dataset, settings, "valid", chunk_rows=1))
    second = list(iter_evaluation_pools(frame.reverse(), dataset, settings, "valid", chunk_rows=47))
    pool = pl.concat([b.candidates for b in first])
    other = pl.concat([b.candidates for b in second])
    assert pool.equals(other)
    assert pool.height == 100 and pool["candidate_item_id"].n_unique() == 100
    assert pool["candidate_label"].sum() == 1
    assert pool.filter(pl.col("candidate_label") == 1)["candidate_item_id"].item() == 131
    assert (pool["catalog_snapshot_ts"] < pool["target_timestamp"]).all()
    negatives = set(pool.filter(pl.col("candidate_label") == 0)["candidate_item_id"])
    oracle = EvaluationOracle(frame, dataset)
    exclusions = oracle.exclusions(1, 131, 30, "valid", policy, False)
    assert negatives.isdisjoint(exclusions)
    assert {11, 12, 131} <= exclusions
    assert (10 in exclusions) == (dataset in ("movielens", "yoochoose"))
    assert 13 in oracle.exclusions(1, 131, 30, "valid", policy, True)
    with pytest.raises(ValueError):
        oracle.exclusions(1, 131, 30, "valid", "invalid", False)
    oracle.close()
    assert pl.concat([b.groups for b in first])["pool_hash"].to_list() == pl.concat([b.groups for b in second])["pool_hash"].to_list()


def test_unseen_target_and_separate_fallback() -> None:
    frame = facts("movielens")
    repeat = frame.tail(1).with_columns(pl.lit(1003).alias("event_id"), pl.lit(10).alias("item_id"), pl.lit(35).alias("timestamp"), pl.lit("valid").alias("split"))
    repeat = repeat.cast(frame.schema)
    frame = pl.concat([frame, repeat])
    assert select_targets(frame, "movielens", "valid", "unseen_only").collect()["event_id"].to_list() == [1002]
    fallback = repeat.with_columns(pl.lit(1004).alias("event_id"), pl.lit(2).alias("user_id"), pl.lit(1).alias("label"), pl.lit("rating_3.5").alias("event_type"))
    frame = pl.concat([frame, fallback.cast(frame.schema)])
    assert select_targets(frame, "movielens", "valid", "unseen_only", fallback=True).collect()["user_id"].to_list() == [2]
    settings = load_config().point_in_time_policy.evaluation
    batches = list(iter_evaluation_pools(frame, "movielens", settings, "valid", fallback=True))
    assert batches[0].groups["mode"].item() == "fallback"
    assert batches[0].groups["primary_target"].item() == "rating_3.0-3.5"


def test_short_catalog_and_oracle_database_guard(tmp_path: Path) -> None:
    settings = load_config().point_in_time_policy.evaluation
    frame = facts("taobao").filter(pl.col("user_id") < 1050)
    batches = list(iter_evaluation_pools(frame, "taobao", settings, "valid", chunk_rows=1))
    assert sum(b.candidates.height for b in batches) == 0
    assert sum(b.skipped.height for b in batches) == 1
    database = tmp_path / "oracle.sqlite"
    oracle = EvaluationOracle(frame, "taobao", database)
    oracle.close()
    with pytest.raises(ValueError, match="not empty"):
        EvaluationOracle(frame, "taobao", database)


@pytest.mark.parametrize("strategy", ["popularity_biased", "hard"])
def test_sampling_variants_and_metadata_availability(strategy: str) -> None:
    settings = load_config().point_in_time_policy.evaluation
    settings = settings.model_copy(update={"negative_sampling": settings.negative_sampling.model_copy(update={"strategy": strategy}), "exclude_future_positives": True})
    metadata = pl.DataFrame({"item_id": [900, 901, 902], "valid_from": [1000, -(2**63), 12]}).lazy()
    batches = list(iter_evaluation_pools(facts("taobao"), "taobao", settings, "valid", metadata_catalog=metadata, category=lambda item, ts: item if item == 131 else 0))
    pool = pl.concat([b.candidates for b in batches])
    assert pool.height == 100
    assert {900, 901, 13}.isdisjoint(set(pool["candidate_item_id"]))
    assert batches[0].groups["oracle_horizon_end"].item() == 50


def test_fenwick_catalog_and_reservoir() -> None:
    catalog = CandidateCatalog()
    with pytest.raises(ValueError, match="empty"):
        catalog.draw(np.random.default_rng(42))
    for item in range(1, 21):
        catalog.add(item, 10, item)
    catalog.add(1, 11, 10000)
    assert np.isclose(catalog._prefix(20), sum((catalog.counts[i] + 1) ** 0.75 for i in catalog.items))
    rng = np.random.default_rng(42)
    samples = [catalog.draw(rng, popularity=True) for _ in range(1000)]
    assert samples.count(1) > 700
    # Tiny predicate forces the bounded hard strategy into the global reservoir.
    pool = _draw_pool(catalog, {1}, 10, rng, "hard", lambda item: False)
    assert len(pool) == 10 and len({i for i, _ in pool}) == 10
    assert _draw_pool(catalog, set(range(1, 21)), 10, rng, "random") == []


def test_explicit_shared_targets_and_invalid_registry() -> None:
    frame = facts("taobao")
    settings = load_config().point_in_time_policy.evaluation
    query = pl.DataFrame({"event_id": [1000]}).lazy()
    batches = list(iter_evaluation_pools(frame, "taobao", settings, "valid", target_event_ids=query))
    assert batches[0].groups["eval_group_id"].item() == 1000
    with pytest.raises(ValueError, match="different split"):
        list(iter_evaluation_pools(frame, "taobao", settings, "test", target_event_ids=query))
    with pytest.raises(ValueError, match="positive"):
        list(iter_evaluation_pools(frame, "taobao", settings, "test", chunk_rows=0))
    duplicate = pl.concat([query, query])
    with pytest.raises(ValueError, match="registry"):
        list(iter_evaluation_pools(frame, "taobao", settings, "valid", target_event_ids=duplicate))
    movie = facts("movielens")
    repeat = movie.tail(1).with_columns(pl.lit(2001, dtype=pl.Int64).alias("event_id"), pl.lit(10, dtype=pl.Int64).alias("item_id"))
    movie = pl.concat([movie, repeat])
    with pytest.raises(ValueError, match="previously seen"):
        list(iter_evaluation_pools(movie, "movielens", settings, "test", target_event_ids=pl.DataFrame({"event_id": [2001]}).lazy()))
