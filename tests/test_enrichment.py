"""Prefix invariance, bounded sampling and isolation of future diagnostics."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config.loader import load_config
from src.generator.enrichment.personas import fit_personas
from src.generator.enrichment.negatives import PrefixState, annotate_future_conflicts, draw_candidates, iter_training_negatives, timestamp_groups
from src.generator.schemas import ITEMS_SCHEMA, NEGATIVES_SCHEMA, USERS_SCHEMA


def events() -> pl.DataFrame:
    """Known catalog before several user anchors and future observations."""
    rows = [(i, 100 + i, i, 10, 0, "pv", "train", "taobao", False, i % 3) for i in range(1, 16)]
    rows += [(100, 1, 1, 20, 2, "buy", "train", "taobao", False, 1),
             (101, 1, 2, 20, 2, "buy", "train", "taobao", False, 2),
             (102, 2, 3, 25, 2, "buy", "train", "taobao", False, 0),
             (103, 1, 12, 30, 2, "buy", "valid", "taobao", False, 0),
             (104, 3, 99, 1000, 2, "buy", "test", "taobao", False, 0)]
    return pl.DataFrame(rows, schema=["event_id", "user_id", "item_id", "timestamp", "label", "event_type", "split", "dataset_source", "is_synthetic", "category_id"], orient="row")


def primary(row: dict) -> bool:
    return row["event_type"] == "buy"


def settings():
    return load_config().point_in_time_policy.synthetic


def test_personas_are_prefix_fitted_and_typed() -> None:
    frame = events()
    config = settings().personas.model_copy(update={"k": 3, "min_prior_interactions": 1})
    full = fit_personas(frame, 26, config)
    prefix = fit_personas(frame.filter(pl.col("timestamp") < 26).reverse(), 26, config)
    assert full.max_input_ts == 25
    assert full.assignments == prefix.assignments
    assert np.allclose(full.centroids, prefix.centroids)
    assert full.metadata("taobao").schema == USERS_SCHEMA
    assert full.features_for(999, 27) == (0, "cold_start", 0.0, 0)
    with pytest.raises(ValueError, match="strictly before"):
        full.features_for(1, 26)
    items = fit_personas(frame, 26, config, entity="item_id")
    assert items.metadata("taobao").schema == ITEMS_SCHEMA
    empty = fit_personas(frame, 0, config)
    assert empty.max_input_ts is None and empty.assignments == {}
    assert empty.metadata("taobao").is_empty()
    no_categories = fit_personas(frame.drop("category_id"), 26, config)
    assert no_categories.assignments
    synthetic = frame.with_columns(pl.lit(True).alias("is_synthetic"))
    assert fit_personas(synthetic, 26, config).assignments == {}


def test_low_history_personas_are_boundary() -> None:
    snapshot = fit_personas(events(), 11, settings().personas)
    assert all(row[1] == "boundary" for row in snapshot.assignments.values())
    assert snapshot.features_for(101, 12)[0] >= 1


def test_training_negatives_ignore_future_and_same_second_positives() -> None:
    frame = events()
    config = settings().negatives
    full = pl.concat(list(iter_training_negatives(frame, config, primary, chunk_rows=2)))
    prefix = pl.concat(list(iter_training_negatives(frame.filter(pl.col("timestamp") < 30), config, primary, chunk_rows=3)))
    assert full.equals(prefix)
    assert full.schema == NEGATIVES_SCHEMA
    assert full["context_event_id"].unique().sort().to_list() == [100, 101, 102]
    assert full.filter((pl.col("user_id") == 1) & pl.col("item_id").is_in([1, 2])).is_empty()
    assert full["future_positive_conflict"].sum() == 0
    assert (full["catalog_snapshot_ts"] < full["context_timestamp"]).all()
    assert full.unique(subset=["context_event_id", "item_id"]).height == full.height
    assert 99 not in full["item_id"]


def test_future_annotation_changes_only_evaluator_columns(tmp_path: Path) -> None:
    frame = events()
    backend = PrefixState(tmp_path / "state.sqlite")
    negatives = pl.concat(list(iter_training_negatives(frame, settings().negatives, primary, state=backend)))
    row = negatives.row(0, named=True)
    backend.observe([(row["user_id"], row["item_id"], 2000)])
    annotated = annotate_future_conflicts(negatives, backend)
    assert annotated["future_positive_conflict"].sum() >= 1
    keep = [c for c in negatives.columns if c not in ("future_positive_conflict", "conflict_first_future_ts")]
    assert annotated.select(keep).equals(negatives.select(keep))
    assert backend.first_future(-1, -1, 0) is None
    with pytest.raises(ValueError, match="backward"):
        backend.observe([(1, 2, 1)])
    with pytest.raises(ValueError, match="collision"):
        backend.register_negative(negatives["negative_id"][0])
    backend.close()
    with pytest.raises(ValueError, match="not empty"):
        PrefixState(tmp_path / "state.sqlite")


def test_artifact_id_backend_preserves_samples_without_sql_id_registry(tmp_path: Path) -> None:
    frame = events()
    reference = pl.concat(list(iter_training_negatives(frame, settings().negatives, primary)))
    state = PrefixState(tmp_path / "artifact.sqlite", negative_id_validation="artifact_gate")
    try:
        output = pl.concat(list(iter_training_negatives(frame, settings().negatives, primary, state=state, chunk_rows=2)))
        assert output.equals(reference)
        assert state.connection.execute("SELECT name FROM sqlite_master WHERE name='negative_ids'").fetchall() == []
        # Artifact mode delegates collision rejection to the generator gate.
        state.register_negative(42)
        state.register_negative(42)
    finally:
        state.close()
    with pytest.raises(ValueError, match="validation backend"):
        PrefixState(tmp_path / "bad.sqlite", negative_id_validation="disabled")


def test_shortfall_and_disabled_sampling_terminate() -> None:
    frame = events().head(2)
    assert list(iter_training_negatives(frame, settings().negatives, primary)) == []
    assert list(iter_training_negatives(events(), settings().negatives.model_copy(update={"enabled": False}), primary)) == []
    state = PrefixState()
    with pytest.raises(ValueError, match="negative"):
        draw_candidates(state, 1, set(), -1, np.random.default_rng(42))
    assert draw_candidates(state, 1, set(), 3, np.random.default_rng(42)) == []
    state.observe([(2, i, 1) for i in range(1, 6)])
    assert len(draw_candidates(state, 1, {1, 2, 3}, 9, np.random.default_rng(42))) == 2
    assert draw_candidates(state, 2, set(), 3, np.random.default_rng(42)) == []
    assert len(draw_candidates(state, 1, set(), 3, np.random.default_rng(42), lambda i: i % 2 == 0)) == 2
    state.observe([(2, 5, 10000)])
    with pytest.raises(ValueError, match="future"):
        list(iter_training_negatives(events(), settings().negatives, primary, state=state))
    state.close()


def test_difficulty_sources_and_atomic_groups() -> None:
    frame = events()
    assert [len(g) for g in timestamp_groups(frame, 2)] == [15, 2, 1, 1, 1]
    assert list(timestamp_groups(frame.head(0))) == []
    for mixture, callbacks in [([0.0, 1.0, 0.0], {"cluster": lambda i, t: i % 3, "cluster_neighbors": lambda c, t: {(c + 1) % 3}}),
                               ([0.0, 0.0, 1.0], {"category": lambda i, t: i % 3}),
                               ([0.0, 0.0, 1.0], {"category": lambda i, t: i + 1000})]:
        config = settings().negatives.model_copy(update={"difficulty_mix": mixture})
        result = pl.concat(list(iter_training_negatives(frame, config, primary, **callbacks)))
        assert result.height > 0
        assert set(result["negative_type"].to_list()) <= {"global", "cluster_mixed", "hard_boundary"}


def test_inverse_category_fallback_preserves_exact_rng_and_prefix(tmp_path: Path) -> None:
    from src.generator.metadata import MetadataIndex
    frame = events()
    # A rare category, a changed/tombstoned membership and a future catalog item.
    metadata = pl.DataFrame([(1, "taobao", "category_id", "7", 1, None, "observed_asof", "raw", 1),
        (2, "taobao", "category_id", "7", 1, None, "observed_asof", "raw", 1),
        (2, "taobao", "category_id", "-1", 20, None, "observed_asof", "raw", 20),
        (3, "taobao", "category_id", "7", 25, None, "observed_asof", "raw", 25),
        (12, "taobao", "category_id", "7", 1, None, "observed_asof", "raw", 1),
        (99, "taobao", "category_id", "7", 1, None, "observed_asof", "raw", 1)], schema=ITEMS_SCHEMA, orient="row")
    index = MetadataIndex(metadata.lazy(), tmp_path / "categories.sqlite")
    assert list(index.items_in_category(7, 20)) == [1, 2, 12, 99]
    assert list(index.items_in_category(7, 25)) == [1, 12, 99]
    config = settings().negatives.model_copy(update={"difficulty_mix": [0., 0., 1.]})
    slow = pl.concat(list(iter_training_negatives(frame, config, primary, category=index.category)))
    fast = pl.concat(list(iter_training_negatives(frame, config, primary, category=index.category, category_candidates=index.items_in_category)))
    assert fast.equals(slow)
    prefix = pl.concat(list(iter_training_negatives(frame.filter(pl.col("timestamp") < 30), config, primary,
        category=index.category, category_candidates=index.items_in_category)))
    assert fast.equals(prefix) and 99 not in fast["item_id"]
    state = PrefixState()
    state.observe([(2, i, 1) for i in range(1, 2001)])
    calls = [0]
    def rare(item):
        calls[0] += 1
        return item in (7, 1999)
    normal = draw_candidates(state, 1, set(), 5, np.random.default_rng(42), rare)
    slow_calls = calls[0]
    calls[0] = 0
    inverse = draw_candidates(state, 1, set(), 5, np.random.default_rng(42), rare, lambda: [7, 1999, 9999])
    assert normal == inverse == [7, 1999] and calls[0] < slow_calls / 2
    state.close()
    index.close()


def test_cluster_inverse_keeps_centroid_spaces_and_exact_sampling_separate(tmp_path: Path):
    from src.generator.metadata import MetadataIndex
    rows = [(item, "taobao", "item_cluster", str(1000 + item), snapshot, None, "derived_pit", "personas", snapshot)
            for snapshot in (5, 22) for item in range(1, 16) if snapshot != 22 or item != 12]
    rows += [(99, "taobao", "item_cluster", "1001", 1000, None, "derived_pit", "personas", 1000)]
    index = MetadataIndex(pl.DataFrame(rows, schema=ITEMS_SCHEMA, orient="row").lazy(), tmp_path / "clusters.sqlite")
    assert list(index.items_in_clusters({1012}, 22, 5)) == [12]
    assert list(index.items_in_clusters({1012}, 23, 22)) == []  # Old inactive membership cannot enter a new space.
    assert list(index.items_in_clusters({1001}, 23, 22)) == [1]  # Future99 remains unavailable.
    assert list(index.items_in_clusters(set(), 23, 22)) == []
    with pytest.raises(ValueError, match="strictly before"):
        list(index.items_in_clusters({1001}, 22, 22))
    def space(timestamp):
        return 5 if timestamp <= 22 else 22
    def cluster(item, timestamp):
        return index.get(item, timestamp, "item_cluster") if index.snapshot_timestamp(item, timestamp, "item_cluster") == space(timestamp) else None
    def candidates(values, timestamp):
        return index.items_in_clusters(values, timestamp, space(timestamp))
    for mixture in ([0., 0., 1.], [0., 1., 0.]):
        config = settings().negatives.model_copy(update={"difficulty_mix": mixture})
        callbacks = {"cluster": cluster, "cluster_neighbors": lambda value, timestamp: {999999}}
        slow = pl.concat(list(iter_training_negatives(events(), config, primary, **callbacks)))
        fast = pl.concat(list(iter_training_negatives(events(), config, primary, cluster_candidates=candidates, **callbacks)))
        assert fast.equals(slow) and 99 not in fast["item_id"]
    index.close()
