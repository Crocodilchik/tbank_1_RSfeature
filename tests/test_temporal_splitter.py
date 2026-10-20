"""Fixed UTC boundaries, lower quantiles and simultaneous timestamp safety."""

import polars as pl
import pytest

from src.config.loader import load_config
from src.generator.splitting.temporal_splitter import SplitBoundaries, assign_splits, is_primary, primary_expression, resolve_boundaries, select_targets


def test_lower_quantile_preserves_ties() -> None:
    frame = pl.DataFrame({"event_id": list(range(10)), "timestamp": [1, 2, 2, 3, 4, 5, 6, 7, 7, 8], "is_synthetic": [False] * 10})
    config = load_config().splits.retailrocket
    with pytest.raises(ValueError, match="strictly before"):
        resolve_boundaries(frame, config)
    assigned = assign_splits(frame, SplitBoundaries(3, 7)).collect()
    assert assigned.filter(pl.col("timestamp") == 2)["split"].unique().to_list() == ["train"]
    assert assigned.filter(pl.col("timestamp") == 7)["split"].unique().to_list() == ["test"]


def test_quantile_tie_failure_and_valid_assignment() -> None:
    config = load_config().splits.retailrocket
    tied = pl.DataFrame({"timestamp": [1] * 10, "is_synthetic": [False] * 10})
    with pytest.raises(ValueError, match="strictly before"):
        resolve_boundaries(tied, config)
    with pytest.raises(ValueError, match="empty"):
        resolve_boundaries(tied.head(0), config)
    frame = pl.DataFrame({"event_id": list(range(20)), "timestamp": list(range(20)), "is_synthetic": [False] * 20})
    boundaries = resolve_boundaries(frame, config)
    assert (boundaries.train_end, boundaries.valid_end) == (15, 17)
    assigned = assign_splits(frame, boundaries).collect()
    assert assigned.filter(pl.col("split") == "train")["timestamp"].max() == 14
    assert assigned.filter(pl.col("split") == "valid")["timestamp"].min() == 15
    assert assigned.filter(pl.col("split") == "test")["timestamp"].min() == 17
    assert assigned.height == frame.height


def test_exact_taobao_boundaries() -> None:
    settings = load_config().splits.taobao
    bounds = resolve_boundaries(pl.DataFrame(), settings)
    assert (bounds.window_start, bounds.train_end, bounds.valid_end, bounds.window_end) == (1511481600, 1512172800, 1512259200, 1512345600)
    frame = pl.DataFrame({"timestamp": [bounds.train_end - 1, bounds.train_end, bounds.valid_end], "is_synthetic": [False] * 3})
    assert assign_splits(frame, bounds).collect()["split"].to_list() == ["train", "valid", "test"]
    for invalid in (bounds.window_start - 1, bounds.window_end):
        with pytest.raises(ValueError, match="out-of-window"):
            assign_splits(frame.with_columns(pl.when(pl.col("timestamp") == bounds.valid_end).then(invalid).otherwise(pl.col("timestamp")).alias("timestamp")), bounds)
    with pytest.raises(ValueError, match="empty"):
        assign_splits(frame.head(1), bounds)
    with pytest.raises(ValueError, match="synthetic"):
        assign_splits(frame.with_columns(pl.lit(True).alias("is_synthetic")), bounds)


@pytest.mark.parametrize("args", [(20, 10, None, None), (10, 20, 10, None), (10, 20, None, 20)])
def test_reject_reversed_boundaries(args: tuple) -> None:
    with pytest.raises(ValueError):
        SplitBoundaries(*args)


def test_exact_primary_and_invalid_targets() -> None:
    row = dict(dataset_source="yoochoose", event_type="buy", label=2)
    assert not is_primary(row)
    assert pl.DataFrame([row]).filter(primary_expression("yoochoose")).is_empty()
    with pytest.raises(ValueError):
        primary_expression("unknown")
    with pytest.raises(ValueError):
        select_targets(pl.DataFrame(), "taobao", "train", "allow_previous")
