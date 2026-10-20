"""Half-open UTC splits without splitting simultaneous timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.config.loader import SplitSettings


@dataclass(frozen=True)
class SplitBoundaries:
    """Resolved timestamps shared by every downstream model."""

    train_end: int
    valid_end: int
    window_start: int | None = None
    window_end: int | None = None

    def __post_init__(self) -> None:
        if self.train_end >= self.valid_end:
            raise ValueError("train_end must be strictly before valid_end")
        if self.window_start is not None and self.window_start >= self.train_end:
            raise ValueError("window_start must be before train_end")
        if self.window_end is not None and self.window_end <= self.valid_end:
            raise ValueError("window_end must be after valid_end")


def resolve_boundaries(events: pl.DataFrame | pl.LazyFrame, settings: SplitSettings) -> SplitBoundaries:
    """Resolve lower quantiles on real clean facts, or use exact configured UTC dates."""
    frame = events.lazy() if isinstance(events, pl.DataFrame) else events
    frame = frame.filter(~pl.col("is_synthetic"))
    if settings.method == "fixed_dates":
        return SplitBoundaries(int(settings.train_end.timestamp()), int(settings.valid_end.timestamp()),
                               int(settings.window_start.timestamp()), int(settings.window_end.timestamp()))
    row = frame.select(pl.col("timestamp").quantile(settings.train_quantile, interpolation="lower").alias("t1"),
                       pl.col("timestamp").quantile(settings.valid_quantile, interpolation="lower").alias("t2")).collect(engine="streaming").row(0)
    if any(x is None for x in row):
        raise ValueError("cannot split an empty real event stream")
    return SplitBoundaries(int(row[0]), int(row[1]))


def assign_splits(events: pl.DataFrame | pl.LazyFrame, boundaries: SplitBoundaries) -> pl.LazyFrame:
    """Assign every real event exactly once, rejecting out-of-window records."""
    frame = events.lazy() if isinstance(events, pl.DataFrame) else events
    outside = pl.col("timestamp").is_null() | pl.col("is_synthetic")
    if boundaries.window_start is not None:
        outside |= pl.col("timestamp") < boundaries.window_start
    if boundaries.window_end is not None:
        outside |= pl.col("timestamp") >= boundaries.window_end
    if frame.filter(outside).limit(1).collect(engine="streaming").height:
        raise ValueError("split input contains synthetic/null/out-of-window facts")
    result = frame.with_columns(pl.when(pl.col("timestamp") < boundaries.train_end).then(pl.lit("train"))
                               .when(pl.col("timestamp") < boundaries.valid_end).then(pl.lit("valid"))
                               .otherwise(pl.lit("test")).alias("split"))
    counts = result.group_by("split").len().collect(engine="streaming")
    if set(counts["split"]) != {"train", "valid", "test"}:
        raise ValueError("temporal split produced an empty train/valid/test partition")
    return result


def primary_expression(dataset: str, fallback: bool = False) -> pl.Expr:
    """Return the exact task predicate; click does not include purchase events."""
    if dataset == "movielens":
        return pl.col("label") == (1 if fallback else 2)
    targets = {"retailrocket": "addtocart" if fallback else "transaction",
               "taobao": "cart" if fallback else "buy", "yoochoose": "click"}
    if dataset not in targets:
        raise ValueError(f"unknown dataset: {dataset}")
    return pl.col("event_type") == targets[dataset]


def is_primary(row: dict) -> bool:
    """Equivalent row predicate for streaming training code."""
    dataset = row["dataset_source"]
    return row["label"] == 2 if dataset == "movielens" else row["event_type"] == {"retailrocket": "transaction", "taobao": "buy", "yoochoose": "click"}[dataset]


def select_targets(events: pl.DataFrame | pl.LazyFrame, dataset: str, split: str,
                   policy: str, fallback: bool = False) -> pl.LazyFrame:
    """Select one common last eligible target per user/session in the split."""
    if split not in ("valid", "test"):
        raise ValueError("evaluation targets must belong to valid or test")
    frame = events.lazy() if isinstance(events, pl.DataFrame) else events
    real = frame.filter(~pl.col("is_synthetic"))
    candidates = real.filter((pl.col("split") == split) & primary_expression(dataset, fallback))
    if policy == "unseen_only":
        pairs = candidates.select("user_id", "item_id").unique()
        first = real.join(pairs, on=["user_id", "item_id"], how="semi").group_by(["user_id", "item_id"]).agg(pl.col("timestamp").min().alias("first_ts"))
        candidates = candidates.join(first, on=["user_id", "item_id"]).filter(pl.col("timestamp") == pl.col("first_ts")).drop("first_ts")
    if fallback:
        users_with_primary = select_targets(real, dataset, split, policy, fallback=False).select("user_id").unique()
        candidates = candidates.join(users_with_primary, on="user_id", how="anti")
    return candidates.sort(["timestamp", "event_id"]).unique(subset="user_id", keep="last", maintain_order=True)
