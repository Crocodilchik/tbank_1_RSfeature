"""Native lazy point-in-time aggregates; query markers never become observations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import polars as pl

from src.generator.metadata import asof_item_field
from src.generator.splitting.temporal_splitter import primary_expression

QUERY_KEYS = ["query_group_id", "candidate_item_id"]
ENTITY_KEYS = {"user": ["user_id"], "item": ["candidate_item_id"],
    "user_item": ["user_id", "candidate_item_id"], "category": ["category_id"],
    "user_category": ["user_id", "category_id"], "root_category": ["root_category_id"]}
EVENT_TYPES = {"retailrocket": ("view", "addtocart", "transaction"), "taobao": ("pv", "cart", "fav", "buy"),
    "yoochoose": ("click", "buy"), "movielens": tuple(f"rating_{i / 2}" for i in range(1, 11))}


@dataclass(frozen=True)
class FeatureSpec:
    """A typed, cutoff-bound aggregation over real historical events."""

    entity: str
    window_days: int | None
    aggregation: str
    event_type: str | None = None

    @property
    def name(self) -> str:
        """Return a stable feature name safe for column projection."""
        window = "all" if self.window_days is None else f"{self.window_days}d"
        kind = "any" if self.event_type is None else self.event_type.replace(".", "_")
        return f"{self.entity}__{window}__{kind}__{self.aggregation}"

    def validate(self, dataset: str) -> None:
        """Reject unchecked fields, unavailable events, and unbounded window expressions."""
        if self.entity not in ENTITY_KEYS or self.aggregation not in {"count", "sum", "mean", "min", "max", "positive_rate", "recency", "target_encoding", "rating_mean"}:
            raise ValueError(f"unsupported feature primitive: {self}")
        if self.window_days is not None and (not isinstance(self.window_days, int) or not 1 <= self.window_days <= 365):
            raise ValueError("feature window must be a positive bounded number of days")
        if self.event_type is not None and self.event_type not in EVENT_TYPES[dataset]:
            raise ValueError("feature event filter is not available in this dataset")
        if self.window_days is None and self.aggregation in {"min", "max"}:
            raise ValueError("expanding min/max is not an approved primitive")


def manual_specs(dataset: str, windows_days: Sequence[int] = (1, 7, 14, 30)) -> list[FeatureSpec]:
    """Return the preregistered manual count/conversion/affinity/recency feature set."""
    specs = [FeatureSpec(entity, day, aggregation) for entity in ENTITY_KEYS for day in windows_days
             for aggregation in ("count", "positive_rate")]
    specs += [FeatureSpec(entity, None, aggregation) for entity in ENTITY_KEYS
              for aggregation in ("count", "recency", "target_encoding")]
    specs += [FeatureSpec(entity, day, "count", kind) for entity in ("user", "item")
              for day in windows_days for kind in EVENT_TYPES[dataset]]
    if dataset == "movielens":
        specs += [FeatureSpec(entity, None, "rating_mean") for entity in ("user", "item", "user_item")]
    return specs


def attach_metadata(queries: pl.LazyFrame, metadata: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
    """Attach only metadata known before each query, with explicit numeric unknowns."""
    frame = queries
    for field in ("category_id", "brand_id", "root_category_id", "release_year"):
        frame = asof_item_field(frame, metadata, field, pl.Int64, item_column="candidate_item_id")
        frame = frame.with_columns(pl.col(field).fill_null(-1)).drop(f"asof_{field}_ts")
    if dataset == "movielens":
        frame = asof_item_field(frame, metadata, "genres", pl.List(pl.String), item_column="candidate_item_id")
        frame = frame.with_columns(pl.col("genres").list.len().fill_null(0).cast(pl.Int64).alias("genre_count"),
            pl.when(pl.col("genres").list.first().is_not_null()).then(pl.col("genres").list.first().hash(seed=42).mod(2**31 - 1).cast(pl.Int64)).otherwise(-1).alias("category_id")).drop("genres", "asof_genres_ts")
    else:
        frame = frame.with_columns(pl.lit(0, dtype=pl.Int64).alias("genre_count"))
    frame = asof_item_field(frame, metadata, "available", pl.Boolean, item_column="candidate_item_id")
    return frame.with_columns(pl.col("available").cast(pl.Int8).fill_null(-1)).drop("asof_available_ts")


def _global_prior(events: pl.LazyFrame, queries: pl.LazyFrame, dataset: str, empty_prior: float) -> pl.LazyFrame:
    real = events.select("timestamp", pl.lit(1, dtype=pl.Int64).alias("n"), primary_expression(dataset).cast(pl.Int64).alias("p"))
    markers = queries.select("timestamp").unique().with_columns(pl.lit(0, dtype=pl.Int64).alias("n"), pl.lit(0, dtype=pl.Int64).alias("p"))
    grouped = pl.concat([real, markers]).group_by("timestamp").agg(pl.col("n").sum(), pl.col("p").sum()).sort("timestamp")
    grouped = grouped.with_columns((pl.col("n").cum_sum() - pl.col("n")).alias("prior_n"), (pl.col("p").cum_sum() - pl.col("p")).alias("prior_p"))
    return grouped.select("timestamp", pl.when(pl.col("prior_n") > 0).then(pl.col("prior_p") / pl.col("prior_n")).otherwise(empty_prior).alias("_prior"))


def build_features(events: pl.LazyFrame, queries: pl.LazyFrame, metadata: pl.LazyFrame,
                   dataset: str, specs: Sequence[FeatureSpec] | None = None,
                   *, smoothing_alpha: float = 10., min_history: int = 5,
                   empty_prior: float = .5,
                   prepared_inputs: tuple[pl.LazyFrame, pl.LazyFrame] | None = None) -> pl.LazyFrame:
    """Build causal features with atomic seconds, prefix TE and native rolling windows.

    The returned frame retains scorer keys/target for training, but the booster
    receives only the named numeric features and explicitly listed metadata.
    No future-conflict or target-dependent row fields enter an aggregation.
    """
    if dataset not in EVENT_TYPES or smoothing_alpha <= 0 or min_history < 1:
        raise ValueError("invalid feature dataset or smoothing settings")
    specs = list(manual_specs(dataset) if specs is None else specs)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("features must be nonempty and uniquely named")
    for spec in specs:
        spec.validate(dataset)
    events = events.filter(~pl.col("is_synthetic"))
    real, prepared = prepare_inputs(events, queries, metadata, dataset, empty_prior) if prepared_inputs is None else prepared_inputs
    result = prepared
    grouped_specs: dict[tuple[str, int | None, str | None], list[FeatureSpec]] = defaultdict(list)
    for spec in specs:
        grouped_specs[spec.entity, spec.window_days, spec.event_type].append(spec)
    for (entity, days, kind), group in grouped_specs.items():
        keys = ENTITY_KEYS[entity]
        observations = real if kind is None else real.filter(pl.col("event_type") == kind)
        for key in keys:
            if key in {"category_id", "root_category_id"}:
                observations = observations.filter(pl.col(key) >= 0)
        past = observations.select(*keys, "timestamp", pl.lit(1, dtype=pl.Int64).alias("_n"),
            pl.col("_positive").alias("_p"), pl.col("_label").alias("_sum"), pl.col("_label").alias("_min"),
            pl.col("_label").alias("_max"), pl.col("_rating").alias("_r"), pl.col("timestamp").alias("_last"))
        markers = prepared.select(*keys, "timestamp").unique().with_columns(
            *[pl.lit(0, dtype=pl.Int64).alias(c) for c in ("_n", "_p")],
            *[pl.lit(0., dtype=pl.Float64).alias(c) for c in ("_sum", "_r")],
            *[pl.lit(None, dtype=pl.Float64).alias(c) for c in ("_min", "_max")], pl.lit(None, dtype=pl.Int64).alias("_last"))
        stream = pl.concat([past, markers]).group_by(*keys, "timestamp").agg(
            *[pl.col(c).sum() for c in ("_n", "_p", "_sum", "_r")], pl.col("_min").min(), pl.col("_max").max(), pl.col("_last").max()).sort([*keys, "timestamp"])
        if days is None:
            aggregate = stream.with_columns(*[(pl.col(c).cum_sum() - pl.col(c)).over(keys).alias(c) for c in ("_n", "_p", "_sum", "_r")],
                pl.col("_last").forward_fill().shift(1).over(keys).alias("_last"))
        else:
            aggregate = stream.rolling("timestamp", period=f"{days * 86400}i", group_by=keys, closed="left").agg(
                *[pl.col(c).sum() for c in ("_n", "_p", "_sum", "_r")], pl.col("_min").min(), pl.col("_max").max(), pl.col("_last").max())
        aggregate = aggregate.join(prepared.select(*keys, "timestamp", "_prior").unique(), on=[*keys, "timestamp"])
        expressions = []
        for spec in group:
            primitives = {"count": pl.col("_n"), "sum": pl.col("_sum"), "mean": pl.col("_sum") / pl.col("_n").clip(lower_bound=1),
                "min": pl.col("_min").fill_null(0.), "max": pl.col("_max").fill_null(0.),
                "positive_rate": pl.col("_p") / pl.col("_n").clip(lower_bound=1),
                "rating_mean": pl.col("_r") / pl.col("_n").clip(lower_bound=1),
                "recency": (pl.col("timestamp") - pl.col("_last")).fill_null(30 * 86400).clip(0, 365 * 86400),
                "target_encoding": pl.when(pl.col("_n") >= min_history).then((pl.col("_p") + smoothing_alpha * pl.col("_prior")) / (pl.col("_n") + smoothing_alpha)).otherwise(pl.col("_prior"))}
            expressions.append(primitives[spec.aggregation].cast(pl.Float32).alias(spec.name))
        result = result.join(aggregate.select(*keys, "timestamp", *expressions), on=[*keys, "timestamp"], how="left")
    return result.drop("_prior")


def prepare_inputs(events: pl.LazyFrame, queries: pl.LazyFrame, metadata: pl.LazyFrame,
                   dataset: str, empty_prior: float = .5) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Prepare reusable PIT metadata and prefix priors before feature block execution."""
    events = events.filter(~pl.col("is_synthetic"))
    prepared = attach_metadata(queries, metadata, dataset).join(_global_prior(events, queries, dataset, empty_prior), on="timestamp")
    real = attach_metadata(events.rename({"item_id": "candidate_item_id"}), metadata, dataset).with_columns(
        primary_expression(dataset).cast(pl.Int64).alias("_positive"), pl.col("label").cast(pl.Float64).alias("_label"),
        pl.col("event_type").str.strip_prefix("rating_").cast(pl.Float64, strict=False).fill_null(0.).alias("_rating"))
    return real, prepared


def feature_columns(specs: Sequence[FeatureSpec]) -> list[str]:
    """Return the explicit booster whitelist, excluding query IDs, times and targets."""
    return [spec.name for spec in specs] + ["category_id", "brand_id", "root_category_id", "release_year", "genre_count", "available"]


def reference_features(events: pl.LazyFrame, queries: pl.LazyFrame, metadata: pl.LazyFrame,
                       dataset: str, specs: Sequence[FeatureSpec], *, smoothing_alpha: float = 10.,
                       min_history: int = 5, empty_prior: float = .5) -> pl.DataFrame:
    """Independent range-join reference for a bounded sample of real query targets.

    It uses direct filtered aggregates, no rolling query markers. Unknown
    category/root IDs do not create an artificial shared behavioral entity.
    """
    real, prepared = prepare_inputs(events, queries, metadata, dataset, empty_prior)
    times = events.filter(~pl.col("is_synthetic")).group_by("timestamp").agg(
        pl.len().cast(pl.Int64).alias("n"), primary_expression(dataset).sum().cast(pl.Int64).alias("p")).sort("timestamp").with_columns(
        pl.col("n").cum_sum(), pl.col("p").cum_sum()).rename({"timestamp": "prefix_ts"})
    prior = prepared.select("timestamp").unique().sort("timestamp").join_asof(times, left_on="timestamp", right_on="prefix_ts", allow_exact_matches=False).select(
        "timestamp", pl.when(pl.col("n") > 0).then(pl.col("p") / pl.col("n")).otherwise(empty_prior).alias("_reference_prior"))
    base = prepared.drop("_prior").join(prior, on="timestamp")
    result = base
    by_entity: dict[str, list[FeatureSpec]] = defaultdict(list)
    for spec in specs:
        by_entity[spec.entity].append(spec)
    for entity, group in by_entity.items():
        keys = ENTITY_KEYS[entity]
        source = real.select(*keys, pl.col("timestamp").alias("_event_time"), "_positive", "_label", "_rating", "event_type")
        for key in keys:
            if key in {"category_id", "root_category_id"}:
                source = source.filter(pl.col(key) >= 0)
        query_columns = list(dict.fromkeys([*QUERY_KEYS, *keys, "timestamp", "_reference_prior"]))
        joined = base.select(query_columns).join(source, on=keys, how="inner").filter(pl.col("_event_time") < pl.col("timestamp"))
        expressions = []
        defaults = {}
        for spec in group:
            condition = pl.col("_event_time").is_not_null()
            if spec.window_days is not None:
                condition &= pl.col("_event_time") >= pl.col("timestamp") - spec.window_days * 86400
            if spec.event_type is not None:
                condition &= pl.col("event_type") == spec.event_type
            n = condition.cast(pl.Int64).sum()
            p = pl.col("_positive").filter(condition).sum()
            label = pl.col("_label").filter(condition)
            prior_value = pl.col("_reference_prior").first()
            values = {"count": n, "sum": label.sum(), "mean": label.sum() / n.clip(lower_bound=1),
                "min": label.min().fill_null(0.), "max": label.max().fill_null(0.), "positive_rate": p / n.clip(lower_bound=1),
                "rating_mean": pl.col("_rating").filter(condition).sum() / n.clip(lower_bound=1),
                "recency": (pl.col("timestamp").first() - pl.col("_event_time").filter(condition).max()).fill_null(30 * 86400).clip(0, 365 * 86400),
                "target_encoding": pl.when(n >= min_history).then((p + smoothing_alpha * prior_value) / (n + smoothing_alpha)).otherwise(prior_value)}
            expressions.append(values[spec.aggregation].cast(pl.Float32).alias(spec.name))
            defaults[spec.name] = pl.col("_reference_prior") if spec.aggregation == "target_encoding" else pl.lit(30 * 86400 if spec.aggregation == "recency" else 0.)
        aggregate = joined.group_by(QUERY_KEYS).agg(expressions)
        result = result.join(aggregate, on=QUERY_KEYS, how="left").with_columns(
            pl.col(name).fill_null(value).cast(pl.Float32) for name, value in defaults.items())
    return result.drop("_reference_prior").collect(engine="streaming")
