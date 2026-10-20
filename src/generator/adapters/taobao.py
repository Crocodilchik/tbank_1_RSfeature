"""Taobao second-level events and explicit category availability modes."""

import polars as pl

from src.generator.adapters.base import BaseAdapter, DataValidationError, item_metadata


class TaobaoAdapter(BaseAdapter):
    """Filter timestamp anomalies before any behavioral derivation."""

    def load_raw(self) -> pl.LazyFrame:
        """Read the headerless five-column CSV lazily."""
        return self.scan("events", {"user_id": pl.Int64, "item_id": pl.Int64, "category_id": pl.Int64,
                                    "behavior": pl.String, "timestamp": pl.Int64})

    def observations_lazy(self) -> pl.LazyFrame:
        """Keep the approved UTC block and map raw behavior facts."""
        frame = self.load_raw().filter(pl.col("timestamp").is_between(1511481600, 1512345599)).unique(
            subset=["user_id", "item_id", "category_id", "behavior", "timestamp"], keep="first", maintain_order=True)
        return self.event_identity(frame.rename({"behavior": "event_type"}).with_columns(
            pl.col("event_type").replace_strict({"pv": 0, "cart": 1, "fav": 1, "buy": 2}, return_dtype=pl.Int8).alias("label"),
            pl.lit(None, dtype=pl.Int64).alias("timestamp_ms")))

    def validate(self) -> list[str]:
        """Check raw fields without constructing a 100M-row deduplication state."""
        try:
            raw = self.load_raw()
            missing = pl.any_horizontal(pl.col(c).is_null() for c in ("user_id", "item_id", "category_id", "behavior", "timestamp"))
            in_window = pl.col("timestamp").is_between(1511481600, 1512345599)
            invalid = (pl.col("user_id") < 0) | (pl.col("item_id") < 0) | (pl.col("category_id") < 0) | ~pl.col("behavior").is_in(["pv", "cart", "fav", "buy"])
            bad = raw.select(missing.any().alias("missing"), (in_window & invalid).any().alias("invalid")).collect(engine="streaming").row(0)
            return ["null required raw field in taobao"] if bad[0] else (["invalid required raw field in taobao"] if bad[1] else [])
        except pl.exceptions.PolarsError as exc:
            return [f"taobao: {exc}"]

    def metadata_lazy(self) -> pl.LazyFrame:
        """Validate static category consistency or keep observed category snapshots."""
        frame = self.restrict_metadata(self.load_raw().filter(pl.col("timestamp").is_between(1511481600, 1512345599)))
        if self.settings.category_id_availability == "static_assumed":
            grouped = frame.group_by("item_id").agg(pl.col("category_id").min().alias("category_id"), pl.col("category_id").max().alias("maximum"))
            conflict = grouped.filter(pl.col("category_id") != pl.col("maximum")).limit(1).collect(engine="streaming")
            if conflict.height:
                raise DataValidationError("Taobao item/category is not static; explicitly select observed_asof")
            return item_metadata(grouped, "taobao", "category_id", pl.col("category_id"))
        observed = frame.group_by(["item_id", "timestamp"]).agg(pl.col("category_id").min().alias("minimum"), pl.col("category_id").max().alias("maximum")).with_columns(
            pl.when(pl.col("minimum") == pl.col("maximum")).then(pl.col("minimum")).otherwise(-1).alias("category_id"))
        changes = observed.sort(["item_id", "timestamp"]).with_columns(pl.col("category_id").shift(1).over("item_id").alias("previous")).filter(
            ~pl.col("category_id").eq_missing(pl.col("previous")))
        conflicts = observed.filter(pl.col("minimum") != pl.col("maximum"))
        return pl.concat([item_metadata(changes, "taobao", "category_id", pl.col("category_id"), pl.col("timestamp"), "observed_snapshot"),
            item_metadata(conflicts, "taobao", "category_conflict", pl.lit(True), pl.col("timestamp"), "observed_snapshot")])
