"""Yoochoose sessions, mixed click context and isolated official test source."""

from collections import Counter, defaultdict
from typing import Iterator

import polars as pl

from src.generator.adapters.base import BaseAdapter, item_metadata
from src.generator.schemas import ITEMS_SCHEMA


class YoochooseAdapter(BaseAdapter):
    """Treat session IDs as sessions, preserving source-specific namespaces."""

    def clicks(self, official: bool = False, *, for_metadata: bool = False) -> pl.LazyFrame:
        """Read and deduplicate clicks without using target context as features."""
        role = "official_test" if official else "clicks"
        raw = self.scan(role, {"session_id": pl.Int64, "timestamp": pl.String, "item_id": pl.Int64, "category": pl.String}, file_code=3 if official else 1)
        if for_metadata:
            raw = self.restrict_metadata(raw)
        # One fact per click key, unknown context if source rows disagree.
        return raw.group_by(["session_id", "timestamp", "item_id"], maintain_order=True).agg(
            pl.col("source_row_id").min(), pl.col("file_code").first(),
            pl.when(pl.col("category").n_unique() == 1).then(pl.col("category").first()).otherwise(pl.lit("0")).alias("category"),
        ).with_columns(pl.lit("click").alias("event_type"), pl.lit(1, dtype=pl.Int8).alias("label"),
                       pl.lit(None, dtype=pl.Int64).alias("price"), pl.lit(None, dtype=pl.Int64).alias("quantity"))

    def load_raw(self) -> pl.LazyFrame:
        """Combine primary clicks and valid purchases, never official test training."""
        buys = self.scan("buys", {"session_id": pl.Int64, "timestamp": pl.String, "item_id": pl.Int64, "price": pl.Int64, "quantity": pl.Int64}, file_code=2)
        buys = buys.filter((pl.col("quantity") > 0) & (pl.col("price") >= 0)).unique(
            subset=["session_id", "timestamp", "item_id", "price", "quantity"], keep="first", maintain_order=True).with_columns(
                pl.lit("0").alias("category"), pl.lit("buy").alias("event_type"), pl.lit(2, dtype=pl.Int8).alias("label"))
        return pl.concat([self.clicks(), buys], how="diagonal")

    def _normalize_source(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        frame = frame.rename({"session_id": "user_id"}).with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="UTC").dt.epoch("ms").alias("timestamp_ms"))
        return self.event_identity(frame.with_columns((pl.col("timestamp_ms") // 1000).alias("timestamp")))

    def observations_lazy(self) -> pl.LazyFrame:
        """Normalize primary session facts with conservative second-level ordering."""
        return self._normalize_source(self.load_raw())

    def validate(self) -> list[str]:
        """Check raw fields/dates directly before quantity filtering or click deduplication."""
        schemas = {"clicks": {"session_id": pl.Int64, "timestamp": pl.String, "item_id": pl.Int64, "category": pl.String},
            "buys": {"session_id": pl.Int64, "timestamp": pl.String, "item_id": pl.Int64, "price": pl.Int64, "quantity": pl.Int64}}
        try:
            for role, schema in schemas.items():
                raw = self.scan(role, schema, file_code=1 if role == "clicks" else 2)
                required = ("session_id", "timestamp", "item_id") if role == "clicks" else tuple(schema)
                invalid = pl.any_horizontal(pl.col(c).is_null() for c in required) | (pl.col("session_id") < 0) | (pl.col("item_id") < 0)
                invalid |= pl.col("timestamp").str.to_datetime(time_zone="UTC").dt.epoch("ms").is_null()
                if role == "buys":
                    invalid |= (pl.col("price") < 0) | (pl.col("quantity") < 0)
                if raw.select(invalid.any()).collect(engine="streaming").item():
                    return [f"invalid required Yoochoose {role} field/price/quantity"]
            return []
        except pl.exceptions.PolarsError as exc:
            return [f"invalid Yoochoose source: {exc}"]

    def official_observations_lazy(self) -> pl.LazyFrame:
        """Read separate official click facts only for supplementary evaluation."""
        if not self.settings.include_official_test:
            raise ValueError("official test is disabled by configuration")
        return self._normalize_source(self.clicks(official=True))

    def metadata_lazy(self) -> pl.LazyFrame:
        """Collect only bounded-sample prefix modes; full export uses iter_metadata."""
        if self.row_limit is None:
            raise ValueError("full Yoochoose metadata requires iter_metadata for bounded memory")
        batches = list(self.iter_metadata())
        return (pl.concat(batches) if batches else pl.DataFrame(schema=ITEMS_SCHEMA)).lazy()

    def iter_metadata(self, chunk_rows: int = 100_000) -> Iterator[pl.DataFrame]:
        """Update category/brand modes atomically by second and emit unique snapshots."""
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        frame = self._normalize_source(self.clicks(for_metadata=True)).select(
            "item_id", "timestamp", pl.col("category").cast(pl.Int64, strict=False).alias("numeric")
        ).filter(pl.col("numeric") > 0).sort(["timestamp", "item_id", "numeric"])
        counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
        touched: set[tuple[int, str]] = set()
        current: int | None = None
        output: list[tuple] = []

        def flush() -> None:
            for item, field in sorted(touched):
                counter = counts[item, field]
                mode = min(counter, key=lambda value: (-counter[value], value))
                output.append((item, "yoochoose", field, str(mode), current, None, "observed_snapshot", "raw", current))
            touched.clear()

        for batch in frame.collect_batches(chunk_size=chunk_rows, maintain_order=True):
            for item, timestamp, numeric in batch.iter_rows():
                if current is not None and timestamp != current:
                    flush()
                    if len(output) >= chunk_rows:
                        yield pl.DataFrame(output, schema=ITEMS_SCHEMA, orient="row")
                        output = []
                current = timestamp
                field = "category_id" if numeric <= 12 else "brand_id"
                counts[item, field][numeric] += 1
                touched.add((item, field))
        flush()
        if output:
            yield pl.DataFrame(output, schema=ITEMS_SCHEMA, orient="row")
