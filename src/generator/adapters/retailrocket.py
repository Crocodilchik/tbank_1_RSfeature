"""RetailRocket events, weekly properties and static category hierarchy."""

from __future__ import annotations

import polars as pl
import pyarrow.parquet as pq
from typing import Iterator

from src.generator.adapters.base import BaseAdapter, DataValidationError, item_metadata, json_value
from src.generator.schemas import CATEGORIES_SCHEMA, ITEMS_SCHEMA
from src.generator.storage import runtime_workspace


class RetailRocketAdapter(BaseAdapter):
    """Preserve observed properties and avoid catalog backfilling."""

    def load_raw(self) -> pl.LazyFrame:
        """Read the original event schema with millisecond precision."""
        return self.scan("events", {"timestamp": pl.Int64, "visitorid": pl.Int64, "event": pl.String,
                                    "itemid": pl.Int64, "transactionid": pl.String})

    def observations_lazy(self) -> pl.LazyFrame:
        """Deduplicate exact original events before conservative second conversion."""
        raw = self.load_raw().unique(subset=["timestamp", "visitorid", "event", "itemid", "transactionid"], keep="first", maintain_order=True)
        frame = raw.rename({"visitorid": "user_id", "itemid": "item_id", "event": "event_type", "timestamp": "timestamp_ms"}).with_columns(
            (pl.col("timestamp_ms") // 1000).alias("timestamp"),
            pl.col("event_type").replace_strict({"view": 0, "addtocart": 1, "transaction": 2}, return_dtype=pl.Int8).alias("label"))
        return self.event_identity(frame)

    def validate(self) -> list[str]:
        """Reject unexpected event years instead of silently deleting them."""
        errors = super().validate()
        if errors:
            return errors
        bad = self.observations_lazy().filter(~pl.col("timestamp").is_between(1420070400, 1451606399)).limit(1).collect(engine="streaming")
        return ["RetailRocket event outside declared 2015 window"] if bad.height else []

    def metadata_lazy(self) -> pl.LazyFrame:
        """Store all weekly properties with strict observed availability."""
        return self._metadata_from_raw(self._raw_metadata())

    def _raw_metadata(self) -> pl.LazyFrame:
        schema = {"timestamp": pl.Int64, "itemid": pl.Int64, "property": pl.String, "value": pl.String}
        return self.restrict_metadata(pl.concat([self.scan(role, schema) for role in ("properties1", "properties2")]).rename({"itemid": "item_id"})).select("timestamp", "item_id", "property", "value")

    def _metadata_from_raw(self, raw: pl.LazyFrame) -> pl.LazyFrame:
        raw = raw.unique(
            subset=["timestamp", "item_id", "property", "value"], keep="first")
        conflicts = raw.group_by(["timestamp", "item_id", "property"]).agg(pl.col("value").n_unique().alias("n")).filter(pl.col("n") > 1).limit(1).collect(engine="streaming")
        if conflicts.height:
            raise DataValidationError("conflicting RetailRocket property values at identical source timestamp")
        # Values within a rounded second become available together; latest original ms wins.
        raw = raw.with_columns((pl.col("timestamp") // 1000).alias("snapshot_timestamp")).sort("timestamp").unique(
            subset=["item_id", "property", "snapshot_timestamp"], keep="last")
        field = pl.when(pl.col("property") == "categoryid").then(pl.lit("category_id")).when(pl.col("property") == "available").then(pl.lit("available")).otherwise(pl.concat_str([pl.lit("property_"), pl.col("property")]))
        value = pl.when(pl.col("property") == "categoryid").then(json_value(pl.col("value").cast(pl.Int64, strict=False))).when(pl.col("property") == "available").then(json_value(pl.col("value") == "1")).otherwise(json_value(pl.col("value").str.split(" ")))
        invalid = raw.filter(((pl.col("property") == "categoryid") & pl.col("value").cast(pl.Int64, strict=False).is_null()) |
                             ((pl.col("property") == "available") & ~pl.col("value").is_in(["0", "1"]))).limit(1).collect(engine="streaming")
        if invalid.height:
            raise DataValidationError("invalid RetailRocket categoryid/available property")
        return raw.select("item_id", pl.lit("retailrocket").alias("dataset_source"), field.alias("field_name"), value.alias("field_value"),
                          pl.col("snapshot_timestamp").alias("valid_from"), pl.lit(None, dtype=pl.Int64).alias("valid_to"),
                          pl.lit("observed_snapshot").alias("availability_class"), pl.lit("raw").alias("source"), "snapshot_timestamp").cast(ITEMS_SCHEMA)

    def iter_metadata(self, chunk_rows: int = 100_000, *, shards: int = 64) -> Iterator[pl.DataFrame]:
        """Normalize all full-volume properties with one bounded item partition at a time.

        The source CSV files are scanned once into runtime Parquet. All rows of
        an item share a partition, preserving exact dedup and conflict rules.
        """
        if chunk_rows <= 0 or not 1 <= shards <= 256:
            raise ValueError("metadata chunk_rows/shards must be positive and bounded")
        if self.row_limit is not None or self.metadata_item_ids is not None:
            yield from super().iter_metadata(chunk_rows)
            return
        raw = self._raw_metadata()
        schema = dict(raw.collect_schema())
        arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema
        with runtime_workspace(self.settings.raw_dir.parents[2]) as temporary:
            writers = {}
            paths = {}
            try:
                for batch in raw.collect_batches(chunk_size=min(chunk_rows, 50_000), maintain_order=True):
                    if batch.select(pl.any_horizontal(pl.all().is_null()).any()).item() or (batch["item_id"] < 0).any():
                        raise DataValidationError("invalid required Retailrocket property field")
                    parts = batch.with_columns((pl.col("item_id") % shards).alias("_partition")).partition_by("_partition", as_dict=True)
                    for (partition,), frame in parts.items():
                        if partition not in writers:
                            path = temporary / f"properties_{partition}.parquet"
                            paths[partition] = path
                            writers[partition] = pq.ParquetWriter(path, arrow_schema, compression="zstd")
                        writers[partition].write_table(frame.drop("_partition").to_arrow())
            finally:
                for writer in writers.values():
                    writer.close()
            for partition in sorted(paths):
                yield from self._metadata_from_raw(pl.scan_parquet(paths[partition])).collect_batches(chunk_size=chunk_rows, maintain_order=True)

    def categories(self) -> pl.DataFrame:
        """Resolve roots/paths; preserve orphan categories and reject cycles."""
        raw = self.scan("categories", {"categoryid": pl.Int64, "parentid": pl.Int64}).collect()
        if raw["categoryid"].n_unique() != raw.height:
            raise DataValidationError("duplicate RetailRocket category ID")
        parents = dict(raw.select("categoryid", "parentid").iter_rows())
        paths: dict[int, list[int]] = {}

        def visit(node: int, trail: set[int]) -> list[int]:
            if node in trail:
                raise DataValidationError("cycle in category hierarchy")
            if node in paths:
                return paths[node]
            parent = parents[node]
            paths[node] = visit(parent, trail | {node}) + [node] if parent in parents else [node]
            return paths[node]

        rows = []
        for node, parent in parents.items():
            path = visit(node, set())
            rows.append((node, "retailrocket", parent if parent in parents else None, len(path) - 1, None, "/".join(map(str, path))))
        return pl.DataFrame(rows, schema=CATEGORIES_SCHEMA, orient="row")
