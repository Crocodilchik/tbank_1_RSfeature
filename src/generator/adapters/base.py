"""Shared bounded adapter interface and lossless temporal normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import polars as pl

from src.config.loader import DatasetSettings
from src.generator.schemas import CATEGORIES_SCHEMA, EVENTS_SCHEMA, ITEMS_SCHEMA, STATIC_TS


class DataValidationError(ValueError):
    """Raw data violates the declared input contract."""


def json_value(expression: pl.Expr) -> pl.Expr:
    """Encode a typed scalar/list as canonical JSON using native Polars."""
    return pl.struct(value=expression).struct.json_encode().str.strip_prefix('{"value":').str.strip_suffix("}")


def item_metadata(frame: pl.LazyFrame, dataset: str, field: str | pl.Expr, value: pl.Expr,
                  timestamp: pl.Expr | None = None, availability: str = "static_assumed") -> pl.LazyFrame:
    """Build typed long metadata without joining it onto the event stream."""
    ts = timestamp if timestamp is not None else pl.lit(STATIC_TS, dtype=pl.Int64)
    snapshot = timestamp if timestamp is not None else pl.lit(None, dtype=pl.Int64)
    return frame.filter(value.is_not_null()).select(
        pl.col("item_id"), pl.lit(dataset).alias("dataset_source"), (field if isinstance(field, pl.Expr) else pl.lit(field)).alias("field_name"),
        json_value(value).alias("field_value"), ts.alias("valid_from"),
        pl.lit(None, dtype=pl.Int64).alias("valid_to"), pl.lit(availability).alias("availability_class"),
        pl.lit("raw").alias("source"), snapshot.alias("snapshot_timestamp"),
    ).cast(ITEMS_SCHEMA)


class BaseAdapter(ABC):
    """Read raw sources lazily and materialize only explicitly bounded samples.

    `split=unassigned` is an ingestion-only state. The splitter must replace it
    before any final core export. source_row_id is assigned before filtering.
    """

    def __init__(self, settings: DatasetSettings, row_limit: int | None = None) -> None:
        self.settings = settings
        self.row_limit = row_limit
        self.metadata_item_ids: pl.LazyFrame | None = None
        self.observation_user_modulus = 1
        if row_limit is not None and row_limit <= 0:
            raise ValueError("row_limit must be positive")

    def scan(self, role: str, schema: dict[str, pl.DataType], *, file_code: int = 1) -> pl.LazyFrame:
        """Scan a source with explicit types, file namespace and original row IDs."""
        path = self.settings.raw_dir / self.settings.files[role]
        if not path.is_file():
            raise FileNotFoundError(f"missing {self.settings.name}/{role}: {path}")
        frame = pl.scan_csv(path, has_header=self.settings.adapter.header, schema=schema,
                           n_rows=self.row_limit, row_index_name="source_row_id", row_index_offset=1,
                           low_memory=True).with_columns(
            pl.col("source_row_id").cast(pl.Int64), pl.lit(file_code, dtype=pl.Int64).alias("file_code"))
        user_field, roles = {"retailrocket": ("visitorid", {"events"}), "taobao": ("user_id", {"events"}),
            "yoochoose": ("session_id", {"clicks", "buys", "official_test"}), "movielens": ("userId", {"ratings"})}[self.settings.name]
        if self.observation_user_modulus > 1 and role in roles:
            frame = frame.filter(pl.col(user_field) % self.observation_user_modulus == 0)
        return frame

    @abstractmethod
    def load_raw(self) -> pl.LazyFrame:
        """Return raw source rows, retaining source lineage."""

    @abstractmethod
    def observations_lazy(self) -> pl.LazyFrame:
        """Return normalized facts with event IDs and raw auxiliary context."""

    @abstractmethod
    def metadata_lazy(self) -> pl.LazyFrame:
        """Return separately stored point-in-time item metadata."""

    def categories(self) -> pl.DataFrame:
        """Return an empty category table when the source has no real hierarchy."""
        return pl.DataFrame(schema=CATEGORIES_SCHEMA)

    def restrict_metadata(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Restrict smoke metadata before costly aggregation; full mode remains complete."""
        return frame if self.metadata_item_ids is None else frame.join(self.metadata_item_ids.select("item_id").unique(), on="item_id", how="semi")

    def iter_metadata(self, chunk_rows: int = 100_000) -> Iterator[pl.DataFrame]:
        """Stream metadata with bounded output batches."""
        yield from self.metadata_lazy().collect_batches(chunk_size=chunk_rows, maintain_order=True)

    def event_identity(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Namespace immutable source IDs across datasets and event files."""
        code = {"retailrocket": 1, "taobao": 2, "yoochoose": 3, "movielens": 4}[self.settings.name]
        return frame.with_columns((pl.lit(code * 10**15) + pl.col("file_code") * 10**12 + pl.col("source_row_id")).alias("event_id"))

    def normalize_lazy(self, source: pl.LazyFrame | None = None) -> pl.LazyFrame:
        """Prepare the unified event table using streaming-capable operations."""
        frame = (source if source is not None else self.observations_lazy()).sort(["timestamp", "event_id"])
        return frame.with_columns(
            pl.lit(self.settings.name).alias("dataset_source"),
            pl.lit("session" if self.settings.name == "yoochoose" else "persistent").alias("user_type"),
            pl.lit(False).alias("is_synthetic"),
            pl.col("event_id").cum_count().over("user_id").cast(pl.Int64).alias("user_event_seq"),
            pl.lit("unassigned").alias("split"), pl.col("timestamp").alias("target_cutoff_ts"),
        ).select(EVENTS_SCHEMA.keys()).cast(EVENTS_SCHEMA)

    def normalize(self, partition: pl.LazyFrame | None = None, max_rows: int = 100_000) -> pl.DataFrame:
        """Collect a bounded partition, rejecting accidentally materialized large logs."""
        errors = self.validate()
        if errors:
            raise DataValidationError("; ".join(errors))
        result = (partition if partition is not None else self.normalize_lazy()).limit(max_rows + 1).collect(engine="streaming")
        if result.height > max_rows:
            raise DataValidationError("normalize exceeds max_rows; use iter_normalized or streaming sink")
        return result

    def iter_normalized(self, chunk_rows: int = 100_000) -> Iterator[pl.DataFrame]:
        """Yield bounded normalized batches without collecting the complete log."""
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        errors = self.validate()
        if errors:
            raise DataValidationError("; ".join(errors))
        yield from self.normalize_lazy().collect_batches(chunk_size=chunk_rows, maintain_order=True)

    def validate(self) -> list[str]:
        """Report malformed IDs, times or enum values before writing artifacts."""
        try:
            required = {
                "retailrocket": ("visitorid", "itemid", "event", "timestamp"),
                "taobao": ("user_id", "item_id", "category_id", "behavior", "timestamp"),
                "yoochoose": ("session_id", "item_id", "timestamp"),
                "movielens": ("userId", "movieId", "rating", "timestamp"),
            }[self.settings.name]
            missing = self.load_raw().filter(pl.any_horizontal(pl.col(c).is_null() for c in required)).limit(1).collect(engine="streaming")
            if missing.height:
                return [f"null required raw field in {self.settings.name}"]
            facts = self.observations_lazy()
            bad = facts.filter(pl.any_horizontal(pl.col(c).is_null() for c in ("user_id", "item_id", "timestamp", "label")) |
                               (pl.col("user_id") < 0) | (pl.col("item_id") < 0) |
                               ~pl.col("label").is_in([0, 1, 2])).limit(1).collect(engine="streaming")
            return [f"invalid required field in {self.settings.name}: {bad.to_dicts()}"] if bad.height else []
        except (pl.exceptions.PolarsError, DataValidationError) as exc:
            return [f"{self.settings.name}: {exc}"]
