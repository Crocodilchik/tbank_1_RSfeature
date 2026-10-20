"""Strict item metadata as-of access, on disk or in vectorized lazy joins."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator

import polars as pl


class MetadataIndex:
    """Runtime-only indexed metadata with exact matches forbidden for every query."""

    def __init__(self, metadata: pl.LazyFrame, database: Path) -> None:
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("CREATE TABLE properties (item INTEGER,field TEXT,ts INTEGER,value TEXT,PRIMARY KEY(item,field,ts)) WITHOUT ROWID")
        self._value_index = False
        self._fields: set[str] = set()
        try:
            for batch in metadata.select("item_id", "field_name", "valid_from", "field_value").collect_batches(chunk_size=100_000):
                self._fields.update(batch["field_name"].unique().to_list())
                self.connection.executemany("INSERT INTO properties VALUES (?,?,?,?)", batch.iter_rows())
            self.connection.commit()
        except BaseException:
            self.connection.close()
            raise

    def get(self, item: int, timestamp: int, field: str, default: Any = None) -> Any:
        """Return the latest known value strictly before timestamp, never future fill."""
        if field not in self._fields:
            return default
        row = self.connection.execute("SELECT value FROM properties WHERE item=? AND field=? AND ts<? ORDER BY ts DESC LIMIT 1", (item, field, timestamp)).fetchone()
        return json.loads(row[0]) if row is not None else default

    def category(self, item: int, timestamp: int) -> int | None:
        """Return the raw known category ID, or an explicit unknown."""
        value = self.get(item, timestamp, "category_id")
        return value if value is not None and value >= 0 else None

    def items_in_category(self, category: int, timestamp: int) -> Iterator[int]:
        """Yield ordered IDs whose latest strict-prefix category equals the value.

        The inverse index narrows rare-category fallback without consulting future
        versions. Later changes and unknown tombstones suppress older memberships.
        Catalog availability remains the caller's separate prefix constraint.
        """
        return self._items_matching("category_id", {category}, timestamp)

    def items_in_clusters(self, clusters: set[int], timestamp: int, snapshot_ts: int) -> Iterator[int]:
        """Yield only memberships fitted in one available centroid space."""
        return self._items_matching("item_cluster", clusters, timestamp, snapshot_ts)

    def _items_matching(self, field: str, values: Iterable[int], timestamp: int,
                        snapshot_ts: int | None = None) -> Iterator[int]:
        if snapshot_ts is not None and snapshot_ts >= timestamp:
            raise ValueError("cluster snapshot must be strictly before query timestamp")
        values = sorted({int(value) for value in values if value >= 0})
        if not values:
            return
        if not self._value_index:
            self.connection.execute("CREATE INDEX properties_by_value ON properties(field,value,ts,item)")
            self._value_index = True
        placeholders = ",".join("?" for _ in values)
        fitted = " AND p.ts=?" if snapshot_ts is not None else ""
        parameters = [field, *[json.dumps(value) for value in values], timestamp]
        if snapshot_ts is not None:
            parameters.append(snapshot_ts)
        parameters.append(timestamp)
        cursor = self.connection.execute(f"""SELECT p.item FROM properties p
            WHERE p.field=? AND p.value IN ({placeholders}) AND p.ts<? {fitted}
            AND NOT EXISTS (SELECT 1 FROM properties q
                WHERE q.item=p.item AND q.field=p.field AND q.ts>p.ts AND q.ts<?)
            ORDER BY p.item""", parameters)
        for row in cursor:
            yield int(row[0])

    def snapshot_timestamp(self, item: int, timestamp: int, field: str) -> int | None:
        """Identify the available fitted space before comparing cluster IDs."""
        if field not in self._fields:
            return None
        row = self.connection.execute("SELECT ts FROM properties WHERE item=? AND field=? AND ts<? ORDER BY ts DESC LIMIT 1", (item, field, timestamp)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        """Close the runtime-only metadata index."""
        self.connection.close()


def asof_item_field(queries: pl.LazyFrame, metadata: pl.LazyFrame, field: str,
                    dtype: pl.DataType, *, item_column: str = "item_id", time_column: str = "timestamp") -> pl.LazyFrame:
    """Join one metadata field natively with strict `<T`, retaining audit timestamps."""
    values = metadata.filter(pl.col("field_name") == field).select(
        pl.col("item_id").alias(item_column), pl.col("valid_from").alias(f"asof_{field}_ts"),
        pl.col("field_value").str.json_decode(dtype).alias(field)).sort(f"asof_{field}_ts")
    return queries.sort(time_column).join_asof(values, left_on=time_column, right_on=f"asof_{field}_ts",
                                              by=item_column, allow_exact_matches=False, strategy="backward", check_sortedness=False)
