"""Disk-bounded sequence samples with atomic timestamp cutoffs."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Callable, Iterator

import polars as pl

from src.generator.enrichment.negatives import timestamp_groups
from src.generator.schemas import SEQUENCES_SCHEMA
from src.generator.splitting.temporal_splitter import primary_expression


def iter_sequence_samples(events: pl.LazyFrame, dataset: str, split: str, database: Path,
                          *, targets: pl.LazyFrame | None = None, max_history: int = 50,
                          category: Callable[[int, int], int | None] | None = None,
                          chunk_rows: int = 100_000) -> Iterator[pl.DataFrame]:
    """Emit train leave_one_out or common last_event_only evaluation samples.

    Each history lookup is a bounded indexed range query. Previously repeated
    items remain in history; the target event and all same-second events do not.
    """
    if not 0 < max_history <= 50 or split not in ("train", "valid", "test"):
        raise ValueError("invalid sequence history limit/split")
    if split != "train" and targets is None:
        raise ValueError("validation/test sequences require common target registry")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("CREATE TABLE history(user INTEGER,ts INTEGER,event INTEGER,item INTEGER,label INTEGER,category INTEGER,PRIMARY KEY(user,ts,event)) WITHOUT ROWID")
    real = events.filter(~pl.col("is_synthetic"))
    if targets is None:
        marked = real.with_columns(((pl.col("split") == split) & primary_expression(dataset)).alias("_sample"))
    else:
        marked = real.join(targets.select("event_id").unique().with_columns(pl.lit(True).alias("_sample")), on="event_id", how="left")
    output = []
    try:
        for same_time in timestamp_groups(marked, chunk_rows):
            timestamp = same_time[0]["timestamp"]
            for row in same_time:
                if not row["_sample"]:
                    continue
                if row["split"] != split:
                    raise ValueError("sequence target belongs to wrong split")
                history = list(reversed(connection.execute(
                    "SELECT event,item,ts,label,category FROM history WHERE user=? AND ts<? ORDER BY ts DESC,event DESC LIMIT ?",
                    (row["user_id"], timestamp, max_history)).fetchall()))
                histories = [[int(h[i]) for h in history] for i in range(5)]
                deltas = [timestamp - h[2] for h in history]
                output.append((row["event_id"], row["user_id"], row["event_id"], row["item_id"], timestamp,
                               row["label"], timestamp, *histories, deltas, history[-1][2] if history else None, dataset, split))
                if len(output) >= chunk_rows:
                    yield pl.DataFrame(output, schema=SEQUENCES_SCHEMA, orient="row")
                    output = []
            history_rows = []
            for row in same_time:
                raw_category = category(row["item_id"], timestamp) if category else None
                history_rows.append((row["user_id"], timestamp, row["event_id"], row["item_id"], row["label"], int(raw_category) + 1 if raw_category is not None else 0))
            connection.executemany("INSERT INTO history VALUES (?,?,?,?,?,?)", history_rows)
        if output:
            yield pl.DataFrame(output, schema=SEQUENCES_SCHEMA, orient="row")
    finally:
        connection.close()
