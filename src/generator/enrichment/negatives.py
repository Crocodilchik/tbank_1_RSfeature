"""Bounded prefix-only sampling with disk-backed history and isolated oracle audit."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
import sqlite3
from typing import Callable, Iterable, Iterator, Literal

import numpy as np
import polars as pl
from sortedcontainers import SortedList

from src.config.loader import NegativeSettings
from src.generator.schemas import NEGATIVES_SCHEMA
from src.utils.reproducibility import stable_seed

LOGGER = logging.getLogger(__name__)


class PrefixState:
    """Catalog order in memory and indexed interaction membership on disk.

    SQLite keeps histories bounded in Python memory. The optional in-memory
    database is intended for small fixtures only. Call close after use.
    """

    def __init__(self, database: Path | None = None, *,
                 negative_id_validation: Literal["immediate", "artifact_gate"] = "immediate") -> None:
        if negative_id_validation not in ("immediate", "artifact_gate"):
            raise ValueError("unknown negative ID validation backend")
        self.negative_id_validation = negative_id_validation
        if database is not None:
            database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database) if database else ":memory:")
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("CREATE TABLE IF NOT EXISTS history (user INTEGER, item INTEGER, ts INTEGER, PRIMARY KEY(user,item,ts)) WITHOUT ROWID")
        if negative_id_validation == "immediate":
            self.connection.execute("CREATE TABLE IF NOT EXISTS negative_ids (id INTEGER PRIMARY KEY)")
        if self.connection.execute("SELECT 1 FROM history LIMIT 1").fetchone() is not None:
            self.connection.close()
            raise ValueError("prefix database is not empty; use a new run-owned state path")
        self.catalog = SortedList()
        self._catalog_set: set[int] = set()
        self.last_timestamp: int | None = None

    def observe(self, rows: Iterable[tuple[int, int, int]]) -> None:
        """Add real completed timestamp groups to prefix history/catalog."""
        batch = list(rows)
        if self.last_timestamp is not None and any(ts < self.last_timestamp for _, _, ts in batch):
            raise ValueError("prefix state cannot move backward in time")
        self.connection.executemany("INSERT OR IGNORE INTO history VALUES (?,?,?)", batch)
        for _, item, ts in batch:
            if item not in self._catalog_set:
                self.catalog.add(item)
                self._catalog_set.add(item)
            self.last_timestamp = max(ts, self.last_timestamp if self.last_timestamp is not None else ts)

    def seen(self, user: int, item: int) -> bool:
        """Test membership of a real user-item pair in the available prefix."""
        return self.connection.execute("SELECT 1 FROM history WHERE user=? AND item=? LIMIT 1", (user, item)).fetchone() is not None

    def first_future(self, user: int, item: int, timestamp: int) -> int | None:
        """Evaluator-only lookup, never consulted during training sampling."""
        row = self.connection.execute("SELECT ts FROM history WHERE user=? AND item=? AND ts>? ORDER BY ts LIMIT 1", (user, item, timestamp)).fetchone()
        return int(row[0]) if row else None

    def register_negative(self, negative_id: int) -> None:
        """Check immediately, or require the caller's complete artifact unique gate."""
        if self.negative_id_validation == "artifact_gate":
            return
        try:
            self.connection.execute("INSERT INTO negative_ids VALUES (?)", (negative_id,))
        except sqlite3.IntegrityError as exc:
            raise ValueError("negative ID collision or duplicate context/item") from exc

    def close(self) -> None:
        """Flush and close the owned state database."""
        self.connection.commit()
        self.connection.close()


def timestamp_groups(events: pl.DataFrame | pl.LazyFrame, chunk_rows: int = 100_000) -> Iterator[list[dict]]:
    """Iterate globally ordered atomic seconds across Polars batch boundaries."""
    frame = events.lazy() if isinstance(events, pl.DataFrame) else events
    current = None
    group: list[dict] = []
    for batch in frame.sort(["timestamp", "event_id"]).collect_batches(chunk_size=chunk_rows, maintain_order=True):
        for row in batch.iter_rows(named=True):
            if current is not None and row["timestamp"] != current:
                yield group
                group = []
            current = row["timestamp"]
            group.append(row)
    if group:
        yield group


def draw_candidates(state: PrefixState, user: int, exclude: set[int], n: int, rng: np.random.Generator,
                    predicate: Callable[[int], bool] | None = None,
                    fallback_candidates: Callable[[], Iterable[int]] | None = None) -> list[int]:
    """Draw unique prefix items with finite rejection and deterministic fallback."""
    if n < 0:
        raise ValueError("candidate count cannot be negative")
    selected: set[int] = set()
    for _ in range(max(100, n * 50)):
        if len(selected) >= n or not len(state.catalog):
            break
        item = state.catalog[int(rng.integers(len(state.catalog)))]
        if item not in exclude and item not in selected and not state.seen(user, item) and (predicate is None or predicate(item)):
            selected.add(item)
    if len(selected) < n:
        # Bounded-memory fallback handles sparse eligibility without looping forever.
        # An inverse metadata stream has exactly the same ascending eligible order.
        # Keep rejection/RNG draws unchanged, including reservoir sampling order.
        catalog = state.catalog if fallback_candidates is None else fallback_candidates()
        eligible = (i for i in catalog if i in state._catalog_set and i not in exclude and i not in selected and not state.seen(user, i) and (predicate is None or predicate(i)))
        reservoir: list[int] = []
        for index, item in enumerate(eligible):
            if len(reservoir) < n - len(selected):
                reservoir.append(item)
            else:
                j = int(rng.integers(index + 1))
                if j < len(reservoir):
                    reservoir[j] = item
        selected.update(reservoir)
    return sorted(selected)


def iter_training_negatives(events: pl.DataFrame | pl.LazyFrame, settings: NegativeSettings,
                            primary: Callable[[dict], bool], seed: int = 42, state: PrefixState | None = None,
                            category: Callable[[int, int], int | None] | None = None,
                            cluster: Callable[[int, int], int | None] | None = None,
                            cluster_neighbors: Callable[[int, int], set[int]] | None = None,
                            chunk_rows: int = 100_000,
                            category_candidates: Callable[[int, int], Iterable[int]] | None = None,
                            cluster_candidates: Callable[[set[int], int], Iterable[int]] | None = None) -> Iterator[pl.DataFrame]:
    """Sample only train anchors, before updating their timestamp group.

    Difficulty pools use prefix category or cluster callbacks. Missing/empty
    pools fall back to global sampling. Future conflict is deliberately false
    until a separate evaluator annotation pass.
    """
    owned = state is None
    state = state or PrefixState()
    output = []
    emitted = 0
    try:
        if not settings.enabled:
            return
        for group in timestamp_groups(events, chunk_rows):
            if state.last_timestamp is not None and state.last_timestamp >= group[0]["timestamp"]:
                raise ValueError("sampler prefix contains simultaneous/future observations")
            simultaneous: dict[int, set[int]] = defaultdict(set)
            for row in group:
                simultaneous[row["user_id"]].add(row["item_id"])
            for row in group:
                if row.get("split") != "train" or not primary(row) or row.get("is_synthetic", False):
                    continue
                user, timestamp = row["user_id"], row["timestamp"]
                exclude = set(simultaneous[user])
                rng = np.random.default_rng(stable_seed(seed, row["dataset_source"], "train", row["event_id"]))
                for _ in range(settings.negatives_per_positive):
                    difficulty = int(rng.choice(3, p=settings.difficulty_mix))
                    kind = ("global", "cluster_mixed", "hard_boundary")[difficulty]
                    callback = cluster if cluster is not None else (category if difficulty == 2 else None)
                    target_value = callback(row["item_id"], timestamp) if callback is not None else None
                    predicate = None
                    fallback_candidates = None
                    if difficulty == 1 and target_value is not None and cluster_neighbors is not None:
                        allowed = cluster_neighbors(target_value, timestamp)
                        predicate = lambda item, cb=callback, values=allowed, ts=timestamp: cb(item, ts) in values
                        if callback is cluster and cluster_candidates is not None:
                            fallback_candidates = lambda values=allowed, ts=timestamp: cluster_candidates(values, ts)
                    elif difficulty == 2 and target_value is not None:
                        predicate = lambda item, cb=callback, target=target_value, ts=timestamp: cb(item, ts) == target
                        if callback is category and category_candidates is not None:
                            fallback_candidates = lambda target=target_value, ts=timestamp: category_candidates(target, ts)
                        elif callback is cluster and cluster_candidates is not None:
                            fallback_candidates = lambda target=target_value, ts=timestamp: cluster_candidates({target}, ts)
                    if difficulty and predicate is None:
                        kind = "global"
                    items = draw_candidates(state, user, exclude, 1, rng, predicate, fallback_candidates)
                    if not items and predicate is not None:
                        kind = "global"
                        items = draw_candidates(state, user, exclude, 1, rng)
                    if not items:
                        continue
                    item = items[0]
                    exclude.add(item)
                    negative_id = stable_seed("negative", row["event_id"], item, kind, seed)
                    state.register_negative(negative_id)
                    output.append((negative_id, user, item, row["event_id"], timestamp, timestamp, 0, kind,
                                   row["dataset_source"], seed, state.last_timestamp, False, None))
                    if len(output) >= chunk_rows:
                        emitted += len(output)
                        LOGGER.info("sampled training negatives rows=%s latest_context_ts=%s", emitted, timestamp)
                        yield pl.DataFrame(output, schema=NEGATIVES_SCHEMA, orient="row")
                        output = []
            state.observe((r["user_id"], r["item_id"], r["timestamp"]) for r in group if not r.get("is_synthetic", False))
        if output:
            LOGGER.info("sampled training negatives complete rows=%s", emitted + len(output))
            yield pl.DataFrame(output, schema=NEGATIVES_SCHEMA, orient="row")
    finally:
        if owned:
            state.close()


def annotate_future_conflicts(negatives: pl.DataFrame, evaluator: PrefixState) -> pl.DataFrame:
    """Annotate already-frozen rows; IDs, labels, order and weights stay unchanged."""
    first = [evaluator.first_future(row["user_id"], row["item_id"], row["context_timestamp"]) for row in negatives.iter_rows(named=True)]
    return negatives.with_columns(pl.Series("conflict_first_future_ts", first, dtype=pl.Int64),
                                  pl.Series("future_positive_conflict", [x is not None for x in first], dtype=pl.Boolean))
