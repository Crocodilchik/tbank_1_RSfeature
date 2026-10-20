"""One immutable candidate pool with evaluator-only future exclusions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
from typing import Callable, Iterator

import numpy as np
import polars as pl
from sortedcontainers import SortedList

from src.config.loader import EvaluationSettings
from src.generator.enrichment.negatives import timestamp_groups
from src.generator.schemas import CANDIDATES_SCHEMA
from src.generator.splitting.temporal_splitter import primary_expression, select_targets
from src.utils.reproducibility import stable_seed

GROUPS_SCHEMA = {"eval_group_id": pl.Int64, "split": pl.String, "mode": pl.String, "primary_target": pl.String,
                 "exclusion_set_size": pl.Int64, "target_is_fallback": pl.Boolean, "catalog_size": pl.Int64,
                 "oracle_horizon_end": pl.Int64, "pool_hash": pl.String}
SKIPS_SCHEMA = {"target_event_id": pl.Int64, "user_id": pl.Int64, "split": pl.String, "reason": pl.String}


class EvaluationOracle:
    """Disk-backed future facts used exclusively by the benchmark constructor."""

    def __init__(self, events: pl.DataFrame | pl.LazyFrame, dataset: str, database: Path | None = None, fallback: bool = False) -> None:
        if database is not None:
            database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database) if database else ":memory:")
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("CREATE TABLE IF NOT EXISTS facts (user INTEGER,item INTEGER,ts INTEGER,split TEXT,positive INTEGER)")
        if self.connection.execute("SELECT 1 FROM facts LIMIT 1").fetchone():
            self.connection.close()
            raise ValueError("evaluation oracle database is not empty")
        frame = events.lazy() if isinstance(events, pl.DataFrame) else events
        real = frame.filter(~pl.col("is_synthetic")).select("user_id", "item_id", "timestamp", "split", primary_expression(dataset, fallback).cast(pl.Int8).alias("positive"))
        self.horizon_end = None
        for batch in real.collect_batches(chunk_size=100_000, maintain_order=True):
            self.connection.executemany("INSERT INTO facts VALUES (?,?,?,?,?)", batch.iter_rows())
            maximum = batch["timestamp"].max()
            self.horizon_end = max(maximum, self.horizon_end if self.horizon_end is not None else maximum)
        self.connection.execute("CREATE INDEX IF NOT EXISTS facts_user ON facts(user,split,positive,item)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS facts_time ON facts(user,ts,item)")
        self.connection.commit()

    def exclusions(self, user: int, item: int, timestamp: int, split: str, policy: str, extra_future: bool) -> set[int]:
        """Union target, repeat exclusions and all positives of the current eval split."""
        items = {item}
        items.update(r[0] for r in self.connection.execute("SELECT item FROM facts WHERE user=? AND split=? AND positive=1", (user, split)))
        if policy == "unseen_only":
            items.update(r[0] for r in self.connection.execute("SELECT item FROM facts WHERE user=?", (user,)))
        elif policy == "allow_within_session_only":
            items.update(r[0] for r in self.connection.execute("SELECT item FROM facts WHERE user=? AND ts<?", (user, timestamp)))
        elif policy != "allow_previous":
            raise ValueError(f"unknown exclusion policy: {policy}")
        if extra_future:
            items.update(r[0] for r in self.connection.execute("SELECT item FROM facts WHERE user=? AND ts>? AND positive=1", (user, timestamp)))
        return items

    def close(self) -> None:
        """Close the evaluator-owned database."""
        self.connection.close()


class CandidateCatalog:
    """Past-known items with uniform order and dynamic Fenwick popularity masses."""

    def __init__(self) -> None:
        self.items = SortedList()
        self.counts: Counter[int] = Counter()
        self.indices: dict[int, int] = {}
        self.values: list[int] = []
        self.tree = [0.0]
        self.weights = [0.0]
        self.last_timestamp: int | None = None

    def _prefix(self, index: int) -> float:
        total = 0.0
        while index:
            total += self.tree[index]
            index -= index & -index
        return total

    def add(self, item: int, timestamp: int, interactions: int = 0) -> None:
        """Activate an observed catalog item and update prefix-only popularity."""
        if item not in self.indices:
            self.items.add(item)
            self.values.append(item)
            index = len(self.values)
            self.indices[item] = index
            self.weights.append(1.0)
            self.tree.append(1.0 + self._prefix(index - 1) - self._prefix(index - (index & -index)))
        index = self.indices[item]
        self.counts[item] += interactions
        new = (self.counts[item] + 1) ** 0.75
        delta = new - self.weights[index]
        self.weights[index] = new
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & -index
        self.last_timestamp = max(timestamp, self.last_timestamp if self.last_timestamp is not None else timestamp)

    def draw(self, rng: np.random.Generator, popularity: bool = False) -> int:
        """Draw an available item in O(log catalog_size), never a future item."""
        if not self.items:
            raise ValueError("cannot draw from empty catalog")
        if not popularity:
            return self.items[int(rng.integers(len(self.items)))]
        mass = rng.random() * self._prefix(len(self.values))
        index = 0
        bit = 1 << (len(self.values).bit_length() - 1)
        while bit:
            next_index = index + bit
            if next_index < len(self.tree) and self.tree[next_index] <= mass:
                mass -= self.tree[next_index]
                index = next_index
            bit >>= 1
        return self.values[min(index, len(self.values) - 1)]


@dataclass
class EvaluationBatch:
    """Separate scoring candidates, evaluator-only metadata and skipped groups."""

    candidates: pl.DataFrame
    groups: pl.DataFrame
    skipped: pl.DataFrame


def _draw_pool(catalog: CandidateCatalog, exclusions: set[int], n: int, rng: np.random.Generator,
               strategy: str, predicate: Callable[[int], bool] | None = None) -> list[tuple[int, str]]:
    if len(catalog.items) - sum(i in catalog.indices for i in exclusions) < n:
        return []
    selected: dict[int, str] = {}
    for _ in range(max(200, n * 50)):
        if len(selected) == n:
            break
        item = catalog.draw(rng, strategy == "popularity_biased")
        if item not in exclusions and item not in selected and (predicate is None or predicate(item)):
            selected[item] = "popularity" if strategy == "popularity_biased" else ("hard" if predicate is not None else "random")
    if len(selected) < n:
        # Exact uniform/weighted reservoir for finite dense-exclusion fallback.
        heap = []
        for item in catalog.items:
            if item in exclusions or item in selected:
                continue
            weight = (catalog.counts[item] + 1) ** 0.75 if strategy == "popularity_biased" else 1.0
            priority = float(rng.random()) ** (1 / weight)
            entry = (priority, item)
            if len(heap) < n - len(selected):
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        for _, item in heap:
            selected[item] = "popularity" if strategy == "popularity_biased" else "random"
    return sorted(selected.items())


def iter_evaluation_pools(events: pl.DataFrame | pl.LazyFrame, dataset: str, settings: EvaluationSettings,
                          split: str, *, mode: str = "primary", fallback: bool = False,
                          database: Path | None = None, chunk_rows: int = 100_000,
                          metadata_catalog: pl.LazyFrame | None = None,
                          catalog_events: pl.LazyFrame | None = None,
                          target_event_ids: pl.LazyFrame | None = None,
                          category: Callable[[int, int], int | None] | None = None) -> Iterator[EvaluationBatch]:
    """Generate one locked pool per eligible common target using only prefix catalog.

    Oracle labels affect exclusions only. Catalog/popularity advance after the
    atomic timestamp batch. Full logs remain lazy and oracle history stays on disk.
    """
    frame = events.lazy() if isinstance(events, pl.DataFrame) else events
    frame = frame.filter(~pl.col("is_synthetic"))
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if fallback and mode == "primary":
        mode = "fallback"
    policy = settings.repeat_purchase_policy[dataset]
    targets = target_event_ids if target_event_ids is not None else select_targets(frame, dataset, split, policy, fallback).select("event_id")
    # The registry has one integer per evaluation user, not one row per event.
    # Freeze it once to avoid repeated selection and empty-join cost-estimator panics.
    targets = targets.collect(engine="streaming", optimizations=pl.QueryOptFlags(row_estimate=False, join_order=False)).lazy()
    targets = targets.with_columns(pl.lit(True).alias("_target"))
    marked = frame.join(targets, on="event_id", how="left")
    if catalog_events is not None:
        marked = marked.join(catalog_events.select("event_id").unique().with_columns(pl.lit(True).alias("_catalog_event")), on="event_id", how="left")
    requested = targets.select(pl.len().alias("n"), pl.col("event_id").n_unique().alias("unique")).collect().row(0)
    if requested[0] == 0:
        return
    matched = marked.filter(pl.col("_target").fill_null(False))
    stats = matched.select(pl.len().alias("n"), pl.col("event_id").n_unique().alias("ids"),
                           pl.col("user_id").n_unique().alias("users"),
                           ((pl.col("split") != split) | ~primary_expression(dataset, fallback)).sum().alias("bad")).collect(engine="streaming").row(0)
    if requested[0] != requested[1] or stats[:3] != (requested[0],) * 3 or stats[3]:
        raise ValueError("invalid target registry: missing/duplicate IDs, repeated user, different split or task")
    if target_event_ids is not None and policy == "unseen_only":
        pairs = matched.select("user_id", "item_id").unique()
        first = frame.join(pairs, on=["user_id", "item_id"], how="semi").group_by(["user_id", "item_id"]).agg(pl.col("timestamp").min().alias("first_ts"))
        if matched.join(first, on=["user_id", "item_id"]).filter(pl.col("timestamp") > pl.col("first_ts")).limit(1).collect(engine="streaming").height:
            raise ValueError("target registry contains a previously seen target under unseen_only")
    oracle = EvaluationOracle(frame, dataset, database, fallback)
    catalog = CandidateCatalog()
    metadata_iter = iter(()) if metadata_catalog is None else (
        row for batch in metadata_catalog.filter(pl.col("valid_from") > -(2**63)).sort(["valid_from", "item_id"]).collect_batches(chunk_size=chunk_rows, maintain_order=True) for row in batch.iter_rows(named=True))
    pending = next(metadata_iter, None)
    rows, groups, skips = [], [], []
    try:
        for same_time in timestamp_groups(marked, chunk_rows):
            timestamp = same_time[0]["timestamp"]
            while pending is not None and pending["valid_from"] < timestamp:
                catalog.add(pending["item_id"], pending["valid_from"])
                pending = next(metadata_iter, None)
            for row in same_time:
                if not row["_target"]:
                    continue
                event_id, user, item = row["event_id"], row["user_id"], row["item_id"]
                exclusions = oracle.exclusions(user, item, timestamp, split, policy, settings.exclude_future_positives)
                seed = stable_seed(settings.seed, dataset, mode, split, event_id, settings.negative_sampling.strategy)
                target_category = category(item, timestamp) if category else None
                predicate = (lambda candidate, ts=timestamp, target=target_category: category(candidate, ts) == target) if settings.negative_sampling.strategy == "hard" and target_category is not None else None
                negatives = _draw_pool(catalog, exclusions, 99, np.random.default_rng(seed), settings.negative_sampling.strategy, predicate)
                if len(negatives) != 99:
                    skips.append((event_id, user, split, "insufficient_candidates"))
                    continue
                candidates = sorted([(item, "ground_truth"), *negatives])
                pool_hash = hashlib.sha256(json.dumps([(i, i == item) for i, _ in candidates], separators=(",", ":")).encode()).hexdigest()
                for candidate, kind in candidates:
                    rows.append((event_id, user, event_id, timestamp, item, candidate, int(candidate == item), kind,
                                 dataset, seed, catalog.last_timestamp, policy))
                groups.append((event_id, split, mode, (settings.fallback_targets if fallback else settings.primary_targets)[dataset], len(exclusions), fallback,
                               len(catalog.items), oracle.horizon_end if settings.exclude_future_positives or policy == "unseen_only" else None, pool_hash))
            for row in same_time:
                if catalog_events is None or row["_catalog_event"]:
                    catalog.add(row["item_id"], timestamp, 1)
            if len(rows) >= chunk_rows or len(skips) >= chunk_rows:
                yield EvaluationBatch(pl.DataFrame(rows, schema=CANDIDATES_SCHEMA, orient="row"), pl.DataFrame(groups, schema=GROUPS_SCHEMA, orient="row"), pl.DataFrame(skips, schema=SKIPS_SCHEMA, orient="row"))
                rows, groups, skips = [], [], []
        if rows or groups or skips:
            yield EvaluationBatch(pl.DataFrame(rows, schema=CANDIDATES_SCHEMA, orient="row"), pl.DataFrame(groups, schema=GROUPS_SCHEMA, orient="row"), pl.DataFrame(skips, schema=SKIPS_SCHEMA, orient="row"))
    finally:
        oracle.close()
