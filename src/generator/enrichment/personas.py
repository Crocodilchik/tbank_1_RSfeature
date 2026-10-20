"""Prefix-fitted sparse user/item personas with auditable availability."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterator, Literal

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, normalize

from src.config.loader import PersonasSettings
from src.generator.schemas import ITEMS_SCHEMA, USERS_SCHEMA
from src.utils.reproducibility import stable_seed


@dataclass(frozen=True)
class PersonaSnapshot:
    """An immutable assignment artifact fitted exclusively before snapshot_ts."""

    snapshot_ts: int
    max_input_ts: int | None
    entity: str
    seed: int
    assignments: dict[int, tuple[int, str, float, int]]
    centroids: np.ndarray

    def features_for(self, entity_id: int, target_ts: int) -> tuple[int, str, float, int]:
        """Return past assignments or explicit cold defaults, rejecting future use."""
        if self.snapshot_ts >= target_ts:
            raise ValueError("persona snapshot must be strictly before target")
        return self.assignments.get(entity_id, (0, "cold_start", 0.0, 0))

    def metadata(self, dataset: str) -> pl.DataFrame:
        """Serialize assignments into the approved user/item metadata schema."""
        if len(self.assignments) > 100_000:
            raise ValueError("large persona snapshots require iter_metadata")
        frames = list(self.iter_metadata(dataset))
        return pl.concat(frames) if frames else pl.DataFrame(schema=USERS_SCHEMA if self.entity == "user_id" else ITEMS_SCHEMA)

    def iter_metadata(self, dataset: str, chunk_rows: int = 100_000) -> Iterator[pl.DataFrame]:
        """Serialize bounded metadata batches instead of millions of Python tuples."""
        rows = []
        for entity_id, (cluster, level, affinity, count) in sorted(self.assignments.items()):
            if self.entity == "user_id":
                for field, value in (("user_cluster", cluster), ("affinity_score", affinity), ("total_interactions", count)):
                    rows.append((entity_id, dataset, field, json.dumps(value), self.snapshot_ts, None,
                                 "derived_pit", "synthetic", self.snapshot_ts, level, self.seed))
            else:
                rows.append((entity_id, dataset, "item_cluster", str(cluster), self.snapshot_ts, None, "derived_pit", "synthetic", self.snapshot_ts))
            if len(rows) >= chunk_rows:
                yield pl.DataFrame(rows, schema=USERS_SCHEMA if self.entity == "user_id" else ITEMS_SCHEMA, orient="row")
                rows = []
        schema = USERS_SCHEMA if self.entity == "user_id" else ITEMS_SCHEMA
        if rows:
            yield pl.DataFrame(rows, schema=schema, orient="row")


def fit_personas(events: pl.DataFrame | pl.LazyFrame, snapshot_ts: int, settings: PersonasSettings,
                 seed: int = 42, entity: Literal["user_id", "item_id"] = "user_id") -> PersonaSnapshot:
    """Fit MiniBatchKMeans on sparse prefix category frequencies and window counts.

    Missing categories enter an explicit unknown bucket, never future lookup.
    No dense user-by-item similarity matrix is constructed.
    """
    frame = events.lazy() if isinstance(events, pl.DataFrame) else events
    prefix = frame.filter(pl.col("timestamp") < snapshot_ts)
    if "is_synthetic" in frame.collect_schema():
        prefix = prefix.filter(~pl.col("is_synthetic"))
    recent = prefix.filter(pl.col("timestamp") >= snapshot_ts - 30 * 86400)
    if "category_ids" in frame.collect_schema():
        expanded = recent.with_columns(pl.when(pl.col("category_ids").list.len() > 0).then(pl.col("category_ids")).otherwise(pl.lit([0], dtype=pl.List(pl.Int64))).alias("_categories")).explode("_categories")
        category = pl.col("_categories")
    else:
        expanded = recent
        category = pl.col("category_id").fill_null(0) if "category_id" in frame.collect_schema() else pl.lit(0)
    frequencies = expanded.with_columns((category % settings.vector_hash_dim).alias("bucket")).group_by([entity, "bucket"]).len().collect(engine="streaming")
    active = frequencies.select(entity).unique().lazy()
    statistics = prefix.join(active, on=entity, how="semi").group_by(entity).agg(
        pl.len().alias("count"), pl.col("timestamp").max().alias("last_ts"),
        *[(pl.col("timestamp") >= snapshot_ts - days * 86400).sum().alias(f"n{days}") for days in (1, 7, 30)]
    ).sort(entity).collect(engine="streaming")
    if statistics.is_empty():
        return PersonaSnapshot(snapshot_ts, None, entity, seed, {}, np.zeros((0, settings.vector_hash_dim + 3)))
    ids = statistics[entity].to_numpy()
    row = np.searchsorted(ids, frequencies[entity].to_numpy())
    matrix = coo_matrix((frequencies["len"].to_numpy().astype(np.float32), (row, frequencies["bucket"].to_numpy())),
                        shape=(len(ids), settings.vector_hash_dim)).tocsr()
    del frequencies, row
    matrix = hstack([normalize(matrix, norm="l1"), csr_matrix(np.log1p(statistics.select("n1", "n7", "n30").to_numpy()).astype(np.float32))], format="csr")
    scaler = StandardScaler(with_mean=False)
    matrix = scaler.fit_transform(matrix).tocsr()
    k = min(settings.k, len(ids))
    model = MiniBatchKMeans(n_clusters=k, batch_size=max(k, settings.batch_size), random_state=seed, n_init=3, max_iter=100)
    model.fit(matrix)
    labels = model.predict(matrix)
    nearest = np.empty(len(ids), dtype=np.float32)
    alternatives = np.empty((len(ids), min(2, max(0, k - 1))), dtype=np.int32)
    for start in range(0, len(ids), settings.batch_size):
        end = min(start + settings.batch_size, len(ids))
        distances = model.transform(matrix[start:end])
        nearest[start:end] = distances[np.arange(end - start), labels[start:end]]
        alternatives[start:end] = np.argsort(distances, axis=1, kind="stable")[:, 1:3]
    thresholds = {label: np.quantile(nearest[labels == label], settings.uncertainty_quantiles) for label in set(labels)}
    assignments = {}
    for i, raw_id in enumerate(ids):
        cluster = int(labels[i])
        q50, q85 = thresholds[cluster]
        level_index = 0 if nearest[i] <= q50 else (1 if nearest[i] <= q85 else 2)
        count = int(statistics["count"][i])
        if count < settings.min_prior_interactions:
            level_index = 2
        rng = np.random.default_rng(stable_seed(seed, entity, int(raw_id), snapshot_ts))
        others = alternatives[i]
        if len(others) and rng.random() < settings.mixing_probabilities[level_index]:
            cluster = int(rng.choice(others))
        assignments[int(raw_id)] = (cluster + 1, ("core", "middle", "boundary")[level_index], float(1 / (1 + nearest[i])), count)
    return PersonaSnapshot(snapshot_ts, int(statistics["last_ts"].max()), entity, seed, assignments, model.cluster_centers_.copy())
