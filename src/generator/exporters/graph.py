"""Sparse graph base/delta edges without full daily graph duplication."""

from typing import Iterator

import polars as pl

from src.generator.schemas import EDGES_SCHEMA, STATIC_TS
from src.utils.reproducibility import stable_seed


def interaction_edges(events: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
    """Export real interactions; consumers must select timestamp<T per query."""
    return events.filter(~pl.col("is_synthetic")).select(
        pl.col("event_id").alias("edge_id"), pl.lit("User").alias("src_type"), pl.col("user_id").alias("src_id"),
        pl.lit("Item").alias("dst_type"), pl.col("item_id").alias("dst_id"), pl.col("event_type").alias("edge_type"),
        "timestamp", pl.col("timestamp").alias("valid_from"), pl.lit(None, dtype=pl.Int64).alias("valid_to"),
        pl.lit(1.0).alias("weight"), pl.lit(dataset).alias("dataset_source"),
        pl.concat_str([pl.lit(dataset + ":delta:"), (pl.col("timestamp") // 86400).cast(pl.String)]).alias("split_snapshot_id"),
        pl.lit("real").alias("source")).cast(EDGES_SCHEMA)


def iter_metadata_edges(metadata: pl.LazyFrame, categories: pl.DataFrame, dataset: str,
                        chunk_rows: int = 100_000) -> Iterator[pl.DataFrame]:
    """Export dated item-category/brand and static hierarchy edges with lineage.

    This is a delta log, not a graph usable wholesale. Item-category consumers
    select the last known relation per item/field, removing obsolete relations.
    """
    import json
    rows = []
    relevant = metadata.filter(pl.col("field_name").is_in(["category_id", "brand_id", "genres"]))
    for batch in relevant.collect_batches(chunk_size=chunk_rows, maintain_order=True):
        for row in batch.iter_rows(named=True):
            value = json.loads(row["field_value"])
            values = value if row["field_name"] == "genres" else [value]
            for raw in values:
                category = stable_seed("genre", raw) if row["field_name"] == "genres" else int(raw)
                kind = "Brand" if row["field_name"] == "brand_id" else "Category"
                ts = row["valid_from"]
                edge_id = stable_seed("metadata_edge", dataset, row["item_id"], kind, category, ts)
                rows.append((edge_id, "Item", row["item_id"], kind, category, "belongs_to", ts, ts, None, 1.0, dataset,
                             dataset + (":base" if ts == STATIC_TS else f":delta:{ts // 86400}"), "real"))
                if len(rows) >= chunk_rows:
                    yield pl.DataFrame(rows, schema=EDGES_SCHEMA, orient="row")
                    rows = []
    for row in categories.iter_rows(named=True):
        if row["parent_id"] is not None:
            edge_id = stable_seed("category_parent", dataset, row["category_id"], row["parent_id"])
            rows.append((edge_id, "Category", row["category_id"], "Category", row["parent_id"], "parent", STATIC_TS,
                         STATIC_TS, None, 1.0, dataset, dataset + ":base", "real"))
    if rows:
        yield pl.DataFrame(rows, schema=EDGES_SCHEMA, orient="row")


def graph_at(edges: pl.LazyFrame, timestamp: int) -> pl.LazyFrame:
    """Select only past edges and latest item metadata relations, never target time."""
    past = edges.filter(pl.col("timestamp") < timestamp)
    interactions = past.filter(pl.col("edge_type") != "belongs_to")
    memberships = past.filter(pl.col("edge_type") == "belongs_to")
    # Genre edges are static and can be multivalued; dated categories/brands use as-of.
    static = memberships.filter(pl.col("timestamp") == STATIC_TS)
    dated = memberships.filter(pl.col("timestamp") != STATIC_TS).sort("timestamp").unique(subset=["src_id", "dst_type"], keep="last")
    return pl.concat([interactions, static, dated.filter(pl.col("dst_id") >= 0)])
