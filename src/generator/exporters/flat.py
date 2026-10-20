"""Common supervised query rows, with feature construction owned by models."""

import polars as pl

from src.generator.splitting.temporal_splitter import primary_expression

QUERY_SCHEMA = {"query_group_id": pl.Int64, "user_id": pl.Int64, "candidate_item_id": pl.Int64,
                "timestamp": pl.Int64, "target": pl.Int8, "split": pl.String, "dataset_source": pl.String}


def training_queries(events: pl.LazyFrame, negatives: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
    """Pair real anchors with frozen sampled negatives, excluding unsampled anchors."""
    contexts = negatives.select(pl.col("context_event_id").alias("event_id")).unique()
    positive = events.filter((pl.col("split") == "train") & primary_expression(dataset)).join(contexts, on="event_id", how="semi").select(
        pl.col("event_id").alias("query_group_id"), "user_id", pl.col("item_id").alias("candidate_item_id"),
        "timestamp", pl.lit(1, dtype=pl.Int8).alias("target"), "split", "dataset_source")
    negative = negatives.select(pl.col("context_event_id").alias("query_group_id"), "user_id", pl.col("item_id").alias("candidate_item_id"),
                                 pl.col("context_timestamp").alias("timestamp"), pl.lit(0, dtype=pl.Int8).alias("target"),
                                 pl.lit("train").alias("split"), "dataset_source")
    return pl.concat([positive, negative]).cast(QUERY_SCHEMA).sort(["timestamp", "query_group_id", "candidate_item_id"])


def evaluation_queries(candidates: pl.LazyFrame, split: str) -> pl.LazyFrame:
    """Project scorer queries; answers remain a separately whitelisted target column."""
    return candidates.select(pl.col("eval_group_id").alias("query_group_id"), "user_id", "candidate_item_id",
                             pl.col("target_timestamp").alias("timestamp"), pl.col("candidate_label").alias("target"),
                             pl.lit(split).alias("split"), "dataset_source").cast(QUERY_SCHEMA)
