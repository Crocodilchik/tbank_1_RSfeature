"""Small complete artifact bundles for model integration, independent of raw files."""

from pathlib import Path
import polars as pl

from src.config.loader import AppConfig, load_config
from src.generator.enrichment.negatives import PrefixState, annotate_future_conflicts, iter_training_negatives
from src.generator.evaluation.sampler import iter_evaluation_pools
from src.generator.exporters.sequences import iter_sequence_samples
from src.generator.exporters.graph import interaction_edges
from src.generator.exporters.flat import evaluation_queries, training_queries
from src.generator.schemas import EVENTS_SCHEMA, ITEMS_SCHEMA
from src.generator.splitting.temporal_splitter import is_primary
from src.generator.storage import write_batches
from src.utils.reproducibility import config_hash, file_hash


def model_bundle(tmp_path: Path) -> tuple[AppConfig, Path]:
    """Create learnable causal train pairs and separate valid/test timestamp groups."""
    config = load_config(dataset="retailrocket")
    config = config.model_copy(update={"project": config.project.model_copy(update={"experiments_dir": tmp_path / "experiments"})})
    root = tmp_path / "processed"
    rows = [(i, 1000 + i, i, 10, 0, "view", "train") for i in range(1, 131)]
    rows += [(3000 + i, 1 + i % 5, 1 + i % 20, 11 + i, 2, "transaction", "train") for i in range(50)]
    rows += [(4000 + i, 1 + i % 5, 31 + i, 70 + i, 2, "transaction", "valid") for i in range(10)]
    rows += [(5000 + i, 1 + i % 5, 51 + i, 90 + i, 2, "transaction", "test") for i in range(10)]
    frame = pl.DataFrame(rows, schema=["event_id", "user_id", "item_id", "timestamp", "label", "event_type", "split"], orient="row").with_columns(
        pl.lit("retailrocket").alias("dataset_source"), pl.lit("persistent").alias("user_type"), pl.lit(False).alias("is_synthetic"),
        pl.lit(None, dtype=pl.Int64).alias("timestamp_ms"), pl.col("event_id").alias("source_row_id"),
        pl.lit(1, dtype=pl.Int64).alias("user_event_seq"), pl.col("timestamp").alias("target_cutoff_ts")).select(EVENTS_SCHEMA.keys()).cast(EVENTS_SCHEMA)
    entries = []
    digest = config_hash(config)

    def output(relative: str, value: pl.DataFrame) -> None:
        path = root / relative
        write_batches(path, [value], dict(value.schema))
        entries.append({"artifact_path": relative, "sha256": file_hash(path), "row_count": value.height, "config_hash": digest})

    for split in ("train", "valid", "test"):
        output(f"core/unified_events_{split}.parquet", frame.filter(pl.col("split") == split))
    output("model_specific/graph/events.parquet", interaction_edges(frame.lazy(), "retailrocket").collect())
    output("metadata/items.parquet", pl.DataFrame(schema=ITEMS_SCHEMA))
    state = PrefixState()
    negatives = pl.concat(list(iter_training_negatives(frame, config.point_in_time_policy.synthetic.negatives, is_primary, state=state)))
    negatives = annotate_future_conflicts(negatives, state)
    state.close()
    output("model_specific/flat/train_queries.parquet", training_queries(frame.lazy(), negatives.lazy(), "retailrocket").collect())
    train_ids = negatives.select(pl.col("context_event_id").alias("event_id")).unique().lazy()
    output("training/sequence_samples.parquet", pl.concat(list(iter_sequence_samples(frame.lazy(), "retailrocket", "train", tmp_path / "seq_train.sqlite", targets=train_ids))))
    for split in ("valid", "test"):
        pools = pl.concat([b.candidates for b in iter_evaluation_pools(frame, "retailrocket", config.point_in_time_policy.evaluation, split)])
        targets = pools.select(pl.col("eval_group_id").alias("event_id")).unique().lazy()
        output(f"evaluation/{split}/sequence_samples.parquet", pl.concat(list(iter_sequence_samples(frame.lazy(), "retailrocket", split, tmp_path / f"seq_{split}.sqlite", targets=targets))))
        output(f"evaluation/{split}/eval_candidates.parquet", pools)
        output(f"model_specific/flat/{split}_queries.parquet", evaluation_queries(pools.lazy(), split).collect())
    output("audit/generation_state.parquet", pl.DataFrame({"status": ["complete"], "config_hash": [digest], "mode": ["smoke"]}))
    manifest = pl.DataFrame(entries)
    write_batches(root / "audit/artifact_manifest.parquet", [manifest], dict(manifest.schema))
    return config, root
