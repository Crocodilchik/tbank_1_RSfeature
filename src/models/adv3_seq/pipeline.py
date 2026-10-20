"""Sequential model sharing the exact GraphSAGE tuning/evaluation protocol."""

from contextlib import contextmanager
from pathlib import Path
from functools import partial
from typing import Iterator

from src.config.loader import AppConfig
from src.generator.artifacts import ArtifactBundle
from src.generator.metadata import MetadataIndex
from src.models.adv3_seq.two_tower_sasrec import SequenceBatchBuilder, SequenceIndex, SequentialTwoTower
from src.models.common.pipeline import run_neural
from src.models.common.objectives import objective_variant


@contextmanager
def sequence_context(bundle: ArtifactBundle, temporary: Path) -> Iterator[SequenceBatchBuilder]:
    """Open generator sequences and dated metadata in disposable indexed stores."""
    import polars as pl
    samples = pl.concat([bundle.scan("training/sequence_samples.parquet"), *[
        bundle.scan(f"evaluation/{split}/sequence_samples.parquet") for split in ("valid", "test")]])
    sequences = SequenceIndex(samples, temporary / "sequences.sqlite")
    try:
        metadata = MetadataIndex(bundle.scan("metadata/items.parquet").filter(pl.col("field_name").is_in(["category_id", "brand_id", "genres"])), temporary / "metadata.sqlite")
        try:
            builder = SequenceBatchBuilder(sequences, metadata)
            try:
                yield builder
            finally:
                builder.close()
        finally:
            metadata.close()
    finally:
        sequences.close()


def run_sequence(config: AppConfig, dataset: str, *, processed_root: Path | None = None, smoke: bool = False,
                 objective: str | None = None) -> Path:
    """Tune a causal Two-Tower and calibrate/score the unchanged locked pools."""
    model_name, loss = objective_variant("adv3", config.runtime.models["adv3"].loss, objective)
    return run_neural(config, dataset, model_name=model_name, model_factory=partial(SequentialTwoTower, loss=loss), builder_context=sequence_context,
        extra_parameters=lambda trial: {"attention_heads": trial.suggest_categorical("attention_heads", [2, 4])},
        processed_root=processed_root, smoke=smoke)
