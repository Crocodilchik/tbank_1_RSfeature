"""Fusion training with separately fitted OOF graph banks for every final seed."""
from __future__ import annotations
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import time
from typing import Iterator
import polars as pl
import torch
from src.config.loader import AppConfig
from src.generator.artifacts import ArtifactBundle
from src.models.adv2_gnn.heterogeneous_gnn import TemporalGraphIndex
from src.models.adv3_seq.pipeline import sequence_context
from src.models.adv3_seq.two_tower_sasrec import SequenceBatchBuilder
from src.models.adv4_fusion.graph_seq_fusion import FusionBatchBuilder, GraphSequentialFusion
from src.models.common.pipeline import run_neural
from src.models.common.temporal_graph import CheckpointBank, PrefixGraphEncoder, fit_graph_bank
from src.models.common.objectives import objective_variant
from src.utils.reproducibility import config_hash, file_hash


class PreparedFusionBuilder(FusionBatchBuilder):
    """Own one seed's frozen teacher and preserve all upstream costs/hashes."""

    def __init__(self, sequence: SequenceBatchBuilder, config: AppConfig, bundle: ArtifactBundle,
                 index: TemporalGraphIndex, root: Path, device: torch.device, smoke: bool,
                 reuse: bool = False) -> None:
        self.sequence, self.config, self.bundle, self.index = sequence, config, bundle, index
        self.root, self.device, self.smoke = root, device, smoke
        self.reuse = reuse
        self.seed = None
        self.encoder = None
        self.provenance = {}
        self.prepare_seed(config.runtime.seed)

    def prepare_seed(self, seed: int) -> None:
        """Rebuild the upstream bank for a changed final-model seed; never reuse future fit."""
        if seed == self.seed:
            return
        if self.encoder is not None:
            self.encoder.close()
        started = time.perf_counter()
        if self.reuse:
            bank = CheckpointBank(self.root / f"seed_{seed}", expected_config=config_hash(self.config),
                expected_parent=file_hash(self.bundle.manifest_path), expected_seed=seed)
            for row in bank.records:
                bank.verified_path(row)
        else:
            bank = fit_graph_bank(self.config, self.bundle, self.index, self.root / f"seed_{seed}", seed=seed, device=self.device, smoke=self.smoke)
        self.encoder = PrefixGraphEncoder(bank, self.index, self.device, tuple(self.config.runtime.training.gnn_fanout))
        self.seed = seed
        self.provenance[str(seed)] = {"checkpoint_manifest": str(bank.manifest_path), "sha256": file_hash(bank.manifest_path),
            "wall_seconds": time.perf_counter() - started, "snapshots": bank.snapshots, "cold_warmup": "zero_graph_vectors",
            "reused": self.reuse, "shared_fit_seconds": sum(row["fit_seconds"] for row in bank.records)}

    def close(self) -> None:
        """Release the active teacher; indexes are closed by their outer contexts."""
        if self.encoder is not None:
            self.encoder.close()


def run_fusion(config: AppConfig, dataset: str, *, processed_root: Path | None = None,
               smoke: bool = False, graph_enabled: bool = True, sequence_enabled: bool = True,
               metadata_enabled: bool = True, objective: str | None = None) -> Path:
    """Fit the full OOF Fusion or an explicitly isolated component ablation."""
    disabled = [name for name, enabled in (("graph", graph_enabled), ("sequence", sequence_enabled), ("metadata", metadata_enabled)) if not enabled]
    model_name = "adv4" + ("_" + "_".join(disabled) + "_off" if disabled else "")
    model_name, loss = objective_variant(model_name, config.runtime.models["adv4"].loss, objective)
    reuse = bool(disabled) or loss != (config.runtime.models["adv4"].loss or "sampled_bce")

    @contextmanager
    def context(bundle: ArtifactBundle, temporary: Path) -> Iterator[PreparedFusionBuilder]:
        mode = "smoke" if smoke else bundle.mode
        primary = config.project.experiments_dir / "adv4" / dataset / mode
        if reuse:
            manifest = pl.read_parquet(primary / "run_manifest.parquet").row(0, named=True)
            state = pl.read_parquet(primary / "run_state.parquet")["status"].item()
            if state != "complete" or any(manifest[key] != value for key, value in {
                "model": "adv4", "dataset": dataset, "mode": mode, "config_hash": config_hash(config),
                "parent_manifest_hash": file_hash(bundle.manifest_path)}.items()):
                raise ValueError("Fusion ablation requires a compatible completed primary donor")
        device = torch.device("cuda" if config.runtime.device == "auto" and torch.cuda.is_available() else "cpu" if config.runtime.device == "auto" else config.runtime.device)
        index = TemporalGraphIndex(bundle.graph(), temporary / "graph.sqlite")
        try:
            with sequence_context(bundle, temporary) as sequence:
                root = primary / "upstream"
                builder = PreparedFusionBuilder(sequence, config, bundle, index, root, device, smoke, reuse=reuse)
                try:
                    yield builder
                finally:
                    builder.close()
        finally:
            index.close()

    factory = partial(GraphSequentialFusion, graph_size=config.runtime.models["adv2"].hidden_size,
        loss=loss,
        graph_enabled=graph_enabled, sequence_enabled=sequence_enabled, metadata_enabled=metadata_enabled)
    return run_neural(config, dataset, model_name=model_name, model_factory=factory, builder_context=context,
        extra_parameters=lambda trial: {"attention_heads": trial.suggest_categorical("attention_heads", [2, 4]),
            "lambda_graph": trial.suggest_categorical("lambda_graph", [0., .1, 1.])}, processed_root=processed_root, smoke=smoke)
