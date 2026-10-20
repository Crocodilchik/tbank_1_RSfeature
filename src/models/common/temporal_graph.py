"""Temporal out-of-fold GraphSAGE checkpoint banks and frozen prefix features."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
import gc
import json
from pathlib import Path
import time

import polars as pl
import torch
from torch.nn import functional as F

from src.config.loader import AppConfig
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import write_batches
from src.models.adv2_gnn.heterogeneous_gnn import HeterogeneousGraphSAGE, TemporalGraphIndex
from src.models.adv2_gnn.pipeline import GraphBatchBuilder
from src.models.common.training import ResourceLimitError, fit_neural
from src.utils.reproducibility import executed_code_revision, config_hash, file_hash, seed_everything


class CheckpointBank:
    """Validated causal model registry: max_input<fit_cutoff<=snapshot<T."""

    def __init__(self, root: Path, *, expected_config: str, expected_parent: str,
                 expected_seed: int | None = None) -> None:
        self.root = root.resolve()
        self.manifest_path = root / "checkpoints.parquet"
        records = pl.read_parquet(self.manifest_path).sort("snapshot_ts")
        self.records = records.to_dicts()
        self.snapshots = records["snapshot_ts"].to_list()
        if not self.records or len(set(self.snapshots)) != len(self.snapshots):
            raise ValueError("checkpoint bank needs nonempty unique snapshots")
        for row in self.records:
            if expected_seed is not None and row["training_seed"] != expected_seed:
                raise ValueError("checkpoint bank training seed differs from requested experiment seed")
            if not row["max_input_ts"] < row["fit_cutoff_ts"] <= row["snapshot_ts"] or row["config_hash"] != expected_config or row["parent_manifest_hash"] != expected_parent:
                raise ValueError("checkpoint bank has incompatible cutoff/config/source provenance")
            if not (self.root / row["checkpoint_path"]).resolve().is_relative_to(self.root):
                raise ValueError("checkpoint path escapes bank root")
        settings = [json.loads(row["settings"]) for row in self.records]
        if any(item != settings[0] for item in settings):
            raise ValueError("checkpoint feature dimensions/settings must be identical across snapshots")
        self.settings = settings[0]

    def at(self, timestamp: int) -> dict | None:
        """Select the latest strictly earlier snapshot; exact boundaries use older fit."""
        index = bisect_left(self.snapshots, timestamp) - 1
        return self.records[index] if index >= 0 else None

    def verified_path(self, row: dict) -> Path:
        """Reject changed checkpoint weights before frozen inference."""
        path = self.root / row["checkpoint_path"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        return path


def fit_graph_bank(config: AppConfig, bundle: ArtifactBundle, index: TemporalGraphIndex,
                   root: Path, *, seed: int, device: torch.device, smoke: bool) -> CheckpointBank:
    """Train independent expanding-prefix models, including final train-only encoder."""
    started = time.perf_counter()
    train = bundle.scan("model_specific/flat/train_queries.parquet")
    cuts = train.select([pl.col("timestamp").quantile(q, interpolation="lower").alias(str(q)) for q in config.runtime.training.inner_train_quantiles]).collect().row(0)
    maximum = train.select(pl.col("timestamp").max()).collect().item()
    if maximum is None:
        raise ValueError("graph OOF bank requires nonempty training queries")
    cutoffs = sorted(set([int(cut) for cut in cuts] + [int(maximum) + 1]))
    settings = config.runtime.models["adv2"]
    records = []
    root.mkdir(parents=True, exist_ok=True)
    for cutoff in cutoffs:
        prefix = train.filter(pl.col("timestamp") < cutoff)
        if not prefix.select(pl.len()).collect().item():
            continue  # No fitted encoder exists before the warmup prefix.
        seed_everything(seed, config.runtime.deterministic)
        model = HeterogeneousGraphSAGE(hidden_size=settings.hidden_size, num_layers=settings.num_layers, dropout=.1)
        destination = root / f"snapshot_{cutoff}/model.pt"
        remaining = config.runtime.training.max_trial_hours * 3600 - (time.perf_counter() - started)
        epochs = 2 if smoke else min(5, config.runtime.training.max_neural_epochs)
        fit = fit_neural(model, GraphBatchBuilder(index, config.runtime.training.gnn_fanout), prefix, None, None, destination,
            model_name="oof_graph", seed=seed, device=device, epochs=epochs, batch_groups=min(64, config.runtime.training.neural_batch_size),
            learning_rate=.001, weight_decay=1e-5, max_seconds=remaining)
        if fit.complete_epochs != epochs:
            raise ResourceLimitError("OOF checkpoint did not complete every declared prefix epoch")
        records.append({"snapshot_ts": cutoff, "fit_cutoff_ts": cutoff, "max_input_ts": fit.fit_end_ts,
            "checkpoint_path": destination.relative_to(root).as_posix(), "sha256": file_hash(destination),
            "settings": json.dumps(model.settings, sort_keys=True), "training_seed": seed,
            "config_hash": config_hash(config), "parent_manifest_hash": file_hash(bundle.manifest_path),
            "code_revision": executed_code_revision(), "fit_seconds": fit.fit_seconds,
            "complete_epochs": fit.complete_epochs, "train_rows": fit.train_rows})
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pl.DataFrame(records)
    write_batches(root / "checkpoints.parquet", [frame], dict(frame.schema))
    return CheckpointBank(root, expected_config=config_hash(config), expected_parent=file_hash(bundle.manifest_path))


@dataclass(frozen=True)
class GraphEncoding:
    """CPU tensors from a frozen past teacher, with explicit snapshot lineage."""

    user: torch.Tensor
    items: torch.Tensor
    scores: torch.Tensor
    snapshot_ts: int | None


class PrefixGraphEncoder:
    """Keep one frozen checkpoint on device and at most128 query tensors on CPU."""

    def __init__(self, bank: CheckpointBank, index: TemporalGraphIndex, device: torch.device,
                 fanout: tuple[int, ...]) -> None:
        self.bank, self.index, self.device, self.fanout = bank, index, device, fanout
        self.width = bank.settings["hidden_size"]
        self.loaded_snapshot = None
        self.model = None
        self.encode = lru_cache(maxsize=128)(self._encode)

    def _encode(self, user: int, items: tuple[int, ...], timestamp: int, target_event: int) -> GraphEncoding:
        row = self.bank.at(timestamp)
        if row is None:
            return GraphEncoding(torch.zeros(self.width), torch.zeros(len(items), self.width), torch.zeros(len(items)), None)
        if row["snapshot_ts"] != self.loaded_snapshot:
            model = HeterogeneousGraphSAGE(**self.bank.settings).to(self.device)
            model.load_state_dict(torch.load(self.bank.verified_path(row), map_location=self.device, weights_only=True))
            model.eval().requires_grad_(False)
            self.model, self.loaded_snapshot = model, row["snapshot_ts"]
        unique = sorted(set(items))
        graph = self.index.sample(user, unique, timestamp, self.fanout, target_event_id=target_event).data.to(self.device)
        with torch.no_grad():
            values = self.model.encode(graph)
            user_vector = F.normalize(values["User"][graph["User"].seed_mask][0], dim=-1)
            item_vectors = F.normalize(values["Item"][graph["Item"].seed_mask], dim=-1)
            positions = {item: index for index, item in enumerate(unique)}
            item_vectors = item_vectors[torch.tensor([positions[item] for item in items], device=self.device)]
            scores = (item_vectors * user_vector).sum(-1) * self.model.log_temperature.exp().clamp(max=100.) + self.model.bias
        return GraphEncoding(user_vector.cpu(), item_vectors.cpu(), scores.cpu(), row["snapshot_ts"])

    def close(self) -> None:
        """Release cached features and the active frozen model."""
        self.encode.cache_clear()
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
