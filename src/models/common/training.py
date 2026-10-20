"""Streaming supervised query groups, neural fitting and exact locked-pool scoring."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn
from scipy.special import expit

from src.evaluation.metrics import PREDICTION_SCHEMA, compare_models
from src.generator.exporters.flat import QUERY_SCHEMA
from src.generator.storage import write_batches

LOGGER = logging.getLogger(__name__)


class ResourceLimitError(RuntimeError):
    """A full neural pass could not finish within the declared compute budget."""


@dataclass(frozen=True)
class QueryGroup:
    """One real context and its complete frozen supervised candidate rows."""

    group_id: int
    user_id: int
    timestamp: int
    rows: pl.DataFrame


def iter_query_groups(queries: pl.LazyFrame, *, expected_size: int | None = None) -> Iterator[QueryGroup]:
    """Read chronological groups across Parquet batch boundaries without a global sort."""
    current: list[dict] = []
    previous_key = None
    group_id = None

    def finish(rows: list[dict]) -> QueryGroup:
        frame = pl.DataFrame(rows, schema=QUERY_SCHEMA)
        if (expected_size is not None and frame.height != expected_size) or frame["user_id"].n_unique() != 1 or frame["timestamp"].n_unique() != 1 or frame["candidate_item_id"].n_unique() != frame.height or frame["target"].sum() != 1 or not frame["target"].is_in([0, 1]).all():
            raise ValueError("incomplete/invalid neural query group")
        return QueryGroup(rows[0]["query_group_id"], rows[0]["user_id"], rows[0]["timestamp"], frame)

    for batch in queries.select(QUERY_SCHEMA.keys()).collect_batches(chunk_size=8192, maintain_order=True):
        for row in batch.iter_rows(named=True):
            key = (row["timestamp"], row["query_group_id"], row["candidate_item_id"])
            if previous_key is not None and key <= previous_key:
                raise ValueError("query rows must be unique and sorted by timestamp/group/candidate")
            previous_key = key
            if group_id is not None and group_id != row["query_group_id"]:
                yield finish(current)
                current = []
            group_id = row["query_group_id"]
            current.append(row)
    if current:
        yield finish(current)


def group_batches(queries: pl.LazyFrame, batch_groups: int, *, expected_size: int | None = None) -> Iterator[list[QueryGroup]]:
    """Batch complete contexts without introducing additional in-batch negatives."""
    if batch_groups <= 0:
        raise ValueError("batch_groups must be positive")
    batch = []
    for group in iter_query_groups(queries, expected_size=expected_size):
        batch.append(group)
        if len(batch) == batch_groups:
            yield batch
            batch = []
    if batch:
        yield batch


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value.to(device)


@dataclass
class NeuralBatch:
    """Model inputs are separate from labels and evaluator-owned row identifiers."""

    inputs: Any
    labels: Tensor
    rows: pl.DataFrame

    def to(self, device: torch.device) -> NeuralBatch:
        """Move only tensors/graph inputs; scorer keys stay on the host."""
        return NeuralBatch(_move(self.inputs, device), self.labels.to(device), self.rows)


BatchBuilder = Callable[[Sequence[QueryGroup]], NeuralBatch]


@dataclass(frozen=True)
class NeuralFitResult:
    """Measured complete-epoch fit, including a possible discarded partial epoch."""

    best_epoch: int
    best_ndcg: float | None
    complete_epochs: int
    fit_seconds: float
    train_rows: int
    fit_end_ts: int


def predict_batches(model: nn.Module, builder: BatchBuilder, queries: pl.LazyFrame,
                    *, model_name: str, seed: int, device: torch.device,
                    batch_groups: int = 3, fit_end_ts: int,
                    probabilities: Callable[[np.ndarray], np.ndarray] | None = None) -> Iterator[pl.DataFrame]:
    """Score every row of each locked pool with frozen weights and no target input."""
    model.eval()
    with torch.no_grad():
        for groups in group_batches(queries, batch_groups, expected_size=100):
            if min(group.timestamp for group in groups) <= fit_end_ts:
                raise ValueError("inference target must be strictly after neural fit_end_ts")
            batch = builder(groups).to(device)
            output = model(batch.inputs)
            if not isinstance(output, Tensor):
                raise ValueError("neural scorer returned misaligned/nonfinite predictions")
            scores = output.detach().cpu().numpy().astype(np.float64)
            if scores.shape != (batch.rows.height,) or not np.isfinite(scores).all():
                raise ValueError("neural scorer returned misaligned/nonfinite predictions")
            p = expit(scores) if probabilities is None else probabilities(scores)
            yield batch.rows.select(pl.col("query_group_id").alias("eval_group_id"), "candidate_item_id").with_columns(
                pl.Series("score", scores), pl.Series("probability", p), pl.lit(model_name).alias("model"),
                pl.lit(seed, dtype=pl.Int64).alias("training_seed")).cast(PREDICTION_SCHEMA)


def _save_state(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.pt")
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, partial)
    partial.replace(path)


def fit_neural(model: nn.Module, builder: BatchBuilder, train: pl.LazyFrame,
               valid: pl.LazyFrame | None, valid_truth: pl.LazyFrame | None,
               checkpoint: Path, *, model_name: str, seed: int, device: torch.device,
               epochs: int, batch_groups: int = 64, learning_rate: float = .001,
               weight_decay: float = 1e-5, patience: int = 3,
               max_seconds: float | None = None) -> NeuralFitResult:
    """Fit all train rows each epoch; stop on earlier valid or declared trial deadline.

    A budget exhausted before one full epoch is a failed cell. Partial later
    epochs are discarded and the best complete checkpoint is restored.
    """
    if epochs <= 0 or batch_groups <= 0 or patience <= 0 or (valid is None) != (valid_truth is None):
        raise ValueError("invalid epoch/batch/patience or validation arguments")
    stats = train.select(pl.len().alias("n"), pl.col("timestamp").max().alias("end")).collect().row(0)
    if not stats[0]:
        raise ValueError("neural training requires nonempty supervised rows")
    fit_end = int(stats[1])
    if valid is not None:
        valid_start = valid.select(pl.col("timestamp").min()).collect().item()
        if valid_start is None or valid_start <= fit_end:
            raise ValueError("neural validation must be nonempty and strictly after train")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=False)
    started = time.perf_counter()
    deadline = float("inf") if max_seconds is None else started + max_seconds
    best_score, best_epoch, stale, complete = -float("inf"), 0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        seen, total_loss = 0, 0.
        timed_out = False
        for groups in group_batches(train, batch_groups):
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            batch = builder(groups).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.inputs)
            if logits.shape != batch.labels.shape:
                raise ValueError("neural train scores and labels are misaligned")
            loss = (model.training_loss(batch.inputs, batch.labels, logits) if hasattr(model, "training_loss") else
                torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite neural training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            seen += batch.rows.height
            total_loss += float(loss.detach()) * batch.rows.height
        if timed_out:
            if complete == 0:
                raise ResourceLimitError("trial budget expired before one full neural epoch; no full-data result exists")
            break
        if seen != stats[0]:
            raise ValueError("neural epoch did not consume every supervised training row")
        complete += 1
        if valid is not None:
            predictions_path = checkpoint.with_name(f"valid_epoch_{epoch}.parquet")
            write_batches(predictions_path, predict_batches(model, builder, valid, model_name=model_name, seed=seed, device=device,
                fit_end_ts=fit_end), PREDICTION_SCHEMA)
            score = compare_models(valid_truth, {model_name: pl.scan_parquet(predictions_path)})["NDCG@10"].item()
        else:
            score = float(epoch)  # Exact final retraining: keep the requested last complete epoch.
        LOGGER.info("neural %s epoch=%s rows=%s mean_BCE=%.6f valid_NDCG10=%s", model_name, epoch, seen, total_loss / seen, score if valid is not None else "not_evaluated")
        if score > best_score + 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            _save_state(model, checkpoint)
        else:
            stale += 1
        if valid is not None and stale >= patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return NeuralFitResult(best_epoch, best_score if valid is not None else None, complete,
        time.perf_counter() - started, int(stats[0]), fit_end)
