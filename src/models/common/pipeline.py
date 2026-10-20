"""Shared full-data neural search, checkpoint, calibration and locked-pool protocol."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import time
from typing import Callable, Any
from contextlib import AbstractContextManager

import numpy as np
import optuna
import polars as pl
import torch

from src.config.loader import AppConfig
from src.evaluation.metrics import PlattCalibrator, PREDICTION_SCHEMA, compare_models
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import runtime_workspace, write_batches
from src.models.common.training import BatchBuilder, ResourceLimitError, fit_neural, predict_batches
from src.utils.reproducibility import executed_code_revision, config_hash, environment_info, file_hash, seed_everything

LOGGER = logging.getLogger(__name__)


def run_neural(config: AppConfig, dataset: str, *, model_name: str,
               model_factory: Callable[..., torch.nn.Module],
               builder_context: Callable[[ArtifactBundle, Path], AbstractContextManager[BatchBuilder]],
               extra_parameters: Callable[[optuna.Trial], dict[str, Any]] | None = None,
               processed_root: Path | None = None, smoke: bool = False) -> Path:
    """Tune the neural model on early valid, calibrate on late valid and score all test rows."""
    start = time.perf_counter()
    bundle = ArtifactBundle(processed_root or config.dataset(dataset).processed_dir, config_hash(config))
    if bundle.mode != "full" and not smoke:
        raise ValueError("smoke/diagnostic data require an explicit smoke model run")
    mode = "smoke" if smoke else bundle.mode
    root = config.project.experiments_dir / model_name / dataset / mode
    root.mkdir(parents=True, exist_ok=True)
    state = pl.DataFrame({"status": ["running"], "config_hash": [config_hash(config)]})
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    device = torch.device("cuda" if config.runtime.device == "auto" and torch.cuda.is_available() else "cpu" if config.runtime.device == "auto" else config.runtime.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_groups = min(64, config.runtime.training.neural_batch_size)
    train = bundle.scan("model_specific/flat/train_queries.parquet")
    valid = bundle.scan("model_specific/flat/valid_queries.parquet").sort(["timestamp", "query_group_id", "candidate_item_id"])
    raw_cutoff = valid.select(pl.col("timestamp").quantile(config.runtime.training.valid_hpo_fraction, interpolation="lower")).collect().item()
    if raw_cutoff is None:
        raise ValueError("neural validation requires nonempty locked query groups")
    cutoff = int(raw_cutoff)
    hpo_valid, calibration_valid = valid.filter(pl.col("timestamp") < cutoff), valid.filter(pl.col("timestamp") >= cutoff)
    if hpo_valid.select(pl.len()).collect().item() == 0 or calibration_valid.select(pl.len()).collect().item() == 0:
        raise ValueError("neural validation needs separate tuning/calibration timestamp blocks")
    truth_valid = bundle.scan("evaluation/valid/eval_candidates.parquet").filter(pl.col("target_timestamp") < cutoff)
    trial_count = 2 if smoke else config.runtime.training.hpo_trials
    hpo_deadline = start + trial_count * config.runtime.training.max_trial_hours * 3600
    with runtime_workspace(config.project.root) as temporary:
        with builder_context(bundle, temporary) as builder:

            def objective(trial: optuna.Trial) -> float:
                if time.perf_counter() >= hpo_deadline:
                    raise ResourceLimitError("total HPO budget including upstream pretraining is exhausted")
                seed_everything(config.runtime.seed, config.runtime.deterministic)
                settings = {"hidden_size": trial.suggest_categorical("hidden_size", [64, 128]),
                    "num_layers": trial.suggest_categorical("num_layers", [1, 2]), "dropout": trial.suggest_categorical("dropout", [0., .1, .2])}
                if extra_parameters is not None:
                    settings.update(extra_parameters(trial))
                lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
                decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                model = model_factory(**settings)
                fit = fit_neural(model, builder, train, hpo_valid, truth_valid, temporary / f"trial_{trial.number}/best.pt", model_name=model_name,
                    seed=config.runtime.seed, device=device, epochs=2 if smoke else config.runtime.training.max_neural_epochs,
                    batch_groups=batch_groups, learning_rate=lr, weight_decay=decay, patience=config.runtime.training.neural_patience,
                    max_seconds=min(config.runtime.training.max_trial_hours * 3600, hpo_deadline - time.perf_counter()))
                trial.set_user_attr("best_epoch", fit.best_epoch)
                trial.set_user_attr("fit_seconds", fit.fit_seconds)
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return fit.best_ndcg

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.runtime.seed))
            study.optimize(objective, n_trials=trial_count, catch=(ResourceLimitError,))
            trials = pl.DataFrame({"trial": [t.number for t in study.trials], "state": [t.state.name for t in study.trials],
                "NDCG@10": [t.value for t in study.trials], "parameters": [json.dumps(t.params, sort_keys=True) for t in study.trials]}).with_columns(pl.col("NDCG@10").cast(pl.Float64))
            write_batches(root / "hpo_trials.parquet", [trials], dict(trials.schema))
            completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                raise ResourceLimitError(f"no {model_name} trial completed one full epoch within the resource budget")
            maximum = max(trial.value for trial in completed)
            best = min((trial for trial in completed if abs(trial.value - maximum) <= 1e-8), key=lambda t: (t.user_attrs["fit_seconds"], t.number))
            # Test inputs are scored after the search is fixed; labels are evaluator-owned.
            test = bundle.scan("model_specific/flat/test_queries.parquet").sort(["timestamp", "query_group_id", "candidate_item_id"])
            seeds = [config.runtime.seed] if smoke else config.runtime.training_seeds
            for seed in seeds:
                if hasattr(builder, "prepare_seed"):
                    builder.prepare_seed(seed)
                seed_everything(seed, config.runtime.deterministic)
                seed_root = root / f"seed_{seed}"
                params = dict(best.params)
                lr, decay = params.pop("learning_rate"), params.pop("weight_decay")
                model = model_factory(**params)
                fit = fit_neural(model, builder, train, None, None, seed_root / "model.pt", model_name=model_name, seed=seed, device=device,
                    epochs=best.user_attrs["best_epoch"], batch_groups=batch_groups, learning_rate=lr, weight_decay=decay)
                calibration_path = temporary / f"calibration_{seed}.parquet"
                write_batches(calibration_path, predict_batches(model, builder, calibration_valid, model_name=model_name, seed=seed, device=device,
                    fit_end_ts=fit.fit_end_ts), PREDICTION_SCHEMA)
                labeled_path = temporary / f"calibration_labeled_{seed}.parquet"
                calibration_data = pl.scan_parquet(calibration_path).join(
                    calibration_valid.select(pl.col("query_group_id").alias("eval_group_id"), "candidate_item_id", "target", "timestamp"),
                    on=["eval_group_id", "candidate_item_id"]).select("score", "target", "timestamp")
                calibration_data.sink_parquet(labeled_path)
                calibration = PlattCalibrator.fit_frame(pl.scan_parquet(labeled_path), split="valid", calibration_start_ts=cutoff)
                calibration_frame = pl.DataFrame([asdict(calibration)])
                write_batches(seed_root / "calibration.parquet", [calibration_frame], dict(calibration_frame.schema))
                test_start = int(test.select(pl.col("timestamp").min()).collect().item())
                write_batches(seed_root / "test_predictions.parquet", predict_batches(model, builder, test, model_name=model_name, seed=seed, device=device,
                    fit_end_ts=fit.fit_end_ts, probabilities=lambda scores: calibration.predict(scores, target_min_ts=test_start)), PREDICTION_SCHEMA)
                metrics = compare_models(bundle.scan("evaluation/test/eval_candidates.parquet"), {model_name: pl.scan_parquet(seed_root / "test_predictions.parquet")}).with_columns(
                    pl.lit(dataset).alias("dataset"), pl.lit(mode).alias("mode"), pl.lit(seed).alias("training_seed"), pl.lit(config_hash(config)).alias("config_hash"))
                write_batches(seed_root / "test_metrics.parquet", [metrics], dict(metrics.schema))
                metadata = pl.DataFrame({"settings": [json.dumps(model.settings, sort_keys=True)], "fit_end_ts": [fit.fit_end_ts], "training_seed": [seed],
                    "model_sha256": [file_hash(seed_root / "model.pt")], "builder_provenance": [json.dumps(getattr(builder, "provenance", {}), sort_keys=True)], "mode": [mode], "config_hash": [config_hash(config)]})
                write_batches(seed_root / "model_metadata.parquet", [metadata], dict(metadata.schema))
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    manifest = pl.DataFrame({"dataset": [dataset], "model": [model_name], "mode": [mode], "config_hash": [config_hash(config)], "parent_manifest_hash": [file_hash(bundle.manifest_path)],
        "code_revision": [executed_code_revision()], "created_at_utc": [datetime.now(timezone.utc).isoformat()], "environment": [json.dumps(environment_info(), sort_keys=True)], "device": [str(device)], "batch_query_groups": [batch_groups], "hash_slots": [65536], "upstream_provenance": [json.dumps(getattr(builder, "provenance", {}), sort_keys=True)], "wall_seconds": [time.perf_counter() - start]})
    manifest = manifest.with_columns(pl.lit(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, dtype=pl.Int64).alias("cuda_peak_allocated_bytes"),
        pl.lit(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None, dtype=pl.Int64).alias("cuda_peak_reserved_bytes"))
    write_batches(root / "run_manifest.parquet", [manifest], dict(manifest.schema))
    state = state.with_columns(pl.lit("complete").alias("status"))
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    LOGGER.info("neural complete dataset=%s mode=%s", dataset, mode)
    return root
