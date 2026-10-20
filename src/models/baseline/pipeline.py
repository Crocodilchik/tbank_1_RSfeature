"""Shared full/smoke GBDT experiment execution with locked-pool evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import optuna
import polars as pl
import psutil
import pyarrow.parquet as pq
from scipy.special import expit

from src.config.loader import AppConfig
from src.evaluation.metrics import PlattCalibrator, PREDICTION_SCHEMA, compare_models, per_group_metrics
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import runtime_workspace, write_batches, write_lazy
from src.models.baseline.booster import fit_booster, temporal_feature_selection
from src.models.baseline.feature_engineering import ENTITY_KEYS, QUERY_KEYS, FeatureSpec, build_features, feature_columns, manual_specs, prepare_inputs, reference_features
from src.utils.reproducibility import executed_code_revision, config_hash, environment_info, file_hash

LOGGER = logging.getLogger(__name__)


def build_feature_artifact(bundle: ArtifactBundle, config: AppConfig, dataset: str,
                           specs: Sequence[FeatureSpec], *, shards: int = 16) -> Path:
    """Spool prepared inputs, compute bounded entity shards and publish numeric Parquet."""
    if shards <= 0:
        raise ValueError("feature shards must be positive")
    signature = hashlib.sha256(json.dumps([asdict(s) for s in specs], sort_keys=True).encode()).hexdigest()[:16]
    output = bundle.root / "model_specific" / "flat" / f"features_{signature}.parquet"
    events = bundle.events()
    metadata = bundle.scan("metadata/items.parquet")
    queries = pl.concat([bundle.scan(f"model_specific/flat/{split}_queries.parquet") for split in ("train", "valid", "test")])
    policy = config.point_in_time_policy.target_encoding
    with runtime_workspace(config.project.root) as temporary:
        real, prepared = prepare_inputs(events, queries, metadata, dataset, policy.empty_prior)
        write_lazy(temporary / "prepared_events.parquet", real)
        write_lazy(temporary / "prepared_queries.parquet", prepared)
        real = pl.scan_parquet(temporary / "prepared_events.parquet")
        prepared = pl.scan_parquet(temporary / "prepared_queries.parquet")
        by_entity: dict[str, list[FeatureSpec]] = defaultdict(list)
        for spec in specs:
            by_entity[spec.entity].append(spec)
        result = prepared.drop("_prior")
        for entity, entity_specs in by_entity.items():
            paths = []
            keys = ENTITY_KEYS[entity]
            for shard in range(shards):
                LOGGER.info("features dataset=%s entity=%s shard=%s/%s", dataset, entity, shard + 1, shards)
                predicate = pl.struct(keys).hash(seed=config.runtime.seed).mod(shards) == shard
                q = prepared.filter(predicate)
                e = real.filter(predicate)
                block = build_features(events, queries, metadata, dataset, entity_specs, smoothing_alpha=policy.smoothing_alpha,
                    min_history=policy.min_history_events, empty_prior=policy.empty_prior, prepared_inputs=(e, q))
                path = temporary / f"{entity}_{shard}.parquet"
                write_lazy(path, block.select(*QUERY_KEYS, *[s.name for s in entity_specs]))
                paths.append(path)
            path = temporary / f"merged_{entity}.parquet"
            write_lazy(path, result.join(pl.scan_parquet(paths), on=QUERY_KEYS, how="left"))
            result = pl.scan_parquet(path)
        write_lazy(output, result)
    manifest = pl.DataFrame({"artifact_path": [output.relative_to(bundle.root).as_posix()], "sha256": [file_hash(output)],
        "parent_manifest_hash": [file_hash(bundle.manifest_path)], "config_hash": [config_hash(config)],
        "code_revision": [executed_code_revision()], "specifications": [json.dumps([asdict(s) for s in specs], sort_keys=True)], "mode": [bundle.mode]})
    write_batches(output.with_name(output.stem + "_manifest.parquet"), [manifest], dict(manifest.schema))
    audit_features(bundle, config, dataset, output, specs)
    return output


def audit_features(bundle: ArtifactBundle, config: AppConfig, dataset: str, features_path: Path,
                    specs: Sequence[FeatureSpec]) -> None:
    """Compare up to 100 targets per split against independent strict prefix formulas."""
    frame = pl.scan_parquet(features_path)
    samples = []
    for split in ("train", "valid", "test"):
        samples.append(frame.filter((pl.col("split") == split) & (pl.col("target") == 1))
            .sort(pl.col("query_group_id").hash(seed=42)).limit(100).collect(engine="streaming"))
    selected = pl.concat(samples)
    input_queries = selected.select("query_group_id", "user_id", "candidate_item_id", "timestamp", "target", "split", "dataset_source")
    policy = config.point_in_time_policy.target_encoding
    reference = reference_features(bundle.events(), input_queries.lazy(), bundle.scan("metadata/items.parquet"), dataset, specs,
        smoothing_alpha=policy.smoothing_alpha, min_history=policy.min_history_events, empty_prior=policy.empty_prior)
    columns = feature_columns(specs)
    left = selected.sort(QUERY_KEYS).select(columns).to_numpy()
    right = reference.sort(QUERY_KEYS).select(columns).to_numpy()
    passed = left.shape == right.shape and np.allclose(left, right, rtol=1e-6, atol=1e-8)
    audit = pl.DataFrame({"gate": ["manual_features_independent_prefix"], "passed": [passed], "checked_targets": [selected.height]})
    write_batches(features_path.with_name(features_path.stem + "_gates.parquet"), [audit], dict(audit.schema))
    if not passed:
        raise ValueError("manual PIT feature values differ from independent prefix formulas")


def collect_budgeted(frame: pl.LazyFrame, n_features: int, *, fraction: float = .65) -> pl.DataFrame:
    """Refuse a dense booster fit that would exceed measured available RAM.

    This is a resource failure, never permission to silently reduce the data.
    The generated Parquet remains available for a larger-memory host.
    """
    rows = frame.select(pl.len()).collect().item()
    required = rows * (n_features * 4 + 64) * 4
    available = psutil.virtual_memory().available
    if required > available * fraction:
        raise MemoryError(f"full GBDT fit needs an estimated {required / 2**30:.2f} GiB working RAM; available {available / 2**30:.2f} GiB. No training rows were dropped. Use a host with more free RAM.")
    return frame.collect(engine="streaming")


def _prediction_frame(queries: pl.DataFrame, scores: np.ndarray, probabilities: np.ndarray,
                      model: str, seed: int) -> pl.DataFrame:
    return queries.select(pl.col("query_group_id").alias("eval_group_id"), "candidate_item_id").with_columns(
        pl.Series("score", scores), pl.Series("probability", probabilities), pl.lit(model).alias("model"),
        pl.lit(seed, dtype=pl.Int64).alias("training_seed")).cast(PREDICTION_SCHEMA)


def run_gbdt(config: AppConfig, dataset: str, *, processed_root: Path | None = None,
              model_name: str = "baseline", specs: Sequence[FeatureSpec] | None = None,
              selected_features: int | None = None, smoke: bool = False,
              backend: str | None = None) -> Path:
    """Tune on early valid, calibrate on late valid, then score every locked test group."""
    started = time.perf_counter()
    bundle = ArtifactBundle(processed_root or config.dataset(dataset).processed_dir, config_hash(config))
    if bundle.mode != "full" and not smoke:
        raise ValueError("smoke/diagnostic inputs require an explicitly marked smoke experiment")
    mode = "smoke" if smoke else bundle.mode
    root = config.project.experiments_dir / model_name / dataset / mode
    root.mkdir(parents=True, exist_ok=True)
    state = pl.DataFrame({"status": ["running"], "config_hash": [config_hash(config)]})
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    specs = list(manual_specs(dataset, config.runtime.models["baseline"].windows_days) if specs is None else specs)
    features_path = build_feature_artifact(bundle, config, dataset, specs, shards=2 if smoke else 16)
    columns = feature_columns(specs)
    all_features = pl.scan_parquet(features_path)
    train_source = all_features.filter(pl.col("split") == "train")
    selection_cutoff = train_source.select(pl.col("timestamp").quantile(.85, interpolation="lower")).collect().item()
    selection_prefix = train_source.filter(pl.col("timestamp") < selection_cutoff)
    max_groups = max(1, 50_000 // (config.point_in_time_policy.synthetic.negatives.negatives_per_positive + 1))
    selection_ids = selection_prefix.select("query_group_id").unique().sort(pl.col("query_group_id").hash(seed=config.runtime.seed)).limit(max_groups)
    selection = collect_budgeted(selection_prefix.join(selection_ids, on="query_group_id", how="semi"), len(columns))
    columns = temporal_feature_selection(selection, columns, seed=config.runtime.seed, top_k=selected_features, already_prefix=True)
    keep = list(dict.fromkeys(["query_group_id", "candidate_item_id", "user_id", "timestamp", "target", "split", "dataset_source", *columns]))
    train = collect_budgeted(train_source.select(keep), len(columns))
    valid = collect_budgeted(all_features.filter(pl.col("split") == "valid").select(keep), len(columns))
    cutoff = int(valid["timestamp"].quantile(config.runtime.training.valid_hpo_fraction, interpolation="lower"))
    hpo_valid, calibration_valid = valid.filter(pl.col("timestamp") < cutoff), valid.filter(pl.col("timestamp") >= cutoff)
    if hpo_valid.is_empty() or calibration_valid.is_empty():
        raise ValueError("valid does not contain separate timestamp blocks for tuning and calibration")
    backend = backend or config.runtime.models["baseline"].backend
    training = config.runtime.training
    trials = 2 if smoke else training.hpo_trials
    maximum_iterations = 30 if smoke else training.max_booster_iterations
    truth_valid = bundle.scan("evaluation/valid/eval_candidates.parquet").join(
        hpo_valid.select(pl.col("query_group_id").alias("eval_group_id")).unique().lazy(), on="eval_group_id", how="semi")

    def objective(trial: optuna.Trial) -> float:
        params = {"depth": trial.suggest_categorical("depth", [4, 6, 8, 10]), "learning_rate": trial.suggest_float("learning_rate", .01, .2, log=True),
                  "l2": trial.suggest_float("l2", 1., 20., log=True)}
        model = fit_booster(train, hpo_valid, columns, backend=backend, seed=config.runtime.seed, iterations=maximum_iterations,
            patience=min(training.booster_patience, 5) if smoke else training.booster_patience,
            max_seconds=training.max_trial_hours * 3600, **params)
        scores = model.predict(hpo_valid)
        predictions = _prediction_frame(hpo_valid, scores, expit(scores), model_name, config.runtime.seed)
        value = compare_models(truth_valid, {model_name: predictions})["NDCG@10"].item()
        trial.set_user_attr("fit_seconds", time.perf_counter() - trial_start[0])
        trial.set_user_attr("iterations", model.estimator.tree_count_ if backend == "catboost" else model.estimator.current_iteration())
        return value

    trial_start = [0.]
    def timed_objective(trial: optuna.Trial) -> float:
        trial_start[0] = time.perf_counter()
        return objective(trial)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.runtime.seed))
    # Each fit enforces its declared budget. A global timeout could silently stop
    # before n_trials because validation and scoring also consume wall time.
    study.optimize(timed_objective, n_trials=trials)
    if len(study.trials) != trials:
        raise RuntimeError("HPO ended before the declared trial count; full run is incomplete")
    best_value = study.best_value
    best = min((t for t in study.trials if abs(t.value - best_value) <= 1e-8), key=lambda t: (t.user_attrs["fit_seconds"], t.number))
    tuning = pl.DataFrame({"trial": [t.number for t in study.trials], "NDCG@10": [t.value for t in study.trials],
        "parameters": [json.dumps(t.params, sort_keys=True) for t in study.trials], "fit_seconds": [t.user_attrs["fit_seconds"] for t in study.trials]})
    write_batches(root / "hpo_trials.parquet", [tuning], dict(tuning.schema))
    # Test data is materialized only after HPO has fixed parameters and iteration count.
    test = collect_budgeted(all_features.filter(pl.col("split") == "test").select(keep), len(columns))
    seeds = [config.runtime.seed] if smoke else config.runtime.training_seeds
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        model = fit_booster(train, None, columns, backend=backend, seed=seed, iterations=int(best.user_attrs["iterations"]), **best.params)
        model.save(seed_root)
        calibration_scores = model.predict(calibration_valid)
        calibration = PlattCalibrator.fit(calibration_scores, calibration_valid["target"].to_list(), calibration_valid["timestamp"].to_list(), split="valid", calibration_start_ts=cutoff)
        calibration_table = pl.DataFrame([asdict(calibration)])
        write_batches(seed_root / "calibration.parquet", [calibration_table], dict(calibration_table.schema))
        scores = model.predict(test)
        predictions = _prediction_frame(test, scores, calibration.predict(scores, target_min_ts=int(test["timestamp"].min())), model_name, seed)
        write_batches(seed_root / "test_predictions.parquet", [predictions], PREDICTION_SCHEMA)
        metrics = compare_models(bundle.scan("evaluation/test/eval_candidates.parquet"), {model_name: predictions}).with_columns(
            pl.lit(dataset).alias("dataset"), pl.lit(mode).alias("mode"), pl.lit(seed).alias("training_seed"), pl.lit(config_hash(config)).alias("config_hash"))
        write_batches(seed_root / "test_metrics.parquet", [metrics], dict(metrics.schema))
    provenance = pl.DataFrame({"dataset": [dataset], "model": [model_name], "mode": [mode], "config_hash": [config_hash(config)],
        "parent_manifest_hash": [file_hash(bundle.manifest_path)], "feature_hash": [file_hash(features_path)],
        "code_revision": [executed_code_revision()], "wall_seconds": [time.perf_counter() - started], "training_seeds": [seeds], "feature_selection_rows": [selection.height],
        "hpo_attempted_trials": [len(study.trials)], "created_at_utc": [datetime.now(timezone.utc).isoformat()],
        "environment": [json.dumps(environment_info(), sort_keys=True)], "device": ["cpu"]})
    write_batches(root / "run_manifest.parquet", [provenance], dict(provenance.schema))
    state = state.with_columns(pl.lit("complete").alias("status"))
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    LOGGER.info("%s complete dataset=%s mode=%s output=%s", model_name, dataset, mode, root)
    return root
