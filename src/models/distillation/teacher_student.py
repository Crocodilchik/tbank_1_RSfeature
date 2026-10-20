"""Audited OOF teacher artifacts, score-feature stacking and soft-target students."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import time
from typing import Iterator, Literal, Sequence

import numpy as np
import optuna
import polars as pl
import torch
from scipy.special import expit

from src.config.loader import AppConfig
from src.evaluation.metrics import PlattCalibrator, PREDICTION_SCHEMA, compare_models, log_loss
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import runtime_workspace, write_batches
from src.models.adv2_gnn.heterogeneous_gnn import TemporalGraphIndex
from src.models.baseline.booster import BoosterModel, fit_booster, temporal_feature_selection
from src.models.baseline.feature_engineering import FeatureSpec, feature_columns, manual_specs
from src.models.baseline.pipeline import build_feature_artifact, collect_budgeted, _prediction_frame
from src.models.common.temporal_graph import CheckpointBank, PrefixGraphEncoder, fit_graph_bank
from src.models.common.training import ResourceLimitError, iter_query_groups
from src.utils.reproducibility import executed_code_revision, config_hash, environment_info, file_hash

LOGGER = logging.getLogger(__name__)
Mode = Literal["score_feature_stacking", "soft_target_distillation"]
TEACHER_SCHEMA = {"query_group_id": pl.Int64, "candidate_item_id": pl.Int64, "user_id": pl.Int64,
    "timestamp": pl.Int64, "split": pl.String, "teacher_score": pl.Float64,
    "teacher_probability": pl.Float64, "teacher_available": pl.Boolean,
    "teacher_snapshot_ts": pl.Int64, "teacher_fit_end_ts": pl.Int64}
KEYS = ["query_group_id", "candidate_item_id", "user_id", "timestamp", "split"]


def validate_teacher_rows(scores: pl.LazyFrame, queries: pl.LazyFrame, bank: CheckpointBank) -> None:
    """Reject key changes, future/in-sample weights, missing lineage and invalid defaults."""
    size, unique = scores.select(pl.len(), pl.struct(KEYS).n_unique().alias("unique")).collect().row(0)
    if size != unique or size != queries.select(pl.len()).collect().item() or scores.select(KEYS).join(queries.select(KEYS), on=KEYS, how="anti").limit(1).collect().height:
        raise ValueError("teacher keys differ from the full frozen query registry")
    known = pl.col("teacher_available")
    invalid = (pl.any_horizontal(pl.col(c).is_null() for c in KEYS + ["teacher_available", "teacher_score", "teacher_probability"]) |
        ~pl.col("teacher_score").is_finite() | ~pl.col("teacher_probability").is_finite() | ~pl.col("teacher_probability").is_between(0, 1) |
        ((pl.col("teacher_probability") - 1 / (1 + (-pl.col("teacher_score")).exp())).abs() > 1e-12) |
        (known & (pl.col("teacher_snapshot_ts").is_null() | pl.col("teacher_fit_end_ts").is_null() |
            (pl.col("teacher_fit_end_ts") >= pl.col("teacher_snapshot_ts")) | (pl.col("teacher_snapshot_ts") >= pl.col("timestamp")))) |
        (~known & (pl.col("teacher_snapshot_ts").is_not_null() | pl.col("teacher_fit_end_ts").is_not_null() |
            (pl.col("teacher_score") != 0.) | (pl.col("teacher_probability") != .5))))
    if scores.filter(invalid).limit(1).collect().height:
        raise ValueError("teacher contains future/in-sample scores or invalid cold defaults")
    # Validate the chosen checkpoint, including exact-boundary and warmup behavior.
    registry = pl.DataFrame({"teacher_snapshot_ts": bank.snapshots,
        "expected_fit": [row["max_input_ts"] for row in bank.records]})
    active = scores.filter(known).join(registry.lazy(), on="teacher_snapshot_ts", how="left")
    if active.filter(pl.col("expected_fit").is_null() | (pl.col("expected_fit") != pl.col("teacher_fit_end_ts"))).limit(1).collect().height:
        raise ValueError("teacher fit lineage differs from checkpoint bank")
    contexts = scores.select("timestamp", "teacher_snapshot_ts", "teacher_available").unique()
    for batch in contexts.collect_batches(chunk_size=65536):
        for timestamp, snapshot, available in batch.iter_rows():
            expected = bank.at(timestamp)
            if (expected is not None) != available or (expected is not None and expected["snapshot_ts"] != snapshot):
                raise ValueError("teacher did not use the latest strictly past checkpoint")


class TeacherArtifact:
    """Checksummed teacher predictions bound to generator inputs and an OOF bank."""

    def __init__(self, root: Path, bundle: ArtifactBundle, config: AppConfig) -> None:
        self.path = root / "teacher_scores.parquet"
        self.manifest_path = root / "teacher_manifest.parquet"
        row = pl.read_parquet(self.manifest_path).row(0, named=True)
        if row["config_hash"] != config_hash(config) or row["parent_manifest_hash"] != file_hash(bundle.manifest_path) or row["sha256"] != file_hash(self.path):
            raise ValueError("teacher artifact checksum/config/source mismatch")
        bank_path = Path(row["bank_manifest"])
        if file_hash(bank_path) != row["bank_sha256"]:
            raise ValueError("teacher checkpoint registry checksum mismatch")
        self.bank = CheckpointBank(bank_path.parent, expected_config=config_hash(config), expected_parent=file_hash(bundle.manifest_path))
        queries = pl.concat([bundle.scan(f"model_specific/flat/{split}_queries.parquet") for split in ("train", "valid", "test")])
        validate_teacher_rows(pl.scan_parquet(self.path), queries, self.bank)

    def scan(self) -> pl.LazyFrame:
        """Read validated teacher values; labels were never teacher inputs."""
        return pl.scan_parquet(self.path)


def export_teacher(config: AppConfig, bundle: ArtifactBundle, bank: CheckpointBank,
                   index: TemporalGraphIndex, root: Path, *, device: torch.device) -> TeacherArtifact:
    """Score all train/valid/test query keys using their strictly prior teacher."""
    encoder = PrefixGraphEncoder(bank, index, device, tuple(config.runtime.training.gnn_fanout))

    def batches() -> Iterator[pl.DataFrame]:
        rows = []
        for split in ("train", "valid", "test"):
            queries = bundle.scan(f"model_specific/flat/{split}_queries.parquet").sort(["timestamp", "query_group_id", "candidate_item_id"])
            for group in iter_query_groups(queries, expected_size=100 if split != "train" else None):
                items = tuple(group.rows["candidate_item_id"].to_list())
                encoded = encoder.encode(group.user_id, items, group.timestamp, group.group_id)
                checkpoint = bank.at(group.timestamp)
                for item, score in zip(items, encoded.scores.tolist(), strict=True):
                    rows.append((group.group_id, item, group.user_id, group.timestamp, split, score, float(expit(score)),
                        checkpoint is not None, encoded.snapshot_ts, checkpoint["max_input_ts"] if checkpoint else None))
                if len(rows) >= 65536:
                    yield pl.DataFrame(rows, schema=TEACHER_SCHEMA, orient="row")
                    rows = []
        if rows:
            yield pl.DataFrame(rows, schema=TEACHER_SCHEMA, orient="row")

    root.mkdir(parents=True, exist_ok=True)
    try:
        write_batches(root / "teacher_scores.parquet", batches(), TEACHER_SCHEMA)
    finally:
        encoder.close()
    frame = pl.DataFrame({"sha256": [file_hash(root / "teacher_scores.parquet")], "config_hash": [config_hash(config)],
        "parent_manifest_hash": [file_hash(bundle.manifest_path)], "bank_manifest": [str(bank.manifest_path)],
        "bank_sha256": [file_hash(bank.manifest_path)], "code_revision": [executed_code_revision()],
        "created_at_utc": [datetime.now(timezone.utc).isoformat()]})
    write_batches(root / "teacher_manifest.parquet", [frame], dict(frame.schema))
    return TeacherArtifact(root, bundle, config)


def mixed_targets(train: pl.DataFrame, alpha: float) -> np.ndarray:
    """Combine real labels with OOF probabilities; the teacher is never a soft input."""
    p = train["teacher_probability"].to_numpy()
    if not 0 <= alpha <= 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid distillation alpha or teacher probabilities")
    return alpha * train["target"].to_numpy() + (1 - alpha) * p


def choose_alpha(train: pl.DataFrame, features: Sequence[str], *, seed: int, smoke: bool) -> tuple[float, pl.DataFrame]:
    """Select alpha exclusively on the last inner-train time block by binary LogLoss."""
    cutoff = train["timestamp"].quantile(.85, interpolation="lower")
    prefix, holdout = train.filter(pl.col("timestamp") < cutoff), train.filter(pl.col("timestamp") >= cutoff)
    if prefix.is_empty() or holdout.is_empty():
        raise ValueError("alpha selection requires distinct inner train timestamp blocks")
    rows = []
    for alpha in (.25, .5, .75, 1.):
        model = fit_booster(prefix, None, features, seed=seed, iterations=30 if smoke else 50, soft_targets=mixed_targets(prefix, alpha))
        rows.append({"alpha": alpha, "inner_LogLoss": log_loss(holdout["target"], expit(model.predict(holdout))),
            "fit_end_ts": model.fit_end_ts, "holdout_start_ts": int(holdout["timestamp"].min())})
    result = pl.DataFrame(rows).sort(["inner_LogLoss", "alpha"], descending=[False, True])
    return result["alpha"][0], result


def run_distillation(config: AppConfig, dataset: str, *, mode: Mode, processed_root: Path | None = None,
                     smoke: bool = False, specs: Sequence[FeatureSpec] | None = None) -> Path:
    """Fit a declared student mode with shared pools, OOF teacher and per-seed costs."""
    if mode not in ("score_feature_stacking", "soft_target_distillation"):
        raise ValueError("unknown teacher/student mode")
    started = time.perf_counter()
    bundle = ArtifactBundle(processed_root or config.dataset(dataset).processed_dir, config_hash(config))
    if bundle.mode != "full" and not smoke:
        raise ValueError("smoke teacher inputs require an explicit smoke run")
    run_mode = "smoke" if smoke else bundle.mode
    root = config.project.experiments_dir / "distillation" / mode / dataset / run_mode
    root.mkdir(parents=True, exist_ok=True)
    state = pl.DataFrame({"status": ["running"], "config_hash": [config_hash(config)]})
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    specs = list(specs or manual_specs(dataset))
    features_path = build_feature_artifact(bundle, config, dataset, specs, shards=2 if smoke else 16)
    source = pl.scan_parquet(features_path)
    columns = feature_columns(specs)
    selection = source.filter(pl.col("split") == "train")
    cutoff = selection.select(pl.col("timestamp").quantile(.85, interpolation="lower")).collect().item()
    prefix = selection.filter(pl.col("timestamp") < cutoff)
    groups = prefix.select("query_group_id").unique().sort(pl.col("query_group_id").hash(seed=config.runtime.seed)).limit(max(1, 50000 // (config.point_in_time_policy.synthetic.negatives.negatives_per_positive + 1)))
    columns = temporal_feature_selection(collect_budgeted(prefix.join(groups, on="query_group_id", how="semi"), len(columns)), columns, seed=config.runtime.seed, already_prefix=True)
    student_columns = columns + (["teacher_score", "teacher_available"] if mode == "score_feature_stacking" else [])
    keep = list(dict.fromkeys([*KEYS, "target", "dataset_source", *columns, "teacher_score", "teacher_available", "teacher_probability"]))
    device = torch.device("cuda" if config.runtime.device == "auto" and torch.cuda.is_available() else "cpu" if config.runtime.device == "auto" else config.runtime.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    costs = []
    with runtime_workspace(config.project.root) as temporary:
        index = TemporalGraphIndex(bundle.graph(), temporary / "graph.sqlite")
        try:
            def teacher(seed: int) -> pl.LazyFrame:
                before = time.perf_counter()
                bank = fit_graph_bank(config, bundle, index, root / f"teacher_seed_{seed}/bank", seed=seed, device=device, smoke=smoke)
                artifact = export_teacher(config, bundle, bank, index, root / f"teacher_seed_{seed}", device=device)
                costs.append({"seed": seed, "wall_seconds": time.perf_counter() - before, "manifest_sha256": file_hash(artifact.manifest_path)})
                return source.join(artifact.scan().select(*KEYS, "teacher_score", pl.col("teacher_available").cast(pl.Float32), "teacher_probability"), on=KEYS).select(keep)

            def partitions(frame: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
                return tuple(collect_budgeted(frame.filter(pl.col("split") == split), len(student_columns) + 3) for split in ("train", "valid"))

            combined = teacher(config.runtime.seed)
            train, valid = partitions(combined)
            alpha, alpha_table = choose_alpha(train, columns, seed=config.runtime.seed, smoke=smoke) if mode == "soft_target_distillation" else (1., pl.DataFrame({"alpha": [1.], "selection": ["not_applicable_stacking"]}))
            write_batches(root / "alpha_selection.parquet", [alpha_table], dict(alpha_table.schema))
            valid_cut = int(valid["timestamp"].quantile(config.runtime.training.valid_hpo_fraction, interpolation="lower"))
            hpo_valid, calibration_valid = valid.filter(pl.col("timestamp") < valid_cut), valid.filter(pl.col("timestamp") >= valid_cut)
            if hpo_valid.is_empty() or calibration_valid.is_empty():
                raise ValueError("student validation needs distinct tuning/calibration timestamp blocks")
            truth = bundle.scan("evaluation/valid/eval_candidates.parquet").filter(pl.col("target_timestamp") < valid_cut)
            trials = 2 if smoke else config.runtime.training.hpo_trials
            deadline = started + trials * config.runtime.training.max_trial_hours * 3600
            def objective(trial: optuna.Trial) -> float:
                before = time.perf_counter()
                if before >= deadline:
                    raise ResourceLimitError("student HPO budget including teacher is exhausted")
                params = {"depth": trial.suggest_categorical("depth", [4, 6, 8, 10]), "learning_rate": trial.suggest_float("learning_rate", .01, .2, log=True), "l2": trial.suggest_float("l2", 1., 20., log=True)}
                model = fit_booster(train, hpo_valid, student_columns, seed=config.runtime.seed, iterations=30 if smoke else config.runtime.training.max_booster_iterations,
                    patience=5 if smoke else config.runtime.training.booster_patience, soft_targets=mixed_targets(train, alpha) if mode == "soft_target_distillation" else None,
                    max_seconds=min(config.runtime.training.max_trial_hours * 3600, deadline - before), **params)
                scores = model.predict(hpo_valid)
                value = compare_models(truth, {mode: _prediction_frame(hpo_valid, scores, expit(scores), mode, config.runtime.seed)})["NDCG@10"].item()
                trial.set_user_attr("iterations", model.estimator.tree_count_)
                trial.set_user_attr("fit_seconds", time.perf_counter() - before)
                return value
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.runtime.seed))
            study.optimize(objective, n_trials=trials, catch=(ResourceLimitError,))
            hpo = pl.DataFrame({"trial": [trial.number for trial in study.trials], "state": [trial.state.name for trial in study.trials], "NDCG@10": [trial.value for trial in study.trials], "parameters": [json.dumps(trial.params, sort_keys=True) for trial in study.trials]}).with_columns(pl.col("NDCG@10").cast(pl.Float64))
            write_batches(root / "hpo_trials.parquet", [hpo], dict(hpo.schema))
            completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                raise ResourceLimitError("no student trial completed")
            best_value = max(trial.value for trial in completed)
            best = min((trial for trial in completed if abs(trial.value - best_value) <= 1e-8), key=lambda trial: (trial.user_attrs["fit_seconds"], trial.number))
            for seed in ([config.runtime.seed] if smoke else config.runtime.training_seeds):
                if seed != config.runtime.seed:
                    combined = teacher(seed)
                    train, valid = partitions(combined)
                    calibration_valid = valid.filter(pl.col("timestamp") >= valid_cut)
                test = collect_budgeted(combined.filter(pl.col("split") == "test"), len(student_columns) + 3)
                model = fit_booster(train, None, student_columns, seed=seed, iterations=best.user_attrs["iterations"],
                    soft_targets=mixed_targets(train, alpha) if mode == "soft_target_distillation" else None, **best.params)
                seed_root = root / f"seed_{seed}"
                model.save(seed_root)
                calibration = PlattCalibrator.fit(model.predict(calibration_valid), calibration_valid["target"].to_numpy(), calibration_valid["timestamp"].to_numpy(), split="valid", calibration_start_ts=valid_cut)
                table = pl.DataFrame([asdict(calibration)])
                write_batches(seed_root / "calibration.parquet", [table], dict(table.schema))
                scores = model.predict(test)
                predictions = _prediction_frame(test, scores, calibration.predict(scores, target_min_ts=int(test["timestamp"].min())), mode, seed)
                write_batches(seed_root / "test_predictions.parquet", [predictions], PREDICTION_SCHEMA)
                metrics = compare_models(bundle.scan("evaluation/test/eval_candidates.parquet"), {mode: predictions}).with_columns(pl.lit(dataset).alias("dataset"), pl.lit(run_mode).alias("mode"), pl.lit(seed).alias("training_seed"), pl.lit(config_hash(config)).alias("config_hash"))
                write_batches(seed_root / "test_metrics.parquet", [metrics], dict(metrics.schema))
                contract = pl.DataFrame({"mode": [mode], "alpha": [alpha], "teacher_required_at_inference": [mode == "score_feature_stacking"], "teacher_inputs": [["teacher_score", "teacher_available"] if mode == "score_feature_stacking" else []]})
                write_batches(seed_root / "inference_contract.parquet", [contract], dict(contract.schema))
        finally:
            index.close()
    manifest = pl.DataFrame({"dataset": [dataset], "model": [mode], "mode": [run_mode], "config_hash": [config_hash(config)],
        "parent_manifest_hash": [file_hash(bundle.manifest_path)], "feature_hash": [file_hash(features_path)], "code_revision": [executed_code_revision()],
        "created_at_utc": [datetime.now(timezone.utc).isoformat()], "environment": [json.dumps(environment_info(), sort_keys=True)], "upstream": [json.dumps(costs, sort_keys=True)], "wall_seconds": [time.perf_counter() - started]})
    manifest = manifest.with_columns(pl.lit(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, dtype=pl.Int64).alias("cuda_peak_allocated_bytes"),
        pl.lit(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None, dtype=pl.Int64).alias("cuda_peak_reserved_bytes"))
    write_batches(root / "run_manifest.parquet", [manifest], dict(manifest.schema))
    state = state.with_columns(pl.lit("complete").alias("status"))
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    LOGGER.info("student complete dataset=%s mode=%s", dataset, mode)
    return root
