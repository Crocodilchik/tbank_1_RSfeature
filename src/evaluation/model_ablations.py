"""Registered neural objective/component contrasts over unchanged primary inputs."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import polars as pl

from src.config.loader import AppConfig
from src.evaluation.experiment_report import (CONTRAST_SCHEMA, METRICS, bootstrap_contrast,
    collect_dataset, holm_adjust)
from src.evaluation.metrics import bind_predictions, per_group_metrics
from src.generator.artifacts import ArtifactBundle
from src.utils.reproducibility import config_hash, file_hash

# Fixed before the first full model result; no architecture selection by test.
MODEL_VARIANTS = {f"{model}_{loss}": (model, ("--loss", loss))
    for model in ("adv2", "adv3", "adv4") for loss in ("bpr", "contrastive")}
MODEL_VARIANTS.update({f"adv4_{component}_off": ("adv4", ("--disable-component", component))
    for component in ("graph", "sequence", "metadata")})
MODEL_ABLATION_MODELS = ("adv2", "adv3", "adv4", *MODEL_VARIANTS)
ABLATION_CONTRAST_SCHEMA = {**CONTRAST_SCHEMA, "control_model": pl.String,
    "status": pl.String, "reason": pl.String}


def validate_model_controls(config: AppConfig) -> None:
    """Require the registered BCE controls without mutating their configuration."""
    if any((config.runtime.models[model].loss or "sampled_bce") != "sampled_bce"
           for model in ("adv2", "adv3", "adv4")):
        raise ValueError("registered objective ablations require sampled_bce primary controls")


def model_ablation_command(source: Path, config_dir: Path, dataset: str, model: str) -> list[str]:
    """Return a full-budget command with explicit isolated model identity."""
    if model not in MODEL_VARIANTS:
        raise ValueError("unknown registered model ablation")
    architecture, flags = MODEL_VARIANTS[model]
    return [sys.executable, str(source / "scripts/run_advanced.py"), "--dataset", dataset,
        "--config-dir", str(config_dir), "--model", architecture, *flags]


def _mean_groups(config: AppConfig, bundle: ArtifactBundle, dataset: str, model: str,
                 mode: str) -> pl.DataFrame:
    truth = bundle.scan("evaluation/test/eval_candidates.parquet")
    seeds = [config.runtime.seed] if mode == "smoke" else config.runtime.training_seeds
    frames = []
    for seed in seeds:
        path = config.project.experiments_dir / model / dataset / mode / f"seed_{seed}/test_predictions.parquet"
        groups = per_group_metrics(bind_predictions(truth, pl.scan_parquet(path)))
        frames.append(groups.select("eval_group_id", "user_id", "NDCG@10"))
    return pl.concat(frames).group_by("eval_group_id", "user_id").agg(pl.col("NDCG@10").mean())


def _fusion_lineage_errors(config: AppConfig, bundle: ArtifactBundle, dataset: str,
                           mode: str, complete: set[str]) -> dict[str, str]:
    """Reject stale/corrupt donors again at collection, without loading PyTorch."""
    variants = sorted(model for model in complete if model.startswith("adv4_"))
    if "adv4" not in complete:
        return {model: "completed primary Fusion donor unavailable" for model in variants}
    seeds = [config.runtime.seed] if mode == "smoke" else config.runtime.training_seeds
    root = config.project.experiments_dir / "adv4" / dataset / mode
    try:
        primary = json.loads(pl.read_parquet(root / "run_manifest.parquet")["upstream_provenance"].item())
        digests = {}
        for seed in seeds:
            bank_root = (root / "upstream" / f"seed_{seed}").resolve()
            path = bank_root / "checkpoints.parquet"
            digests[str(seed)] = file_hash(path)
            if primary[str(seed)]["sha256"] != digests[str(seed)]:
                raise ValueError("primary Fusion shared bank checksum differs")
            records = pl.read_parquet(path)
            if records.is_empty():
                raise ValueError("primary Fusion shared bank is empty")
            for row in records.iter_rows(named=True):
                if row["training_seed"] != seed or row["config_hash"] != config_hash(config) or row["parent_manifest_hash"] != file_hash(bundle.manifest_path) or not row["max_input_ts"] < row["fit_cutoff_ts"] <= row["snapshot_ts"]:
                    raise ValueError("primary Fusion shared bank seed/config/source/cutoff differs")
                checkpoint = (bank_root / row["checkpoint_path"]).resolve()
                if not checkpoint.is_relative_to(bank_root) or file_hash(checkpoint) != row["sha256"]:
                    raise ValueError("primary Fusion shared checkpoint path/checksum differs")
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as error:
        return {model: str(error) for model in ("adv4", *variants)}
    errors = {}
    for model in variants:
        try:
            path = config.project.experiments_dir / model / dataset / mode / "run_manifest.parquet"
            donor = json.loads(pl.read_parquet(path)["upstream_provenance"].item())
            if any(donor[str(seed)]["sha256"] != digests[str(seed)] or donor[str(seed)].get("reused") is not True for seed in seeds):
                raise ValueError("Fusion variant uses a different shared OOF donor")
        except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as error:
            errors[model] = str(error)
    return errors


def collect_model_ablations(config: AppConfig, dataset: str, *, processed_root: Path | None = None,
                            mode: str = "full", repetitions: int = 2000) -> tuple[pl.DataFrame, ...]:
    """Verify all cells, then pair every variant against its own primary control.

    Missing contrasts stay in the nine-member Holm family with p=1. Stored mean
    scores alone never suffice: group metrics are recomputed from shared truth.
    """
    validate_model_controls(config)
    cells, metrics, _, coverage = collect_dataset(config, dataset, processed_root=processed_root,
        mode=mode, models=MODEL_ABLATION_MODELS, repetitions=repetitions)
    complete = set(cells.filter(pl.col("status") == "complete")["model"])
    paired = {}
    if complete:
        bundle = ArtifactBundle(processed_root or config.dataset(dataset).processed_dir, config_hash(config))
        errors = _fusion_lineage_errors(config, bundle, dataset, mode, complete)
        for model, reason in errors.items():
            selected = pl.col("model") == model
            cells = cells.with_columns(
                pl.when(selected).then(pl.lit("invalid_shared_upstream")).otherwise(pl.col("status")).alias("status"),
                pl.when(selected).then(pl.lit(reason)).otherwise(pl.col("reason")).alias("reason"),
                *[pl.when(selected).then(None).otherwise(pl.col(name + suffix)).alias(name + suffix)
                  for name in METRICS for suffix in ("_mean", "_std")])
        complete.difference_update(errors)
        metrics = metrics.filter(~pl.col("model").is_in(list(errors)))
        paired = {model: _mean_groups(config, bundle, dataset, model, mode) for model in sorted(complete)}
    contrasts = []
    for model, (control, _) in MODEL_VARIANTS.items():
        row = {key: None for key in ABLATION_CONTRAST_SCHEMA}
        row.update(dataset=dataset, model=model, control_model=control, mode=mode,
            status="unavailable_comparison", significant=False, n_groups=0,
            reason="both verified control and variant are required")
        if model in paired and control in paired:
            groups = paired[control].rename({"NDCG@10": "control"}).join(
                paired[model].rename({"NDCG@10": "variant"}), on=["eval_group_id", "user_id"], validate="1:1")
            if groups.height != paired[control].height or groups.height != paired[model].height:
                raise ValueError("ablation control/variant group keys differ")
            statistics = bootstrap_contrast(groups.with_columns((pl.col("variant") - pl.col("control")).alias("difference")),
                repetitions=repetitions, seed=config.runtime.seed)
            baseline = groups["control"].mean()
            row.update(**statistics, relative_delta=statistics["delta"] / baseline if baseline else None,
                status="complete", reason=None, n_groups=groups.height)
        contrasts.append(row)
    adjusted = holm_adjust([row["p_value"] if row["p_value"] is not None else 1. for row in contrasts])
    for row, value in zip(contrasts, adjusted, strict=True):
        row["holm_p"] = float(value)
        row["significant"] = bool(row["status"] == "complete" and value < .05 and row["ci_lower"] > 0)
    return cells, metrics, pl.DataFrame(contrasts, schema=ABLATION_CONTRAST_SCHEMA), coverage


def render_model_ablation_report(summary: Path, destination: Path) -> None:
    """Publish incomplete work alongside valid paired effects and their limits."""
    cells = pl.read_parquet(summary / "cells.parquet")
    comparisons = pl.read_parquet(summary / "comparisons.parquet")
    count = cells.filter(pl.col("status") == "complete").height
    lines = ["# Neural objective и Fusion component ablations", "",
        f"Проверены {count}/{cells.height} cells, включая primary controls. Полный run требует три seeds; null не заменяется smoke.", "",
        "Все варианты используют canonical train anchors, negatives и primary eval pools. Fusion читает исходные OOF checkpoints primary Adv4; manifests/weights проверяются по config, source и seed.", "",
        "Metadata-off означает отключение metadata в Item Tower. Metadata relations графа остаются; полная Taobao metadata-off проверяется отдельно.", "",
        "| Dataset | Model | Status | NDCG@10 | Seed std | Groups |",
        "|---|---|---|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{value:.6f}"
    for row in cells.iter_rows(named=True):
        lines.append(f'| {row["dataset"]} | {row["model"]} | {row["status"]} | {fmt(row["NDCG@10_mean"])} | {fmt(row["NDCG@10_std"])} | {row["n_groups"]} |')
    lines += ["", "Paired user-cluster bootstrap:2000 resamples, seed42, mean по training seeds до resampling. Holm family: все9 запланированных контрастов внутри dataset, отсутствующие p=1.", "",
        "| Dataset | Variant − control | Status | Δ NDCG@10 | Relative Δ | 95% CI | Holm p | Significant gain |",
        "|---|---|---|---:|---:|---|---:|---|"]
    for row in comparisons.iter_rows(named=True):
        interval = "—" if row["ci_lower"] is None else f'[{fmt(row["ci_lower"])}, {fmt(row["ci_upper"])}]'
        lines.append(f'| {row["dataset"]} | {row["model"]} − {row["control_model"]} | {row["status"]} | {fmt(row["delta"])} | {fmt(row["relative_delta"])} | {interval} | {fmt(row["holm_p"])} | {row["significant"]} |')
    lines += ["", "Время shared upstream fit сохранено в builder provenance отдельно от incremental wall варианта; reuse не означает бесплатный upstream. Другие обязательные ablations и итоговая приёмка остаются отдельными проверками.", ""]
    for row in cells.filter(pl.col("status") != "complete").iter_rows(named=True):
        reason = (row["reason"] or row["status"]).replace("\n", " ")
        lines.append(f'- {row["dataset"]}/{row["model"]}: {reason}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
