"""Verified full-matrix collection, point-in-time cohorts and paired uncertainty."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Mapping
import numpy as np
import polars as pl

from src.config.loader import AppConfig
from src.evaluation.metrics import bind_predictions, per_group_metrics, pool_fingerprint
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import runtime_workspace, write_batches
from src.utils.reproducibility import config_hash, file_hash

MODELS = ("baseline", "adv1", "adv2", "adv3", "adv4")
METRICS = ("NDCG@3", "NDCG@5", "NDCG@10", "NDCG@20", "Recall@3", "Recall@5", "Recall@10", "Recall@20",
    "HitRate@3", "HitRate@5", "HitRate@10", "HitRate@20", "MRR", "AUC", "LogLoss")
COHORTS = ("all", "train_cold_user", "train_cold_item", "event_cold_user", "event_cold_item", "short_session", "long_tail")
LONG_SCHEMA = {"dataset": pl.String, "model": pl.String, "mode": pl.String, "cohort": pl.String,
    "metric": pl.String, "value": pl.Float64, "seed_std": pl.Float64, "n_groups": pl.Int64,
    "n_seeds": pl.Int64, "reason": pl.String, "config_hash": pl.String, "pool_hash": pl.String}
CONTRAST_SCHEMA = {"dataset": pl.String, "model": pl.String, "mode": pl.String, "delta": pl.Float64,
    "relative_delta": pl.Float64, "ci_lower": pl.Float64, "ci_upper": pl.Float64, "p_value": pl.Float64,
    "holm_p": pl.Float64, "significant": pl.Boolean, "n_groups": pl.Int64}
COVERAGE_SCHEMA = {"dataset": pl.String, "cohort": pl.String, "n_groups": pl.Int64,
    "total_groups": pl.Int64, "train_popularity_threshold": pl.Int64}


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values in their original order."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Holm requires finite probabilities")
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values)
    result[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return result


def bootstrap_contrast(groups: pl.DataFrame, *, repetitions: int = 2000, seed: int = 42) -> dict:
    """Paired user-cluster percentile CI and smoothed two-sided sign probability."""
    if repetitions < 2 or groups.is_empty() or not groups["difference"].is_finite().all():
        raise ValueError("bootstrap needs finite nonempty pairs and at least2 repetitions")
    clusters = groups.sort(["user_id", "difference"]).group_by("user_id", maintain_order=True).agg(pl.col("difference").sum().alias("total"), pl.len().alias("n"))
    totals, counts = clusters["total"].to_numpy(), clusters["n"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for index in range(repetitions):
        chosen = rng.integers(len(clusters), size=len(clusters))
        samples[index] = totals[chosen].sum() / counts[chosen].sum()
    low, high = np.quantile(samples, [.025, .975])
    p = min(1., 2 * min(((samples <= 0).sum() + 1) / (repetitions + 1), ((samples >= 0).sum() + 1) / (repetitions + 1)))
    return {"delta": float(totals.sum() / counts.sum()), "ci_lower": float(low), "ci_upper": float(high), "p_value": float(p)}


def _prefix_counts(events: pl.LazyFrame, queries: pl.DataFrame, entity: str, *, chunk_rows: int = 100_000) -> np.ndarray:
    """Merge two ordered streams, retaining only O(query count + batch) counters.

    Event-level cumulative windows would materialize a counter for every source
    event. This merge counts atomic seconds strictly before each query instead.
    """
    relevant = queries.select(entity).unique().lazy()
    ordered = events.select(entity, "timestamp").join(relevant, on=entity, how="semi").sort([entity, "timestamp"])
    rows = (row for batch in ordered.collect_batches(chunk_size=chunk_rows, maintain_order=True) for row in batch.iter_rows())
    current = next(rows, None)
    result = np.zeros(queries.height, dtype=np.int64)
    previous, count = None, 0
    targets = queries.with_row_index("_position").select(entity, "timestamp", "_position").sort([entity, "timestamp", "_position"])
    for identity, timestamp, position in targets.iter_rows():
        if identity != previous:
            previous, count = identity, 0
        while current is not None and current[0] < identity:
            current = next(rows, None)
        while current is not None and current[0] == identity and current[1] < timestamp:
            count += 1
            current = next(rows, None)
        result[position] = count
    return result


def cohort_registry(bundle: ArtifactBundle, dataset: str, *, runtime_root: Path | None = None) -> pl.DataFrame:
    """Compute PIT slices with spill files confined to a run-owned workspace."""
    with runtime_workspace(runtime_root or Path(__file__).resolve().parents[2]):
        return _cohort_registry(bundle, dataset)


def _cohort_registry(bundle: ArtifactBundle, dataset: str) -> pl.DataFrame:
    """Retain query-level counters rather than source-event cumulative windows."""
    events = bundle.events().filter(~pl.col("is_synthetic"))
    queries = bundle.scan("evaluation/test/eval_candidates.parquet").filter(pl.col("candidate_label") == 1).select(
        "eval_group_id", "user_id", pl.col("candidate_item_id").alias("item_id"), pl.col("target_timestamp").alias("timestamp")).collect(engine="streaming")
    train = events.filter(pl.col("split") == "train")
    for entity in ("user_id", "item_id"):
        seen = train.select(entity).unique().with_columns(pl.lit(True).alias("_seen")).collect(engine="streaming")
        queries = queries.join(seen, on=entity, how="left").with_columns(pl.col("_seen").is_null().alias("train_cold_" + entity.removesuffix("_id"))).drop("_seen")
        queries = queries.with_columns(pl.Series("past_" + entity, _prefix_counts(events, queries, entity)))
    threshold = train.group_by("item_id").agg(pl.len().alias("popularity")).select(pl.col("popularity").quantile(.8, interpolation="lower")).collect(engine="streaming").item()
    if threshold is None:
        raise ValueError("cohort popularity threshold needs nonempty train events")
    return queries.with_columns(pl.lit(True).alias("all"),
        (pl.col("past_user_id") == 0).alias("event_cold_user"), (pl.col("past_item_id") == 0).alias("event_cold_item"),
        ((pl.col("past_user_id") <= 2) & pl.lit(dataset == "yoochoose")).alias("short_session"),
        ((pl.col("past_item_id") > 0) & (pl.col("past_item_id") <= threshold)).alias("long_tail"),
        pl.lit(int(threshold)).alias("train_popularity_threshold")).sort("eval_group_id")


def _empty_cell(dataset: str, model: str, mode: str, status: str, reason: str) -> dict:
    row = {"dataset": dataset, "model": model, "mode": mode, "status": status, "reason": reason,
        "n_groups": 0, "n_seeds": 0, "pool_hash": None, "wall_seconds": None, "human_minutes": None,
        "peak_rss_bytes": None, "cpu_seconds": None, "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None, "artifact_bytes": None, "latency_p50_ms": None,
        "latency_p95_ms": None, "resource_measurement": "unmeasured"}
    row.update({name + suffix: None for name in METRICS for suffix in ("_mean", "_std")})
    return row


def _latest_job(config: AppConfig, dataset: str, model: str) -> dict | None:
    journal = config.project.experiments_dir / "execution/journal.parquet"
    if not journal.exists():
        return None
    rows = pl.read_parquet(journal).filter((pl.col("dataset") == dataset) & (pl.col("model") == model) &
        (pl.col("stage") == "model") & (pl.col("config_hash") == config_hash(config))).sort("attempt", descending=True)
    return rows.row(0, named=True) if rows.height else None


def unavailable_collection(dataset: str, *, status: str, reason: str, mode: str = "full",
                           models: Sequence[str] = MODELS) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Represent every planned model with null scores when its task is unavailable."""
    return (_cell_frame([_empty_cell(dataset, model, mode, status, reason) for model in models]),
            pl.DataFrame(schema=LONG_SCHEMA), pl.DataFrame(schema=CONTRAST_SCHEMA), pl.DataFrame(schema=COVERAGE_SCHEMA))


def collect_dataset(config: AppConfig, dataset: str, *, mode: str = "full", processed_root: Path | None = None,
                    repetitions: int = 2000, models: Sequence[str] = MODELS,
                    model_roots: Mapping[str, Path] | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Recompute every completed seed from predictions; keep missing/invalid cells visible."""
    if mode not in ("full", "smoke"):
        raise ValueError("collector mode must be full or smoke")
    cells, long, comparisons, paired = [], [], [], {}
    try:
        dataset_root = processed_root or config.dataset(dataset).processed_dir
        state_path = dataset_root / "audit/generation_state.parquet"
        if state_path.exists():
            state = pl.read_parquet(state_path).row(0, named=True)
            if state["config_hash"] == config_hash(config) and state["status"] in ("running", "failed"):
                cells = [_empty_cell(dataset, model, mode, state["status"] + "_dataset", state.get("message") or "generation " + state["status"]) for model in models]
                return _cell_frame(cells), pl.DataFrame(schema=LONG_SCHEMA), pl.DataFrame(schema=CONTRAST_SCHEMA), pl.DataFrame(schema=COVERAGE_SCHEMA)
        bundle = ArtifactBundle(dataset_root, config_hash(config))
        if (mode == "full") != (bundle.mode == "full"):
            raise ValueError("full/smoke dataset mode differs from requested report")
        truth = bundle.scan("evaluation/test/eval_candidates.parquet")
        fingerprint = pool_fingerprint(truth)
        cohorts = cohort_registry(bundle, dataset, runtime_root=config.project.root)
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        cells = [_empty_cell(dataset, model, mode, "missing_dataset" if isinstance(error, FileNotFoundError) else "invalid_dataset", str(error)) for model in models]
        return _cell_frame(cells), pl.DataFrame(schema=LONG_SCHEMA), pl.DataFrame(schema=CONTRAST_SCHEMA), pl.DataFrame(schema=COVERAGE_SCHEMA)
    seeds = [config.runtime.seed] if mode == "smoke" else config.runtime.training_seeds
    parent = file_hash(bundle.manifest_path)
    for model in models:
        root = (model_roots or {}).get(model, config.project.experiments_dir / model / dataset / mode)
        row = _empty_cell(dataset, model, mode, "missing_run", "run manifest absent")
        job = _latest_job(config, dataset, model) if mode == "full" else None
        if job is not None and job["status"] in ("running", "failed", "interrupted"):
            row.update(status=job["status"] + "_run", reason=job["reason"] or "execution " + job["status"])
            cells.append(row)
            continue
        if not (root / "run_manifest.parquet").exists():
            cells.append(row)
            continue
        try:
            manifest = pl.read_parquet(root / "run_manifest.parquet").row(0, named=True)
            state_path = root / "run_state.parquet"
            if state_path.exists() and pl.read_parquet(state_path)["status"].item() != "complete":
                raise ValueError("model run is still running/incomplete")
            if manifest["config_hash"] != config_hash(config) or manifest["parent_manifest_hash"] != parent or manifest["mode"] != mode or manifest["dataset"] != dataset or manifest["model"] != model:
                raise ValueError("model manifest dataset/config/source/mode mismatch")
            group_frames, seed_metrics = [], []
            for seed in seeds:
                predictions = pl.scan_parquet(root / f"seed_{seed}/test_predictions.parquet")
                identities = predictions.select(pl.col("model").unique(), pl.col("training_seed").unique()).collect()
                if identities.height != 1 or identities["model"].item() != model or identities["training_seed"].item() != seed:
                    raise ValueError("prediction model/seed identity mismatch")
                metrics = per_group_metrics(bind_predictions(truth, predictions))
                stated = pl.read_parquet(root / f"seed_{seed}/test_metrics.parquet").row(0, named=True)
                if stated["pool_hash"] != fingerprint or stated["n_groups"] != metrics.height or stated["config_hash"] != config_hash(config) or stated["training_seed"] != seed or any(not np.isclose(stated[name], metrics[name].mean(), rtol=1e-10, atol=1e-12) for name in METRICS):
                    raise ValueError("published model metrics differ from verified predictions")
                seed_metrics.append({name: metrics[name].mean() for name in METRICS})
                group_frames.append(metrics.select("eval_group_id", "user_id", *METRICS).with_columns(pl.lit(seed).alias("seed")))
            all_groups = pl.concat(group_frames)
            mean_groups = all_groups.group_by("eval_group_id", "user_id").agg(pl.col(name).mean() for name in METRICS)
            paired[model] = mean_groups.select("eval_group_id", "user_id", pl.col("NDCG@10").alias(model))
            row.update(status="complete", reason=None, n_groups=mean_groups.height, n_seeds=len(seeds), pool_hash=fingerprint, wall_seconds=manifest["wall_seconds"])
            row.update(cuda_peak_allocated_bytes=manifest.get("cuda_peak_allocated_bytes"),
                       cuda_peak_reserved_bytes=manifest.get("cuda_peak_reserved_bytes"),
                       artifact_bytes=sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))
            if job is not None:
                row.update(peak_rss_bytes=job["peak_rss_bytes"], cpu_seconds=job["cpu_seconds"],
                           resource_measurement=job["measurement"])
            for name in METRICS:
                values = [record[name] for record in seed_metrics]
                row[name + "_mean"] = float(np.mean(values))
                row[name + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            grouped = all_groups.join(cohorts.select("eval_group_id", *COHORTS), on="eval_group_id")
            for cohort in COHORTS:
                selected = grouped.filter(pl.col(cohort))
                count = selected["eval_group_id"].n_unique()
                for name in METRICS:
                    values = selected.group_by("seed").agg(pl.col(name).mean())[name].to_numpy()
                    long.append({"dataset": dataset, "model": model, "mode": mode, "cohort": cohort, "metric": name,
                        "value": float(values.mean()) if count else None, "seed_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                        "n_groups": count, "n_seeds": len(seeds), "reason": None if count else "empty_cohort",
                        "config_hash": config_hash(config), "pool_hash": fingerprint})
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            row["status"] = "incomplete" if isinstance(error, FileNotFoundError) else "invalid_run"
            row["reason"] = str(error)
            paired.pop(model, None)
        cells.append(row)
    if "baseline" in paired:
        baseline = paired["baseline"]
        mean_baseline = baseline["baseline"].mean()
        for model in MODELS[1:]:
            if model not in paired:
                continue
            groups = baseline.join(paired[model], on=["eval_group_id", "user_id"]).with_columns((pl.col(model) - pl.col("baseline")).alias("difference"))
            statistics = bootstrap_contrast(groups, repetitions=repetitions, seed=config.runtime.seed)
            comparisons.append({"dataset": dataset, "model": model, "mode": mode, **statistics,
                "relative_delta": statistics["delta"] / mean_baseline if mean_baseline else None,
                "holm_p": None, "significant": False, "n_groups": groups.height})
        # Missing contrasts are p=1 in the four-member prespecified family.
        pvalues = [next((record["p_value"] for record in comparisons if record["model"] == model), 1.) for model in MODELS[1:]]
        adjusted = dict(zip(MODELS[1:], holm_adjust(pvalues), strict=True))
        for record in comparisons:
            record["holm_p"] = float(adjusted[record["model"]])
            record["significant"] = record["holm_p"] < .05 and record["ci_lower"] > 0
    coverage = pl.DataFrame({"dataset": [dataset] * len(COHORTS), "cohort": list(COHORTS), "n_groups": [int(cohorts[cohort].sum()) for cohort in COHORTS],
        "total_groups": [cohorts.height] * len(COHORTS), "train_popularity_threshold": [cohorts["train_popularity_threshold"][0]] * len(COHORTS)})
    return _cell_frame(cells), pl.DataFrame(long, schema=LONG_SCHEMA), pl.DataFrame(comparisons, schema=CONTRAST_SCHEMA), coverage


def _cell_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col(name + suffix).cast(pl.Float64) for name in METRICS for suffix in ("_mean", "_std")).with_columns(
        pl.col("wall_seconds", "human_minutes", "cpu_seconds", "latency_p50_ms", "latency_p95_ms").cast(pl.Float64),
        pl.col("peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "artifact_bytes").cast(pl.Int64),
        pl.col("pool_hash").cast(pl.String))


def save_collection(root: Path, results: Sequence[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]]) -> None:
    """Publish four deterministic Parquet result tables without inventing missing values."""
    for index, name in enumerate(("cells", "metrics", "comparisons", "coverage")):
        frames = [result[index] for result in results if result[index].width]
        if frames:
            frame = pl.concat(frames, how="vertical_relaxed")
            write_batches(root / f"{name}.parquet", [frame], dict(frame.schema))


def render_report(full_root: Path, smoke_root: Path | None, destination: Path) -> None:
    """Write a factual Markdown report; incomplete full runs never inherit smoke scores."""
    full = pl.read_parquet(full_root / "cells.parquet")
    completed = full.filter(pl.col("status") == "complete").height
    lines = ["# Результаты экспериментов", "", f"Полная матрица: завершено {completed}/{full.height} cells. Полная приемка проекта {'не выполнена' if completed != 20 else 'требует отдельной проверки ablations и ресурсов'}.", "",
        "Все цифры пересчитаны по сохраненным predictions и общим candidate keys. Основной full требует три seeds; smoke — только техническая проверка одного seed.", ""]
    for dataset in full["dataset"].unique(maintain_order=True):
        lines += [f"## {dataset}: full", "", "| Модель | Статус | NDCG@10 | seed std | Recall/HR@10 | MRR | AUC | LogLoss | Groups |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in full.filter(pl.col("dataset") == dataset).iter_rows(named=True):
            fields = ("NDCG@10_mean", "NDCG@10_std", "Recall@10_mean", "MRR_mean", "AUC_mean", "LogLoss_mean")
            values = [f'{row[name]:.6f}' if row[name] is not None else "—" for name in fields]
            lines.append(f'| {row["model"]} | {row["status"]} | ' + " | ".join(values) + f' | {row["n_groups"]} |')
        lines.append("")
    contrast_path = full_root / "comparisons.parquet"
    contrasts = pl.read_parquet(contrast_path) if contrast_path.exists() else pl.DataFrame()
    lines += ["## Парные сравнения с Baseline", ""]
    if contrasts.height:
        lines += ["| Dataset | Advanced | Δ NDCG@10 | Relative % | 95% CI | Holm p |", "|---|---|---:|---:|---|---:|"]
        for row in contrasts.iter_rows(named=True):
            relative = f'{row["relative_delta"] * 100:.2f}' if row["relative_delta"] is not None else "undefined (Baseline=0)"
            lines.append(f'| {row["dataset"]} | {row["model"]} | {row["delta"]:.6f} | {relative} | [{row["ci_lower"]:.6f}, {row["ci_upper"]:.6f}] | {row["holm_p"]:.6f} |')
    else:
        lines.append("Недоступны: нужны завершенные full Baseline и Advanced на одинаковых pools и seeds.")
    _append_cold_table(lines, full_root, "Cold-start: full")
    lines += ["", "## Ресурсы завершённых full cells", "", "| Dataset | Модель | Wall s | CPU s | Peak RSS MiB | Peak CUDA allocated MiB | Artifacts MiB |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in full.filter(pl.col("status") == "complete").iter_rows(named=True):
        values = [f'{row[name] / scale:.2f}' if row.get(name) is not None else "unmeasured" for name, scale in
                  (("wall_seconds", 1), ("cpu_seconds", 1), ("peak_rss_bytes", 2**20), ("cuda_peak_allocated_bytes", 2**20), ("artifact_bytes", 2**20))]
        lines.append(f'| {row["dataset"]} | {row["model"]} | ' + " | ".join(values) + " |")
    lines += ["## Незавершенные прогоны", ""]
    for row in full.filter(pl.col("status") != "complete").iter_rows(named=True):
        reason = row["reason"].replace("\n", " ") if row["reason"] else row["status"]
        lines.append(f'- {row["dataset"]}/{row["model"]}: {reason}')
    if smoke_root is not None and (smoke_root / "cells.parquet").exists():
        smoke = pl.read_parquet(smoke_root / "cells.parquet").filter(pl.col("status") == "complete")
        lines += ["", "## Ограниченный smoke — отдельно от full", "", "Это не исследовательский benchmark. Нельзя переносить эти scores на полный dataset или доказывать ими преимущество.", "", "| Dataset | Модель | NDCG@10 | AUC | Groups | Wall s |", "|---|---|---:|---:|---:|---:|"]
        for row in smoke.iter_rows(named=True):
            lines.append(f'| {row["dataset"]} | {row["model"]} | {row["NDCG@10_mean"]:.6f} | {row["AUC_mean"]:.6f} | {row["n_groups"]} | {row["wall_seconds"]:.2f} |')
        _append_cold_table(lines, smoke_root, "Cold-start: smoke, только техническая проверка")
    students_path = full_root.parent / "distillation_smoke/cells.parquet"
    if students_path.exists():
        students = pl.read_parquet(students_path).filter(pl.col("status") == "complete")
        lines += ["", "## Teacher/student smoke", "", "| Dataset | Режим | NDCG@10 | AUC | Groups | alpha | Teacher на inference |", "|---|---|---:|---:|---:|---:|---|"]
        for row in students.iter_rows(named=True):
            contracts = list((full_root.parent.parent / "distillation" / row["model"] / row["dataset"] / "smoke").glob("seed_*/inference_contract.parquet"))
            contract = pl.read_parquet(contracts[0]).row(0, named=True) if len(contracts) == 1 else {}
            lines.append(f'| {row["dataset"]} | {row["model"]} | {row["NDCG@10_mean"]:.6f} | {row["AUC_mean"]:.6f} | {row["n_groups"]} | {contract.get("alpha", "unavailable")} | {contract.get("teacher_required_at_inference", "unavailable")} |')
        lines += ["", "Alpha выбран внутри train. Alpha=1 означает отказ от teacher mixture; такой результат не подтверждает полезность дистилляции."]
    lines += ["", "## Ограничения и область выводов", "", "Полные synthetic/repeat/negative/objective/component ablations еще не подтверждены. Fallback, official Yoochoose и genome diagnostic не объединяются с primary. Статус каждой обязательной ветки должен быть проверен отдельно.", "",
        "Human minutes, peak RAM/VRAM и latency не восстанавливаются из времени обучения. Если измерений нет, экономия ручного труда и production latency не считаются доказанными.", "",
        "При неполной матрице гипотеза об улучшении Advanced не проверена. Нулевой или отрицательный эффект публикуется без изменения test и удаления неудобных groups.", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _append_cold_table(lines: list[str], root: Path, title: str) -> None:
    lines += ["", "## " + title, ""]
    path = root / "metrics.parquet"
    rows = pl.read_parquet(path).filter((pl.col("cohort") != "all") & (pl.col("metric") == "NDCG@10")) if path.exists() else pl.DataFrame()
    if rows.is_empty():
        lines.append("Пока нет завершенных predictions для этих срезов.")
        return
    lines += ["| Dataset | Модель | Cohort | Groups | NDCG@10 |", "|---|---|---|---:|---:|"]
    for row in rows.iter_rows(named=True):
        value = f'{row["value"]:.6f}' if row["value"] is not None else "undefined (empty cohort)"
        lines.append(f'| {row["dataset"]} | {row["model"]} | {row["cohort"]} | {row["n_groups"]} | {value} |')
