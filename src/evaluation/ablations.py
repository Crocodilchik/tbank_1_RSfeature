"""Isolated synthetic branches and exact common-pool/common-anchor contracts."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import numpy as np
import polars as pl
import yaml

from src.config.loader import AppConfig, load_config, read_yaml
from src.evaluation.metrics import pool_fingerprint
from src.evaluation.experiment_report import collect_dataset, unavailable_collection
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import write_batches
from src.utils.reproducibility import config_hash, file_hash

SYNTHETIC_BRANCHES = {"real_only": (False, 4), "extra_negatives": (False, 8),
                      "personas": (True, 4), "full_enrichment": (True, 8)}


def prepare_synthetic_config(config_dir: Path, branch: str) -> Path:
    """Create immutable variant inputs without editing canonical configuration.

    Real-only still uses the four mandatory contrastive negatives. Variant model
    artifacts have a separate experiments root; generator output is isolated too.
    """
    if branch not in SYNTHETIC_BRANCHES:
        raise ValueError("unknown synthetic branch")
    config_dir = config_dir.resolve()
    base = load_config(config_dir)
    synthetic = base.point_in_time_policy.synthetic
    if synthetic.personas.enabled or not synthetic.negatives.enabled or synthetic.negatives.negatives_per_positive != 4:
        raise ValueError("canonical real_only control requires personas off and four mandatory negatives")
    if branch == "real_only":
        return config_dir
    inputs = sorted(config_dir.rglob("*.yaml"))
    digest = hashlib.sha256()
    digest.update(b"synthetic_branch_config_v1_relative_paths")
    for path in inputs:
        digest.update(path.relative_to(config_dir).as_posix().encode())
        digest.update(file_hash(path).encode())
    version = digest.hexdigest()[:16]
    target = base.project.experiments_dir / "ablation_configs" / version / branch / "configs"
    main = read_yaml(config_dir / "main_config.yaml")
    main["project"]["root"] = Path(os.path.relpath(base.project.root, target.parent)).as_posix()
    main["project"]["experiments_dir"] = (base.project.experiments_dir / "ablations/synthetic" / version / branch).relative_to(base.project.root).as_posix()
    personas, negatives = SYNTHETIC_BRANCHES[branch]
    main["point_in_time_policy"]["synthetic"]["personas"]["enabled"] = personas
    main["point_in_time_policy"]["synthetic"]["negatives"]["negatives_per_positive"] = negatives
    payload = yaml.safe_dump(main, allow_unicode=True, sort_keys=False)
    if target.exists():
        expected = {path.relative_to(config_dir).as_posix() for path in inputs}
        actual = {path.relative_to(target).as_posix() for path in target.rglob("*.yaml")}
        if actual != expected or (target / "main_config.yaml").read_text(encoding="utf-8") != payload or any(
            file_hash(path) != file_hash(target / path.relative_to(config_dir)) for path in inputs if path.name != "main_config.yaml"):
            raise ValueError("existing ablation config differs; preserve it and use a new version")
    else:
        target.mkdir(parents=True)
        for path in inputs:
            relative = path.relative_to(config_dir)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative.as_posix() == "main_config.yaml":
                destination.write_text(payload, encoding="utf-8")
            else:
                shutil.copyfile(path, destination)
    load_config(target)
    return target


def synthetic_processed_root(config: AppConfig, dataset: str, branch: str) -> Path:
    """Resolve separate variant data using its full validated configuration hash."""
    if branch not in SYNTHETIC_BRANCHES:
        raise ValueError("unknown synthetic branch")
    if branch == "real_only":
        return config.dataset(dataset).processed_dir
    return config.dataset(dataset).processed_dir / "ablations" / ("synthetic_" + branch) / config_hash(config)[:16]


def anchor_fingerprint(bundle: ArtifactBundle) -> tuple[str, int]:
    """Hash all real positive train contexts, ignoring additional negative rows."""
    columns = ("query_group_id", "user_id", "timestamp", "candidate_item_id")
    anchors = bundle.scan("model_specific/flat/train_queries.parquet").filter(pl.col("target") == 1).select(columns)
    count, unique = anchors.select(pl.len(), pl.col("query_group_id").n_unique().alias("unique")).collect().row(0)
    if count == 0 or count != unique or anchors.select(pl.any_horizontal(pl.all().is_null()).any()).collect().item():
        raise ValueError("train anchors must be nonempty unique complete real contexts")
    digest = hashlib.sha256()
    for batch in anchors.sort(list(columns)).collect_batches(chunk_size=100_000, maintain_order=True):
        digest.update(np.asarray(batch.select(pl.all().cast(pl.Int64)).to_numpy(), dtype="<i8").tobytes(order="C"))
    return digest.hexdigest(), count


def verify_synthetic_contract(control: ArtifactBundle, variant: ArtifactBundle, *, dataset: str, branch: str,
                              destination: Path) -> pl.DataFrame:
    """Fail before model execution when any branch changes real anchors or eval pools."""
    if branch not in SYNTHETIC_BRANCHES or control.mode != "full" or variant.mode != "full":
        raise ValueError("synthetic comparison requires registered branches and complete full bundles")
    control_anchors, count = anchor_fingerprint(control)
    variant_anchors, variant_count = anchor_fingerprint(variant)
    if control_anchors != variant_anchors or count != variant_count:
        raise ValueError("synthetic branch changed real training anchors")
    records = []
    for split in ("valid", "test"):
        path = f"evaluation/{split}/eval_candidates.parquet"
        original, changed = pool_fingerprint(control.scan(path)), pool_fingerprint(variant.scan(path))
        if original != changed:
            raise ValueError("synthetic branch changed frozen " + split + " candidates/targets")
        records.append({"dataset": dataset, "branch": branch, "split": split, "pool_hash": original,
            "train_anchor_hash": control_anchors, "train_anchors": count, "control_manifest_hash": file_hash(control.manifest_path),
            "variant_manifest_hash": file_hash(variant.manifest_path), "control_config_hash": control.config_hash,
            "variant_config_hash": variant.config_hash, "status": "passed"})
    result = pl.DataFrame(records)
    write_batches(destination, [result], dict(result.schema))
    return result


def collect_synthetic_dataset(config_dir: Path, dataset: str, branches: tuple[str, ...] = tuple(SYNTHETIC_BRANCHES)) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Collect isolated branches only after rechecking exact comparison contracts."""
    if not branches or len(set(branches)) != len(branches):
        raise ValueError("synthetic branches must be nonempty and unique")
    base = load_config(config_dir, dataset)
    results = []
    for branch in branches:
        variant_dir = prepare_synthetic_config(config_dir, branch)
        config = load_config(variant_dir, dataset)
        processed = synthetic_processed_root(config, dataset, branch)
        try:
            control = ArtifactBundle(base.dataset(dataset).processed_dir, config_hash(base))
            variant = ArtifactBundle(processed, config_hash(config))
            contract_path = base.project.experiments_dir / "ablations/synthetic/contracts" / dataset / (branch + ".parquet")
            verify_synthetic_contract(control, variant, dataset=dataset, branch=branch, destination=contract_path)
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            result = unavailable_collection(dataset, status="unavailable_comparison", reason=str(error))
        else:
            result = collect_dataset(config, dataset, processed_root=processed)
        results.append(tuple(frame.with_columns(pl.lit(branch).alias("synthetic_branch")) for frame in result))
    return tuple(pl.concat([result[index] for result in results], how="vertical_relaxed") for index in range(4))


def render_synthetic_report(summary: Path, destination: Path) -> None:
    """Publish branch scores and paired-task mean differences without significance claims."""
    cells = pl.read_parquet(summary / "cells.parquet")
    lines = ["# Синтетические ablations", "", "Real-only включает четыре обязательных contrastive negatives. Extra добавляет ещё четыре; personas и full enrichment используют только fitted prefix snapshots.", "",
        "Одинаковые train anchors, valid/test pools и seeds обязательны. Контракты перепроверены перед сборкой. Недоступные сравнения имеют null scores.", "",
        "| Dataset | Ветка | Модель | Статус | NDCG@10 | seed std | Δ к real_only той же модели | Groups |", "|---|---|---|---|---:|---:|---:|---:|"]
    for row in cells.iter_rows(named=True):
        control = cells.filter((pl.col("dataset") == row["dataset"]) & (pl.col("model") == row["model"]) &
                               (pl.col("synthetic_branch") == "real_only") & (pl.col("status") == "complete"))
        value = row["NDCG@10_mean"]
        delta = value - control["NDCG@10_mean"].item() if value is not None and control.height == 1 else None
        values = [f"{number:.6f}" if number is not None else "—" for number in (value, row["NDCG@10_std"], delta)]
        lines.append(f'| {row["dataset"]} | {row["synthetic_branch"]} | {row["model"]} | {row["status"]} | ' + " | ".join(values) + f' | {row["n_groups"]} |')
    lines += ["", "Mean delta здесь описательная. Она не заменяет paired bootstrap CI или Holm correction для вывода об эффекте синтетики. Primary гипотеза проверяется отдельно; неполные ветки не считаются завершёнными ablations.", ""]
    for row in cells.filter(pl.col("status") != "complete").iter_rows(named=True):
        reason = (row["reason"] or row["status"]).replace("\n", " ")
        lines.append(f'- {row["dataset"]}/{row["synthetic_branch"]}/{row["model"]}: {reason}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
