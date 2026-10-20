"""Execute registered full synthetic branches with identical anchors and eval pools."""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import sys
for name, value in (("POLARS_MAX_THREADS", "4"), ("OMP_NUM_THREADS", "4"),
                    ("OPENBLAS_NUM_THREADS", "4"), ("POLARS_OOC_MEMORY_BUDGET_MB", "512"),
                    ("POLARS_OVERRIDE_TOTAL_MEMORY_MB", "1024")):
    os.environ.setdefault(name, value)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from src.config.loader import DATASETS, load_config
from src.evaluation.ablations import (SYNTHETIC_BRANCHES, collect_synthetic_dataset, prepare_synthetic_config,
    render_synthetic_report, synthetic_processed_root, verify_synthetic_contract)
from src.evaluation.execution import run_job
from src.evaluation.experiment_report import MODELS, collect_dataset, save_collection
from src.generator.artifacts import ArtifactBundle
from src.utils.logging_setup import configure_logging
from src.utils.reproducibility import config_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--branches", nargs="+", choices=tuple(SYNTHETIC_BRANCHES), default=list(SYNTHETIC_BRANCHES))
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)
    source = Path(__file__).resolve().parents[1]
    config_dir = (args.config_dir or source / "configs").resolve()
    results = {}

    def refresh(dataset: str) -> None:
        results[dataset] = collect_synthetic_dataset(config_dir, dataset, tuple(args.branches))
        base = load_config(config_dir, dataset)
        summary = base.project.experiments_dir / "summary/synthetic_ablations"
        save_collection(summary, list(results.values()))
        render_synthetic_report(summary, base.project.root / "docs/synthetic_ablation_results.md")

    for dataset in args.datasets:
        base = load_config(config_dir, dataset)
        try:
            control = ArtifactBundle(base.dataset(dataset).processed_dir, config_hash(base))
            if control.mode != "full":
                raise ValueError("complete full control required")
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            logger.error("control unavailable %s: %s", dataset, error)
            refresh(dataset)
            continue
        for branch in args.branches:
            if branch == "real_only" or args.collect_only:
                continue
            variant_dir = prepare_synthetic_config(config_dir, branch)
            config = load_config(variant_dir, dataset)
            processed = synthetic_processed_root(config, dataset, branch)
            journal = config.project.experiments_dir / "execution/journal.parquet"
            try:
                bundle = ArtifactBundle(processed, config_hash(config))
                if bundle.mode != "full":
                    raise ValueError("complete full variant required")
            except (OSError, ValueError, pl.exceptions.PolarsError):
                if args.skip_generation:
                    logger.error("variant data unavailable %s/%s", dataset, branch)
                    continue
                result = run_job([sys.executable, str(source / "scripts/run_generator.py"), "--dataset", dataset,
                    "--config-dir", str(variant_dir), "--output-root", str(processed)], cwd=source, journal=journal,
                    dataset=dataset, stage="generator", model="all", config_hash=config_hash(config))
                if result["status"] != "complete":
                    logger.error("variant generation failed %s/%s", dataset, branch)
                    refresh(dataset)
                    continue
                bundle = ArtifactBundle(processed, config_hash(config))
            try:
                verify_synthetic_contract(control, bundle, dataset=dataset, branch=branch,
                    destination=base.project.experiments_dir / "ablations/synthetic/contracts" / dataset / (branch + ".parquet"))
            except ValueError as error:
                logger.error("comparison rejected %s/%s: %s", dataset, branch, error)
                refresh(dataset)
                continue
            verified = collect_dataset(config, dataset, processed_root=processed)[0]
            for model in args.models:
                if verified.filter((pl.col("model") == model) & (pl.col("status") == "complete")).height:
                    logger.info("reuse verified ablation %s/%s/%s", dataset, branch, model)
                    continue
                script = "run_baseline.py" if model == "baseline" else "run_advanced.py"
                command = [sys.executable, str(source / "scripts" / script), "--dataset", dataset,
                    "--config-dir", str(variant_dir), "--processed-root", str(processed)]
                if model != "baseline":
                    command += ["--model", model]
                result = run_job(command, cwd=source, journal=journal, dataset=dataset, stage="model",
                    model=model, config_hash=config_hash(config))
                logger.info("ablation %s/%s/%s: %s", dataset, branch, model, result["status"])
                refresh(dataset)
        refresh(dataset)


if __name__ == "__main__":
    main()
