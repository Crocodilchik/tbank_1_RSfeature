"""Run complete generators/model cells sequentially, recording failures and resumable results."""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import sys
for name, value in (("POLARS_MAX_THREADS", "4"), ("POLARS_OOC_MEMORY_BUDGET_MB", "512"),
                    ("POLARS_OVERRIDE_TOTAL_MEMORY_MB", "1024"), ("OMP_NUM_THREADS", "4")):
    os.environ.setdefault(name, value)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from src.config.loader import DATASETS, load_config
from src.evaluation.execution import recover_preboot_jobs, run_job
from src.evaluation.experiment_report import MODELS, collect_dataset, render_report, save_collection
from src.generator.artifacts import ArtifactBundle
from src.utils.logging_setup import configure_logging
from src.utils.reproducibility import config_hash, file_hash


def main() -> None:
    """Resume only verified complete full inputs; smoke artifacts never satisfy a cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generation-only", action="store_true", help="Prepare full datasets sequentially without starting models")
    parser.add_argument("--force-models", action="store_true")
    args = parser.parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)
    source_root = Path(__file__).resolve().parents[1]
    config_dir = (args.config_dir or source_root / "configs").resolve()
    initial_config = load_config(config_dir)
    recover_preboot_jobs(initial_config.project.experiments_dir / "execution/journal.parquet",
        generation_roots={name: initial_config.dataset(name).processed_dir for name in DATASETS})
    for dataset in args.datasets:
        config = load_config(config_dir, dataset)
        journal = config.project.experiments_dir / "execution/journal.parquet"
        try:
            bundle = ArtifactBundle(config.dataset(dataset).processed_dir, config_hash(config))
            if bundle.mode != "full":
                raise ValueError("complete full data required")
        except (OSError, ValueError, pl.exceptions.PolarsError):
            if args.skip_generation:
                logger.error("full dataset unavailable: %s", dataset)
                continue
            result = run_job([sys.executable, str(source_root / "scripts/run_generator.py"), "--dataset", dataset,
                "--config-dir", str(config_dir)], cwd=source_root, journal=journal, dataset=dataset,
                stage="generator", model="all", config_hash=config_hash(config))
            if result["status"] != "complete":
                logger.error("generator failed: %s; see %s", dataset, result["log_path"])
                continue
        if args.generation_only:
            continue
        verified = collect_dataset(config, dataset)[0]
        for model in args.models:
            if not args.force_models and verified.filter((pl.col("model") == model) & (pl.col("status") == "complete")).height:
                logger.info("reuse verified complete cell %s/%s", dataset, model)
                continue
            script = "run_baseline.py" if model == "baseline" else "run_advanced.py"
            command = [sys.executable, str(source_root / "scripts" / script), "--dataset", dataset, "--config-dir", str(config_dir)]
            if model != "baseline":
                command += ["--model", model]
            result = run_job(command, cwd=source_root, journal=journal, dataset=dataset, stage="model", model=model, config_hash=config_hash(config))
            logger.info("cell %s/%s: %s; log=%s", dataset, model, result["status"], result["log_path"])
            summaries = [collect_dataset(load_config(config_dir, name), name) for name in DATASETS]
            root = config.project.experiments_dir / "summary/full"
            save_collection(root, summaries)
            render_report(root, config.project.experiments_dir / "summary/smoke", config.project.root / "docs/experiment_results.md")
    config = load_config(config_dir, args.datasets[0])
    results = [collect_dataset(load_config(config_dir, name), name) for name in DATASETS]
    root = config.project.experiments_dir / "summary/full"
    save_collection(root, results)
    render_report(root, config.project.experiments_dir / "summary/smoke", config.project.root / "docs/experiment_results.md")
    logger.info("full execution finished; inspect cells and journal for failures/incomplete work")


if __name__ == "__main__":
    main()
