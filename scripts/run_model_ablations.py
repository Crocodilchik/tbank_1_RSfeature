"""Run preregistered full neural objective/component variants sequentially."""
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
from src.evaluation.execution import run_job
from src.evaluation.experiment_report import save_collection
from src.evaluation.model_ablations import (MODEL_VARIANTS, collect_model_ablations,
    model_ablation_command, render_model_ablation_report, validate_model_controls)
from src.utils.logging_setup import configure_logging
from src.utils.reproducibility import config_hash


def main() -> None:
    """Never silently reduce budgets or train a substitute for a missing control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)
    source = Path(__file__).resolve().parents[1]
    config_dir = (args.config_dir or source / "configs").resolve()

    def refresh() -> None:
        # Keep all48 planned cells visible even during a single-dataset resume.
        results = [collect_model_ablations(load_config(config_dir, name), name) for name in DATASETS]
        config = load_config(config_dir)
        summary = config.project.experiments_dir / "summary/model_ablations"
        save_collection(summary, results)
        render_model_ablation_report(summary, config.project.root / "docs/model_ablation_results.md")

    for dataset in args.datasets:
        config = load_config(config_dir, dataset)
        validate_model_controls(config)
        if args.collect_only:
            continue
        cells = collect_model_ablations(config, dataset)[0]
        for model, (control, _) in MODEL_VARIANTS.items():
            if cells.filter((pl.col("model") == model) & (pl.col("status") == "complete")).height:
                logger.info("reuse verified model ablation %s/%s", dataset, model)
                continue
            if not cells.filter((pl.col("model") == control) & (pl.col("status") == "complete")).height:
                logger.error("skip %s/%s: verified control %s unavailable", dataset, model, control)
                continue
            result = run_job(model_ablation_command(source, config_dir, dataset, model), cwd=source,
                journal=config.project.experiments_dir / "execution/journal.parquet", dataset=dataset,
                stage="model", model=model, config_hash=config_hash(config))
            logger.info("model ablation %s/%s: %s", dataset, model, result["status"])
            refresh()
    refresh()


if __name__ == "__main__":
    main()
