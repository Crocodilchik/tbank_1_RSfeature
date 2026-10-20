"""Collect verified full/smoke experiments and render an honest result report."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("POLARS_OOC_MEMORY_BUDGET_MB", "512")
os.environ.setdefault("POLARS_OVERRIDE_TOTAL_MEMORY_MB", "1024")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.loader import DATASETS, load_config
from src.evaluation.experiment_report import collect_dataset, render_report, save_collection
from src.utils.logging_setup import configure_logging
import logging
import polars as pl


def main() -> None:
    """Keep absent cells visible; --strict fails until the requested matrix is complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--include-distillation", action="store_true")
    args = parser.parse_args()
    if args.processed_root is not None and args.dataset is None:
        parser.error("--processed-root requires --dataset")
    if args.report and args.mode != "full":
        parser.error("--report requires --mode full; pass smoke via --smoke-summary")
    configure_logging()
    results = []
    students = []
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        config = load_config(dataset=dataset)
        results.append(collect_dataset(config, dataset, mode=args.mode, processed_root=args.processed_root))
        if args.include_distillation:
            names = ("score_feature_stacking", "soft_target_distillation")
            roots = {name: config.project.experiments_dir / "distillation" / name / dataset / args.mode for name in names}
            students.append(collect_dataset(config, dataset, mode=args.mode, processed_root=args.processed_root, models=names, model_roots=roots))
    root = args.output_root or config.project.experiments_dir / "summary" / args.mode
    save_collection(root, results)
    if students:
        save_collection(root.parent / ("distillation_" + args.mode), students)
    if args.report:
        render_report(root, args.smoke_summary, args.report)
    complete = sum(result[0].filter(pl.col("status") == "complete").height for result in results)
    expected = sum(result[0].height for result in results)
    logging.getLogger(__name__).info("verified cells=%s/%s mode=%s", complete, expected, args.mode)
    if args.strict and complete != expected:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
