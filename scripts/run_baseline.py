"""Run the baseline on all four datasets or an explicitly selected dataset."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("POLARS_OOC_MEMORY_BUDGET_MB", "512")
os.environ.setdefault("POLARS_OVERRIDE_TOTAL_MEMORY_MB", "1024")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loader import DATASETS, load_config
from src.models.baseline.pipeline import run_gbdt
from src.utils.logging_setup import configure_logging


def main() -> None:
    """Execute full experiments by default; reduced probes need --smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backend", choices=("catboost", "lightgbm"))
    args = parser.parse_args()
    if args.processed_root is not None and args.dataset is None:
        parser.error("--processed-root requires --dataset")
    configure_logging()
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        run_gbdt(load_config(args.config_dir, dataset), dataset, processed_root=args.processed_root, smoke=args.smoke, backend=args.backend)


if __name__ == "__main__":
    main()
