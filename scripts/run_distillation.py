"""Run temporal OOF teacher/student experiments on declared full or smoke data."""
import argparse
import os
from pathlib import Path
import sys
for name, value in (("POLARS_MAX_THREADS", "4"), ("POLARS_OOC_MEMORY_BUDGET_MB", "512"), ("POLARS_OVERRIDE_TOTAL_MEMORY_MB", "1024"), ("OMP_NUM_THREADS", "4")):
    os.environ.setdefault(name, value)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.loader import DATASETS, load_config
from src.models.distillation.teacher_student import run_distillation
from src.utils.logging_setup import configure_logging


def main() -> None:
    """Execute an explicit inference mode; never silently reuse sample inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("score_feature_stacking", "soft_target_distillation"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.processed_root is not None and args.dataset is None:
        parser.error("--processed-root requires --dataset")
    configure_logging()
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        run_distillation(load_config(args.config_dir, dataset), dataset, mode=args.mode, processed_root=args.processed_root, smoke=args.smoke)

if __name__ == "__main__":
    main()
