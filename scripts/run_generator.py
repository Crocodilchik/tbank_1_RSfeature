"""Generate a full or explicitly marked smoke recommendation benchmark."""

from __future__ import annotations

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
from src.generator.pipeline import generate
from src.utils.logging_setup import configure_logging


def main() -> None:
    """Parse dataset selection; smoke modes never masquerade as full runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--sample-modulus", type=int, default=1)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--output-root", type=Path, help="Isolated full ablation output below processed/ablations")
    args = parser.parse_args()
    configure_logging()
    generate(load_config(args.config_dir, args.dataset), args.dataset, sample_modulus=args.sample_modulus, row_limit=args.row_limit,
             output_root=args.output_root)


if __name__ == "__main__":
    main()
