"""Run one advanced recommender on all datasets or one selected dataset."""

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
from src.models.adv1_automl.auto_feature_generator import AutoFeatureGenerator
from src.models.baseline.pipeline import run_gbdt
from src.generator.storage import write_batches
from src.utils.logging_setup import configure_logging


def main() -> None:
    """Execute an explicitly named model with common full/smoke protocols."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("adv1", "adv2", "adv3", "adv4"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--loss", choices=("sampled_bce", "bpr", "contrastive"))
    parser.add_argument("--disable-component", choices=("graph", "sequence", "metadata"))
    args = parser.parse_args()
    if args.processed_root is not None and args.dataset is None:
        parser.error("--processed-root requires --dataset")
    if args.loss is not None and args.model == "adv1":
        parser.error("--loss requires a neural model")
    if args.disable_component is not None and (args.model != "adv4" or args.loss is not None):
        parser.error("--disable-component requires adv4 and cannot be combined with --loss")
    configure_logging()
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        config = load_config(args.config_dir, dataset)
        if args.model == "adv2":
            from src.models.adv2_gnn.pipeline import run_graph
            run_graph(config, dataset, processed_root=args.processed_root, smoke=args.smoke, objective=args.loss)
            continue
        if args.model == "adv3":
            from src.models.adv3_seq.pipeline import run_sequence
            run_sequence(config, dataset, processed_root=args.processed_root, smoke=args.smoke, objective=args.loss)
            continue
        if args.model == "adv4":
            from src.models.adv4_fusion.pipeline import run_fusion
            run_fusion(config, dataset, processed_root=args.processed_root, smoke=args.smoke, objective=args.loss,
                graph_enabled=args.disable_component != "graph", sequence_enabled=args.disable_component != "sequence",
                metadata_enabled=args.disable_component != "metadata")
            continue
        settings = config.runtime.models[args.model]
        factory = AutoFeatureGenerator(dataset, windows_days=settings.windows_days, max_candidate_features=settings.max_candidate_features,
            max_feature_depth=settings.max_feature_depth, seed=config.runtime.seed)
        candidates = factory.generate()
        result = run_gbdt(config, dataset, processed_root=args.processed_root, model_name=args.model, specs=candidates,
            selected_features=settings.selected_features, smoke=args.smoke, backend=settings.backend)
        audit = factory.audit(candidates)
        write_batches(result / "feature_search.parquet", [audit], dict(audit.schema))


if __name__ == "__main__":
    main()
