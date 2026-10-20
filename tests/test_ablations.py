"""Registered synthetic branches preserve configuration, anchors and frozen pools."""
from pathlib import Path
import shutil
from types import SimpleNamespace
import polars as pl
import pytest
from src.config.loader import load_config
from src.evaluation.ablations import (SYNTHETIC_BRANCHES, anchor_fingerprint, collect_synthetic_dataset,
    prepare_synthetic_config, render_synthetic_report, synthetic_processed_root, verify_synthetic_contract)
from src.evaluation.experiment_report import save_collection
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import write_batches
from src.utils.reproducibility import file_hash
from tests.helpers import model_bundle


def test_isolated_config_variants_are_immutable(tmp_path: Path):
    directory = tmp_path / "configs"
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", directory)
    hashes = {p.relative_to(directory): file_hash(p) for p in directory.rglob("*.yaml")}
    base = load_config(directory, "retailrocket")
    assert prepare_synthetic_config(directory, "real_only") == directory
    for branch, (personas, negatives) in SYNTHETIC_BRANCHES.items():
        path = prepare_synthetic_config(directory, branch)
        changed = load_config(path, "retailrocket")
        assert changed.point_in_time_policy.synthetic.personas.enabled == personas
        assert changed.point_in_time_policy.synthetic.negatives.negatives_per_positive == negatives
        assert changed.runtime == base.runtime and changed.dataset("retailrocket") == base.dataset("retailrocket")
        assert changed.project.root == base.project.root
        assert prepare_synthetic_config(directory, branch) == path
        processed = synthetic_processed_root(changed, "retailrocket", branch)
        assert processed == base.dataset("retailrocket").processed_dir if branch == "real_only" else processed.is_relative_to(base.dataset("retailrocket").processed_dir / "ablations")
    assert hashes == {p.relative_to(directory): file_hash(p) for p in directory.rglob("*.yaml")}
    # Generated inputs remain valid after moving the project to another directory.
    relocated = tmp_path / "relocated"
    shutil.copytree(directory, relocated / "configs")
    shutil.copytree(tmp_path / "experiments/ablation_configs", relocated / "experiments/ablation_configs")
    moved = load_config(prepare_synthetic_config(relocated / "configs", "extra_negatives"), "retailrocket")
    assert moved.project.root == relocated and moved.project.experiments_dir.is_relative_to(relocated / "experiments")
    with pytest.raises(ValueError, match="unknown"):
        prepare_synthetic_config(directory, "unknown")
    with pytest.raises(ValueError, match="unknown"):
        synthetic_processed_root(base, "retailrocket", "unknown")
    changed = prepare_synthetic_config(directory, "personas") / "main_config.yaml"
    changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        prepare_synthetic_config(directory, "personas")
    main = directory / "main_config.yaml"
    main.write_text(main.read_text(encoding="utf-8").replace("negatives_per_positive: 4", "negatives_per_positive: 8"), encoding="utf-8")
    with pytest.raises(ValueError, match="control requires"):
        prepare_synthetic_config(directory, "real_only")


def test_exact_contract_ignores_extra_negatives_but_rejects_changed_tasks(tmp_path: Path):
    _, processed = model_bundle(tmp_path)
    bundle = ArtifactBundle(processed)
    bundle.mode = "full"  # This test-only fixture represents a complete full comparison.
    path = "model_specific/flat/train_queries.parquet"
    queries = bundle.scan(path).collect()
    def proxy(scan):
        return SimpleNamespace(mode="full", manifest_path=bundle.manifest_path, config_hash=bundle.config_hash, scan=scan)
    extras = queries.filter(pl.col("target") == 0).with_columns((pl.col("candidate_item_id") + 1000000).alias("candidate_item_id"))
    augmented = proxy(lambda name: pl.concat([queries, extras]).reverse().lazy() if name == path else bundle.scan(name))
    receipt = verify_synthetic_contract(bundle, augmented, dataset="retailrocket", branch="extra_negatives", destination=tmp_path / "receipt.parquet")
    assert receipt.height == 2 and (receipt["status"] == "passed").all()
    assert anchor_fingerprint(bundle) == anchor_fingerprint(augmented)
    changed = proxy(lambda name: queries.with_columns((pl.col("timestamp") + 1).alias("timestamp")).lazy() if name == path else bundle.scan(name))
    with pytest.raises(ValueError, match="training anchors"):
        verify_synthetic_contract(bundle, changed, dataset="retailrocket", branch="personas", destination=tmp_path / "bad.parquet")
    changed = proxy(lambda name: bundle.scan(name).with_columns((pl.col("candidate_item_id") + 1000000).alias("candidate_item_id")) if name.startswith("evaluation/test/") else bundle.scan(name))
    with pytest.raises(ValueError, match="test candidates"):
        verify_synthetic_contract(bundle, changed, dataset="retailrocket", branch="personas", destination=tmp_path / "bad.parquet")
    for invalid in (queries.head(0), pl.concat([queries, queries.filter(pl.col("target") == 1)]), queries.with_columns(pl.lit(None, dtype=pl.Int64).alias("timestamp"))):
        changed = proxy(lambda name, frame=invalid: frame.lazy() if name == path else bundle.scan(name))
        with pytest.raises(ValueError, match="anchors must"):
            anchor_fingerprint(changed)
    with pytest.raises(ValueError, match="registered branches"):
        verify_synthetic_contract(bundle, bundle, dataset="retailrocket", branch="unknown", destination=tmp_path / "bad.parquet")


def test_missing_control_keeps_all_twenty_dataset_cells_visible(tmp_path: Path):
    directory = tmp_path / "configs"
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", directory)
    for invalid in ((), ("real_only", "real_only")):
        with pytest.raises(ValueError, match="nonempty and unique"):
            collect_synthetic_dataset(directory, "retailrocket", invalid)
    result = collect_synthetic_dataset(directory, "retailrocket")
    assert result[0].height == 20 and (result[0]["status"] == "unavailable_comparison").all()
    assert result[0]["NDCG@10_mean"].null_count() == 20
    assert result[1].is_empty() and result[2].is_empty() and result[3].is_empty()
    root = tmp_path / "summary"
    save_collection(root, [result])
    destination = tmp_path / "report.md"
    render_synthetic_report(root, destination)
    assert "четыре обязательных" in destination.read_text(encoding="utf-8")
    # Explicit toy aggregate fixtures exercise descriptive delta formatting only.
    cells = result[0].with_columns(pl.lit("complete").alias("status"), pl.lit(.01).alias("NDCG@10_std"),
        pl.when(pl.col("synthetic_branch") == "real_only").then(.3).otherwise(.4).alias("NDCG@10_mean"))
    write_batches(root / "cells.parquet", [cells], dict(cells.schema))
    render_synthetic_report(root, destination)
    assert "0.100000" in destination.read_text(encoding="utf-8")
