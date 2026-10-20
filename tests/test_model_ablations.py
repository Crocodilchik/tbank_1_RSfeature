"""Known-rank fixtures verify pairing, seed averaging and incomplete Holm families."""
from pathlib import Path
import json
import numpy as np
import polars as pl
import pytest

from src.evaluation import model_ablations as module
from src.evaluation.experiment_report import holm_adjust, save_collection
from src.evaluation.metrics import PREDICTION_SCHEMA, compare_models
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import write_batches
from src.models.common.objectives import objective_variant
from src.utils.reproducibility import config_hash, file_hash
from tests.helpers import model_bundle


def publish_runs(config, root):
    """Publish three seeds of controlled toy ranks, with no actual result reuse."""
    state_path = root / "audit/generation_state.parquet"
    state = pl.read_parquet(state_path).with_columns(pl.lit("full").alias("mode"))
    write_batches(state_path, [state], dict(state.schema))
    manifest_path = root / "audit/artifact_manifest.parquet"
    manifest = pl.read_parquet(manifest_path).with_columns(pl.when(pl.col("artifact_path") == "audit/generation_state.parquet")
        .then(pl.lit(file_hash(state_path))).otherwise(pl.col("sha256")).alias("sha256"))
    write_batches(manifest_path, [manifest], dict(manifest.schema))
    bundle = ArtifactBundle(root)
    truth = bundle.scan("evaluation/test/eval_candidates.parquet").collect()
    upstream = {}
    for seed in config.runtime.training_seeds:
        bank = config.project.experiments_dir / "adv4/retailrocket/full/upstream" / f"seed_{seed}"
        bank.mkdir(parents=True)
        checkpoint = bank / "fixture.pt"
        checkpoint.write_bytes(f"test-only frozen donor {seed}".encode())
        records = pl.DataFrame({"checkpoint_path": [checkpoint.name], "sha256": [file_hash(checkpoint)],
            "training_seed": [seed], "config_hash": [config_hash(config)],
            "parent_manifest_hash": [file_hash(bundle.manifest_path)], "max_input_ts": [19],
            "fit_cutoff_ts": [20], "snapshot_ts": [20]})
        write_batches(bank / "checkpoints.parquet", [records], dict(records.schema))
        upstream[str(seed)] = {"sha256": file_hash(bank / "checkpoints.parquet")}
    for model in module.MODEL_ABLATION_MODELS:
        result = config.project.experiments_dir / model / "retailrocket/full"
        for seed in config.runtime.training_seeds:
            good = model not in ("adv3", "adv2_bpr") and not (model == "adv3_contrastive" and seed == 43)
            labels = truth["candidate_label"].cast(pl.Float64)
            predictions = truth.select("eval_group_id", "candidate_item_id").with_columns(
                (labels if good else -labels).alias("score"), (.05 + .9 * labels).alias("probability"),
                pl.lit(model).alias("model"), pl.lit(seed, dtype=pl.Int64).alias("training_seed")).cast(PREDICTION_SCHEMA)
            write_batches(result / f"seed_{seed}/test_predictions.parquet", [predictions], PREDICTION_SCHEMA)
            metrics = compare_models(truth, {model: predictions}).with_columns(
                pl.lit(config_hash(config)).alias("config_hash"), pl.lit(seed).alias("training_seed"))
            write_batches(result / f"seed_{seed}/test_metrics.parquet", [metrics], dict(metrics.schema))
        provenance = pl.DataFrame({"model": [model], "dataset": ["retailrocket"], "mode": ["full"],
            "config_hash": [config_hash(config)], "parent_manifest_hash": [file_hash(bundle.manifest_path)], "wall_seconds": [1.]})
        if model.startswith("adv4"):
            provenance = provenance.with_columns(pl.lit(json.dumps({key: {**value, "reused": model != "adv4"}
                for key, value in upstream.items()})).alias("upstream_provenance"))
        write_batches(result / "run_manifest.parquet", [provenance], dict(provenance.schema))
    return bundle


def test_registered_variants_do_not_mutate_control_or_inject_limits(tmp_path):
    config, _ = model_bundle(tmp_path)
    module.validate_model_controls(config)
    before = config_hash(config)
    assert len(module.MODEL_ABLATION_MODELS) == 12 and len(module.MODEL_VARIANTS) == 9
    for model in module.MODEL_VARIANTS:
        command = module.model_ablation_command(tmp_path, tmp_path / "configs", "retailrocket", model)
        assert "--smoke" not in command and "--row-limit" not in command
        assert command[-2:] == list(module.MODEL_VARIANTS[model][1])
    assert config_hash(config) == before
    with pytest.raises(ValueError, match="unknown registered"):
        module.model_ablation_command(tmp_path, tmp_path, "retailrocket", "adv1")
    assert objective_variant("adv2", None, None) == ("adv2", "sampled_bce")
    assert objective_variant("adv2", "sampled_bce", "sampled_bce")[0] == "adv2"
    assert objective_variant("adv2", None, "bpr") == ("adv2_bpr", "bpr")
    for values in (("bad", None), (None, "bad")):
        with pytest.raises(ValueError, match="objective"):
            objective_variant("adv2", *values)
    models = dict(config.runtime.models)
    models["adv3"] = models["adv3"].model_copy(update={"loss": "bpr"})
    bad = config.model_copy(update={"runtime": config.runtime.model_copy(update={"models": models})})
    with pytest.raises(ValueError, match="BCE|sampled_bce"):
        module.collect_model_ablations(bad, "retailrocket")


def test_three_seed_pairing_holm_and_missing_contrasts(tmp_path, monkeypatch):
    config, root = model_bundle(tmp_path)
    publish_runs(config, root)
    result = module.collect_model_ablations(config, "retailrocket", processed_root=root, repetitions=2000)
    cells, _, comparisons, _ = result
    assert cells.height == 12 and (cells["status"] == "complete").all()
    contrast = comparisons.filter(pl.col("model") == "adv3_contrastive").row(0, named=True)
    assert contrast["delta"] == pytest.approx(2 / 3) and contrast["relative_delta"] is None
    assert contrast["ci_lower"] == pytest.approx(2 / 3) and contrast["significant"]
    assert comparisons.filter(pl.col("model") == "adv2_bpr")["delta"].item() == -1.
    assert not comparisons.filter(pl.col("model") == "adv2_bpr")["significant"].item()
    assert np.allclose(comparisons["holm_p"], holm_adjust(comparisons["p_value"].to_list()))
    save_collection(tmp_path / "summary", [result])
    module.render_model_ablation_report(tmp_path / "summary", tmp_path / "result.md")
    report = (tmp_path / "result.md").read_text(encoding="utf-8")
    assert "12/12" in report and "metadata" in report and "Holm" in report
    # An absent final seed cannot leave a favourable contrast from two seeds.
    (config.project.experiments_dir / "adv3_contrastive/retailrocket/full/seed_44/test_predictions.parquet").unlink()
    result = module.collect_model_ablations(config, "retailrocket", processed_root=root, repetitions=30)
    absent = result[2].filter(pl.col("model") == "adv3_contrastive").row(0, named=True)
    assert absent["status"] == "unavailable_comparison" and absent["delta"] is None
    assert absent["holm_p"] == 1. and not absent["significant"]
    save_collection(tmp_path / "summary", [result])
    module.render_model_ablation_report(tmp_path / "summary", tmp_path / "result.md")
    assert "adv3_contrastive" in (tmp_path / "result.md").read_text(encoding="utf-8")
    original = module._mean_groups
    def changed(*args):
        frame = original(*args)
        return frame.head(0) if args[3] == "adv2_bpr" else frame
    monkeypatch.setattr(module, "_mean_groups", changed)
    with pytest.raises(ValueError, match="group keys differ"):
        module.collect_model_ablations(config, "retailrocket", processed_root=root, repetitions=30)


def test_unavailable_datasets_preserve_full_registered_family(tmp_path):
    config, _ = model_bundle(tmp_path)
    result = module.collect_model_ablations(config, "retailrocket", processed_root=tmp_path / "missing")
    assert result[0].height == 12 and result[0]["NDCG@10_mean"].null_count() == 12
    assert result[2].height == 9 and result[2]["delta"].null_count() == 9
    assert (result[2]["holm_p"] == 1.).all()
    save_collection(tmp_path / "summary", [result])
    module.render_model_ablation_report(tmp_path / "summary", tmp_path / "result.md")
    assert "0/12" in (tmp_path / "result.md").read_text(encoding="utf-8")


def test_collector_rejects_corrupt_and_stale_shared_fusion_donors(tmp_path):
    config, root = model_bundle(tmp_path)
    publish_runs(config, root)
    source = config.project.experiments_dir
    checkpoint = source / "adv4/retailrocket/full/upstream/seed_42/fixture.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed after fitting the variant")
    result = module.collect_model_ablations(config, "retailrocket", processed_root=root, repetitions=30)
    rejected = result[0].filter(pl.col("model").str.starts_with("adv4"))
    assert (rejected["status"] == "invalid_shared_upstream").all()
    assert rejected["NDCG@10_mean"].null_count() == rejected.height
    assert result[1].filter(pl.col("model").str.starts_with("adv4")).is_empty()
    assert result[2].filter(pl.col("model") == "adv4_graph_off")["delta"].item() is None
    checkpoint.write_bytes(original)
    path = source / "adv4_graph_off/retailrocket/full/run_manifest.parquet"
    manifest = pl.read_parquet(path)
    donor = json.loads(manifest["upstream_provenance"].item())
    donor["42"]["sha256"] = "stale donor from an older primary fit"
    changed = manifest.with_columns(pl.lit(json.dumps(donor)).alias("upstream_provenance"))
    write_batches(path, [changed], dict(changed.schema))
    result = module.collect_model_ablations(config, "retailrocket", processed_root=root, repetitions=30)
    assert result[0].filter(pl.col("model") == "adv4_graph_off")["status"].item() == "invalid_shared_upstream"
    assert result[0].filter(pl.col("model") == "adv4")["status"].item() == "complete"
    state = pl.DataFrame({"status": ["running"]})
    write_batches(source / "adv4/retailrocket/full/run_state.parquet", [state], dict(state.schema))
    result = module.collect_model_ablations(config, "retailrocket", processed_root=root, repetitions=30)
    assert result[0].filter(pl.col("model") == "adv4_contrastive")["status"].item() == "invalid_shared_upstream"


@pytest.mark.parametrize("architecture,objective", [("adv2", "bpr"), ("adv3", "contrastive")])
def test_alternative_objective_fit_isolated_from_primary(tmp_path, monkeypatch, architecture, objective):
    from src.models.adv2_gnn import pipeline as graph
    from src.models.adv2_gnn.heterogeneous_gnn import HeterogeneousGraphSAGE
    from src.models.adv3_seq import pipeline as sequence
    from src.models.adv3_seq.two_tower_sasrec import SequentialTwoTower
    config, processed = model_bundle(tmp_path)
    def small_graph(**kw):
        kw.update(hidden_size=8, hash_slots=128)
        return HeterogeneousGraphSAGE(**kw)
    def small_sequence(**kw):
        kw.update(hidden_size=8, hash_slots=128)
        return SequentialTwoTower(**kw)
    monkeypatch.setattr(graph, "HeterogeneousGraphSAGE", small_graph)
    monkeypatch.setattr(sequence, "SequentialTwoTower", small_sequence)
    run = graph.run_graph if architecture == "adv2" else sequence.run_sequence
    parent = file_hash(processed / "audit/artifact_manifest.parquet")
    result = run(config, "retailrocket", processed_root=processed, smoke=True, objective=objective)
    assert result.parts[-3] == f"{architecture}_{objective}"
    assert not (config.project.experiments_dir / architecture / "retailrocket/smoke").exists()
    metadata = pl.read_parquet(result / "seed_42/model_metadata.parquet").row(0, named=True)
    assert json.loads(metadata["settings"])["loss"] == objective
    assert file_hash(processed / "audit/artifact_manifest.parquet") == parent
    assert pl.read_parquet(result / "seed_42/test_predictions.parquet").height == 500
