"""GraphSAGE batches and common full/smoke experiment entry point."""
from pathlib import Path
from typing import Sequence
import polars as pl
import torch
from torch_geometric.data import Batch
from src.config.loader import AppConfig
from src.generator.artifacts import ArtifactBundle
from src.models.adv2_gnn.heterogeneous_gnn import HeterogeneousGraphSAGE, TemporalGraphIndex
from src.models.common.training import NeuralBatch, QueryGroup


class GraphBatchBuilder:
    """Build disjoint causal PyG graphs; labels remain outside model inputs."""

    def __init__(self, index: TemporalGraphIndex, fanout: Sequence[int]) -> None:
        self.index = index
        self.fanout = tuple(fanout)

    def __call__(self, groups: Sequence[QueryGroup]) -> NeuralBatch:
        """Encode complete contexts with their frozen training/evaluation candidates."""
        graphs = [self.index.sample(group.user_id, group.rows["candidate_item_id"].to_list(), group.timestamp,
            self.fanout, target_event_id=group.group_id).data for group in groups]
        rows = pl.concat([group.rows for group in groups])
        return NeuralBatch(Batch.from_data_list(graphs), torch.tensor(rows["target"].to_numpy(), dtype=torch.float32), rows)




def run_graph(config: AppConfig, dataset: str, *, processed_root: Path | None = None,
              smoke: bool = False, objective: str | None = None) -> Path:
    """Tune GraphSAGE, calibrate on late valid and score all locked test rows."""
    from contextlib import contextmanager
    from functools import partial
    from src.models.common.pipeline import run_neural
    from src.models.common.objectives import objective_variant
    model_name, loss = objective_variant("adv2", config.runtime.models["adv2"].loss, objective)

    @contextmanager
    def context(bundle: ArtifactBundle, temporary: Path):
        index = TemporalGraphIndex(bundle.graph(), temporary / "graph.sqlite")
        try:
            yield GraphBatchBuilder(index, config.runtime.training.gnn_fanout)
        finally:
            index.close()

    return run_neural(config, dataset, model_name=model_name, model_factory=partial(HeterogeneousGraphSAGE, loss=loss),
        builder_context=context, processed_root=processed_root, smoke=smoke)
