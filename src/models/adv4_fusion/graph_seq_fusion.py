"""Two-stage graph/sequence fusion with explicit component ablation switches."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from src.models.adv3_seq.two_tower_sasrec import SequenceBatchBuilder, SequentialTwoTower
from src.models.common.temporal_graph import PrefixGraphEncoder
from src.models.common.training import NeuralBatch, QueryGroup


class FusionBatchBuilder:
    """Augment each causal generator context with frozen temporal OOF graph inputs."""

    def __init__(self, sequence: SequenceBatchBuilder, encoder: PrefixGraphEncoder) -> None:
        self.sequence, self.encoder = sequence, encoder

    def __call__(self, groups: Sequence[QueryGroup]) -> NeuralBatch:
        """Compute one bounded graph per query, including past history-item roots."""
        batch = self.sequence(groups)
        inputs = batch.inputs
        width = self.encoder.width
        inputs["graph_history"] = torch.zeros(*inputs["history_item_ids"].shape, width)
        inputs["graph_candidates"] = torch.zeros(inputs["candidate_item_ids"].shape[0], width)
        inputs["graph_users"] = torch.zeros(len(groups), width)
        offset = 0
        for index, group in enumerate(groups):
            length = int(inputs["history_lengths"][index])
            history = inputs["history_item_ids"][index, :length].tolist()
            candidates = group.rows["candidate_item_id"].to_list()
            encoded = self.encoder.encode(group.user_id, tuple(history + candidates), group.timestamp, group.group_id)
            if encoded.snapshot_ts is not None and encoded.snapshot_ts >= group.timestamp:
                raise ValueError("fusion graph snapshot must be strictly before target")
            inputs["graph_history"][index, :length] = encoded.items[:length]
            inputs["graph_candidates"][offset:offset + len(candidates)] = encoded.items[length:]
            inputs["graph_users"][index] = encoded.user
            offset += len(candidates)
        return batch


class GraphSequentialFusion(SequentialTwoTower):
    """Trainable graph projections fused into causal history and candidate towers."""

    def __init__(self, graph_size: int = 64, lambda_graph: float = .1, graph_enabled: bool = True,
                 sequence_enabled: bool = True, metadata_enabled: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        if graph_size <= 0 or lambda_graph not in (0., .1, 1.):
            raise ValueError("invalid graph feature size or auxiliary loss weight")
        hidden = self.settings["hidden_size"]
        self.settings.update(graph_size=graph_size, lambda_graph=lambda_graph, graph_enabled=graph_enabled,
            sequence_enabled=sequence_enabled, metadata_enabled=metadata_enabled)
        self.history_fusion = nn.Linear(hidden + graph_size, hidden)
        self.candidate_fusion = nn.Linear(hidden + graph_size, hidden)
        self.graph_user_projection = nn.Linear(graph_size, hidden)
        self.graph_item_projection = nn.Linear(graph_size, hidden)

    def _metadata(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.settings["metadata_enabled"]:
            return inputs
        return {key: torch.full_like(value, -1) if key.endswith("category_ids") or key.endswith("brand_ids") else value for key, value in inputs.items()}

    def _graph(self, inputs: dict[str, Tensor], key: str) -> Tensor:
        return inputs[key] if self.settings["graph_enabled"] else torch.zeros_like(inputs[key])

    def history_tokens(self, inputs: dict[str, Tensor]) -> Tensor:
        """Fuse event-time content with graph vectors available at query cutoff."""
        base = super().history_tokens(self._metadata(inputs))
        return self.history_fusion(torch.cat([base, self._graph(inputs, "graph_history")], dim=-1))

    def candidate_vectors(self, inputs: dict[str, Tensor]) -> Tensor:
        """Project the available graph and content representations of each candidate."""
        base = super().candidate_vectors(self._metadata(inputs))
        return self.candidate_fusion(torch.cat([base, self._graph(inputs, "graph_candidates")], dim=-1))

    def user_vectors(self, inputs: dict[str, Tensor]) -> Tensor:
        """Use the causal Transformer or explicit graph-only context ablation."""
        if self.settings["sequence_enabled"]:
            return super().user_vectors(inputs)
        return self.graph_user_projection(self._graph(inputs, "graph_users"))

    def training_loss(self, inputs: dict[str, Tensor], labels: Tensor, logits: Tensor) -> Tensor:
        """Auxiliary BCE trains Fusion graph projections; the teacher stays frozen."""
        loss = super().training_loss(inputs, labels, logits)
        if self.settings["graph_enabled"] and self.settings["lambda_graph"]:
            users = F.normalize(self.graph_user_projection(inputs["graph_users"]), dim=-1)
            items = F.normalize(self.graph_item_projection(inputs["graph_candidates"]), dim=-1)
            graph_scores = (users[inputs["candidate_owners"]] * items).sum(-1)
            loss = loss + self.settings["lambda_graph"] * F.binary_cross_entropy_with_logits(graph_scores, labels)
        return loss
