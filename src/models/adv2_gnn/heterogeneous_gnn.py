"""Sparse PyG GraphSAGE and disk-backed strict temporal subgraph sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterator, Sequence

import polars as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from src.generator.schemas import STATIC_TS
from src.models.common.embeddings import HashEmbedding
from src.models.common.objectives import supervised_loss

NODE_TYPES = ("User", "Item", "Category", "Brand")
RELATIONS = tuple([(a, kind, b) for level in range(3) for a, kind, b in (
    ("User", f"event_{level}", "Item"), ("Item", f"rev_event_{level}", "User"))] + [
    ("Item", "category", "Category"), ("Category", "rev_category", "Item"),
    ("Item", "brand", "Brand"), ("Brand", "rev_brand", "Item"),
    ("Category", "parent", "Category"), ("Category", "child", "Category")])


def _event_level(kind: str) -> int:
    if kind.startswith("rating_"):
        value = float(kind.removeprefix("rating_"))
        return 2 if value >= 4 else (1 if value >= 3 else 0)
    return {"view": 0, "pv": 0, "cart": 1, "addtocart": 1, "fav": 1, "click": 1,
            "transaction": 2, "buy": 2}[kind]


def _reverse_kind(kind: str) -> str:
    return {"parent": "child", "child": "parent"}.get(kind, kind)


def _relation(src: str, dst: str, kind: str) -> tuple[str, str, str]:
    if kind.startswith("event_"):
        return src, kind if src == "User" else "rev_" + kind, dst
    if kind in ("category", "brand"):
        return src, kind if src == "Item" else "rev_" + kind, dst
    return src, kind, dst


@dataclass(frozen=True)
class SampledGraph:
    """A bounded per-query graph plus independently inspectable input timestamps."""

    data: HeteroData
    cutoff_ts: int
    max_edge_ts: int | None
    sampled_edge_ids: frozenset[int]


class TemporalGraphIndex:
    """Runtime-only bidirectional adjacency; every lookup requires timestamp<T."""

    def __init__(self, edges: pl.LazyFrame, database: Path) -> None:
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("CREATE TABLE adjacency(src_type TEXT,src INTEGER,dst_type TEXT,dst INTEGER,kind TEXT,ts INTEGER,edge INTEGER,PRIMARY KEY(src_type,src,kind,ts,edge,dst)) WITHOUT ROWID")
        self.connection.execute("CREATE TABLE membership(item INTEGER,kind TEXT,ts INTEGER,dst INTEGER,edge INTEGER,PRIMARY KEY(item,kind,ts,edge)) WITHOUT ROWID")
        try:
            for batch in edges.filter(pl.col("source") == "real").collect_batches(chunk_size=50_000):
                adjacency, membership = [], []
                for row in batch.iter_rows(named=True):
                    a, b = row["src_type"], row["dst_type"]
                    if a not in NODE_TYPES or b not in NODE_TYPES or row["valid_from"] != row["timestamp"]:
                        raise ValueError("invalid graph node type or edge provenance")
                    kind = row["edge_type"]
                    if a == "User" and b == "Item":
                        kind = f"event_{_event_level(kind)}"
                    elif a == "Item" and b in ("Category", "Brand"):
                        kind = "category" if b == "Category" else "brand"
                        membership.append((row["src_id"], kind, row["timestamp"], row["dst_id"], row["edge_id"]))
                    elif (a, b, kind) != ("Category", "Category", "parent"):
                        raise ValueError(f"unsupported graph relation: {(a, kind, b)}")
                    if row["src_id"] < 0 or row["dst_id"] < 0:
                        continue  # Membership tombstone still resets prior associations.
                    adjacency.extend([(a, row["src_id"], b, row["dst_id"], kind, row["timestamp"], row["edge_id"]),
                        (b, row["dst_id"], a, row["src_id"], _reverse_kind(kind), row["timestamp"], row["edge_id"])])
                self.connection.executemany("INSERT INTO adjacency VALUES (?,?,?,?,?,?,?)", adjacency)
                self.connection.executemany("INSERT INTO membership VALUES (?,?,?,?,?)", membership)
            self.connection.execute("CREATE INDEX neighbors ON adjacency(src_type,src,ts DESC,edge DESC)")
            self.connection.commit()
        except BaseException:
            self.connection.close()
            raise

    def _active(self, edge: tuple, timestamp: int) -> bool:
        src_type, src, dst_type, dst, kind, ts, _ = edge
        if kind not in ("category", "brand") or ts == STATIC_TS:
            return True
        item = src if src_type == "Item" else dst
        latest = self.connection.execute("SELECT MAX(ts) FROM membership WHERE item=? AND kind=? AND ts<? AND ts>?", (item, kind, timestamp, STATIC_TS)).fetchone()[0]
        return ts == latest

    def _neighbors(self, node: tuple[str, int], timestamp: int, fanout: int) -> list[tuple]:
        # Interactions have a total per-node budget, not a separate fanout per label.
        rows = self.connection.execute("SELECT src_type,src,dst_type,dst,kind,ts,edge FROM adjacency WHERE src_type=? AND src=? AND ts<? ORDER BY ts DESC,edge DESC LIMIT ?", (*node, timestamp, max(32, fanout * 16))).fetchall()
        result, interactions, other = [], 0, 0
        for row in rows:
            if not self._active(row, timestamp):
                continue
            if row[4].startswith("event_"):
                if interactions >= fanout:
                    continue
                interactions += 1
            else:
                if other >= fanout:
                    continue
                other += 1
            result.append(row)
        if node[0] == "Item":
            # Always retain available current content, including static multi-genre edges.
            content = self.connection.execute("SELECT src_type,src,dst_type,dst,kind,ts,edge FROM adjacency WHERE src_type='Item' AND src=? AND kind IN ('category','brand') AND ts<? ORDER BY ts DESC,edge DESC LIMIT 64", (node[1], timestamp)).fetchall()
            content += self.connection.execute("SELECT src_type,src,dst_type,dst,kind,ts,edge FROM adjacency WHERE src_type='Item' AND src=? AND kind IN ('category','brand') AND ts=? AND ts<?", (node[1], STATIC_TS, timestamp)).fetchall()
            result.extend(row for row in content if self._active(row, timestamp))
        return list(dict.fromkeys(result))

    def sample(self, user_id: int, candidate_item_ids: Sequence[int], timestamp: int,
               fanout: Sequence[int] = (15, 10), *, target_event_id: int | None = None) -> SampledGraph:
        """Sample separate causal neighborhoods for one query group and all its candidates."""
        if user_id < 0 or not candidate_item_ids or len(set(candidate_item_ids)) != len(candidate_item_ids) or any(i < 0 for i in candidate_item_ids) or not fanout or any(f < 1 for f in fanout):
            raise ValueError("invalid graph query IDs or neighbor fanout")
        frontier = {("User", user_id), *[("Item", int(i)) for i in candidate_item_ids]}
        nodes = set(frontier)
        visited: set[tuple[str, int]] = set()
        selected: set[tuple] = set()
        for count in fanout:
            following = set()
            for node in sorted(frontier):
                if node in visited:
                    continue
                visited.add(node)
                for edge in self._neighbors(node, timestamp, count):
                    a, src, b, dst, kind, ts, edge_id = edge
                    if ts >= timestamp or edge_id == target_event_id:
                        raise ValueError("graph sample contains a target/future edge")
                    selected.add(edge)
                    selected.add((b, dst, a, src, _reverse_kind(kind), ts, edge_id))
                    following.add((b, dst))
            nodes.update(following)
            frontier = following - visited
        data = HeteroData()
        mapping = {}
        for kind in NODE_TYPES:
            ids = sorted(value for node_kind, value in nodes if node_kind == kind)
            mapping[kind] = {value: index for index, value in enumerate(ids)}
            data[kind].num_nodes = len(ids)
            data[kind].raw_id = torch.tensor(ids, dtype=torch.long)
            data[kind].seed_mask = torch.tensor([value == user_id if kind == "User" else value in candidate_item_ids if kind == "Item" else False for value in ids], dtype=torch.bool)
            if kind == "Item":
                positions = {value: index for index, value in enumerate(candidate_item_ids)}
                data[kind].seed_order = torch.tensor([positions.get(value, -1) for value in ids], dtype=torch.long)
        for relation in RELATIONS:
            pairs = sorted((mapping[a][src], mapping[b][dst]) for a, src, b, dst, kind, _, _ in selected if _relation(a, b, kind) == relation)
            data[relation].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
        data.cutoff_ts = torch.tensor([timestamp], dtype=torch.long)
        maximum = max((edge[5] for edge in selected), default=None)
        data.max_edge_ts = torch.tensor([maximum if maximum is not None else STATIC_TS], dtype=torch.long)
        return SampledGraph(data, timestamp, maximum, frozenset(edge[6] for edge in selected))

    def close(self) -> None:
        """Close the disposable runtime adjacency index."""
        self.connection.close()


class HeterogeneousGraphSAGE(nn.Module):
    """Relation-specific PyG GraphSAGE with an inductive shared user initializer."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, hash_slots: int = 65536,
                 dropout: float = .1, loss: str = "sampled_bce") -> None:
        super().__init__()
        if hidden_size <= 0 or num_layers not in (1, 2) or not 0 <= dropout < 1 or loss not in ("sampled_bce", "bpr", "contrastive"):
            raise ValueError("invalid GraphSAGE width, layer count or dropout")
        self.settings = {"hidden_size": hidden_size, "num_layers": num_layers, "hash_slots": hash_slots, "dropout": dropout, "loss": loss}
        self.embeddings = nn.ModuleDict({kind: HashEmbedding(hidden_size, hash_slots, 2 if kind == "Item" else 1) for kind in ("Item", "Category", "Brand")})
        self.cold_user = nn.Parameter(torch.zeros(hidden_size))
        self.layers = nn.ModuleList([HeteroConv({relation: SAGEConv((hidden_size, hidden_size), hidden_size) for relation in RELATIONS}, aggr="mean") for _ in range(num_layers)])
        self.dropout = dropout
        self.log_temperature = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))

    def encode(self, graph: HeteroData) -> dict[str, Tensor]:
        """Encode sampled nodes without reading any labels or target IDs."""
        values = {kind: self.embeddings[kind](graph[kind].raw_id) for kind in ("Item", "Category", "Brand")}
        values["User"] = self.cold_user.unsqueeze(0).expand(graph["User"].num_nodes, -1)
        for layer in self.layers:
            values = {kind: F.dropout(F.relu(value), self.dropout, self.training) for kind, value in layer(values, graph.edge_index_dict).items()}
        return values

    def forward(self, graph: HeteroData) -> Tensor:
        """Score each seed Item against its own query's seed User in a PyG batch."""
        values = self.encode(graph)
        users = F.normalize(values["User"][graph["User"].seed_mask], dim=-1)
        items = F.normalize(values["Item"][graph["Item"].seed_mask], dim=-1)
        if hasattr(graph["Item"], "batch"):
            owners = graph["Item"].batch[graph["Item"].seed_mask]
        else:
            owners = torch.zeros(items.shape[0], device=items.device, dtype=torch.long)
        scores = (users[owners] * items).sum(-1) * self.log_temperature.exp().clamp(max=100.) + self.bias
        order = torch.argsort(owners * (items.shape[0] + 1) + graph["Item"].seed_order[graph["Item"].seed_mask])
        return scores[order]

    def training_loss(self, graph: HeteroData, labels: Tensor, logits: Tensor) -> Tensor:
        """Use only candidates belonging to each graph's own frozen query group."""
        owners = graph["Item"].batch[graph["Item"].seed_mask] if hasattr(graph["Item"], "batch") else torch.zeros_like(labels, dtype=torch.long)
        return supervised_loss(logits, labels, owners, self.settings["loss"])
