"""Causal Transformer user tower and content-aware fixed-hash item tower."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterator, Sequence

import polars as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.generator.metadata import MetadataIndex
from src.models.common.embeddings import HashEmbedding
from src.models.common.objectives import supervised_loss
from src.models.common.training import NeuralBatch, QueryGroup
from src.utils.reproducibility import stable_seed


class SequenceIndex:
    """Disposable on-disk lookup of the generator's exact supervised sequences."""

    def __init__(self, samples: pl.LazyFrame, database: Path) -> None:
        if database.exists():
            raise ValueError("sequence index requires a fresh runtime database")
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("CREATE TABLE samples(id INTEGER PRIMARY KEY,payload TEXT) WITHOUT ROWID")
        try:
            for batch in samples.collect_batches(chunk_size=4096):
                self.connection.executemany("INSERT INTO samples VALUES (?,?)", [(row["sample_id"], json.dumps(row)) for row in batch.iter_rows(named=True)])
            self.connection.commit()
        except BaseException:
            self.connection.close()
            raise
        self.get = lru_cache(maxsize=4096)(self._get)

    def _get(self, sample_id: int) -> dict:
        row = self.connection.execute("SELECT payload FROM samples WHERE id=?", (sample_id,)).fetchone()
        if row is None:
            raise ValueError(f"generator sequence is missing for query group {sample_id}")
        return json.loads(row[0])

    def close(self) -> None:
        """Clear the bounded read cache and close the runtime database."""
        self.get.cache_clear()
        self.connection.close()


class SequenceBatchBuilder:
    """Read one sequence per query; metadata for history uses each event's own T."""

    def __init__(self, sequences: SequenceIndex, metadata: MetadataIndex, max_history: int = 50) -> None:
        if not 0 < max_history <= 50:
            raise ValueError("sequence length must be in [1,50]")
        self.sequences, self.metadata, self.max_history = sequences, metadata, max_history
        self.content = lru_cache(maxsize=4096)(self._content)

    def _content(self, item: int, timestamp: int) -> tuple[tuple[int, ...], int]:
        category = self.metadata.category(item, timestamp)
        genres = self.metadata.get(item, timestamp, "genres", []) or []
        if not isinstance(genres, list) or any(not isinstance(genre, str) for genre in genres):
            raise ValueError("genres metadata must be a list of strings")
        categories = ([category] if category is not None else []) + [stable_seed("genre", genre) for genre in genres]
        brand = self.metadata.get(item, timestamp, "brand_id", -1)
        return tuple(categories), int(brand) if brand is not None else -1

    def __call__(self, groups: Sequence[QueryGroup]) -> NeuralBatch:
        """Validate generator provenance and construct separate context/candidate tensors."""
        samples = []
        histories, candidates = [], []
        for group in groups:
            sample = self.sequences.get(group.group_id)
            fields = [sample[key] for key in ("history_event_ids", "history_item_ids", "history_timestamps", "history_labels", "history_time_deltas")]
            size = len(fields[0])
            if sample["user_id"] != group.user_id or sample["target_timestamp"] != group.timestamp or sample["cutoff_ts"] != group.timestamp or sample["target_event_id"] != group.group_id:
                raise ValueError("sequence context does not match query group/cutoff")
            if (any(len(values) != size for values in fields) or size > self.max_history or
                any(t >= group.timestamp for t in fields[2]) or group.group_id in fields[0] or
                fields[4] != [group.timestamp - t for t in fields[2]] or any(label not in (0, 1, 2) for label in fields[3]) or
                any(item < 0 for item in fields[1]) or fields[2] != sorted(fields[2]) or
                sample["max_history_timestamp"] != (fields[2][-1] if size else None)):
                raise ValueError("sequence contains target/future tokens or invalid history alignment")
            samples.append(sample)
            histories.append([self.content(item, ts) for item, ts in zip(fields[1], fields[2], strict=True)])
            candidates.append([self.content(item, group.timestamp) for item in group.rows["candidate_item_id"]])
        length = max(1, max(len(sample["history_item_ids"]) for sample in samples))
        categories = max(1, max((len(content[0]) for row in histories + candidates for content in row), default=0))
        count = sum(group.rows.height for group in groups)
        inputs = {"history_item_ids": torch.full((len(groups), length), -1, dtype=torch.long),
            "history_category_ids": torch.full((len(groups), length, categories), -1, dtype=torch.long),
            "history_brand_ids": torch.full((len(groups), length), -1, dtype=torch.long),
            "history_labels": torch.zeros((len(groups), length), dtype=torch.long),
            "history_time_buckets": torch.zeros((len(groups), length), dtype=torch.long),
            "history_lengths": torch.tensor([len(sample["history_item_ids"]) for sample in samples], dtype=torch.long),
            "candidate_item_ids": torch.empty(count, dtype=torch.long),
            "candidate_category_ids": torch.full((count, categories), -1, dtype=torch.long),
            "candidate_brand_ids": torch.empty(count, dtype=torch.long),
            "candidate_owners": torch.empty(count, dtype=torch.long)}
        offset = 0
        for index, (group, sample) in enumerate(zip(groups, samples, strict=True)):
            size = len(sample["history_item_ids"])
            inputs["history_item_ids"][index, :size] = torch.tensor(sample["history_item_ids"], dtype=torch.long)
            inputs["history_labels"][index, :size] = torch.tensor(sample["history_labels"], dtype=torch.long) + 1
            inputs["history_time_buckets"][index, :size] = torch.tensor([min(31, int(math.log2(1 + delta))) + 1 for delta in sample["history_time_deltas"]], dtype=torch.long)
            for position, (cats, brand) in enumerate(histories[index]):
                inputs["history_category_ids"][index, position, :len(cats)] = torch.tensor(cats, dtype=torch.long)
                inputs["history_brand_ids"][index, position] = brand
            end = offset + group.rows.height
            inputs["candidate_item_ids"][offset:end] = torch.tensor(group.rows["candidate_item_id"].to_numpy(), dtype=torch.long)
            inputs["candidate_owners"][offset:end] = index
            for position, (cats, brand) in enumerate(candidates[index], offset):
                inputs["candidate_category_ids"][position, :len(cats)] = torch.tensor(cats, dtype=torch.long)
                inputs["candidate_brand_ids"][position] = brand
            offset = end
        rows = pl.concat([group.rows for group in groups])
        return NeuralBatch(inputs, torch.tensor(rows["target"].to_numpy(), dtype=torch.float32), rows)

    def close(self) -> None:
        """Release bounded metadata cache; ownership of indexes stays with the context."""
        self.content.cache_clear()


class ItemTower(nn.Module):
    """Shared ID, multivalue category/genre and brand embeddings plus an MLP."""

    def __init__(self, hidden_size: int, hash_slots: int) -> None:
        super().__init__()
        self.item = HashEmbedding(hidden_size, hash_slots, 2)
        self.category = HashEmbedding(hidden_size, hash_slots)
        self.brand = HashEmbedding(hidden_size, hash_slots)
        self.project = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, items: Tensor, categories: Tensor, brands: Tensor) -> Tensor:
        """Mean-pool all known metadata values; missing fields yield zero vectors."""
        pooled = self.category(categories).sum(-2) / (categories >= 0).sum(-1, keepdim=True).clamp(min=1)
        return self.project(torch.cat([self.item(items), pooled, self.brand(brands)], dim=-1))


class SequentialTwoTower(nn.Module):
    """SASRec-style causal history and inductive content-aware candidate tower."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, attention_heads: int = 2,
                 hash_slots: int = 65536, dropout: float = .1, max_history: int = 50, loss: str = "sampled_bce") -> None:
        super().__init__()
        if hidden_size <= 0 or attention_heads not in (2, 4) or hidden_size % attention_heads or num_layers not in (1, 2) or not 0 <= dropout < 1 or not 0 < max_history <= 50 or loss not in ("sampled_bce", "bpr", "contrastive"):
            raise ValueError("invalid sequential width, layers, heads, dropout or history length")
        self.settings = {"hidden_size": hidden_size, "num_layers": num_layers, "attention_heads": attention_heads,
            "hash_slots": hash_slots, "dropout": dropout, "max_history": max_history, "loss": loss}
        self.item_tower = ItemTower(hidden_size, hash_slots)
        self.event = nn.Embedding(4, hidden_size, padding_idx=0)
        self.time = nn.Embedding(33, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_history + 1, hidden_size)
        self.cold_context = nn.Parameter(torch.zeros(hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, attention_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.user_projection = nn.Linear(hidden_size, hidden_size)
        self.log_temperature = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))

    def history_tokens(self, inputs: dict[str, Tensor]) -> Tensor:
        """Encode historical item content and observable event/time context."""
        return (self.item_tower(inputs["history_item_ids"], inputs["history_category_ids"], inputs["history_brand_ids"]) +
            self.event(inputs["history_labels"]) + self.time(inputs["history_time_buckets"]))

    def encode_history(self, inputs: dict[str, Tensor]) -> Tensor:
        """Return every state with strict causal and right-padding attention masks."""
        tokens = self.history_tokens(inputs)
        batch, length, _ = tokens.shape
        if length > self.settings["max_history"]:
            raise ValueError("history exceeds configured positional vocabulary")
        tokens = torch.cat([self.cold_context.expand(batch, 1, -1), tokens], dim=1)
        positions = torch.arange(length + 1, device=tokens.device)
        tokens = tokens + self.position(positions)
        padding = positions.unsqueeze(0) > inputs["history_lengths"].unsqueeze(1)
        causal = torch.ones(length + 1, length + 1, device=tokens.device, dtype=torch.bool).triu(1)
        # Math attention has deterministic CUDA backward; padding always leaves cold token.
        with sdpa_kernel(SDPBackend.MATH):
            return self.transformer(tokens, mask=causal, src_key_padding_mask=padding)

    def user_vectors(self, inputs: dict[str, Tensor]) -> Tensor:
        """Select the last genuine history state or the learned cold context."""
        states = self.encode_history(inputs)
        return self.user_projection(states[torch.arange(states.shape[0], device=states.device), inputs["history_lengths"]])

    def candidate_vectors(self, inputs: dict[str, Tensor]) -> Tensor:
        """Encode only candidate identity and point-in-time content."""
        return self.item_tower(inputs["candidate_item_ids"], inputs["candidate_category_ids"], inputs["candidate_brand_ids"])

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """Score each candidate against its own context without label inputs."""
        users = F.normalize(self.user_vectors(inputs), dim=-1)
        items = F.normalize(self.candidate_vectors(inputs), dim=-1)
        return (users[inputs["candidate_owners"]] * items).sum(-1) * self.log_temperature.exp().clamp(max=100.) + self.bias

    def training_loss(self, inputs: dict[str, Tensor], labels: Tensor, logits: Tensor) -> Tensor:
        """Apply the declared objective within each original candidate group."""
        return supervised_loss(logits, labels, inputs["candidate_owners"], self.settings["loss"])
