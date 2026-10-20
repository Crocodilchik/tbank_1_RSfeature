"""Fixed-size ID encoders that do not fit a vocabulary on future activity."""

import math
import torch
from torch import Tensor, nn


class HashEmbedding(nn.Module):
    """One/two fixed hash tables with index zero reserved for missing values."""

    def __init__(self, hidden_size: int, hash_slots: int = 65536, tables: int = 1) -> None:
        super().__init__()
        if hidden_size <= 0 or hash_slots < 8 or hash_slots & (hash_slots - 1) or tables not in (1, 2):
            raise ValueError("hash embeddings require positive width, power-of-two slots>=8 and one/two tables")
        self.hash_slots = hash_slots
        self.tables = nn.ModuleList([nn.Embedding(hash_slots, hidden_size, padding_idx=0) for _ in range(tables)])

    def forward(self, raw_ids: Tensor) -> Tensor:
        """Map nonnegative raw IDs deterministically; negative IDs yield zero vectors."""
        if raw_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("raw IDs must be integer tensors")
        result = None
        for i, table in enumerate(self.tables):
            indices = torch.where(raw_ids >= 0, raw_ids.remainder(self.hash_slots - 1 - 2 * i) + 1, 0)
            value = table(indices)
            result = value if result is None else result + value
        return result / math.sqrt(len(self.tables))
