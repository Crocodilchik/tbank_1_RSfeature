"""Declared objectives over the same frozen per-query candidates."""
import torch
from torch import Tensor
from torch.nn import functional as F


def objective_variant(model: str, configured: str | None, requested: str | None) -> tuple[str, str]:
    """Keep controls unchanged and isolate an explicit alternative objective."""
    configured = configured or "sampled_bce"
    effective = requested or configured
    if configured not in ("sampled_bce", "bpr", "contrastive") or effective not in ("sampled_bce", "bpr", "contrastive"):
        raise ValueError("unknown supervised recommendation objective")
    return (model if effective == configured else f"{model}_{effective}"), effective


def supervised_loss(logits: Tensor, labels: Tensor, owners: Tensor, mode: str) -> Tensor:
    """Compute BCE, per-query BPR or contrastive loss without in-batch negatives."""
    if logits.shape != labels.shape or owners.shape != labels.shape or logits.ndim != 1:
        raise ValueError("objective requires aligned one-dimensional scores/labels/owners")
    if mode == "sampled_bce":
        return F.binary_cross_entropy_with_logits(logits, labels)
    if mode not in ("bpr", "contrastive"):
        raise ValueError("unknown supervised recommendation objective")
    losses = []
    for owner in owners.unique(sorted=True):
        scores, target = logits[owners == owner], labels[owners == owner]
        if not torch.isin(target, torch.tensor([0., 1.], device=target.device)).all() or target.sum() != 1 or target.numel() < 2:
            raise ValueError("pairwise objectives need one positive and at least one frozen negative per query")
        positive = scores[target == 1][0]
        losses.append(F.softplus(scores[target == 0] - positive).mean() if mode == "bpr" else torch.logsumexp(scores, 0) - positive)
    if not losses:
        raise ValueError("pairwise objective cannot average empty query groups")
    return torch.stack(losses).mean()
