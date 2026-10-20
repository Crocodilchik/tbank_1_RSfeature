"""Objective ablations preserve the original query's negatives and gradient directions."""
import pytest
import torch
from src.models.common.objectives import supervised_loss


@pytest.mark.parametrize("mode", ["sampled_bce", "bpr", "contrastive"])
def test_frozen_query_objectives_and_gradients(mode):
    logits = torch.tensor([2., 1., -1., 1., 0.], requires_grad=True)
    labels = torch.tensor([1., 0., 0., 1., 0.])
    owners = torch.tensor([0, 0, 0, 1, 1])
    loss = supervised_loss(logits, labels, owners, mode)
    loss.backward()
    assert torch.isfinite(loss) and (logits.grad[labels == 1] < 0).all() and (logits.grad[labels == 0] > 0).all()
    if mode != "sampled_bce":
        isolated = (supervised_loss(logits[:3], labels[:3], owners[:3], mode) + supervised_loss(logits[3:], labels[3:], owners[3:], mode)) / 2
        assert torch.allclose(loss, isolated)
    if mode == "bpr":
        expected = ((torch.nn.functional.softplus(torch.tensor([-1., -3.]))).mean() + torch.nn.functional.softplus(torch.tensor(-1.))) / 2
        assert torch.allclose(loss, expected)


def test_invalid_pairwise_groups():
    for logits, labels, owners, mode in ((torch.zeros(2), torch.zeros(2), torch.zeros(2), "bpr"),
        (torch.zeros(1), torch.ones(1), torch.zeros(1), "contrastive"), (torch.zeros(0), torch.zeros(0), torch.zeros(0), "bpr"),
        (torch.zeros(2), torch.zeros(1), torch.zeros(1), "bpr"), (torch.zeros(2), torch.tensor([0., 1.]), torch.zeros(2), "wrong")):
        with pytest.raises(ValueError):
            supervised_loss(logits, labels, owners, mode)
