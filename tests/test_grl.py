"""Unit tests for src/models/grl.py (ECS-21)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.grl import (  # noqa: E402
    IGNORE_INDEX, ArtistHead, ArtistLabelEncoder, GradientReversal,
    ganin_lambda, grad_reverse,
)
from src.models.interface import EmbeddingModel  # noqa: E402


def grad_through(fn, x, upstream):
    x = x.clone().requires_grad_(True)
    fn(x).backward(upstream)
    return x.grad


@pytest.fixture()
def xg():
    g = torch.Generator().manual_seed(0)
    return torch.randn(4, 8, generator=g), torch.randn(4, 8, generator=g)


def test_forward_is_identity(xg):
    x, _ = xg
    for lambd in (0.0, 0.5, 3.0):
        torch.testing.assert_close(grad_reverse(x, lambd), x)


@pytest.mark.parametrize("lambd", [1.0, 0.1, 2.5])
def test_gradient_is_flipped_and_scaled_by_lambda(xg, lambd):
    x, up = xg
    torch.testing.assert_close(grad_through(lambda t: grad_reverse(t, lambd), x, up), -lambd * up)


def test_lambda_zero_gives_zero_gradient(xg):
    x, up = xg
    g = grad_through(lambda t: grad_reverse(t, 0.0), x, up)
    assert torch.count_nonzero(g) == 0


def test_jacobian_is_negated_scaled_identity():
    x = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    lambd = 0.7
    y = grad_reverse(x, lambd)
    jac = torch.stack([torch.autograd.grad(y.flatten()[i], x, retain_graph=True)[0].flatten()
                       for i in range(y.numel())])
    torch.testing.assert_close(jac, -lambd * torch.eye(x.numel(), dtype=torch.double))


def test_module_lambda_can_be_changed(xg):
    x, up = xg
    grl = GradientReversal(1.0)
    grl.lambd = 0.25
    torch.testing.assert_close(grad_through(grl, x, up), -0.25 * up)


class ToyModel(nn.Module):
    """Minimal model following src/models/interface.py."""

    def __init__(self, in_dim=10, embed_dim=6, n_genres=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.enc = nn.Linear(in_dim, embed_dim)
        self.cls = nn.Linear(embed_dim, n_genres)

    def embed(self, x):
        return self.enc(x)

    def head(self, z):
        return self.cls(z)

    def forward(self, x):
        return self.head(self.embed(x))


def test_artist_head_attaches_to_embed_and_reverses_encoder_gradient():
    torch.manual_seed(0)
    model = ToyModel()
    assert isinstance(model, EmbeddingModel)
    x = torch.randn(5, 10)
    y_artist = torch.tensor([0, 1, 2, 1, 0])

    head = ArtistHead(model.embed_dim, n_artists=3, hidden_dim=16, dropout=0.0, lambd=0.5)
    assert head(model.embed(x)).shape == (5, 3)

    def encoder_grad(lambd, reverse):
        model.zero_grad()
        head.zero_grad()
        head.lambd = lambd
        z = model.embed(x)
        logits = head(z) if reverse else head.mlp(z)  # mlp alone = no GRL
        F.cross_entropy(logits, y_artist).backward()
        return model.enc.weight.grad.clone(), head.mlp[0].weight.grad.clone()

    plain_enc, plain_head = encoder_grad(1.0, reverse=False)
    rev_enc, rev_head = encoder_grad(0.5, reverse=True)
    zero_enc, zero_head = encoder_grad(0.0, reverse=True)

    # encoder: sign flipped and scaled by lambda; artist MLP itself: unchanged
    torch.testing.assert_close(rev_enc, -0.5 * plain_enc)
    torch.testing.assert_close(rev_head, plain_head)
    assert torch.count_nonzero(zero_enc) == 0
    torch.testing.assert_close(zero_head, plain_head)


def test_artist_label_encoder_ignores_non_train_artists():
    enc = ArtistLabelEncoder([42, 7, 7, 13])
    assert len(enc) == 3 and enc.classes == [7, 13, 42]
    labels = enc.encode([13, 99, 42, 7])
    assert labels.tolist() == [1, IGNORE_INDEX, 2, 0]

    # ignored rows contribute nothing to the loss
    logits = torch.randn(4, 3)
    full = F.cross_entropy(logits, labels)
    kept = F.cross_entropy(logits[[0, 2, 3]], labels[[0, 2, 3]])
    torch.testing.assert_close(full, kept)


def test_ganin_lambda_schedule():
    assert ganin_lambda(0.0) == 0.0
    assert ganin_lambda(1.0, max_lambda=0.3) == pytest.approx(0.3, abs=1e-4)
    vals = [ganin_lambda(p / 10) for p in range(11)]
    assert vals == sorted(vals)
