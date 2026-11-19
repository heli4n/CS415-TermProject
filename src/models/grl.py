"""Gradient Reversal Layer and adversarial artist head (ECS-21).

GRL (Ganin & Lempitsky, 2015): identity in the forward pass, multiplies the
incoming gradient by ``-lambda`` in the backward pass. Putting it between a
model's embedding and an artist classifier trains the classifier to predict the
artist while pushing the encoder to *remove* artist information:

    forward:   GRL(z) = z
    backward:  dL/dz  = -lambda * dL/dGRL(z)

Usage with any model that follows ``src/models/interface.py``::

    artist_head = ArtistHead(model.embed_dim, n_artists=len(encoder), lambd=0.1)
    z = model.embed(x)
    loss = ce(model.head(z), genre) + ce(artist_head(z), encoder.encode(artist_ids))

The artist head only knows the artists of the *train* partition (with
artist-disjoint splits, val/test artists are unseen by construction), so the
artist loss is computed on train batches only; ``ArtistLabelEncoder`` maps any
other artist to ``IGNORE_INDEX`` so ``cross_entropy`` skips it.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

IGNORE_INDEX = -100  # torch.nn.functional.cross_entropy default ignore_index


class GradientReversalFunction(torch.autograd.Function):
    """y = x forward; dx = -lambd * dy backward. ``lambd`` gets no gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module wrapper; ``lambd`` can be changed during training (schedules)."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


class ArtistHead(nn.Module):
    """GRL -> MLP -> artist logits over the train-partition artists."""

    def __init__(self, embed_dim: int, n_artists: int, hidden_dim: int = 256,
                 dropout: float = 0.3, lambd: float = 1.0):
        super().__init__()
        if n_artists < 2:
            raise ValueError(f"need at least 2 artists, got {n_artists}")
        self.grl = GradientReversal(lambd)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_artists),
        )

    @property
    def lambd(self) -> float:
        return self.grl.lambd

    @lambd.setter
    def lambd(self, value: float) -> None:
        self.grl.lambd = float(value)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.grl(z))


class ArtistLabelEncoder:
    """artist_id -> class index for the train-partition artists.

    Artists not seen in train map to ``IGNORE_INDEX``.
    """

    def __init__(self, train_artist_ids: Iterable[int]):
        self.classes = sorted({int(a) for a in train_artist_ids})
        self._index = {a: i for i, a in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.classes)

    def encode(self, artist_ids) -> torch.Tensor:
        ids = artist_ids.tolist() if hasattr(artist_ids, "tolist") else list(artist_ids)
        return torch.tensor([self._index.get(int(a), IGNORE_INDEX) for a in ids],
                            dtype=torch.long)


def ganin_lambda(progress: float, max_lambda: float = 1.0, gamma: float = 10.0) -> float:
    """DANN schedule: 0 at the start of training, -> max_lambda at the end.

    ``progress`` is the fraction of training done, in [0, 1].
    """
    p = min(max(progress, 0.0), 1.0)
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
