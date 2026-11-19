"""Model interface shared by the baseline/CNN models and the adversarial heads.

DRAFT (ECS-21): written from the ECS-21 spec because the shared
``embed()/head()`` interface has not been pushed yet. To be moved into
``docs/data_contract.md`` / replaced by İpek's version once it lands.

Every genre model exposes two steps so that extra heads (e.g. the adversarial
``ArtistHead``) can be attached to the embedding without touching the model:

    z = model.embed(x)       # (B, embed_dim) float tensor
    logits = model.head(z)   # (B, n_genres) genre logits
    model(x) == model.head(model.embed(x))

``embed_dim`` must be an attribute of the model so heads can be built from it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class EmbeddingModel(Protocol):
    embed_dim: int

    def embed(self, x: torch.Tensor) -> torch.Tensor:  # (B, embed_dim)
        ...

    def head(self, z: torch.Tensor) -> torch.Tensor:  # (B, n_genres)
        ...
