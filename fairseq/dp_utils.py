#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import examples.pointer_generator.pointer_generator_src.transformer_pg as transformer_pg

from opacus.grad_sample.utils import create_or_accumulate_grad_sample, register_grad_sampler
from opacus.grad_sample import utils


@register_grad_sampler(transformer_pg.Embedding)
def compute_embedding_grad_sample(
    layer: transformer_pg.Embedding, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Add grad sample for transformer_pg.Embedding the same way it is added to nn.Embedding
    Computes per sample gradients for ``transformer_pg.Embedding`` layer.
    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    A = torch.where(A >= layer.num_embeddings, torch.ones_like(A) * layer.unk_idx, A)

    saved = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    batch_size = A.shape[batch_dim]
    index = (
        A.unsqueeze(-1)
        .expand(*A.shape, layer.embedding_dim)
        .reshape(batch_size, -1, layer.embedding_dim)
    )
    grad_sample = torch.zeros(
        batch_size, *layer.weight.shape, device=layer.weight.device
    )
    grad_sample.scatter_add_(1, index, B.reshape(batch_size, -1, layer.embedding_dim))
    torch.backends.cudnn.deterministic = saved

    create_or_accumulate_grad_sample(layer.weight, grad_sample, batch_dim)

def register_grad_sampler_transformer_pg_embedding() -> None:
    utils.register_grad_sampler(transformer_pg.Embedding)(compute_embedding_grad_sample)