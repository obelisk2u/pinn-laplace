from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import qmc


@dataclass(frozen=True)
class SampleBatch:
    interior: torch.Tensor
    boundary_points: torch.Tensor
    boundary_values: torch.Tensor


def sample_interior_lhs(
    num_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    points = sampler.random(n=num_points)
    return torch.tensor(points, device=device, dtype=dtype)


def sample_boundary_uniform(
    num_points_per_edge: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    grid = torch.linspace(0.0, 1.0, num_points_per_edge, device=device, dtype=dtype).unsqueeze(1)
    zeros = torch.zeros_like(grid)
    ones = torch.ones_like(grid)

    bottom = torch.cat((grid, zeros), dim=1)
    top = torch.cat((grid, ones), dim=1)
    left = torch.cat((zeros, grid), dim=1)
    right = torch.cat((ones, grid), dim=1)

    return torch.cat((bottom, top, left, right), dim=0)


def sample_training_batch(
    num_interior: int,
    num_boundary_per_edge: int,
    *,
    boundary_function,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> SampleBatch:

    interior = sample_interior_lhs(num_interior, device=device, dtype=dtype, seed=seed)
    boundary_points = sample_boundary_uniform(
        num_boundary_per_edge,
        device=device,
        dtype=dtype,
    )
    boundary_values = boundary_function(boundary_points)
    return SampleBatch(
        interior=interior,
        boundary_points=boundary_points,
        boundary_values=boundary_values,
    )


def make_evaluation_grid(
    grid_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    axis = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    return xx, yy, points