from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.sampling import make_evaluation_grid


def exact_solution(points: torch.Tensor, mode: int = 1) -> torch.Tensor:
    x = points[:, 0:1]
    y = points[:, 1:2]
    frequency = float(mode) * torch.pi
    return torch.sin(frequency * x) * torch.exp(-frequency * y)


def boundary_condition(points: torch.Tensor, mode: int = 1) -> torch.Tensor:
    x = points[:, 0:1]
    y = points[:, 1:2]
    frequency = float(mode) * torch.pi
    top_scale = torch.exp(-points.new_tensor(frequency))
    values = torch.zeros_like(x)

    bottom_mask = torch.isclose(y, torch.zeros_like(y))
    top_mask = torch.isclose(y, torch.ones_like(y))
    values = torch.where(bottom_mask, torch.sin(frequency * x), values)
    values = torch.where(top_mask, torch.sin(frequency * x) * top_scale, values)
    return values


@torch.no_grad()
def predict(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    return model(points)


def l2_relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.linalg.norm(prediction - target)
    denominator = torch.linalg.norm(target)
    return float((numerator / denominator).item())


def max_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.max(torch.abs(prediction - target)).item())


def evaluate_on_grid(
    model: torch.nn.Module,
    residual_function,
    *,
    grid_size: int,
    device: torch.device,
    mode: int = 1,
    dtype: torch.dtype = torch.float32,
) -> dict[str, np.ndarray | float]:
    xx, yy, points = make_evaluation_grid(grid_size, device=device, dtype=dtype)
    with torch.no_grad():
        prediction = model(points)
        target = exact_solution(points, mode=mode)
        abs_error = torch.abs(prediction - target)
        l2_error = l2_relative_error(prediction, target)
        max_error = max_absolute_error(prediction, target)

    residual = residual_function(model, points).abs()

    return {
        "x": xx.detach().cpu().numpy(),
        "y": yy.detach().cpu().numpy(),
        "prediction": prediction.reshape(grid_size, grid_size).detach().cpu().numpy(),
        "target": target.reshape(grid_size, grid_size).detach().cpu().numpy(),
        "absolute_error": abs_error.reshape(grid_size, grid_size).detach().cpu().numpy(),
        "residual": residual.reshape(grid_size, grid_size).detach().cpu().numpy(),
        "l2_relative_error": l2_error,
        "max_absolute_error": max_error,
    }


def plot_loss_curves(history: dict[str, list[float]], output_path: Path) -> None:
    steps = history["step"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(steps, history["pde_loss"], color="#1f77b4")
    axes[0].set_title("PDE Residual Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")

    axes[1].plot(steps, history["bc_loss"], color="#d62728")
    axes[1].set_title("Boundary Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_field(field: np.ndarray, output_path: Path, title: str = "Absolute Error") -> None:
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_solution_and_residual(
    evaluation: dict[str, np.ndarray | float],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fields = [
        ("Prediction", evaluation["prediction"], "viridis"),
        ("Analytical", evaluation["target"], "viridis"),
        ("Residual |∇²u|", evaluation["residual"], "magma"),
    ]
    for ax, (label, field, cmap) in zip(axes, fields):
        image = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap=cmap)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
