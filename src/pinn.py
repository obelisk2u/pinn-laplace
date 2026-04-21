from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.analysis import boundary_condition
from src.sampling import SampleBatch, sample_training_batch


@dataclass(frozen=True)
class TrainingConfig:
    adam_steps: int = 10_000
    lbfgs_steps: int = 5_000
    learning_rate: float = 1e-3
    lbfgs_learning_rate: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 100.0
    num_interior: int = 2_000
    num_boundary_per_edge: int = 200
    log_every: int = 100
    seed: int = 7
    mode: int = 1
    device: str = "cpu"
    dtype: str = "float32"
    verbose: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def laplace_residual(model: nn.Module, points: torch.Tensor) -> torch.Tensor:
    points = points.detach().clone().requires_grad_(True)
    prediction = model(points)
    gradients = torch.autograd.grad(
        prediction,
        points,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]
    du_dx = gradients[:, 0:1]
    du_dy = gradients[:, 1:2]

    d2u_dx2 = torch.autograd.grad(
        du_dx,
        points,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(
        du_dy,
        points,
        grad_outputs=torch.ones_like(du_dy),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    return d2u_dx2 + d2u_dy2


def compute_loss_terms(
    model: nn.Module,
    batch: SampleBatch,
    *,
    lambda_pde: float,
    lambda_bc: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = laplace_residual(model, batch.interior)
    boundary_prediction = model(batch.boundary_points)
    pde_loss = torch.mean(residual.square())
    bc_loss = torch.mean((boundary_prediction - batch.boundary_values).square())
    total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss
    return total_loss, pde_loss, bc_loss


def create_training_batch(config: TrainingConfig) -> SampleBatch:
    device = torch.device(config.device)
    dtype = resolve_dtype(config.dtype)
    return sample_training_batch(
        config.num_interior,
        config.num_boundary_per_edge,
        boundary_function=lambda points: boundary_condition(points, mode=config.mode),
        device=device,
        dtype=dtype,
        seed=config.seed,
    )


def train_pinn(
    model: nn.Module,
    config: TrainingConfig,
) -> tuple[nn.Module, dict[str, list[float]], SampleBatch]:
    set_seed(config.seed)
    device = torch.device(config.device)
    dtype = resolve_dtype(config.dtype)
    model.to(device=device, dtype=dtype)
    batch = create_training_batch(config)
    history: dict[str, list[float]] = {"step": [], "total_loss": [], "pde_loss": [], "bc_loss": []}

    def record(step: int, total_loss: torch.Tensor, pde_loss: torch.Tensor, bc_loss: torch.Tensor) -> None:
        history["step"].append(step)
        history["total_loss"].append(float(total_loss.detach().cpu().item()))
        history["pde_loss"].append(float(pde_loss.detach().cpu().item()))
        history["bc_loss"].append(float(bc_loss.detach().cpu().item()))
        if config.verbose:
            print(
                f"step={step:6d} total={history['total_loss'][-1]:.6e} "
                f"pde={history['pde_loss'][-1]:.6e} bc={history['bc_loss'][-1]:.6e}"
            )

    adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for step in range(1, config.adam_steps + 1):
        adam.zero_grad()
        total_loss, pde_loss, bc_loss = compute_loss_terms(
            model,
            batch,
            lambda_pde=config.lambda_pde,
            lambda_bc=config.lambda_bc,
        )
        total_loss.backward()
        adam.step()
        if step == 1 or step % config.log_every == 0:
            record(step, total_loss, pde_loss, bc_loss)

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=config.lbfgs_learning_rate,
        max_iter=1,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    for step in range(config.adam_steps + 1, config.adam_steps + config.lbfgs_steps + 1):
        losses: dict[str, torch.Tensor] = {}

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            total_loss, pde_loss, bc_loss = compute_loss_terms(
                model,
                batch,
                lambda_pde=config.lambda_pde,
                lambda_bc=config.lambda_bc,
            )
            total_loss.backward()
            losses["total"] = total_loss.detach()
            losses["pde"] = pde_loss.detach()
            losses["bc"] = bc_loss.detach()
            return total_loss

        lbfgs.step(closure)
        if step % config.log_every == 0:
            record(step, losses["total"], losses["pde"], losses["bc"])

    return model, history, batch


def save_training_outputs(
    model: nn.Module,
    config: TrainingConfig,
    history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2))


def summarize_history(history: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "final_total_loss": history["total_loss"][-1],
        "final_pde_loss": history["pde_loss"][-1],
        "final_bc_loss": history["bc_loss"][-1],
        "logged_steps": len(history["step"]),
    }
