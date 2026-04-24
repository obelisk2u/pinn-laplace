from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.analysis import evaluate_on_grid, plot_error_field, plot_loss_curves, plot_solution_and_residual, save_metrics
from src.model import FullyConnectedPINN, ModelConfig
from src.pinn import TrainingConfig, laplace_residual, set_seed, train_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce known PINN failure modes.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="experiments/artifacts/failure_modes")
    parser.add_argument("--adam-steps", type=int, default=4_000)
    parser.add_argument("--lbfgs-steps", type=int, default=1_000)
    return parser.parse_args()


def run_case(
    *,
    name: str,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    output_dir: Path,
    figure_title: str,
) -> dict[str, float | str]:
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_config.seed)
    model = FullyConnectedPINN(model_config)
    trained_model, history, _ = train_pinn(model, train_config)
    evaluation = evaluate_on_grid(
        trained_model,
        laplace_residual,
        grid_size=100,
        device=torch.device(train_config.device),
        mode=train_config.mode,
    )
    plot_loss_curves(history, run_dir / "loss_curves.png")
    plot_solution_and_residual(evaluation, run_dir / "comparison_residual.png", figure_title)
    plot_error_field(evaluation["absolute_error"], run_dir / "absolute_error.png")
    metrics = {
        "name": name,
        "mode": train_config.mode,
        "l2_relative_error": evaluation["l2_relative_error"],
        "max_absolute_error": evaluation["max_absolute_error"],
        "final_pde_loss": history["pde_loss"][-1],
        "final_bc_loss": history["bc_loss"][-1],
    }
    save_metrics(metrics, run_dir / "metrics.json")
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "name": "loss_imbalance_unweighted",
            "model_config": ModelConfig(hidden_layers=4, hidden_units=64),
            "train_config": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=1.0,
            ),
            "title": "Loss Weight Imbalance: λ_bc=1, λ_pde=1",
        },
        {
            "name": "loss_imbalance_weighted",
            "model_config": ModelConfig(hidden_layers=4, hidden_units=64),
            "train_config": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=100.0,
            ),
            "title": "Loss Weight Imbalance Fix: λ_bc=100, λ_pde=1",
        },
        {
            "name": "insufficient_capacity",
            "model_config": ModelConfig(hidden_layers=1, hidden_units=10),
            "train_config": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=100.0,
            ),
            "title": "Insufficient Capacity: 1x10 Network",
        },
        {
            "name": "spectral_bias_high_frequency",
            "model_config": ModelConfig(hidden_layers=4, hidden_units=64),
            "train_config": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=100.0,
                mode=10,
            ),
            "title": "Spectral Bias: High-Frequency Boundary Conditions",
        },
        {
            "name": "spectral_bias_fourier_fix",
            "model_config": ModelConfig(hidden_layers=4, hidden_units=64, fourier_features=32, fourier_scale=4.0),
            "train_config": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=100.0,
                mode=10,
            ),
            "title": "Spectral Bias Fix: Fourier Features",
        },
        {
            "name": "collocation_starvation_50",
            "model_config": ModelConfig(hidden_layers=4, hidden_units=64),
            "train_config": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=100.0,
                num_interior=50,
            ),
            "title": "Collocation Starvation: 50 Interior Points",
        },
        {
            "name": "collocation_starvation_2000",
            "model_config": ModelConfig(hidden_layers=4, hidden_units=64),
            "train_config": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=100.0,
                num_interior=2_000,
            ),
            "title": "Collocation Density Reference: 2000 Interior Points",
        },
    ]

    rows = [
        run_case(
            name=case["name"],
            model_config=case["model_config"],
            train_config=case["train_config"],
            output_dir=output_dir,
            figure_title=case["title"],
        )
        for case in cases
    ]

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
