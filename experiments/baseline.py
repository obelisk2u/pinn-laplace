from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.analysis import evaluate_on_grid, plot_error_field, plot_loss_curves, plot_solution_and_residual, save_metrics
from src.model import FullyConnectedPINN, ModelConfig
from src.pinn import TrainingConfig, laplace_residual, save_training_outputs, set_seed, summarize_history, train_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline PINN run for the 2D Laplace equation.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="experiments/artifacts/baseline")
    parser.add_argument("--adam-steps", type=int, default=10_000)
    parser.add_argument("--lbfgs-steps", type=int, default=5_000)
    parser.add_argument("--num-interior", type=int, default=2_000)
    parser.add_argument("--num-boundary-per-edge", type=int, default=200)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--lambda-pde", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    set_seed(args.seed)
    model = FullyConnectedPINN(
        ModelConfig(hidden_layers=args.hidden_layers, hidden_units=args.hidden_units)
    )
    config = TrainingConfig(
        device=args.device,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        num_interior=args.num_interior,
        num_boundary_per_edge=args.num_boundary_per_edge,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        seed=args.seed,
    )
    model, history, _ = train_pinn(model, config)
    evaluation = evaluate_on_grid(
        model,
        laplace_residual,
        grid_size=100,
        device=torch.device(args.device),
        mode=config.mode,
    )

    save_training_outputs(model, config, history, output_dir)
    plot_loss_curves(history, output_dir / "loss_curves.png")
    plot_solution_and_residual(evaluation, output_dir / "solution_residual.png", "Baseline PINN")
    plot_error_field(evaluation["absolute_error"], output_dir / "absolute_error.png")
    metrics = {
        "l2_relative_error": evaluation["l2_relative_error"],
        "max_absolute_error": evaluation["max_absolute_error"],
        **summarize_history(history),
    }
    save_metrics(metrics, output_dir / "metrics.json")


if __name__ == "__main__":
    main()
