from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.analysis import evaluate_on_grid, save_metrics
from src.model import FullyConnectedPINN, ModelConfig
from src.pinn import TrainingConfig, laplace_residual, set_seed, train_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PINN ablations.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="experiments/artifacts/ablations")
    parser.add_argument("--adam-steps", type=int, default=4_000)
    parser.add_argument("--lbfgs-steps", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "baseline",
            "model": ModelConfig(hidden_layers=4, hidden_units=64),
            "train": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=100.0,
            ),
        },
        {
            "name": "unweighted_loss",
            "model": ModelConfig(hidden_layers=4, hidden_units=64),
            "train": TrainingConfig(
                device=args.device,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                lambda_pde=1.0,
                lambda_bc=1.0,
            ),
        },
        {
            "name": "wider_network",
            "model": ModelConfig(hidden_layers=4, hidden_units=128),
            "train": TrainingConfig(device=args.device, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps),
        },
        {
            "name": "deeper_network",
            "model": ModelConfig(hidden_layers=6, hidden_units=64),
            "train": TrainingConfig(device=args.device, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps),
        },
        {
            "name": "adam_only",
            "model": ModelConfig(hidden_layers=4, hidden_units=64),
            "train": TrainingConfig(device=args.device, adam_steps=args.adam_steps + args.lbfgs_steps, lbfgs_steps=0),
        },
    ]

    rows: list[dict[str, float | str]] = []
    for experiment in experiments:
        run_dir = output_dir / str(experiment["name"])
        set_seed(experiment["train"].seed)
        model = FullyConnectedPINN(experiment["model"])
        trained_model, history, _ = train_pinn(model, experiment["train"])
        evaluation = evaluate_on_grid(
            trained_model,
            laplace_residual,
            grid_size=100,
            device=torch.device(args.device),
            mode=experiment["train"].mode,
        )
        metrics = {
            "l2_relative_error": evaluation["l2_relative_error"],
            "max_absolute_error": evaluation["max_absolute_error"],
            "final_pde_loss": history["pde_loss"][-1],
            "final_bc_loss": history["bc_loss"][-1],
        }
        save_metrics(metrics, run_dir / "metrics.json")
        rows.append({"name": experiment["name"], **metrics})

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
