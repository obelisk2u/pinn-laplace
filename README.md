# PINN Laplace

PyTorch implementation of a Physics-Informed Neural Network (PINN) for the 2D steady-state heat equation on the unit square:

$$
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0, \quad (x, y) \in [0,1]^2
$$

with Dirichlet boundary conditions

$$
u(x,0)=\sin(\pi x), \quad u(x,1)=\sin(\pi x)e^{-\pi}, \quad u(0,y)=u(1,y)=0
$$

The analytical solution is

$$
u(x,y)=\sin(\pi x)e^{-\pi y}
$$

and is used only for validation and visualization, never for interior training targets.

## Repository Layout

```text
pinn-laplace/
├── src/
│   ├── model.py
│   ├── pinn.py
│   ├── sampling.py
│   └── analysis.py
├── experiments/
│   ├── baseline.py
│   ├── ablations.py
│   └── failure_modes.py
├── notebooks/
│   └── results.ipynb
├── tests/
├── README.md
└── requirements.txt
```

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Method

The PINN is a fully connected `tanh` network with Xavier initialization and no batch normalization. Interior collocation points are sampled with Latin hypercube sampling, and boundary points are sampled uniformly on each edge. The loss is

$$
\mathcal{L} = \lambda_{\text{pde}} \, \mathcal{L}_{\text{pde}} + \lambda_{\text{bc}} \, \mathcal{L}_{\text{bc}}
$$

where

$$
\mathcal{L}_{\text{pde}} = \frac{1}{N_f}\sum_i (\partial_{xx}u_\theta + \partial_{yy}u_\theta)^2,
\quad
\mathcal{L}_{\text{bc}} = \frac{1}{N_b}\sum_j (u_\theta - u_{\text{bc}})^2
$$

All derivatives are computed with `torch.autograd.grad(..., create_graph=True)`.

## Reproducing Experiments

Baseline training:

```bash
python experiments/baseline.py --device cpu
```

Ablations:

```bash
python experiments/ablations.py --device cpu
```

Failure modes:

```bash
python experiments/failure_modes.py --device cpu
```

Artifacts are written under `experiments/artifacts/`.

## Failure Modes

`Loss weight imbalance`
With `lambda_bc=1` and `lambda_pde=1`, the optimizer can reduce the smoother PDE residual term while still leaving visible boundary mismatch. In practice the network drifts toward a low-curvature interior field that is not tightly anchored to the Dirichlet data. Increasing `lambda_bc` to `100` corrects this by forcing the optimizer to satisfy the boundary manifold before refining the interior residual. The separate loss traces make the imbalance visible: boundary loss stays elevated much longer in the unweighted setting.

`Insufficient network capacity`
A `1 x 10` network has limited expressive power for simultaneously matching the boundary data and producing a smooth harmonic field. The optimization landscape becomes effectively constrained by representation error rather than only training dynamics. Even when the boundary fit improves locally, the residual heatmap exposes systematic interior error. The result is a visibly underfit solution with structured residual bands.

`Stiff gradients / spectral bias`
For the high-frequency target $u(x,y)=\sin(10\pi x)e^{-10\pi y}$, vanilla `tanh` PINNs exhibit the standard spectral bias toward low-frequency functions. The boundary condition oscillates rapidly in `x`, but the network and optimizer preferentially fit a smoother surrogate. That mismatch appears both in elevated boundary loss and in persistent residual concentration near the lower boundary. Optional Fourier features improve the input representation and make the high-frequency mode easier to optimize.

`Collocation point starvation`
Using only `50` interior points undersamples the residual field and leaves large regions of the domain effectively unconstrained. The model can interpolate the sparse residual checks while violating the PDE between those points. This is visible in the residual heatmap as localized error pockets away from sampled locations. Increasing the interior budget to `2000` materially improves spatial coverage and reduces those blind spots.

## Results Table

Populate after running the scripts:

| Condition | L2 Relative Error | Max Absolute Error |
| --- | ---: | ---: |
| Baseline (`lambda_bc=100`, `2000` collocation) | `TBD` | `TBD` |
| Loss imbalance (`lambda_bc=1`) | `TBD` | `TBD` |
| Insufficient capacity (`1 x 10`) | `TBD` | `TBD` |
| Spectral bias (`10π`) | `TBD` | `TBD` |
| Spectral bias + Fourier features | `TBD` | `TBD` |
| Collocation starvation (`50` points) | `TBD` | `TBD` |

## Testing

```bash
pytest
```
