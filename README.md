# Multilevel Time-Dependent Optimal Transport (Haber & Horesh, 2015)

A high-performance Python implementation, replication, and extension of the multilevel Sequential Quadratic Programming (SQP) solver for time-dependent optimal transport, based on the work of **E. Haber & R. Horesh (2015)**: [*A Multilevel Method for the Solution of Time Dependent Optimal Transport*](https://doi.org/10.4208/nmtma.2015.w02si).

> **Accompanying Report**: For complete theoretical derivations, empirical hyperparameter sweeps, mesh-independence studies, and extended experiments (asymmetric transport, topology changes, and mass merging), refer to [Report.pdf](Report.pdf).

---

## Table of Contents

- [Overview & Problem Formulation](#overview--problem-formulation)
- [Discretization & Solver Architecture](#discretization--solver-architecture)
- [Implementation Guide: Module & Function Reference](#implementation-guide-module--function-reference)
  - [`grid.py` — Staggered Grid Operators](#optimal_transportgridpy--staggered-grid-operators)
  - [`objective.py` — Energy, Gradients, and Gauss-Newton Hessian](#optimal_transportobjectivepy--energy-gradients-and-gauss-newton-hessian)
  - [`constraint.py` — Continuity Residual](#optimal_transportconstraintpy--continuity-residual)
  - [`filter.py` — Globalization via Fletcher-Leyffer Filter](#optimal_transportfilterpy--globalization-via-fletcher-leyffer-filter)
  - [`linear_solver.py` — Saddle-Point & Schur Complement Solvers](#optimal_transportlinear_solverpy--saddle-point--schur-complement-solvers)
  - [`sqp.py` — Outer SQP Optimization Loop](#optimal_transportsqppy--outer-sqp-optimization-loop)
  - [`multilevel.py` — Coarse-to-Fine Prolongation & Warm-Starting](#optimal_transportmultilevelpy--coarse-to-fine-prolongation--warm-starting)
  - [`test_images.py` — Benchmark & Extended Problem Geometries](#optimal_transporttest_imagespy--benchmark--extended-problem-geometries)
  - [`experiments.py` — Paper Replication Suites](#optimal_transportexperimentspy--paper-replication-suites)
  - [`utils.py` — Diagnostic Helpers](#optimal_transportutilspy--diagnostic-helpers)
- [Crucial Numerical Fixes & Engineering Insights](#crucial-numerical-fixes--engineering-insights)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Running Experiments](#running-experiments)

---

## Overview & Problem Formulation

Dynamic optimal transport (the **Benamou–Brenier** formulation) seeks the minimal-kinetic-energy transport path connecting a source density $\mu_0$ at $t = 0$ to a target density $\mu_1$ at $t = 1$. By working with momentum $m = \rho v$ instead of velocity $v$, the problem becomes convex:

$$\min_{m, \rho} \int_0^1 \int_\Omega \frac{|m(x, t)|^p}{\rho(x, t)^{p-1}} \, dx \, dt \quad \text{s.t.} \quad \partial_t \rho + \nabla \cdot m = 0, \quad \rho(\cdot, 0) = \mu_0, \quad \rho(\cdot, 1) = \mu_1$$

For $p = 2$, the minimum value equals the squared 2-Wasserstein distance $W_2^2(\mu_0, \mu_1)$.

Stacking space and time into $w = (m, \rho)^\top$, the constraint simplifies to a single space-time divergence $\nabla_{st} \cdot w = 0$. This gives the system the variational structure of mixed nonlinear flow in porous media, enabling the use of saddle-point linear solvers, Schur complement reduction, and multigrid preconditioning.

---

## Discretization & Solver Architecture

### 1. Staggered Space-Time Grid
Variables live where their natural fluxes reside on an $n_1 \times n_2$ spatial grid with $n_3$ time intervals (uniform mesh spacing $h$):

| Variable | Staggered Location | Array Shape | Description |
| :--- | :--- | :--- | :--- |
| $m_1$ | $x$-faces | $(n_1+1, n_2, n_3)$ | Horizontal momentum flux |
| $m_2$ | $y$-faces | $(n_1, n_2+1, n_3)$ | Vertical momentum flux |
| $\rho_{\text{free}}$ | interior $t$-faces | $(n_1, n_2, n_3-1)$ | Dynamic density (free unknowns) |
| $\rho_{\text{full}}$ | all $t$-faces | $(n_1, n_2, n_3+1)$ | Density including fixed boundaries $\rho_0 = \mu_0, \rho_{n_3} = \mu_1$ |
| $\lambda$ | cell centers | $(n_1, n_2, n_3)$ | Lagrange multipliers for the continuity constraint |

> **Why Staggered?** A collocated/nodal grid creates spurious checkerboard null spaces and is not $h$-elliptic. Staggered placement guarantees stable discrete divergence and gradient operators without artificial stabilization.

### 2. Averaging Order
Because faces and cell centers are offset, evaluation of $|m|^p / \rho^{p-1}$ requires spatial and temporal averaging ($A_s, A_t$). The order of operations is critical:
- **Momentum**: Square first at faces, then average spatially: $A_s(m_1^2 + m_2^2)$.
- **Density**: Invert first at time faces, then average temporally: $A_t(1/\rho^{p-1})$.

This prevents alternating $\pm 1$ momentum fields from falsely averaging to zero, and forces $f(m, \rho) \to \infty$ as $\rho \to 0$, naturally enforcing physical density positivity.

### 3. Outer SQP Loop & Linear Solvers
The KKT optimality conditions are solved using an inexact-Newton **Sequential Quadratic Programming (SQP)** outer loop. Dropping mixed $m$-$\rho$ second derivatives yields a positive block-diagonal Gauss-Newton Hessian approximation $\hat{A}$. Each outer step solves the saddle-point system:

$$\begin{pmatrix} \hat{A} & D^\top \\ D & 0 \end{pmatrix} \begin{pmatrix} \delta w \\ \delta \lambda \end{pmatrix} = -\begin{pmatrix} \nabla_w L \\ \nabla_\lambda L \end{pmatrix}$$

Two inner linear solver paths are implemented:
1. **`cg_sgs` (Reduced Schur Complement)**: Eliminates $\delta w$ analytically ($\delta w = -\hat{A}^{-1}(D^\top \delta\lambda + \nabla_w L)$) and solves the symmetric positive definite Schur system $S \delta\lambda = \text{rhs}$ ($S = D \hat{A}^{-1} D^\top$) with Conjugate Gradients preconditioned by Symmetric Gauss-Seidel (SGS).
2. **`gmres_amg` (Full Saddle-Point System)**: Solves the unreduced system matrix-free using GMRES with a block-triangular preconditioner utilizing one Ruge-Stüben Algebraic Multigrid (AMG) V-cycle on $S$ via `pyamg`.

---

## Implementation Guide: Module & Function Reference

The core logic is located in the [`optimal_transport/`](optimal_transport/) package. Below is a high-level walkthrough of each module and function.

### `optimal_transport/grid.py` — Staggered Grid Operators

Provides discrete differential and averaging operators on the staggered space-time mesh.

- **`div_st(m1, m2, rho, h)`**:
  - *Purpose*: Computes the discrete space-time divergence $\nabla_{st} \cdot (m, \rho) = D_1 m_1 + D_2 m_2 + D_3 \rho$ at cell centers.
  - *Returns*: Array of shape $(n_1, n_2, n_3)$.
- **`grad_st(lam, h)`**:
  - *Purpose*: Adjoint of `div_st` ($D^\top$). Maps cell-centered Lagrange multipliers $\lambda$ to gradient fields on faces $(g_{m1}, g_{m2}, g_{\rho,\text{free}})$.
  - *No-Flux Boundary Condition*: Boundary momentum face components ($m_1$ at $x=0, 1$ and $m_2$ at $y=0, 1$) are explicitly zeroed out to enforce hard no-flux boundaries ($m \cdot n = 0$ on $\partial\Omega$).
- **`avg_x(f)`, `avg_y(f)`, `avg_t(f)`**:
  - *Purpose*: Midpoint averaging operators moving face-centered quantities to cell centers (or intermediate faces) along the specified axis.
- **`avg_x_adj(f)`, `avg_y_adj(f)`, `avg_t_adj(f)`**:
  - *Purpose*: Adjoint averaging operators mapping cell-centered quantities back to staggered faces.

---

### `optimal_transport/objective.py` — Energy, Gradients, and Gauss-Newton Hessian

Implements the Benamou-Brenier discrete objective function and its derivatives.

- **`objective(m1, m2, rho, h, p=2, eps=1e-8)`**:
  - *Purpose*: Evaluates the discrete kinetic energy integral $h^3 \sum \left[ A_s(|m|^p) \cdot A_t(1/\rho^{p-1}) \right]$.
  - *Details*: Uses regularized $|m|_\epsilon = \sqrt{m^2 + \epsilon^2}$ to support $p \in (1, 2]$. Takes the full $\rho$ array $(n_1, n_2, n_3+1)$.
- **`grad_m(m1, m2, rho, lam, h, p=2, eps=1e-8)`**:
  - *Purpose*: Evaluates the gradient of the Lagrangian $L$ with respect to momentum fields $m_1$ and $m_2$:
    $$\nabla_{m} L = \nabla_m f(m, \rho) + D_{1,2}^\top \lambda$$
- **`grad_rho(m1, m2, rho, lam, h, p=2, eps=1e-8)`**:
  - *Purpose*: Evaluates the gradient of the Lagrangian $L$ with respect to the *free interior* density time-faces ($\rho[:, :, 1:-1]$):
    $$\nabla_{\rho_{\text{free}}} L = \nabla_{\rho} f(m, \rho) + D_3^\top \lambda$$
- **`hessian_diag(m1, m2, rho, h, p=2, eps=1e-8)`**:
  - *Purpose*: Calculates the block-diagonal Gauss-Newton Hessian approximations $\hat{A}_{m1}, \hat{A}_{m2}, \hat{A}_\rho$.
  - *Key Feature*: Enforces a strictly positive floor of $h^3$ on $\hat{A}_\rho$. This prevents the $\rho$-block from vanishing at $m = 0$, preserving the conditioning of the Schur complement.

---

### `optimal_transport/filter.py` — Globalization via Fletcher-Leyffer Filter

Replaces heuristic penalty parameters with a two-dimensional filter that trades off objective reduction $f$ and constraint violation $h_{\text{viol}} = \|C\|_1$.

- **`Filter(gamma_f, gamma_h, beta, max_h_factor)`**:
  - `initialize(h_0)`: Sets the maximum allowed violation $h_{\max} = \beta h_0$. Initialized *empty* to prevent the initial $(0, h_0)$ iterate from degenerately blocking small early steps.
  - `is_acceptable(f_trial, h_trial)`: Checks whether a candidate step is dominated by existing filter entries or exceeds the constraint increase cap `max_h_factor * h_current`.
  - `add(f_new, h_new)`: Inserts accepted iterates into the Pareto frontier, conditioned on a relative threshold ($f_{\text{trial}} > f_{\text{cur}}(1 + 10^{-4})$) to prevent floating-point staircase buildup.
  - `accept(h_new)`: Updates the current constraint violation tracking level.

---

### `optimal_transport/linear_solver.py` — Saddle-Point & Schur Complement Solvers

Handles the inner linear solves for Newton directions.

- **`schur_matvec(x_flat, A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3)`**:
  - *Purpose*: Matrix-free matrix-vector product for the Schur complement operator $S x = D \hat{A}^{-1} D^\top x$.
- **`_build_derivative_matrices(n1, n2, n3, h)`**:
  - *Purpose*: Builds explicit sparse CSR difference matrices $D_1, D_2, D_{3,\text{free}}$ mapping interior face DOFs to cell centers.
- **`build_schur_sparse(A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3)`**:
  - *Purpose*: Constructs the explicit sparse matrix $S = D_1 \hat{A}_{m1}^{-1} D_1^\top + D_2 \hat{A}_{m2}^{-1} D_2^\top + D_{3f} \hat{A}_\rho^{-1} D_{3f}^\top$ used to build preconditioners.
- **`build_sgs_preconditioner(S_sparse)`**:
  - *Purpose*: Implements a Symmetric Gauss-Seidel (SGS) preconditioner using triangular forward/backward solves on $(L + D) D^{-1} (U + D)$.
- **`solve_schur_system(rhs, ... tol=1e-4, maxiter=500)`**:
  - *Purpose*: Solves $S \delta\lambda = \text{rhs}$ via Preconditioned Conjugate Gradients (PCG) using the matrix-free matvec and SGS preconditioner.
- **`recover_dw(delta_lam, g_m1, g_m2, g_rho, ...)`**:
  - *Purpose*: Back-substitutes $\delta\lambda$ to recover primal updates: $\delta w = -\hat{A}^{-1}(D^\top \delta\lambda + \nabla_w L)$.
- **`solve_saddle_system_gmres(rhs_m1, rhs_m2, rhs_rho, rhs_lam, ... tol=0.1)`**:
  - *Purpose*: Solves the unreduced saddle-point system matrix-free using GMRES with a Ruge-Stüben AMG V-cycle on $S$ (via `pyamg`) as the Schur block preconditioner.

---

### `optimal_transport/sqp.py` — Outer SQP Optimization Loop

Orchestrates inexact-Newton optimization.

- **`initialize(mu0, mu1, n3, h)`**:
  - *Purpose*: Forms the initial state: zero momentum ($m_1=0, m_2=0$), zero potential ($\lambda=0$), and linear density interpolation $\rho(x, t) = (1-t)\mu_0(x) + t\mu_1(x)$.
- **`_make_rho_full(mu0, mu1, rho_free)`**:
  - *Purpose*: Combines boundary density conditions $\mu_0, \mu_1$ with interior active unknowns $\rho_{\text{free}}$ into a unified $(n_1, n_2, n_3+1)$ array.
- **`sqp(m1, m2, rho_free, lam, mu0, mu1, h, ...)`**:
  - *Purpose*: Main SQP solver loop.
  - *Workflow*:
    1. Computes Lagrangian gradients and Gauss-Newton Hessian diagonals.
    2. Checks stopping criteria: requires **both** relative primal feasibility ($\|C\|_1 / \|C\|_1^{(0)} < \text{tol}$) and normalized dual feasibility ($\|\nabla_w L\|_{\text{int}} / \text{scale}_w < \text{tol}_w$).
    3. Dispatches inner solve (`cg_sgs` or `gmres_amg`) to obtain Newton step $(\delta m, \delta\rho, \delta\lambda)$.
    4. Backtracking line search with positivity checking ($\rho + \alpha \delta\rho > 0$) and filter acceptance.
    5. Fallback restoration phase: if line search fails, attempts pure constraint reduction steps before skipping.

---

### `optimal_transport/multilevel.py` — Coarse-to-Fine Prolongation & Warm-Starting

Accelerates convergence and enables high-contrast optimization.

- **`prolong_solution(m1_c, m2_c, rho_free_c, lam_c, n1_f, n2_f, n3_f)`**:
  - *Purpose*: Bilinear/trilinear interpolation (`scipy.ndimage.zoom`, order=1) of all variables from a coarse grid to a fine grid, respecting individual staggered face locations.
- **`multilevel_sqp(levels, contrast, p=2, ...)`**:
  - *Purpose*: Cascades through a hierarchy of grids (e.g., $16^2 \times 8 \to 32^2 \times 16 \to 64^2 \times 32$), warm-starting each finer level with the prolonged solution from the previous level.

---

## Crucial Numerical Fixes & Engineering Insights

Implementing Haber & Horesh (2015) revealed several numerical edge cases left unspecified in the original paper. As documented in [Report.pdf](Report.pdf), the following adjustments were required for convergence:

1. **Stabilizing $\hat{A}_\rho$ at Zero Momentum**:
   The analytical $\rho$-Hessian diagonal is proportional to $|m|^2$. At initialization ($m = 0$), $\hat{A}_\rho$ collapses to machine epsilon. In the Schur complement $S = D \hat{A}^{-1} D^\top$, the $\rho$ component scales as $\hat{A}_\rho^{-1}$, dominating the momentum contribution by a factor of $\sim 10^8$. This causes CG to return erroneous search directions that increase constraint violations.
   *Fix*: Floor $\hat{A}_\rho$ at $h^3$ (the natural scale of the $m$-block diagonal).

2. **Dual Feasibility in Stopping Criteria**:
   Because $\partial f / \partial m = 2m/\rho = 0$ at $m = 0$, the very first Newton step acts entirely on the continuity constraint. On isotropic grids, an exact solve of $S \delta\lambda = -C$ yields $\delta\rho = 0$. If checking only primal feasibility $\|C\|_1 < \text{tol}$, the solver terminates after a single step, returning the static linear interpolation ("crossfade") rather than the true dynamic optimal transport geodesic.
   *Fix*: Require convergence of both primal feasibility and dual gradient feasibility ($\|\nabla_w L\|_{\text{int}} / \text{scale}_w < 10\sqrt{\text{tol}}$), tracked using a running maximum over iterations.

3. **No-Flux Spatial Boundary Conditions ($m \cdot n = 0$)**:
   The unconstrained formulation allows boundary momentum faces to act as free variables. For symmetric problems, boundary flux naturally cancels. However, for asymmetric problems (such as diagonal translation), the optimizer finds it energetically cheaper to "leak" mass out of the domain walls and re-inject it elsewhere.
   *Fix*: Pin spatial boundary momentum faces to zero and restrict derivative matrices $D_1, D_2$ strictly to interior face DOFs.

4. **Filter Adjustments**:
   - Initializing the Fletcher-Leyffer filter empty avoids blocking valid early steps under small step sizes.
   - Adding a relative threshold ($f_{\text{trial}} > f_{\text{cur}}(1 + 10^{-4})$) prevents floating-point noise near the optimum from accumulating a dense filter staircase.

---

## Repository Structure

```
.
├── Report.pdf                    # Complete final project report & experimental analysis
├── Presentation.pdf              # Summary presentation slides
├── optimal_transport/            # Core library package
│   ├── __init__.py
│   ├── grid.py                   # Staggered grid operators (div, grad, averaging)
│   ├── objective.py              # Benamou-Brenier energy, gradients, Gauss-Newton Hessian
│   ├── constraint.py             # Continuity constraint residual
│   ├── filter.py                 # Fletcher-Leyffer filter & line search acceptance
│   ├── linear_solver.py          # Schur complement (CG+SGS) & Saddle-Point (GMRES+AMG)
│   ├── sqp.py                    # Main SQP optimization loop
│   ├── multilevel.py             # Solution prolongation & coarse-to-fine solver
│   ├── test_images.py            # Benchmark image generation (mu0, mu1)
│   ├── experiments.py            # Paper replication experiments (Exp 1, 2, 3, 5)
│   └── utils.py                  # Statistics logging and utility routines
├── scripts/                      # Experiment runners and tuning sweeps
│   ├── run_experiments.py        # Unified CLI for replication and new test problems
│   ├── test_new_problems.py      # Runs asymmetric, ring->disk, and two-blob tests
│   ├── test_cg_sweep.py          # CG tolerance parameter sweeps
│   ├── test_max_h_factor.py      # Filter step cap parameter sweep
│   └── tune_gmres.py             # GMRES tolerance parameter sweeps
└── output/                       # Generated figures, convergence logs, and animations
```

---

## Installation & Setup

Make sure Python 3.9+ is installed. Install required dependencies:

```bash
pip install numpy scipy matplotlib pillow
```

*(Optional, recommended for large grids)* To run the matrix-free `gmres_amg` solver path:
```bash
pip install pyamg
```

---

## Running Experiments

### 1. Paper Replication Suite (`scripts/run_experiments.py`)

Run all replication experiments using the default `cg_sgs` path:
```bash
python scripts/run_experiments.py
```

Run specific experiments or use the `gmres_amg` path:
```bash
# Run Experiment 1 (Mesh independence) and Experiment 3 (Multilevel)
python scripts/run_experiments.py --exp 1 3

# Compare CG+SGS vs GMRES+AMG
python scripts/run_experiments.py --exp 1 --method both

# Fast smoke run skipping 64x64x40 grids
python scripts/run_experiments.py --no-64
```

### 2. Extended Test Problems (`scripts/test_new_problems.py`)

Run the three extended test cases (Asymmetric Gaussian, Ring to Disk, Two Blobs):
```bash
# Quick smoke test (16x16x10 grid)
python scripts/test_new_problems.py --smoke

# Full evaluation across all grids with GIF generation
python scripts/test_new_problems.py
```

Outputs are automatically saved to `output/new_problems/`.

---

## References

1. **E. Haber and R. Horesh**, *A Multilevel Method for the Solution of Time Dependent Optimal Transport*, Numerical Mathematics: Theory, Methods and Applications, 8(1), pp. 97–111, 2015. [DOI: 10.4208/nmtma.2015.w02si](https://doi.org/10.4208/nmtma.2015.w02si).
2. **J.-D. Benamou and Y. Brenier**, *A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem*, Numerische Mathematik, 84(3), pp. 375–393, 2000.
