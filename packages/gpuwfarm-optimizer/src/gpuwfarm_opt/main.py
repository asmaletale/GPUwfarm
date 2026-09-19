"""
Wind Farm Layout + Yaw Optimiser — GPU Genetic Algorithm
FLORIS-traceable physics layer.

Usage:
    python main.py [--combination SOSFS|FLS|MAX] [--generations N]
                   [--pop N] [--turbines N] [--multispeed]
                   [--optimize both|layout|yaw]

Physics sources:
    Wake velocity:   FLORIS floris/core/wake_velocity/gauss.py
    Turbulence:      FLORIS floris/core/wake_turbulence/crespo_hernandez.py
    Deflection:      FLORIS floris/core/wake_deflection/gauss.py
    Combination:     FLORIS floris/core/wake_combination/{sosfs,fls,max}.py
    Power curve:     FLORIS floris/core/turbine/operation_models.py (NREL 5 MW)
    Wind rose / AEP: FLORIS floris/wind_data.py + floris/floris_model.py
"""
#todo: verify that aep is consistent with floris and that the setup is the same (config-wise)
#todo: add visualization like in legacy code
#todo: add bathymetry
#todo: visualize min lcoe and min vi layouts on the pareto front
#todo: add wake visualization like in floris (wake velocity deficit, turbulence intensity)
#todo: add rotor discretization (like in floris) to have more accurate wake modeling and power calculation

from __future__ import annotations
import argparse
import os
import sys

# On Windows without a full CUDA toolkit, PyTorch ships the CUDA DLLs
# (curand, cublas, etc.) that CuPy needs. Register that directory before
# importing CuPy so cuda-pathfinder can locate them.
if sys.platform == "win32":
    _torch_lib = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib"
    )
    _torch_lib = os.path.normpath(_torch_lib)
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig, CostConfig, VisualImpactConfig
from gpuwfarm_opt.config import GAConfig, PSOConfig, GDConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.swarm import ParticleSwarm
from gpuwfarm_opt.gradient import GradientDescent
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_core.loaders.floris_yaml import load_floris_yaml


# Each optimizer pairs with its own config; every run() returns the same
# 4-tuple, so nothing below the call has to branch on which one was picked.
_CONFIGS = {"ga": GAConfig, "pso": PSOConfig, "gd": GDConfig}
_OPTIMIZERS = {
    "ga": GeneticAlgorithm, "pso": ParticleSwarm, "gd": GradientDescent,
}


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GPU Wind Farm Optimiser")
    p.add_argument("--floris-yaml",  default=None, metavar="PATH",
                   help="Path to a FLORIS v4 input YAML; overrides --combination, "
                        "--turbines, and wind-rose flags")
    p.add_argument("--optimizer", default="ga", choices=["ga", "pso", "gd"],
                   help="Search algorithm: genetic (default), multi-objective "
                        "particle swarm, or SPSA gradient descent with Chebyshev "
                        "decomposition. PSO and GD are inherently multi-objective")
    p.add_argument("--objectives", default="lcoe_vi", choices=["lcoe_vi", "aep_vi"],
                   help="Objective pair to minimise")
    p.add_argument("--combination",  default="SOSFS", choices=["SOSFS", "FLS", "MAX"])
    p.add_argument("--generations",  type=int,   default=150)
    p.add_argument("--pop",          type=int,   default=256)
    p.add_argument("--turbines",     type=int,   default=20)
    p.add_argument("--optimize",     default="both", choices=["both", "layout", "yaw"],
                   help="Decision variables: both, layout only (yaw fixed at 0), "
                        "or yaw only (layout fixed — from --floris-yaml if given)")
    p.add_argument("--multispeed",   action="store_true",
                   help="Use 12-sector × 11-speed Weibull wind rose")
    p.add_argument("--multi-objective", action="store_true",
                   help="Enable multi-objective optimization (LCOE + VI)")
    p.add_argument("--history-file", default="ga_history.h5", metavar="PATH",
                   help="File to save per-generation surviving populations (HDF5)")
    p.add_argument("--evals-file", default="ga_evals.h5", metavar="PATH",
                   help="File to save every evaluated genome, discarded offspring "
                        "included (HDF5). Pass '' to skip it — it costs a second "
                        "full (P, T, 3) D2H copy per generation")
    p.add_argument("--resume", action="store_true",
                   help="Continue from the last generation in --history-file. "
                        "Raise --generations to extend a converged-too-early run; "
                        "starts from scratch if the file does not exist yet")
    p.add_argument("--no-plot",      action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────

def plot_layout(
    sol: cp.ndarray,
    title: str = "Best Layout + Yaw",
    save_path: str = "best_layout.png",
) -> None:
    sol_np = cp.asnumpy(sol)   # (T, 3)
    x, y, yaw = sol_np[:, 0], sol_np[:, 1], sol_np[:, 2]
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=80, zorder=3)
    for i in range(len(x)):
        plt.arrow(x[i], y[i], 60 * np.cos(yaw[i]), 60 * np.sin(yaw[i]),
                  head_width=20, head_length=15, fc="tab:blue", ec="tab:blue")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Layout saved to {save_path}")


def plot_convergence(
    history: list,
    title: str = "AEP Convergence",
    save_path: str = "convergence.png",
) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best AEP (kWh)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Convergence saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    seed_layout = None

    if args.floris_yaml:
        # ── Load everything from a FLORIS YAML ─────────────────────────
        inp = load_floris_yaml(args.floris_yaml)
        farm_cfg     = inp["farm_cfg"]
        wake_cfg     = inp["wake_cfg"]
        turbine_cfg  = inp["turbine_cfg"]
        turbine_data = inp["turbine_data"]
        wind_rose    = inp["wind_rose"]
        seed_layout  = inp["layout_xy"]
        opt_cfg = _CONFIGS[args.optimizer](
            pop_size=args.pop, n_generations=args.generations,
            optimize=args.optimize,
        )
        print(f"Loaded FLORIS YAML: {args.floris_yaml}")
        print(f"Wind rose: {len(wind_rose.wind_dirs)} dirs × "
              f"{len(wind_rose.wind_speeds)} speeds")
    else:
        # ── Manual configuration ────────────────────────────────────────
        wake_cfg    = WakeConfig(combination=args.combination)
        farm_cfg    = FarmConfig(n_turbines=args.turbines)
        turbine_cfg = TurbineConfig()
        turbine_data = TurbineData.nrel_5mw()
        opt_cfg     = _CONFIGS[args.optimizer](
            pop_size=args.pop, n_generations=args.generations,
            optimize=args.optimize,
        )

        if args.multispeed:
            wind_rose = WindRose.default_12sector_multispeed()
            print(f"Wind rose: 12 sectors × {len(wind_rose.wind_speeds)} speed bins")
        else:
            wind_rose = WindRose.default_12sector()
            print(f"Wind rose: 12 sectors × 1 speed bin")

    # ── Physics layer ──────────────────────────────────────────────────
    evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)

    # ── Projection chain ───────────────────────────────────────────────
    projection = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=10),
        BoundaryProjection(farm_cfg),
    ])

    # ── Optimizer ─────────────────────────────────────────────────────
    # PSO and GD are inherently Pareto searches, so they always need the
    # visual-impact config: without it the VI column is all zeros, every
    # individual lands on the same front, and the search silently degenerates.
    multi_objective = args.multi_objective or args.optimizer in ("pso", "gd")
    if multi_objective and not args.multi_objective:
        print(f"  ({args.optimizer} is inherently multi-objective; "
              f"enabling LCOE + VI)")

    cost_cfg = CostConfig()
    vi_cfg   = VisualImpactConfig() if multi_objective else None
    optimizer = _OPTIMIZERS[args.optimizer](
        farm_cfg, opt_cfg, evaluator, projection, wind_rose,
        cost_cfg=cost_cfg,
        vi_cfg=vi_cfg,
        objectives_mode=args.objectives,
        history_file=args.history_file,
        evals_file=args.evals_file,
        resume=args.resume,
    )

    print(f"\nStarting optimisation:")
    print(f"  Optimizer:   {args.optimizer}")
    print(f"  Turbines:    {farm_cfg.n_turbines}")
    print(f"  Population:  {opt_cfg.pop_size}")
    print(f"  Iterations:  {opt_cfg.n_generations}")
    print(f"  Wake combo:  {wake_cfg.combination}")
    print(f"  Optimising:  {opt_cfg.optimize}")
    print(f"  Multi-obj:   {multi_objective} ({args.objectives})")
    print(f"  GPU:         {cp.cuda.Device().id}\n")

    # Only the GA takes multi_objective — PSO and GD always are.
    extra = {"multi_objective": multi_objective} if args.optimizer == "ga" else {}
    best, history, pareto_obj, _ = optimizer.run(
        verbose=True, seed_layout=seed_layout, **extra
    )

    # ── Final results ──────────────────────────────────────────────────
    print(f"\nOptimisation complete.")
    print(f"Best AEP: {history[-1]:.4e} kWh/yr")

    if pareto_obj is not None and len(pareto_obj):
        print(f"Pareto front size: {len(pareto_obj)}")
        for m, name in enumerate(optimizer.objectives.names):
            lo, hi = pareto_obj[:, m].min(), pareto_obj[:, m].max()
            print(f"  {name:12s} {lo:12.4f} – {hi:.4f}")

    if not args.no_plot:
        plot_layout(best)
        plot_convergence(history)


if __name__ == "__main__":
    main()
