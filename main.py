"""
Wind Farm Layout + Yaw Optimiser — GPU Genetic Algorithm
FLORIS-traceable physics layer.

Usage:
    python main.py [--combination SOSFS|FLS|MAX] [--generations N]
                   [--pop N] [--turbines N] [--multispeed]

Physics sources:
    Wake velocity:   FLORIS floris/core/wake_velocity/gauss.py
    Turbulence:      FLORIS floris/core/wake_turbulence/crespo_hernandez.py
    Deflection:      FLORIS floris/core/wake_deflection/gauss.py
    Combination:     FLORIS floris/core/wake_combination/{sosfs,fls,max}.py
    Power curve:     FLORIS floris/core/turbine/operation_models.py (NREL 5 MW)
    Wind rose / AEP: FLORIS floris/wind_data.py + floris/floris_model.py
"""
from __future__ import annotations
import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from config import WakeConfig, FarmConfig, GAConfig, TurbineConfig
from physics.farm_evaluator import FarmEvaluator
from physics.turbine.power_curve import TurbineData
from projection.base import CompositeProjection
from projection.spacing import PairwiseSpacingProjection
from projection.boundary import BoundaryProjection
from optimizer.genetic import GeneticAlgorithm
from wind.wind_rose import WindRose
from loaders.floris_yaml import load_floris_yaml


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GPU Wind Farm Optimiser")
    p.add_argument("--floris-yaml",  default=None, metavar="PATH",
                   help="Path to a FLORIS v4 input YAML; overrides --combination, "
                        "--turbines, and wind-rose flags")
    p.add_argument("--combination",  default="SOSFS", choices=["SOSFS", "FLS", "MAX"])
    p.add_argument("--generations",  type=int,   default=150)
    p.add_argument("--pop",          type=int,   default=256)
    p.add_argument("--turbines",     type=int,   default=20)
    p.add_argument("--multispeed",   action="store_true",
                   help="Use 12-sector × 11-speed Weibull wind rose")
    p.add_argument("--no-plot",      action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────

def plot_layout(sol: cp.ndarray, title: str = "Best Layout + Yaw") -> None:
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
    plt.show()


def plot_convergence(history: list, title: str = "AEP Convergence") -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best AEP (kWh)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
        ga_cfg = GAConfig(pop_size=args.pop, n_generations=args.generations)
        print(f"Loaded FLORIS YAML: {args.floris_yaml}")
        print(f"Wind rose: {len(wind_rose.wind_dirs)} dirs × "
              f"{len(wind_rose.wind_speeds)} speeds")
    else:
        # ── Manual configuration ────────────────────────────────────────
        wake_cfg    = WakeConfig(combination=args.combination)
        farm_cfg    = FarmConfig(n_turbines=args.turbines)
        turbine_cfg = TurbineConfig()
        turbine_data = TurbineData.nrel_5mw()
        ga_cfg      = GAConfig(pop_size=args.pop, n_generations=args.generations)

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

    # ── GA ────────────────────────────────────────────────────────────
    ga = GeneticAlgorithm(farm_cfg, ga_cfg, evaluator, projection, wind_rose)

    print(f"\nStarting optimisation:")
    print(f"  Turbines:    {farm_cfg.n_turbines}")
    print(f"  Population:  {ga_cfg.pop_size}")
    print(f"  Generations: {ga_cfg.n_generations}")
    print(f"  Wake combo:  {wake_cfg.combination}")
    print(f"  GPU:         {cp.cuda.Device().id}\n")

    best, history = ga.run(verbose=True, seed_layout=seed_layout)

    # ── Final results ──────────────────────────────────────────────────
    print(f"\nOptimisation complete.")
    print(f"Best AEP: {history[-1]:.4e} kWh/yr")

    if not args.no_plot:
        plot_layout(best)
        plot_convergence(history)


if __name__ == "__main__":
    main()
