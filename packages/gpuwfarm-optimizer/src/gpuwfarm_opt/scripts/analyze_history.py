"""
Post-process GA history: compute AEP convergence statistics per generation.

Usage:
    python analyze_history.py [--history ga_history.h5] [--output analysis.json]

The HDF5 file contains:
    genomes    — (n_gens, n_individuals, genome_size) float32  [x, y, yaw flattened]
    fitnesses  — (n_gens, n_individuals)              float32  [AEP in kWh]
    objectives — (n_gens, n_individuals, 2)           float32  [obj0, obj1] (multi-obj only)
"""
import json
import argparse
import numpy as np
import h5py
import hdf5plugin  # registers LZ4 codec so h5py can decompress on read
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Multi-objective convergence helpers
# ──────────────────────────────────────────────────────────────────────

def _pareto_mask_2d(objectives: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated solutions (2-objective minimisation)."""
    obj_i = objectives[:, np.newaxis, :]   # (n, 1, 2)
    obj_j = objectives[np.newaxis, :, :]   # (1, n, 2)
    dominates_i = np.all(obj_j <= obj_i, axis=2) & np.any(obj_j < obj_i, axis=2)
    np.fill_diagonal(dominates_i, False)
    return ~dominates_i.any(axis=1)


def _hypervolume_2d(front_obj: np.ndarray, ref: np.ndarray) -> float:
    """
    Hypervolume indicator for a 2-objective minimisation front.

    Sweep-line: sort by obj-0 ascending, sum L-shaped strips to the reference.
    ref must satisfy ref[i] >= max(front_obj[:, i]) for i in {0, 1}.
    """
    if len(front_obj) == 0:
        return 0.0
    order = np.argsort(front_obj[:, 0])
    f1 = front_obj[order, 0]
    f2 = front_obj[order, 1]
    r1, r2 = float(ref[0]), float(ref[1])
    hv = 0.0
    for i in range(len(f1) - 1):
        hv += (f1[i + 1] - f1[i]) * (r2 - f2[i])
    hv += (r1 - f1[-1]) * (r2 - f2[-1])
    return max(0.0, hv)


def load_mo_convergence(
    history_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-generation VI and hypervolume convergence from an HDF5 history file.

    The file must contain an `objectives` dataset written by AsyncPopulationLogger
    when running in multi-objective mode.

    The reference point for hypervolume is derived from the worst objectives
    observed across the entire run (max per objective + 10 % margin), so all
    per-generation HV values are comparable on the same scale.

    Args:
        history_file: path to the HDF5 file written by AsyncPopulationLogger

    Returns:
        gens:       (n_gens,) generation indices
        vi_per_gen: (n_gens,) minimum VI on the Pareto front each generation
        hv_per_gen: (n_gens,) hypervolume of the Pareto front each generation
    """
    with h5py.File(history_file, "r") as f:
        if "objectives" not in f:
            raise KeyError(
                f"'{history_file}' has no 'objectives' dataset — "
                "rerun with multi-objective mode and a history file."
            )
        objectives_all = f["objectives"][:]  # (n_gens, n_individuals, 2)

    n_gens = objectives_all.shape[0]

    # Global reference point: worst finite objective + 10 % range margin
    flat = objectives_all.reshape(-1, 2)
    finite = flat[np.isfinite(flat).all(axis=1)]
    obj_max = finite.max(axis=0)
    obj_range = obj_max - finite.min(axis=0)
    ref = obj_max + np.maximum(obj_range * 0.1, 1e-6)

    vi_per_gen = np.full(n_gens, np.nan)
    hv_per_gen = np.zeros(n_gens)

    for g in range(n_gens):
        objs_g = objectives_all[g]                         # (n_individuals, 2)
        finite_mask = np.isfinite(objs_g).all(axis=1)
        objs_g = objs_g[finite_mask]
        if len(objs_g) == 0:
            continue
        front = objs_g[_pareto_mask_2d(objs_g)]
        if len(front) > 0:
            vi_per_gen[g] = front[:, 1].min()
            hv_per_gen[g] = _hypervolume_2d(front, ref)

    return np.arange(n_gens), vi_per_gen, hv_per_gen


def analyze_generation(fitnesses_row: np.ndarray, generation: int) -> dict:
    """
    Compute per-generation AEP statistics from a fitnesses row.

    Args:
        fitnesses_row: (n_individuals,) AEP values for one generation
        generation:    generation index

    Returns:
        dict with AEP statistics
    """
    return {
        "generation": int(generation),
        "pop_size": int(len(fitnesses_row)),
        "best_aep": float(fitnesses_row.max()),
        "mean_aep": float(fitnesses_row.mean()),
        "std_aep":  float(fitnesses_row.std()),
        "worst_aep": float(fitnesses_row.min()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GA history for AEP convergence"
    )
    parser.add_argument(
        "--history",
        default="ga_history.h5",
        help="Path to generation history HDF5 file",
    )
    parser.add_argument(
        "--output", default="analysis.json", help="Output analysis file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate convergence plots (requires matplotlib)",
    )
    args = parser.parse_args()

    history_file = Path(args.history)
    if not history_file.exists():
        print(f"Error: {history_file} not found")
        return

    with h5py.File(history_file, "r") as f:
        fitnesses = f["fitnesses"][:]  # (n_gens, n_individuals)

    n_gens = fitnesses.shape[0]
    print(f"Loaded {n_gens} generations from {history_file}")

    analysis = [
        analyze_generation(fitnesses[g], g)
        for g in range(n_gens)
    ]

    # Summary
    print("\n" + "=" * 60)
    print("CONVERGENCE SUMMARY")
    print("=" * 60)
    print(f"\n{'Gen':>4} {'Best AEP':>14} {'Mean AEP':>14} {'Std AEP':>12}")
    print("-" * 48)
    for stats in analysis:
        print(
            f"{stats['generation']:4d} "
            f"{stats['best_aep']:14.4e} "
            f"{stats['mean_aep']:14.4e} "
            f"{stats['std_aep']:12.4e}"
        )

    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {args.output}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            gens = [s["generation"] for s in analysis]
            best_aeps = [s["best_aep"] for s in analysis]
            mean_aeps = [s["mean_aep"] for s in analysis]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(gens, best_aeps, "o-")
            axes[0].set_xlabel("Generation")
            axes[0].set_ylabel("Best AEP (kWh)")
            axes[0].set_title("Best AEP Convergence")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(gens, mean_aeps, "o-", color="tab:orange")
            axes[1].set_xlabel("Generation")
            axes[1].set_ylabel("Mean AEP (kWh)")
            axes[1].set_title("Population Mean AEP")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = Path(args.output).with_suffix(".png")
            plt.savefig(plot_file, dpi=150)
            print(f"Convergence plot saved to {plot_file}")

        except ImportError:
            print("Matplotlib not available; skipping plots")


if __name__ == "__main__":
    main()
