"""
Extract best layouts from generation history and optionally visualize.

Usage:
    python extract_pareto.py --history ga_history.h5
    python extract_pareto.py --history ga_history.h5 --plot --generation 50
    python extract_pareto.py --history ga_history.h5 --top 5

The HDF5 file contains:
    genomes   — (n_gens, n_individuals, genome_size) float32  [x, y, yaw flattened]
    fitnesses — (n_gens, n_individuals)              float32  [AEP in kWh]

genome_size = n_turbines * 3 (columns: x_0..x_T, y_0..y_T, yaw_0..yaw_T in
the flat order produced by reshaping (P, T, 3) → (P, T*3)).
"""
import json
import argparse
import numpy as np
import h5py
import hdf5plugin  # registers LZ4 codec so h5py can decompress on read
from pathlib import Path


def extract_best_from_history(
    history_file: str,
    generation: int | None = None,
    top_n: int = 1,
) -> dict:
    """
    Extract the top-N individuals by AEP from a given generation.

    Args:
        history_file: path to ga_history.h5
        generation:   generation index (None = final generation)
        top_n:        how many top individuals to return

    Returns:
        dict with keys: generation, n_turbines, individuals (list of dicts)
    """
    with h5py.File(history_file, "r") as f:
        genomes_all   = f["genomes"][:]    # (n_gens, n_individuals, genome_size)
        fitnesses_all = f["fitnesses"][:]  # (n_gens, n_individuals)

    n_gens, n_individuals, genome_size = genomes_all.shape
    n_turbines = genome_size // 3

    gen_idx = (n_gens - 1) if generation is None else generation
    if gen_idx < 0 or gen_idx >= n_gens:
        raise ValueError(f"generation {gen_idx} out of range [0, {n_gens - 1}]")

    fitnesses = fitnesses_all[gen_idx]   # (n_individuals,)
    genomes   = genomes_all[gen_idx]     # (n_individuals, genome_size)

    top_k = min(top_n, n_individuals)
    top_idx = np.argsort(fitnesses)[::-1][:top_k]  # descending by AEP

    individuals = []
    for rank, idx in enumerate(top_idx):
        genome = genomes[idx].reshape(n_turbines, 3)
        individuals.append({
            "rank":      int(rank),
            "id":        int(idx),
            "aep_kwh":   float(fitnesses[idx]),
            "x":         genome[:, 0].tolist(),
            "y":         genome[:, 1].tolist(),
            "yaw":       genome[:, 2].tolist(),
        })

    return {
        "generation": int(gen_idx),
        "n_turbines": int(n_turbines),
        "pop_size":   int(n_individuals),
        "individuals": individuals,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract best layouts from GA history (HDF5)"
    )
    parser.add_argument("--history", default="ga_history.h5", help="History HDF5 file")
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help="Specific generation (default: final)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="Number of top individuals to extract (default: 1)",
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON file (default: best_gen_N.json)"
    )
    parser.add_argument("--plot", action="store_true", help="Plot best layout")
    args = parser.parse_args()

    result = extract_best_from_history(args.history, args.generation, args.top)

    output_file = args.output or f"best_gen_{result['generation']}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Generation {result['generation']}:")
    print(f"  Population size: {result['pop_size']}")
    print(f"  Turbines:        {result['n_turbines']}")
    print(f"\nTop {len(result['individuals'])} individuals by AEP:")
    for ind in result["individuals"]:
        print(f"  rank {ind['rank']}  id {ind['id']:4d}  AEP {ind['aep_kwh']:.4e} kWh")
    print(f"\nSaved to {output_file}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            best = result["individuals"][0]
            x   = np.array(best["x"])
            y   = np.array(best["y"])
            yaw = np.array(best["yaw"])

            plt.figure(figsize=(7, 7))
            plt.scatter(x, y, s=80, zorder=3)
            for i in range(len(x)):
                plt.arrow(
                    x[i], y[i],
                    60 * np.cos(yaw[i]), 60 * np.sin(yaw[i]),
                    head_width=20, head_length=15,
                    fc="tab:blue", ec="tab:blue",
                )
            plt.xlabel("Easting (m)")
            plt.ylabel("Northing (m)")
            plt.title(
                f"Best Layout – Gen {result['generation']}  "
                f"AEP {best['aep_kwh']:.4e} kWh"
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_file = Path(output_file).with_suffix(".png")
            plt.savefig(plot_file, dpi=150)
            print(f"Plot saved to {plot_file}")

        except ImportError:
            print("Matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
