"""Quick smoke-test: 10 turbines, 20 gen, 64 pop. Saves plots to PNG."""
import os, sys
if sys.platform == "win32":
    _p = os.path.normpath(os.path.join(os.path.dirname(sys.executable),
                          "..", "Lib", "site-packages", "torch", "lib"))
    if os.path.isdir(_p):
        os.add_dll_directory(_p)

import matplotlib
matplotlib.use("Agg")   # no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from config import WakeConfig, FarmConfig, GAConfig, TurbineConfig, CostConfig, VisualImpactConfig
from physics.farm_evaluator import FarmEvaluator
from physics.turbine.power_curve import TurbineData
from projection.base import CompositeProjection
from projection.spacing import PairwiseSpacingProjection
from projection.boundary import BoundaryProjection
from optimizer.genetic import GeneticAlgorithm
from wind.wind_rose import WindRose
from analyze_history import load_mo_convergence

N_TURB, POP, GENS = 10, 200, 200

farm_cfg     = FarmConfig(n_turbines=N_TURB)
wake_cfg     = WakeConfig(combination="SOSFS")
turbine_cfg  = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()
ga_cfg       = GAConfig(pop_size=POP, n_generations=GENS, crossover_rate=0)
wind_rose    = WindRose.default_12sector()

# ── wind rose plot ───────────────────────────────────────────────────
def _plot_wind_rose(wr: WindRose, path: str) -> None:
    dirs_rad = np.deg2rad(wr.wind_dirs)
    width = 2 * np.pi / len(wr.wind_dirs) * 0.9
    marginal = wr.freq_table.sum(axis=1)   # (n_wd,)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    if wr.freq_table.shape[1] == 1:
        ax.bar(dirs_rad, marginal, width=width, bottom=0.0,
               color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.85)
    else:
        speeds = wr.wind_speeds
        cmap = plt.get_cmap("YlOrRd", len(speeds))
        bottom = np.zeros(len(dirs_rad))
        for j, ws in enumerate(speeds):
            heights = wr.freq_table[:, j]
            ax.bar(dirs_rad, heights, width=width, bottom=bottom,
                   color=cmap(j), edgecolor="white", linewidth=0.3,
                   label=f"{ws:.0f} m/s", alpha=0.9)
            bottom += heights
        ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.0),
                  fontsize=7, title="Wind speed")

    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title("Wind Rose", pad=15)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%")
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path} ({os.path.getsize(path)} bytes)")
    plt.close()

_plot_wind_rose(wind_rose, "test_wind_rose.png")

evaluator  = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
projection = CompositeProjection([
    PairwiseSpacingProjection(farm_cfg, n_passes=10),
    BoundaryProjection(farm_cfg),
])

# ── single-objective ──────────────────────────────────────────────────
print("=== single-objective ===")
ga = GeneticAlgorithm(farm_cfg, ga_cfg, evaluator, projection, wind_rose)
best, history, pareto, _ = ga.run(verbose=True, multi_objective=False)
assert pareto is None
print(f"Best AEP: {history[-1]:.4e} kWh  (generations: {len(history)})")

sol = cp.asnumpy(best)
x, y, yaw = sol[:, 0], sol[:, 1], sol[:, 2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(x, y, s=80, zorder=3)
for i in range(N_TURB):
    axes[0].arrow(x[i], y[i], 60*np.cos(yaw[i]), 60*np.sin(yaw[i]),
                  head_width=20, head_length=15, fc="tab:blue", ec="tab:blue")
axes[0].set_title("Best Layout + Yaw")
axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")
axes[0].grid(True, alpha=0.3)
axes[1].plot(history)
axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Best AEP (kWh)")
axes[1].set_title("AEP Convergence")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_single_obj.png", dpi=150)
print(f"Saved test_single_obj.png ({os.path.getsize('test_single_obj.png')} bytes)")
plt.close()

# ── multi-objective: max AEP + min VI ────────────────────────────────
print("\n=== multi-objective (max AEP + min VI) ===")

# Farm occupies roughly 0–2000 m × 0–2000 m.
# Place three observers outside the farm perimeter.
vi_cfg = VisualImpactConfig(
    obs_coords  = [[1000.0, 5000.0]],   # single observer due north of farm
    obs_heights = [1.77],
    obs_weights = [1.0],
)

ga2 = GeneticAlgorithm(
    farm_cfg, ga_cfg, evaluator, projection, wind_rose,
    cost_cfg=CostConfig(),
    vi_cfg=vi_cfg,
    objectives_mode="aep_vi",
    history_file="test_history.h5",
)
best2, history2, pareto2, best_vi2 = ga2.run(verbose=True, multi_objective=True)
assert pareto2 is not None, "pareto_obj should not be None in multi-objective mode"
assert best_vi2 is not None, "best_vi individual should not be None in multi-objective mode"

# pareto2[:, 0] = -AEP (GWh)  →  real AEP = -pareto2[:, 0]
aep_pf = -pareto2[:, 0]   # GWh
vi_pf  =  pareto2[:, 1]

print(f"Best AEP:       {history2[-1]:.4e} kWh")
print(f"Pareto size:    {len(pareto2)}")
print(f"AEP range:      {aep_pf.min():.4f} – {aep_pf.max():.4f} GWh")
print(f"VI range:       {vi_pf.min():.4f} – {vi_pf.max():.4f}")
print(f"History file:   {os.path.getsize('test_history.h5')} bytes")

# ── multi-objective plots  (3 rows × 2 cols) ──────────────────────────
sol2    = cp.asnumpy(best2)           # best-AEP layout
sol_vi2 = cp.asnumpy(best_vi2)        # best-VI  layout
x2,  y2,  yaw2  = sol2[:,    0], sol2[:,    1], sol2[:,    2]
xv2, yv2, yawv2 = sol_vi2[:, 0], sol_vi2[:, 1], sol_vi2[:, 2]

def _draw_layout(ax, x, y, yaw, color, title):
    ax.scatter(x, y, s=80, zorder=3)
    for i in range(len(x)):
        ax.arrow(x[i], y[i], 60*np.cos(yaw[i]), 60*np.sin(yaw[i]),
                 head_width=20, head_length=15, fc=color, ec=color)
    for obs_xy in vi_cfg.obs_coords:
        ax.plot(obs_xy[0], obs_xy[1], "rv", ms=10, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# Row 0 — best AEP layout / best VI layout
_draw_layout(axes[0, 0], x2,  y2,  yaw2,  "tab:green", "Best-AEP Layout")
_draw_layout(axes[0, 1], xv2, yv2, yawv2, "tab:purple", "Best-VI Layout")

# Row 1 — AEP convergence / Pareto front
axes[1, 0].plot(history2, color="tab:green")
axes[1, 0].set_xlabel("Generation"); axes[1, 0].set_ylabel("Best AEP (kWh)")
axes[1, 0].set_title("AEP Convergence")
axes[1, 0].grid(True, alpha=0.3)

sc = axes[1, 1].scatter(aep_pf, vi_pf, c=aep_pf, cmap="viridis",
                         s=80, edgecolors="k", linewidths=0.5, zorder=3)
plt.colorbar(sc, ax=axes[1, 1], label="AEP (GWh)")
axes[1, 1].set_xlabel("AEP (GWh)"); axes[1, 1].set_ylabel("Visual Impact")
axes[1, 1].set_title(f"Pareto Front — AEP vs VI  (n={len(pareto2)})")
axes[1, 1].grid(True, alpha=0.3)

# Row 2 — VI convergence / Hypervolume convergence (loaded from history file)
gens_mo, vi_conv, hv_conv = load_mo_convergence("test_history.h5")

axes[2, 0].plot(gens_mo, vi_conv, color="tab:orange")
axes[2, 0].set_xlabel("Generation"); axes[2, 0].set_ylabel("Min VI (Pareto front)")
axes[2, 0].set_title("VI Convergence")
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(gens_mo, hv_conv, color="tab:red")
axes[2, 1].set_xlabel("Generation"); axes[2, 1].set_ylabel("Hypervolume")
axes[2, 1].set_title("Hypervolume Convergence")
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("test_multi_obj.png", dpi=150)
print(f"Saved test_multi_obj.png ({os.path.getsize('test_multi_obj.png')} bytes)")
plt.close()

print("\nAll checks passed.")
