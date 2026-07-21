"""
Visual + tabular cross-validation report: GPUwfarm vs FLORIS 4.x.

Reuses the exact harness from tests/test_floris_comparison.py (same FLORIS
turbine table, same layouts, same wake parameters fed into both codes) and
renders:

    floris_comparison.png   — wake efficiency vs. separation and freestream
                               power vs. wind speed, ours vs. FLORIS
    floris_comparison.csv   — every comparison metric with its relative
                               error and pass/fail against that test's
                               tolerance (see tests/test_floris_comparison.py)
    floris_comparison_per_turbine.png / .csv
                            — 3-turbine row: (1) AEP per turbine over the
                               same wind rose, (2) power per turbine at the
                               same wind direction + speed, both codes.

Requires: cupy, floris (`pip install floris`)
Run (from packages/gpuwfarm-core/):
    python plot_floris_comparison.py
"""
from __future__ import annotations
import csv
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from tests.test_floris_comparison import (
    D_FLORIS, HH, TI_AMB, WS_TEST, FLORIS_WD, OUR_WD,
    _load_floris_turbine_data, _floris_model, _floris_farm_power_kw,
    _our_farm_power_kw, _our_evaluator, TestAEPWindRose,
)
from gpuwfarm_core.wind.wind_rose import WindRose

# Two-series categorical palette, fixed assignment (FLORIS = reference, ours = subject)
COLOR_FLORIS = "#4C72B0"   # blue
COLOR_OURS   = "#DD8452"   # orange


def _confirm_same_setup(td):
    print("=" * 72)
    print("Setup check -- same turbine & layout fed to both codes")
    print("=" * 72)
    print("Turbine table  : FLORIS turbine_library/nrel_5MW.yaml, loaded once and")
    print("                 passed as `turbine_data` into both _floris_model()")
    print(f"                 (via turbine_type) and our TabulatedPowerCurve.")
    print(f"Rotor diameter : {D_FLORIS} m        (identical on both sides)")
    print(f"Hub height     : {HH} m         (identical on both sides)")
    print(f"Air density    : 1.225 kg/m^3   (identical on both sides)")
    print(f"Ambient TI     : {TI_AMB}          (identical on both sides)")
    print(f"Wake params    : WakeConfig() defaults == FLORIS cfg overrides (see")
    print(f"                 _floris_model() docstring in test_floris_comparison.py)")
    print("Layouts        : the same (layout_x, layout_y) arrays are passed into")
    print("                 both _floris_model(...) and _our_farm_power_kw(...)")
    print("                 for every case below -- no separate layout generation.")
    print()


def _wake_efficiency_pair(sep_m, td):
    fm_waked = _floris_model([0.0, sep_m], [0.0, 0.0])
    p_fl_waked, _ = _floris_farm_power_kw(fm_waked, FLORIS_WD, WS_TEST, TI_AMB)
    fm_free = _floris_model([0.0, 50_000.0], [0.0, 0.0])
    p_fl_free, _ = _floris_farm_power_kw(fm_free, FLORIS_WD, WS_TEST, TI_AMB)
    eta_fl = p_fl_waked / p_fl_free

    p_us_waked = _our_farm_power_kw([0.0, sep_m], [0.0, 0.0], td)
    p_us_free = _our_farm_power_kw([0.0, 50_000.0], [0.0, 0.0], td)
    eta_us = p_us_waked / p_us_free
    return eta_fl, eta_us


def _freestream_power_pair(ws, td):
    fm = _floris_model([0.0], [0.0])
    fm.set(
        wind_directions=[FLORIS_WD],
        wind_speeds=[ws],
        turbulence_intensities=[TI_AMB],
    )
    fm.run()
    p_floris = fm.get_turbine_powers()[0, 0] / 1000.0

    ev = _our_evaluator(1, td)
    import cupy as cp
    pop = cp.zeros((1, 1, 3), dtype=cp.float32)
    rose = WindRose.from_uniform_ti(
        wind_dirs=np.array([OUR_WD], dtype=np.float32),
        wind_speeds=np.array([ws], dtype=np.float32),
        freq_table=np.array([[1.0]], dtype=np.float32),
        ti_ambient=TI_AMB,
    )
    p_ours = float(cp.asnumpy(ev.evaluate(pop, rose))[0]) / 8760.0
    return p_floris, p_ours


def _row(rows, metric, ours, floris, tol):
    rel_err = abs(ours - floris) / abs(floris) if floris != 0 else float("nan")
    rows.append({
        "metric": metric,
        "ours": ours,
        "floris": floris,
        "rel_err_pct": rel_err * 100.0,
        "tol_pct": tol * 100.0,
        "pass": rel_err <= tol,
    })


def build_table(td):
    rows = []

    # Freestream power at the primary test speed
    p_fl, p_us = _freestream_power_pair(WS_TEST, td)
    _row(rows, f"Freestream power @ {WS_TEST:.0f} m/s", p_us, p_fl, 0.01)

    # Freestream power curve at several speeds
    for ws in [5.0, 7.0, 9.0, 11.0, 13.0]:
        p_fl, p_us = _freestream_power_pair(ws, td)
        _row(rows, f"Freestream power @ {ws:.0f} m/s", p_us, p_fl, 0.01)

    # Wake efficiency at the three tested separations
    for sep_D, sep_m in [(5, 5 * D_FLORIS), (7, 7 * D_FLORIS), (10, 10 * D_FLORIS)]:
        eta_fl, eta_us = _wake_efficiency_pair(sep_m, td)
        _row(rows, f"Wake efficiency @ {sep_D}D", eta_us, eta_fl, 0.10)

    # 3-turbine row
    layout_x = [0.0, 5 * D_FLORIS, 2 * 5 * D_FLORIS]
    layout_y = [0.0, 0.0, 0.0]
    fm = _floris_model(layout_x, layout_y)
    p_fl_row, _ = _floris_farm_power_kw(fm, FLORIS_WD, WS_TEST, TI_AMB)
    p_us_row = _our_farm_power_kw(layout_x, layout_y, td)
    _row(rows, "3-turbine row farm power @ 5D", p_us_row, p_fl_row, 0.15)

    free_x = [0.0, 50_000.0, 100_000.0]
    fm_free = _floris_model(free_x, layout_y)
    p_fl_free, _ = _floris_farm_power_kw(fm_free, FLORIS_WD, WS_TEST, TI_AMB)
    p_us_free = _our_farm_power_kw(free_x, layout_y, td)
    _row(rows, "3-turbine row wake efficiency @ 5D",
         p_us_row / p_us_free, p_fl_row / p_fl_free, 0.15)

    # AEP over a Weibull wind-speed distribution, 2-turbine aligned row
    speeds = np.arange(3.0, 25.0, 1.0, dtype=np.float32)
    freqs = TestAEPWindRose._weibull_freq(speeds).astype(np.float32)
    aep_layout_x = [0.0, 7 * D_FLORIS]
    fm = _floris_model(aep_layout_x, [0.0, 0.0])
    fm.set(
        wind_directions=np.full(len(speeds), FLORIS_WD, dtype=float),
        wind_speeds=speeds.astype(float),
        turbulence_intensities=np.full(len(speeds), TI_AMB, dtype=float),
    )
    fm.run()
    aep_floris_kwh = fm.get_farm_AEP(freq=freqs.astype(float)) / 1000.0

    ev = _our_evaluator(2, td)
    import cupy as cp
    pop = cp.zeros((1, 2, 3), dtype=cp.float32)
    pop[0, :, 0] = cp.asarray(np.array(aep_layout_x, dtype=np.float32))
    rose = WindRose.from_uniform_ti(
        wind_dirs=np.array([OUR_WD], dtype=np.float32),
        wind_speeds=speeds,
        freq_table=freqs.reshape(1, -1),
        ti_ambient=TI_AMB,
    )
    aep_ours_kwh = float(cp.asnumpy(ev.evaluate(pop, rose))[0])
    _row(rows, "AEP, 2-turbine @ 7D, Weibull (1 dir)", aep_ours_kwh, aep_floris_kwh, 0.12)

    return rows


# ═════════════════════════════════════════════════════════════════════════
# Per-turbine comparison, 3-turbine row (same layout & turbine table as above)
# ═════════════════════════════════════════════════════════════════════════

LAYOUT_X_3T = [0.0, 5 * D_FLORIS, 2 * 5 * D_FLORIS]
LAYOUT_Y_3T = [0.0, 0.0, 0.0]
N_3T = len(LAYOUT_X_3T)


def per_turbine_power_3t(td):
    """Power (kW) per turbine, same wind direction + speed (WD=270 deg met /
    0 deg ours, WS=9 m/s), both codes."""
    fm = _floris_model(LAYOUT_X_3T, LAYOUT_Y_3T)
    _, p_floris_per_turb = _floris_farm_power_kw(fm, FLORIS_WD, WS_TEST, TI_AMB)

    ev = _our_evaluator(N_3T, td)
    pop = cp.zeros((1, N_3T, 3), dtype=cp.float32)
    pop[0, :, 0] = cp.asarray(np.array(LAYOUT_X_3T, dtype=np.float32))
    pop[0, :, 1] = cp.asarray(np.array(LAYOUT_Y_3T, dtype=np.float32))
    rose = WindRose.from_uniform_ti(
        wind_dirs=np.array([OUR_WD], dtype=np.float32),
        wind_speeds=np.array([WS_TEST], dtype=np.float32),
        freq_table=np.array([[1.0]], dtype=np.float32),
        ti_ambient=TI_AMB,
    )
    p_ours_per_turb = cp.asnumpy(ev.evaluate(pop, rose, per_turbine=True))[0] / 8760.0
    return p_ours_per_turb, np.asarray(p_floris_per_turb)


def per_turbine_aep_3t(td):
    """AEP (kWh) per turbine, same wind rose (Weibull A=9.5, k=2, single
    direction — same distribution as the 2-turbine AEP case above), both
    codes."""
    speeds = np.arange(3.0, 25.0, 1.0, dtype=np.float32)
    freqs = TestAEPWindRose._weibull_freq(speeds).astype(np.float32)

    fm = _floris_model(LAYOUT_X_3T, LAYOUT_Y_3T)
    fm.set(
        wind_directions=np.full(len(speeds), FLORIS_WD, dtype=float),
        wind_speeds=speeds.astype(float),
        turbulence_intensities=np.full(len(speeds), TI_AMB, dtype=float),
    )
    fm.run()
    powers_kw = fm.get_turbine_powers() / 1000.0          # (n_findex, n_turbines)
    aep_floris_per_turb = (powers_kw * freqs.astype(float)[:, None] * 8760.0).sum(axis=0)

    ev = _our_evaluator(N_3T, td)
    pop = cp.zeros((1, N_3T, 3), dtype=cp.float32)
    pop[0, :, 0] = cp.asarray(np.array(LAYOUT_X_3T, dtype=np.float32))
    pop[0, :, 1] = cp.asarray(np.array(LAYOUT_Y_3T, dtype=np.float32))
    rose = WindRose.from_uniform_ti(
        wind_dirs=np.array([OUR_WD], dtype=np.float32),
        wind_speeds=speeds,
        freq_table=freqs.reshape(1, -1),
        ti_ambient=TI_AMB,
    )
    aep_ours_per_turb = cp.asnumpy(ev.evaluate(pop, rose, per_turbine=True))[0]
    return aep_ours_per_turb, aep_floris_per_turb


def build_per_turbine_rows(power_ours, power_floris, aep_ours, aep_floris):
    rows = []
    for i in range(N_3T):
        rel_err = abs(power_ours[i] - power_floris[i]) / power_floris[i]
        rows.append({
            "turbine": i + 1, "quantity": "power_kW",
            "ours": float(power_ours[i]), "floris": float(power_floris[i]),
            "rel_err_pct": rel_err * 100.0,
        })
    for i in range(N_3T):
        rel_err = abs(aep_ours[i] - aep_floris[i]) / aep_floris[i]
        rows.append({
            "turbine": i + 1, "quantity": "AEP_kWh",
            "ours": float(aep_ours[i]), "floris": float(aep_floris[i]),
            "rel_err_pct": rel_err * 100.0,
        })
    return rows


def print_and_save_per_turbine_table(rows, csv_path="floris_comparison_per_turbine.csv"):
    header = f"{'Turbine':>7} {'Quantity':<10} {'Ours':>14} {'FLORIS':>14} {'Rel err':>9}"
    print("=" * len(header))
    print(f"Per-turbine comparison -- 3-turbine row @ 5D spacing (D={D_FLORIS} m)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['turbine']:>7} {r['quantity']:<10} {r['ours']:>14.4g} "
              f"{r['floris']:>14.4g} {r['rel_err_pct']:>8.2f}%")
    print("=" * len(header))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turbine", "quantity", "ours", "floris", "rel_err_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-turbine table saved to {csv_path}\n")


def plot_per_turbine(power_ours, power_floris, aep_ours, aep_floris,
                      save_path="floris_comparison_per_turbine.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    turbines = np.arange(1, N_3T + 1)
    width = 0.35

    def _grouped_bars(ax, floris_vals, ours_vals, ylabel, title):
        b1 = ax.bar(turbines - width / 2, floris_vals, width, color=COLOR_FLORIS, label="FLORIS")
        b2 = ax.bar(turbines + width / 2, ours_vals, width, color=COLOR_OURS, label="GPUwfarm")
        ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=8)
        ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
        ax.set_xticks(turbines)
        ax.set_xticklabels([f"T{i}" for i in turbines])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        ax.margins(y=0.15)

    _grouped_bars(ax1, power_floris, power_ours, "Power (kW)",
                  f"Power per turbine\n(WD={FLORIS_WD:.0f} deg met, WS={WS_TEST:.0f} m/s)")
    _grouped_bars(ax2, aep_floris, aep_ours, "AEP (kWh)",
                  "AEP per turbine\n(Weibull A=9.5, k=2, single direction)")

    fig.suptitle("GPUwfarm vs FLORIS 4.x -- 3-turbine row, same layout & turbine table")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Per-turbine plot saved to {save_path}")
    plt.show()


def print_and_save_table(rows, csv_path="floris_comparison.csv"):
    header = f"{'Metric':<38} {'Ours':>14} {'FLORIS':>14} {'Rel err':>9} {'Tol':>7}  {'Result'}"
    print("=" * len(header))
    print("GPUwfarm vs FLORIS -- comparison table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        result = "PASS" if r["pass"] else "FAIL"
        print(f"{r['metric']:<38} {r['ours']:>14.4g} {r['floris']:>14.4g} "
              f"{r['rel_err_pct']:>8.2f}% {r['tol_pct']:>6.0f}%  {result}")
    print("=" * len(header))
    n_pass = sum(r["pass"] for r in rows)
    print(f"{n_pass}/{len(rows)} within tolerance\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "ours", "floris", "rel_err_pct", "tol_pct", "pass"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Table saved to {csv_path}")


def plot_comparison(td, save_path="floris_comparison.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: wake efficiency vs separation ─────────────────────────
    seps_D = np.arange(3.0, 15.5, 0.5)
    seps_m = seps_D * D_FLORIS
    eta_fl_list, eta_us_list = [], []
    for sm in seps_m:
        eta_fl, eta_us = _wake_efficiency_pair(sm, td)
        eta_fl_list.append(eta_fl)
        eta_us_list.append(eta_us)
    eta_fl_arr = np.array(eta_fl_list)
    eta_us_arr = np.array(eta_us_list)

    ax1.fill_between(seps_D, eta_fl_arr * 0.90, eta_fl_arr * 1.10,
                      color=COLOR_FLORIS, alpha=0.12, label="±10% tolerance")
    ax1.plot(seps_D, eta_fl_arr, color=COLOR_FLORIS, lw=2, label="FLORIS")
    ax1.plot(seps_D, eta_us_arr, color=COLOR_OURS, lw=2, label="GPUwfarm")
    for sD in (5, 7, 10):
        i = int(np.argmin(np.abs(seps_D - sD)))
        ax1.scatter([seps_D[i]], [eta_fl_arr[i]], color=COLOR_FLORIS, s=28, zorder=3)
        ax1.scatter([seps_D[i]], [eta_us_arr[i]], color=COLOR_OURS, s=28, zorder=3)
    ax1.set_xlabel("Turbine separation (rotor diameters)")
    ax1.set_ylabel("Wake efficiency  η = P_waked / P_freestream")
    ax1.set_title(f"2-turbine wake efficiency @ {WS_TEST:.0f} m/s")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)

    # ── Panel 2: freestream power vs wind speed ────────────────────────
    speeds = np.arange(3.0, 25.5, 0.5)
    p_fl_list, p_us_list = [], []
    for ws in speeds:
        p_fl, p_us = _freestream_power_pair(ws, td)
        p_fl_list.append(p_fl)
        p_us_list.append(p_us)

    ax2.plot(speeds, p_fl_list, color=COLOR_FLORIS, lw=2, label="FLORIS")
    ax2.plot(speeds, p_us_list, color=COLOR_OURS, lw=2, ls="--", label="GPUwfarm")
    ax2.set_xlabel("Wind speed (m/s)")
    ax2.set_ylabel("Single-turbine power (kW)")
    ax2.set_title("Freestream power curve (no wake)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False)

    fig.suptitle("GPUwfarm vs FLORIS 4.x -- NREL 5MW, same layout & turbine table")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════
# Layout + wind context, 3-turbine row
# ═════════════════════════════════════════════════════════════════════════

def plot_layout_and_wind(save_path="floris_comparison_layout.png"):
    """
    Two panels giving spatial/wind context for the 3-turbine per-turbine
    comparison above:

      Left  — the layout itself (turbines to scale, T1/T2/T3 labelled) with
              an arrow showing the single wind direction + speed used for
              the per-turbine POWER comparison.
      Right — a wind rose (compass convention, N up, clockwise) showing the
              direction + Weibull speed distribution used for the
              per-turbine AEP comparison. Our test rose has a single
              direction, so this is one spoke split into speed bins.
    """
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    # ── Left: layout + single-condition wind arrow ─────────────────────
    R = D_FLORIS / 2.0
    for i, (tx, ty) in enumerate(zip(LAYOUT_X_3T, LAYOUT_Y_3T)):
        ax1.add_patch(plt.Circle((tx, ty), R, facecolor=COLOR_OURS, edgecolor="k",
                                  linewidth=1.0, alpha=0.8, zorder=3))
        ax1.annotate(f"T{i + 1}", (tx, ty), ha="center", va="center",
                     fontsize=9, fontweight="bold", zorder=4)

    # Wind arrow: our convention, wd measured from +x axis, blowing TOWARD that heading
    wd_rad = np.radians(OUR_WD)
    span_x = max(LAYOUT_X_3T) - min(LAYOUT_X_3T)
    arrow_len = max(span_x * 0.35, 4 * R)
    x0 = min(LAYOUT_X_3T) - arrow_len * 1.3
    y0 = max(LAYOUT_Y_3T) + 3 * R
    dx, dy = arrow_len * np.cos(wd_rad), arrow_len * np.sin(wd_rad)
    ax1.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="-|>", color="0.2", lw=2.5))
    ax1.text(x0, y0 + 3 * R,
             f"WD = {FLORIS_WD:.0f}° (met) / {OUR_WD:.0f}° (ours)\nWS = {WS_TEST:.0f} m/s",
             fontsize=9, ha="left", va="bottom")

    ax1.set_xlabel("x, east (m)")
    ax1.set_ylabel("y, north (m)")
    ax1.set_title("3-turbine row layout\n(power-comparison wind condition)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.25)
    pad = 3 * R
    ax1.set_xlim(min(LAYOUT_X_3T) - arrow_len * 1.5, max(LAYOUT_X_3T) + pad)
    ax1.set_ylim(-6 * R, y0 + 6 * R)

    # ── Right: wind rose for the AEP Weibull distribution ──────────────
    speeds = np.arange(3.0, 25.0, 1.0, dtype=np.float32)
    freqs = TestAEPWindRose._weibull_freq(speeds).astype(np.float32)

    bin_edges = [3, 7, 11, 15, 19, 25]
    cmap = plt.get_cmap("Blues")
    bottom = 0.0
    theta = np.radians(FLORIS_WD)   # compass bearing wind blows FROM
    width = np.radians(18.0)        # wedge width, purely cosmetic (single direction)

    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    for b in range(len(bin_edges) - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (speeds >= lo) & (speeds < hi)
        freq_bin = float(freqs[mask].sum()) * 100.0   # %
        color = cmap(0.35 + 0.55 * b / (len(bin_edges) - 2))
        ax2.bar([theta], [freq_bin], width=width, bottom=bottom,
                color=color, edgecolor="white", linewidth=0.5,
                label=f"{lo}-{hi} m/s")
        bottom += freq_bin

    ax2.set_title(f"AEP wind rose\n(single direction, WD={FLORIS_WD:.0f}° met)", pad=20)
    ax2.set_xticks(np.radians([0, 90, 180, 270]))
    ax2.set_xticklabels(["N", "E", "S", "W"])
    ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False, fontsize=8,
               title="Speed bin")

    fig.suptitle("GPUwfarm vs FLORIS 4.x -- layout & wind context, 3-turbine row")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Layout/wind plot saved to {save_path}")
    plt.show()


def main():
    td = _load_floris_turbine_data()
    _confirm_same_setup(td)
    rows = build_table(td)
    print_and_save_table(rows)
    plot_comparison(td)

    power_ours, power_floris = per_turbine_power_3t(td)
    aep_ours, aep_floris = per_turbine_aep_3t(td)
    per_turbine_rows = build_per_turbine_rows(power_ours, power_floris, aep_ours, aep_floris)
    print_and_save_per_turbine_table(per_turbine_rows)
    plot_per_turbine(power_ours, power_floris, aep_ours, aep_floris)
    plot_layout_and_wind()


if __name__ == "__main__":
    main()
