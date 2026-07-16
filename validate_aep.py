"""
Validate AEP consistency between current CuPy implementation and legacy FLORIS.

This script evaluates a test layout with both implementations and compares results.
"""
import numpy as np
import sys

try:
    from floris.tools import FlorisInterface
    FLORIS_AVAILABLE = True
except ImportError:
    FLORIS_AVAILABLE = False
    print("Warning: FLORIS not available. Skipping legacy comparison.")

import cupy as cp
from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig, CostConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose


def get_test_layout(n_turbines: int = 9) -> np.ndarray:
    """Return a simple grid layout for testing."""
    side = int(np.sqrt(n_turbines))
    x = np.tile(np.linspace(0, 2000, side), side)
    y = np.repeat(np.linspace(0, 2000, side), side)
    return np.column_stack([x[:n_turbines], y[:n_turbines]])


def aep_cupy_implementation(
    layout_xy: np.ndarray,
    wind_rose: WindRose,
    wake_cfg: WakeConfig,
    farm_cfg: FarmConfig,
    turbine_cfg: TurbineConfig,
) -> float:
    """
    Compute AEP using current CuPy implementation.

    Args:
        layout_xy:  (T, 2) x, y coordinates
        wind_rose:  WindRose object
        wake_cfg:   WakeConfig
        farm_cfg:   FarmConfig
        turbine_cfg: TurbineConfig

    Returns:
        AEP in kWh
    """
    turbine_data = TurbineData.nrel_5mw()
    evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)

    T = len(layout_xy)
    pop = cp.zeros((1, T, 3), dtype=cp.float32)
    pop[0, :, 0] = cp.asarray(layout_xy[:, 0], dtype=cp.float32)
    pop[0, :, 1] = cp.asarray(layout_xy[:, 1], dtype=cp.float32)
    pop[0, :, 2] = cp.float32(0)  # no yaw

    aep = evaluator.evaluate(pop)
    return float(cp.asnumpy(aep[0]))


def aep_floris_implementation(
    layout_xy: np.ndarray,
    wind_dirs: np.ndarray,
    wind_speeds: np.ndarray,
    freq: np.ndarray,
    floris_yaml: str,
) -> float:
    """
    Compute AEP using legacy FLORIS implementation.

    Args:
        layout_xy:    (T, 2) x, y coordinates
        wind_dirs:    (n_wd,) degrees
        wind_speeds:  (n_ws,) m/s
        freq:         (n_wd, n_ws) frequency table
        floris_yaml:  path to FLORIS v4 YAML file

    Returns:
        AEP in kWh
    """
    fi = FlorisInterface(floris_yaml)
    fi.reinitialize(
        layout_x=layout_xy[:, 0],
        layout_y=layout_xy[:, 1],
        wind_directions=wind_dirs,
        wind_speeds=wind_speeds,
    )
    aep = fi.get_farm_AEP(freq=freq)
    return float(aep)


def main():
    print("=" * 70)
    print("AEP Validation: CuPy Implementation vs. Legacy FLORIS")
    print("=" * 70)

    # Test layout
    n_turbines = 9
    layout_xy = get_test_layout(n_turbines)

    print(f"\nTest layout: {n_turbines} turbines in 3×3 grid")
    print(f"Layout bounds: [{layout_xy[:, 0].min():.0f}, {layout_xy[:, 0].max():.0f}] × "
          f"[{layout_xy[:, 1].min():.0f}, {layout_xy[:, 1].max():.0f}] m")

    # Configuration
    wake_cfg = WakeConfig(combination="SOSFS")
    farm_cfg = FarmConfig(n_turbines=n_turbines, area_width=2500, area_height=2500)
    turbine_cfg = TurbineConfig()

    # Wind rose
    wind_rose = WindRose.default_12sector()
    wind_dirs = wind_rose.wind_dirs
    wind_speeds = wind_rose.wind_speeds
    freq = wind_rose.freq_table

    print(f"\nWind rose: {len(wind_dirs)} directions × {len(wind_speeds)} speeds")

    # Compute AEP with CuPy
    print("\n--- CuPy Implementation ---")
    try:
        aep_cupy = aep_cupy_implementation(
            layout_xy, wind_rose, wake_cfg, farm_cfg, turbine_cfg
        )
        print(f"AEP (CuPy):    {aep_cupy:.4e} kWh")
    except Exception as e:
        print(f"Error computing CuPy AEP: {e}")
        aep_cupy = None

    # Compute AEP with FLORIS (if available)
    aep_floris = None
    if FLORIS_AVAILABLE:
        print("\n--- Legacy FLORIS Implementation ---")
        try:
            aep_floris = aep_floris_implementation(
                layout_xy, wind_dirs, wind_speeds, freq, "gch_iea_15MW.yaml"
            )
            print(f"AEP (FLORIS):  {aep_floris:.4e} kWh")
        except Exception as e:
            print(f"Error computing FLORIS AEP: {e}")

    # Comparison
    if aep_cupy is not None and aep_floris is not None:
        print("\n--- Comparison ---")
        diff = aep_cupy - aep_floris
        pct_diff = 100 * diff / aep_floris
        print(f"Difference:    {diff:.4e} kWh ({pct_diff:+.2f}%)")
        print(f"Relative RMSE: {abs(pct_diff):.2f}%")

        if abs(pct_diff) < 5.0:
            print("✓ AEP consistency check PASSED (< 5% difference)")
        else:
            print("✗ AEP consistency check FAILED (> 5% difference)")
    elif aep_cupy is not None:
        print(f"\n✓ CuPy AEP computed successfully: {aep_cupy:.4e} kWh")
        print("  (FLORIS comparison skipped)")


if __name__ == "__main__":
    main()
