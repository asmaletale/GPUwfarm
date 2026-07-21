"""
Load a FLORIS v4 input YAML and convert to GPUwfarm config objects.

FLORIS input reference: https://github.com/NREL/floris
Turbine YAML schema:    floris/turbine_library/*.yaml
Farm input schema:      floris/core/default_inputs.yaml

Wind direction convention
-------------------------
FLORIS uses meteorological convention: 270 deg = from west = blows east (+x).
GPUwfarm uses mathematical convention: 0 deg = blows east (+x).
Mapping: our_wd = (270 - floris_wd_met) % 360
"""
from __future__ import annotations
import pathlib
import numpy as np
import yaml

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose


def _resolve_turbine_path(turbine_type: str, yaml_dir: pathlib.Path) -> pathlib.Path | None:
    # 1. flat file next to the input YAML (e.g. examples/nrel_5MW.yaml)
    flat = yaml_dir / f"{turbine_type}.yaml"
    if flat.exists():
        return flat
    # 2. local turbine_library/ next to the input YAML
    local = yaml_dir / "turbine_library" / f"{turbine_type}.yaml"
    if local.exists():
        return local
    # 3. FLORIS package turbine library (if floris is installed)
    try:
        from floris import FlorisModel
        import inspect
        pkg_dir = pathlib.Path(inspect.getfile(FlorisModel)).parent
        pkg_path = pkg_dir / "turbine_library" / f"{turbine_type}.yaml"
        if pkg_path.exists():
            return pkg_path
    except ImportError:
        pass
    return None


def _load_turbine(turbine_type: str, yaml_dir: pathlib.Path) -> tuple[TurbineConfig, TurbineData]:
    path = _resolve_turbine_path(turbine_type, yaml_dir)
    if path is None:
        return TurbineConfig(), TurbineData.nrel_5mw()

    with open(path) as f:
        t = yaml.safe_load(f)

    return TurbineConfig.from_turbine_dict(t), TurbineData.from_turbine_yaml(path)


def load_floris_yaml(path: str | pathlib.Path) -> dict:
    """
    Parse a FLORIS v4 input YAML and return a dict:

        farm_cfg     : FarmConfig
        wake_cfg     : WakeConfig
        turbine_cfg  : TurbineConfig
        turbine_data : TurbineData  (power/Ct tables)
        wind_rose    : WindRose
        layout_xy    : np.ndarray  (N, 2) turbine positions in metres

    Only the fields that GPUwfarm uses are extracted; the rest of the FLORIS
    YAML (solver, logging, wind_shear, wind_veer, secondary_steering, …) is
    silently ignored.

    Wind direction conversion
    -------------------------
    FLORIS wind_directions are in meteorological degrees (270 = from west).
    They are converted to our mathematical convention before building WindRose.
    """
    path = pathlib.Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    yaml_dir = path.parent

    # ── Farm layout ───────────────────────────────────────────────────────────
    layout_x = np.array(cfg["farm"]["layout_x"], dtype=np.float32)
    layout_y = np.array(cfg["farm"]["layout_y"], dtype=np.float32)
    layout_xy = np.stack([layout_x, layout_y], axis=1)   # (N, 2)
    n_turbines = len(layout_x)

    # ── Flow field ────────────────────────────────────────────────────────────
    ff = cfg["flow_field"]
    air_density = float(ff.get("air_density", FarmConfig().air_density))

    # In FLORIS v4 each element of these three lists is one simulation condition
    # (they are not a grid — they are paired).
    wd_met = np.array(ff["wind_directions"], dtype=np.float32)
    ws_arr = np.array(ff["wind_speeds"], dtype=np.float32)
    ti_arr = np.array(ff["turbulence_intensities"], dtype=np.float32)

    # Met → math direction
    wd_our = (270.0 - wd_met) % 360.0

    # Build a (n_wd, n_ws) grid from the listed conditions
    unique_dirs   = np.unique(wd_our)
    unique_speeds = np.unique(ws_arr)
    n_wd, n_ws = len(unique_dirs), len(unique_speeds)

    freq_table  = np.zeros((n_wd, n_ws), dtype=np.float32)
    ti_sum      = np.zeros((n_wd, n_ws), dtype=np.float64)
    count_table = np.zeros((n_wd, n_ws), dtype=np.int32)

    for wd, ws, ti in zip(wd_our, ws_arr, ti_arr):
        i = int(np.searchsorted(unique_dirs, wd))
        j = int(np.searchsorted(unique_speeds, ws))
        freq_table[i, j] += 1.0
        ti_sum[i, j]     += ti
        count_table[i, j] += 1

    ti_ambient = float(ti_arr.mean())
    with np.errstate(invalid="ignore"):
        ti_table = np.where(
            count_table > 0,
            ti_sum / np.maximum(count_table, 1),
            ti_ambient,
        ).astype(np.float32)

    # WindRose.__post_init__ normalises freq_table
    wind_rose = WindRose(
        wind_dirs=unique_dirs,
        wind_speeds=unique_speeds,
        freq_table=freq_table,
        ti_table=ti_table,
    )

    # ── Wake config ───────────────────────────────────────────────────────────
    wake_cfg = WakeConfig.from_wake_dict(cfg.get("wake", {}))

    # ── Turbine ───────────────────────────────────────────────────────────────
    turbine_types = cfg["farm"].get("turbine_type", ["nrel_5MW"])
    turbine_type  = turbine_types[0] if isinstance(turbine_types, list) else turbine_types
    turbine_cfg, turbine_data = _load_turbine(turbine_type, yaml_dir)

    # ── FarmConfig ────────────────────────────────────────────────────────────
    if n_turbines > 1:
        x_span = float(layout_x.max() - layout_x.min())
        y_span = float(layout_y.max() - layout_y.min())
    else:
        x_span = y_span = 0.0
    # 20 % padding around the existing layout so mutated turbines stay in-bounds
    area_width  = max(x_span * 1.2, turbine_cfg.rotor_diameter * 10)
    area_height = max(y_span * 1.2, turbine_cfg.rotor_diameter * 10)
    min_spacing = turbine_cfg.rotor_diameter * 2.0

    farm_cfg = FarmConfig(
        n_turbines  =n_turbines,
        area_width  =area_width,
        area_height =area_height,
        min_spacing =min_spacing,
        air_density =air_density,
        ti_ambient  =ti_ambient,
    )

    return {
        "farm_cfg":     farm_cfg,
        "wake_cfg":     wake_cfg,
        "turbine_cfg":  turbine_cfg,
        "turbine_data": turbine_data,
        "wind_rose":    wind_rose,
        "layout_xy":    layout_xy,
    }
