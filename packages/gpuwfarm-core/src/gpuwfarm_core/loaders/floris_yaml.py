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


_COMBINATION_MAP = {
    "sosfs": "SOSFS",
    "freestream_linear_superposition": "FLS",
    "maximum_velocity_deficit": "MAX",
}


def _resolve_turbine_path(turbine_type: str, yaml_dir: pathlib.Path) -> pathlib.Path | None:
    # 1. local turbine_library/ next to the input YAML
    local = yaml_dir / "turbine_library" / f"{turbine_type}.yaml"
    if local.exists():
        return local
    # 2. FLORIS package turbine library (if floris is installed)
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

    turbine_cfg = TurbineConfig(
        rotor_diameter=float(t["rotor_diameter"]),
        hub_height=float(t["hub_height"]),
        cosine_loss_exponent_yaw=float(
            t.get("power_thrust_table", {}).get("cosine_loss_exponent_yaw", 1.88)
        ),
    )

    pt = t["power_thrust_table"]
    turbine_data = TurbineData(
        wind_speeds=np.array(pt["wind_speed"], dtype=np.float32),
        power_kw=np.array(pt["power"], dtype=np.float32),
        ct_values=np.array(pt["thrust_coefficient"], dtype=np.float32),
        ref_air_density=float(pt.get("ref_air_density", 1.225)),
        cosine_loss_exponent_yaw=float(pt.get("cosine_loss_exponent_yaw", 1.88)),
    )
    return turbine_cfg, turbine_data


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
    air_density = float(ff.get("air_density", 1.225))

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
    wake = cfg.get("wake", {})
    model_strings = wake.get("model_strings", {})
    combo_raw = model_strings.get("combination_model", "sosfs").lower()
    combination = _COMBINATION_MAP.get(combo_raw, "SOSFS")

    vel_p  = wake.get("wake_velocity_parameters",  {}).get("gauss", {})
    def_p  = wake.get("wake_deflection_parameters", {}).get("gauss", {})
    turb_p = wake.get("wake_turbulence_parameters", {}).get("crespo_hernandez", {})

    wake_cfg = WakeConfig(
        combination=combination,
        alpha=float(vel_p.get("alpha", 0.58)),
        beta =float(vel_p.get("beta",  0.077)),
        ka   =float(vel_p.get("ka",    0.38)),
        kb   =float(vel_p.get("kb",    0.004)),
        ad   =float(def_p.get("ad",    0.0)),
        bd   =float(def_p.get("bd",    0.0)),
        dm   =float(def_p.get("dm",    1.0)),
        ch_initial   =float(turb_p.get("initial",    0.1)),
        ch_constant  =float(turb_p.get("constant",   0.9)),
        ch_ai        =float(turb_p.get("ai",         0.8)),
        ch_downstream=float(turb_p.get("downstream", -0.32)),
    )

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
