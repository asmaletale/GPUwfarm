"""
Central configuration dataclasses for the GPUwfarm optimizer.

Physics defaults (WakeConfig, TurbineConfig, FarmConfig.air_density/ti_ambient)
are NOT hardcoded here — they are parsed once from the two FLORIS input YAMLs
that are this project's actual source of truth:

    examples/gch.yaml       — farm / flow-field / wake model parameters
    examples/nrel_5MW.yaml  — turbine geometry + power/Ct table (FLORIS's own file)

These are kept as separate files on purpose (mirrors FLORIS's own farm-input
vs. turbine-library split — do not merge them). loaders/floris_yaml.py reuses
the exact same WakeConfig.from_wake_dict / TurbineConfig.from_turbine_dict
parsers when loading a user-supplied YAML, so `WakeConfig()` and a YAML load
of examples/gch.yaml always produce identical objects.
"""
from __future__ import annotations
import functools
import pathlib
from dataclasses import dataclass, field
from typing import Literal

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_GCH_YAML = _REPO_ROOT / "examples" / "gch.yaml"
_DEFAULT_TURBINE_YAML = _REPO_ROOT / "examples" / "nrel_5MW.yaml"

# FLORIS wake_combination model_strings -> our WakeConfig.combination values
_COMBINATION_MAP = {
    "sosfs": "SOSFS",
    "freestream_linear_superposition": "FLS",
    "maximum_velocity_deficit": "MAX",
}


@functools.lru_cache(maxsize=1)
def _gch_yaml() -> dict:
    with open(_GCH_YAML) as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _default_turbine_yaml() -> dict:
    with open(_DEFAULT_TURBINE_YAML) as f:
        return yaml.safe_load(f)


@dataclass
class WakeConfig:
    # Wake combination method — select from FLORIS implementations
    # FLORIS source: floris/core/wake_velocity/gauss.py and
    #                floris/core/wake_deflection/gauss.py and
    #                floris/core/wake_turbulence/crespo_hernandez.py
    # Defaults are parsed from examples/gch.yaml's `wake:` block — see module
    # docstring. Note: crespo_hernandez.constant=0.9 there matches the
    # CrespoHernandez model CLASS default, not FLORIS's packaged
    # default_inputs.yaml (0.5), which is stale — see NREL/floris#773.
    combination: Literal["SOSFS", "FLS", "MAX"] = field(
        default_factory=lambda: _COMBINATION_MAP.get(
            _gch_yaml()["wake"]["model_strings"]["combination_model"].lower(), "SOSFS"
        )
    )
    alpha: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_velocity_parameters"]["gauss"]["alpha"]))
    beta: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_velocity_parameters"]["gauss"]["beta"]))
    ka: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_velocity_parameters"]["gauss"]["ka"]))
    kb: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_velocity_parameters"]["gauss"]["kb"]))
    ad: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_deflection_parameters"]["gauss"]["ad"]))
    bd: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_deflection_parameters"]["gauss"]["bd"]))
    dm: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_deflection_parameters"]["gauss"]["dm"]))
    ch_initial: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_turbulence_parameters"]["crespo_hernandez"]["initial"]))
    ch_constant: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_turbulence_parameters"]["crespo_hernandez"]["constant"]))
    ch_ai: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_turbulence_parameters"]["crespo_hernandez"]["ai"]))
    ch_downstream: float = field(default_factory=lambda: float(
        _gch_yaml()["wake"]["wake_turbulence_parameters"]["crespo_hernandez"]["downstream"]))

    @classmethod
    def from_wake_dict(cls, wake: dict) -> "WakeConfig":
        """Build a WakeConfig from a FLORIS input YAML's `wake:` section.

        Any key missing from `wake` falls back to this same class's defaults
        (i.e. examples/gch.yaml), never to a separate hardcoded literal.
        """
        default = cls()
        combo_raw = wake.get("model_strings", {}).get("combination_model")
        combination = _COMBINATION_MAP.get(combo_raw.lower(), default.combination) \
            if combo_raw else default.combination
        vel  = wake.get("wake_velocity_parameters", {}).get("gauss", {})
        defl = wake.get("wake_deflection_parameters", {}).get("gauss", {})
        turb = wake.get("wake_turbulence_parameters", {}).get("crespo_hernandez", {})
        return cls(
            combination=combination,
            alpha=float(vel.get("alpha", default.alpha)),
            beta =float(vel.get("beta",  default.beta)),
            ka   =float(vel.get("ka",    default.ka)),
            kb   =float(vel.get("kb",    default.kb)),
            ad   =float(defl.get("ad", default.ad)),
            bd   =float(defl.get("bd", default.bd)),
            dm   =float(defl.get("dm", default.dm)),
            ch_initial   =float(turb.get("initial",    default.ch_initial)),
            ch_constant  =float(turb.get("constant",   default.ch_constant)),
            ch_ai        =float(turb.get("ai",         default.ch_ai)),
            ch_downstream=float(turb.get("downstream", default.ch_downstream)),
        )


@dataclass
class TurbineConfig:
    # Defaults parsed from examples/nrel_5MW.yaml (FLORIS's own turbine file).
    rotor_diameter: float = field(default_factory=lambda: float(
        _default_turbine_yaml()["rotor_diameter"]))
    hub_height: float = field(default_factory=lambda: float(
        _default_turbine_yaml()["hub_height"]))
    cosine_loss_exponent_yaw: float = field(default_factory=lambda: float(
        _default_turbine_yaml()["power_thrust_table"]["cosine_loss_exponent_yaw"]))

    @classmethod
    def from_turbine_dict(cls, t: dict) -> "TurbineConfig":
        """Build a TurbineConfig from a parsed turbine_library/*.yaml dict.

        Any key missing from `t` falls back to this same class's defaults
        (i.e. examples/nrel_5MW.yaml), never to a separate hardcoded literal.
        """
        default = cls()
        pt = t.get("power_thrust_table", {})
        return cls(
            rotor_diameter=float(t.get("rotor_diameter", default.rotor_diameter)),
            hub_height=float(t.get("hub_height", default.hub_height)),
            cosine_loss_exponent_yaw=float(
                pt.get("cosine_loss_exponent_yaw", default.cosine_loss_exponent_yaw)
            ),
        )


@dataclass
class FarmConfig:
    n_turbines:   int   = 20
    area_width:   float = 2000.0  # m
    area_height:  float = 2000.0  # m
    min_spacing:  float = 480.0   # m  (4 * D for D=120)
    # air_density / ti_ambient defaults parsed from examples/gch.yaml's flow_field
    air_density:  float = field(default_factory=lambda: float(
        _gch_yaml()["flow_field"]["air_density"]))
    ti_ambient:   float = field(default_factory=lambda: float(
        sum(_gch_yaml()["flow_field"]["turbulence_intensities"])
        / len(_gch_yaml()["flow_field"]["turbulence_intensities"])
    ))


@dataclass
class GAConfig:
    pop_size:       int   = 256
    n_generations:  int   = 150
    mutation_rate:  float = 0.15
    crossover_rate:      float = 0.7   # probability a parent pair undergoes crossover
    gene_swap_rate:      float = 0.0   # per-turbine swap probability (0 = use 1/T)
    elite:          int   = 6
    max_yaw_deg:    float = 30.0  # degrees


@dataclass
class VisualImpactConfig:
    """Observer and geometry parameters for visual impact assessment.

    Legacy source: legacy/AEP.get_farm_VI() + legacy/configBinary.py.
    Turbine geometry (hub_height, rotor_diameter) is read from TurbineConfig,
    not duplicated here.
    """
    earth_radius: float = 6.371e6   # m
    xfov_deg:     float = 120.0     # degrees — horizontal field of view
    zfov_deg:     float = 40.0      # degrees — vertical field of view

    # Observer locations [(x, y)], heights above ground (m), relative weights
    obs_coords:  list = field(default_factory=lambda: [[0.0, 0.0]])
    obs_heights: list = field(default_factory=lambda: [1.77])
    obs_weights: list = field(default_factory=lambda: [1.0])


@dataclass
class CostConfig:
    """Cost parameters for LCOE calculation. All in millions EUR unless noted."""
    lifetime:              float = 25.0      # years
    discount_rate:         float = 0.05      # 5% annual discount

    # CAPEX per turbine
    dev_consenting_1wt:    float = 3.15      # 0.21 * 15 MW
    turb_substructure_1wt: float = 32.0      # 1.6 * 15 + 8

    # Transmission (internal & export cables, substations)
    c_intcab:              float = 0.3035    # millions/km internal cable
    c_expcable_ac:         float = 2.336     # millions/km export cable AC
    c_expcable_dc:         float = 1.168     # millions/km export cable DC
    c_offsub_ac:           float = 39.0      # offshore substation AC
    c_offsub_dc:           float = 142.75    # offshore substation DC
    c_onsub_dc:            float = 84.35     # onshore substation DC
    ac_dc_threshold:       float = 55.0      # km threshold for AC/DC choice
    n_expcables_ac:        float = 1.0       # cables per 300 MW
    n_expcables_dc:        float = 1.0       # cables per 300 MW

    # Mooring (per turbine, normalized)
    n_lines:               float = 3.0       # mooring lines per turbine
    mbl_chain:             float = 22286.0   # kN (breaking load)
    mbl_dea:               float = 9800.0    # kN (dynamic event analyzer)
    f_usd_eur:             float = 0.92      # USD to EUR conversion

    # Installation
    t_inst:                float = 48.0      # hours per turbine
    v_ahts:                float = 10.0      # km/h (AHTS vessel speed)
    v_psv:                 float = 61.7      # km/h (PSV vessel speed)
    c_boat:                float = 0.011979  # millions/hour rental
    n_turtrip:             float = 3.0       # turbines per trip
    n_fltrip:              float = 2.0       # floaters per trip
    d_port:                float = 10.0      # km distance to port
    c_inst_intcab:         float = 0.115     # millions/km internal cable
    c_inst_expcab:         float = 0.637     # millions/km export cable
    c_inst_offsub:         float = 20.0      # millions (offshore substation)
    c_inst_moo_per_turb:   float = 0.24      # millions per turbine

    # Decommissioning (salvage value, negative)
    r_dec:                 float = 0.23      # millions/MW

    # OPEX (annual, per turbine)
    opex_1wt:              float = 1.965     # 0.131 * 15 MW (Myhr model)
