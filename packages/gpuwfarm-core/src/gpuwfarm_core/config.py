"""
Configuration dataclasses for the GPUwfarm evaluation core.

Physics/cost/visual-impact defaults are NOT hardcoded here — they are parsed
once from focused FLORIS-style input YAMLs that are this project's actual
source of truth, kept as separate files on purpose (one concern per file,
easy to hand-edit for a new site/case):

    examples/gch.yaml           — farm / flow-field / wake model parameters
    examples/<turbine>.yaml     — turbine geometry + power/Ct table (FLORIS's
                                   own turbine_library format, e.g. nrel_5MW.yaml)
    examples/costs.yaml         — LCOE cost model parameters
    examples/visual_impact.yaml — observer geometry for visual-impact assessment

loaders/floris_yaml.py reuses the exact same WakeConfig.from_wake_dict /
TurbineConfig.from_turbine_dict parsers when loading a user-supplied YAML, so
`WakeConfig()` and a YAML load of examples/gch.yaml always produce identical
objects. These configs are consumed by the wake physics, the farm evaluator,
and the LCOE/visual-impact objectives. The optimizer-only ``GAConfig`` lives
in the optimizer package.
"""
from __future__ import annotations
import functools
import pathlib
from dataclasses import dataclass, field
from typing import Literal

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_GCH_YAML = _REPO_ROOT / "examples" / "gch.yaml"
_DEFAULT_TURBINE_YAML = _REPO_ROOT / "examples" / "nrel_5MW.yaml"
_COSTS_YAML = _REPO_ROOT / "examples" / "costs.yaml"
_VI_YAML = _REPO_ROOT / "examples" / "visual_impact.yaml"

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


@functools.lru_cache(maxsize=1)
def _costs_yaml() -> dict:
    with open(_COSTS_YAML) as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _vi_yaml() -> dict:
    with open(_VI_YAML) as f:
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
class VisualImpactConfig:
    """Observer and geometry parameters for visual impact assessment.

    Legacy source: legacy/AEP.get_farm_VI() + legacy/configBinary.py.
    Turbine geometry (hub_height, rotor_diameter) is read from TurbineConfig,
    not duplicated here. Defaults parsed from examples/visual_impact.yaml.
    """
    earth_radius: float = field(default_factory=lambda: float(
        _vi_yaml()["earth_radius"]))
    xfov_deg: float = field(default_factory=lambda: float(
        _vi_yaml()["xfov_deg"]))
    zfov_deg: float = field(default_factory=lambda: float(
        _vi_yaml()["zfov_deg"]))

    # Observer locations [(x, y)], heights above ground (m), relative weights
    obs_coords:  list = field(default_factory=lambda: list(_vi_yaml()["obs_coords"]))
    obs_heights: list = field(default_factory=lambda: list(_vi_yaml()["obs_heights"]))
    obs_weights: list = field(default_factory=lambda: list(_vi_yaml()["obs_weights"]))

    @classmethod
    def from_vi_dict(cls, vi: dict) -> "VisualImpactConfig":
        """Build a VisualImpactConfig from a parsed visual_impact.yaml-shaped dict.

        Any key missing from `vi` falls back to this same class's defaults
        (i.e. examples/visual_impact.yaml), never to a separate hardcoded literal.
        """
        default = cls()
        return cls(
            earth_radius=float(vi.get("earth_radius", default.earth_radius)),
            xfov_deg=float(vi.get("xfov_deg", default.xfov_deg)),
            zfov_deg=float(vi.get("zfov_deg", default.zfov_deg)),
            obs_coords=list(vi.get("obs_coords", default.obs_coords)),
            obs_heights=list(vi.get("obs_heights", default.obs_heights)),
            obs_weights=list(vi.get("obs_weights", default.obs_weights)),
        )


@dataclass
class CostConfig:
    """Cost parameters for LCOE calculation. All in millions EUR unless noted.

    Defaults parsed from examples/costs.yaml.
    """
    lifetime:              float = field(default_factory=lambda: float(_costs_yaml()["lifetime"]))
    discount_rate:         float = field(default_factory=lambda: float(_costs_yaml()["discount_rate"]))

    # CAPEX per turbine
    dev_consenting_1wt:    float = field(default_factory=lambda: float(_costs_yaml()["dev_consenting_1wt"]))
    turb_substructure_1wt: float = field(default_factory=lambda: float(_costs_yaml()["turb_substructure_1wt"]))

    # Transmission (internal & export cables, substations)
    c_intcab:              float = field(default_factory=lambda: float(_costs_yaml()["c_intcab"]))
    c_expcable_ac:         float = field(default_factory=lambda: float(_costs_yaml()["c_expcable_ac"]))
    c_expcable_dc:         float = field(default_factory=lambda: float(_costs_yaml()["c_expcable_dc"]))
    c_offsub_ac:           float = field(default_factory=lambda: float(_costs_yaml()["c_offsub_ac"]))
    c_offsub_dc:           float = field(default_factory=lambda: float(_costs_yaml()["c_offsub_dc"]))
    c_onsub_dc:            float = field(default_factory=lambda: float(_costs_yaml()["c_onsub_dc"]))
    ac_dc_threshold:       float = field(default_factory=lambda: float(_costs_yaml()["ac_dc_threshold"]))
    n_expcables_ac:        float = field(default_factory=lambda: float(_costs_yaml()["n_expcables_ac"]))
    n_expcables_dc:        float = field(default_factory=lambda: float(_costs_yaml()["n_expcables_dc"]))

    # Mooring (per turbine, normalized)
    n_lines:               float = field(default_factory=lambda: float(_costs_yaml()["n_lines"]))
    mbl_chain:             float = field(default_factory=lambda: float(_costs_yaml()["mbl_chain"]))
    mbl_dea:               float = field(default_factory=lambda: float(_costs_yaml()["mbl_dea"]))
    f_usd_eur:             float = field(default_factory=lambda: float(_costs_yaml()["f_usd_eur"]))

    # Installation
    t_inst:                float = field(default_factory=lambda: float(_costs_yaml()["t_inst"]))
    v_ahts:                float = field(default_factory=lambda: float(_costs_yaml()["v_ahts"]))
    v_psv:                 float = field(default_factory=lambda: float(_costs_yaml()["v_psv"]))
    c_boat:                float = field(default_factory=lambda: float(_costs_yaml()["c_boat"]))
    n_turtrip:             float = field(default_factory=lambda: float(_costs_yaml()["n_turtrip"]))
    n_fltrip:              float = field(default_factory=lambda: float(_costs_yaml()["n_fltrip"]))
    d_port:                float = field(default_factory=lambda: float(_costs_yaml()["d_port"]))
    c_inst_intcab:         float = field(default_factory=lambda: float(_costs_yaml()["c_inst_intcab"]))
    c_inst_expcab:         float = field(default_factory=lambda: float(_costs_yaml()["c_inst_expcab"]))
    c_inst_offsub:         float = field(default_factory=lambda: float(_costs_yaml()["c_inst_offsub"]))
    c_inst_moo_per_turb:   float = field(default_factory=lambda: float(_costs_yaml()["c_inst_moo_per_turb"]))

    # Decommissioning (salvage value, negative)
    r_dec:                 float = field(default_factory=lambda: float(_costs_yaml()["r_dec"]))

    # OPEX (annual, per turbine)
    opex_1wt:              float = field(default_factory=lambda: float(_costs_yaml()["opex_1wt"]))

    @classmethod
    def from_costs_dict(cls, c: dict) -> "CostConfig":
        """Build a CostConfig from a parsed costs.yaml-shaped dict.

        Any key missing from `c` falls back to this same class's defaults
        (i.e. examples/costs.yaml), never to a separate hardcoded literal.
        """
        default = cls()
        kwargs = {}
        for f in default.__dataclass_fields__:
            kwargs[f] = float(c.get(f, getattr(default, f)))
        return cls(**kwargs)
