"""
Central configuration dataclasses for the GPUwfarm optimizer.

All physical constants use FLORIS defaults unless noted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WakeConfig:
    # Wake combination method — select from FLORIS implementations
    combination: Literal["SOSFS", "FLS", "MAX"] = "SOSFS"

    # Gauss wake / deflection shared parameters
    # FLORIS source: floris/core/wake_velocity/gauss.py and
    #                floris/core/wake_deflection/gauss.py
    alpha: float = 0.58   # near/far-wake boundary dependence on TI
    beta:  float = 0.077  # near/far-wake boundary dependence on Ct
    ka:    float = 0.38   # wake expansion linear coefficient with TI
    kb:    float = 0.004  # wake expansion base rate

    # Deflection-only parameters
    # FLORIS source: floris/core/wake_deflection/gauss.py
    ad:    float = 0.0    # lateral offset tuning parameter
    bd:    float = 0.0    # lateral offset tuning parameter
    dm:    float = 1.0    # deflection magnitude scaling

    # Crespo-Hernandez turbulence parameters
    # FLORIS source: floris/core/wake_turbulence/crespo_hernandez.py
    # Note: original Crespo 1996 paper uses initial=0.0325, constant=0.73, ai=0.8325
    # FLORIS defaults differ; see https://github.com/NREL/floris/issues/773
    ch_initial:    float = 0.1    # exponent on ambient TI
    ch_constant:   float = 0.9   # overall scaling constant
    ch_ai:         float = 0.8   # exponent on axial induction factor
    ch_downstream: float = -0.32 # exponent on normalised downstream distance


@dataclass
class TurbineConfig:
    rotor_diameter: float = 120.0   # m
    hub_height:     float = 90.0    # m
    cosine_loss_exponent_yaw: float = 1.88  # FLORIS CosineLossTurbine default


@dataclass
class FarmConfig:
    n_turbines:   int   = 20
    area_width:   float = 2000.0  # m
    area_height:  float = 2000.0  # m
    min_spacing:  float = 480.0   # m  (4 * D for D=120)
    air_density:  float = 1.225   # kg/m³
    ti_ambient:   float = 0.06    # ambient turbulence intensity


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
