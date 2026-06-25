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
    pop_size:      int   = 256
    n_generations: int   = 150
    mutation_rate: float = 0.15
    elite:         int   = 6
    max_yaw_deg:   float = 30.0   # degrees
