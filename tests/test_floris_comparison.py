"""
Cross-validation: GPUwfarm vs FLORIS 4.x reference implementation.

Tests that our GPU physics pipeline (GaussVelocityDeficit + CrespoHernandez +
GaussVelocityDeflection + SOSFS) matches the FLORIS reference simulator for
identical layouts, wind conditions, and physics parameters.

Methodology
-----------
To isolate wake-model accuracy from power-curve differences, we load the NREL
5MW power / Ct table directly from FLORIS's own turbine library and feed it
into our TabulatedPowerCurve.  After equalising the tables, any remaining
deviation stems purely from wake-model differences.

Primary metric: wake efficiency
    eta = P_waked_farm / P_freestream_farm
This ratio is insensitive to the absolute power level and directly measures
how accurately each code predicts velocity deficit propagation.

Wind direction convention
-------------------------
FLORIS (meteorological):  270° = from west = blows east (+x axis)
Ours   (mathematical):      0° = blows east (+x axis)
Mapping:  our_wd = (270 - floris_wd_met) % 360
So FLORIS wind_dir=270° ↔ our wd=0°.

FLORIS configuration used in every test
----------------------------------------
  velocity_model    : gauss
  deflection_model  : gauss
  turbulence_model  : crespo_hernandez
  combination_model : sosfs
  turbine_grid_points: 1   (hub-height point only, matches ours)
  secondary_steering: off
  yaw_added_recovery: off
  transverse_velocities: off
  All Gauss parameters identical to our WakeConfig defaults.
  CrespoHernandez constant = 0.9 (overrides FLORIS v4 default of 0.5).

Known residual deviations (see CLAUDE.md)
-----------------------------------------
  1. FLORIS evaluates turbines sequentially (sorted downstream) and
     recomputes each turbine's Ct/axial-induction from its true LOCAL
     (already-waked) inflow speed before using it as a wake source. We
     evaluate all pairs simultaneously and compute every source turbine's
     Ct/axial-induction from FREESTREAM speed (farm_evaluator.py step 3),
     so a turbine's own waking is not reflected in the wake IT generates
     downstream. This only affects turbines with >=1 turbine upstream of
     THEM that also has turbines further downstream (e.g. the middle
     turbine of a 3-turbine row) -- verified to cause ~5% single-condition
     power error and up to ~20% AEP error on the affected turbine, with
     zero effect on turbines whose sources are all unwaked (e.g. any
     2-turbine case, or the first two turbines of a longer row).

     (An earlier version of this note also blamed RSS-vs-sequential TI
     combination for the residual; verified NOT to matter here -- swapping
     the combination method made zero difference on this layout.)

Tolerances
----------
  Single-turbine freestream power: ≤ 1 % (same table, only interp differs)
  Wake efficiency (2-turbine aligned): ≤ 10 % relative
  Multi-turbine AEP ratio vs FLORIS: ≤ 12 % relative
  Direction that these tests fail → suggests a porting bug, not a known deviation.

Skip conditions
---------------
  - ``floris`` package not installed
  - ``cupy`` not available

Install FLORIS:
    pip install floris
Then run:
    pytest tests/test_floris_comparison.py -v -s
"""
from __future__ import annotations
import sys, os, pathlib, inspect
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── availability guards ──────────────────────────────────────────────────────

try:
    import cupy as cp
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

try:
    from floris import FlorisModel as _FM
    _HAS_FLORIS = True
except ImportError:
    _HAS_FLORIS = False

pytestmark = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_FLORIS),
    reason="Requires both cupy and floris packages"
)

# ── constants ─────────────────────────────────────────────────────────────────

# FLORIS nrel_5MW turbine geometry (from nrel_5MW.yaml)
D_FLORIS = 125.88   # m   (rotor diameter in FLORIS turbine library)
HH       = 90.0    # m   (hub height)
TI_AMB   = 0.06    # ambient turbulence intensity
WS_TEST  = 9.0     # m/s  (below-rated → high Ct, strong wakes)

# FLORIS meteorological convention → our mathematical convention
# FLORIS wd=270° (from west, blows east) ↔ our wd=0° (blows east)
FLORIS_WD = 270.0
OUR_WD    = 0.0     # (270 - 270) % 360

# Separation distances in diameters
SEP_5D  = 5 * D_FLORIS    # ~629 m
SEP_7D  = 7 * D_FLORIS    # ~881 m
SEP_10D = 10 * D_FLORIS   # ~1259 m


# ── FLORIS turbine data loader ────────────────────────────────────────────────

def _load_floris_turbine_data():
    """
    Load the NREL 5MW power / Ct table from FLORIS's own turbine library.

    Using this table in both FLORIS and our evaluator equalises the power-curve
    so any remaining error comes purely from wake physics differences.
    """
    from physics.turbine.power_curve import TurbineData

    fpath = pathlib.Path(inspect.getfile(_FM)).parent
    import yaml
    with open(fpath / "turbine_library" / "nrel_5MW.yaml") as f:
        turb = yaml.safe_load(f)

    pt = turb["power_thrust_table"]
    return TurbineData(
        wind_speeds=np.array(pt["wind_speed"],            dtype=np.float32),
        power_kw   =np.array(pt["power"],                 dtype=np.float32),
        ct_values  =np.array(pt["thrust_coefficient"],    dtype=np.float32),
        ref_air_density       =float(pt["ref_air_density"]),
        cosine_loss_exponent_yaw=float(pt["cosine_loss_exponent_yaw"]),
    )


# ── FLORIS model builder ──────────────────────────────────────────────────────

def _floris_model(layout_x, layout_y):
    """
    Build a FlorisModel with parameters matching our WakeConfig defaults.

    Key overrides from FLORIS v4 defaults:
      - turbine_grid_points = 1   (hub height only, matches our evaluator)
      - enable_secondary_steering = False
      - enable_yaw_added_recovery = False
      - enable_transverse_velocities = False
      - crespo_hernandez constant = 0.9  (our value; FLORIS v4 default is 0.5)
    """
    fpath = pathlib.Path(inspect.getfile(_FM)).parent
    import yaml
    with open(fpath / "default_inputs.yaml") as f:
        cfg = yaml.safe_load(f)

    # Single hub-height evaluation point (matches our point-evaluation)
    cfg["solver"]["turbine_grid_points"] = 1

    # Farm layout
    cfg["farm"]["layout_x"] = [float(x) for x in layout_x]
    cfg["farm"]["layout_y"] = [float(y) for y in layout_y]
    cfg["farm"]["turbine_type"] = ["nrel_5MW"] * len(layout_x)

    # Zero wind-shear and veer (our model is flat-shear, no veer)
    cfg["flow_field"]["wind_shear"] = 0.0
    cfg["flow_field"]["wind_veer"]  = 0.0

    # Disable GCH extras — we only implement core Gauss wake
    cfg["wake"]["enable_secondary_steering"]    = False
    cfg["wake"]["enable_yaw_added_recovery"]     = False
    cfg["wake"]["enable_transverse_velocities"]  = False

    # Match our WakeConfig.ch_constant = 0.9
    cfg["wake"]["wake_turbulence_parameters"]["crespo_hernandez"]["constant"] = 0.9

    return _FM(cfg)


def _floris_farm_power_kw(fm, wd_met, ws, ti):
    """Run FLORIS for one wind condition; return (farm_power_kW, per_turbine_kW)."""
    fm.set(
        wind_directions=[float(wd_met)],
        wind_speeds=[float(ws)],
        turbulence_intensities=[float(ti)],
    )
    fm.run()
    per_turb_w = fm.get_turbine_powers()[0]          # (n_turbines,) Watts
    return per_turb_w.sum() / 1000.0, per_turb_w / 1000.0


# ── our evaluator builder ─────────────────────────────────────────────────────

def _our_evaluator(n_turbines, turbine_data=None):
    from config import WakeConfig, FarmConfig, TurbineConfig
    from physics.farm_evaluator import FarmEvaluator

    wake_cfg    = WakeConfig(combination="SOSFS")
    farm_cfg    = FarmConfig(n_turbines=n_turbines, air_density=1.225, ti_ambient=TI_AMB)
    turbine_cfg = TurbineConfig(rotor_diameter=D_FLORIS, hub_height=HH)
    return FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)


def _our_farm_power_kw(layout_x, layout_y, turbine_data=None):
    """
    Run our evaluator for a single wind condition (wd=0°, blows +x).
    Returns total farm power in kW.
    """
    from wind.wind_rose import WindRose

    n = len(layout_x)
    ev = _our_evaluator(n, turbine_data)

    pop = cp.zeros((1, n, 3), dtype=cp.float32)
    pop[0, :, 0] = cp.asarray(np.array(layout_x, dtype=np.float32))
    pop[0, :, 1] = cp.asarray(np.array(layout_y, dtype=np.float32))

    # wd=0° in our convention = wind blows east (+x) = FLORIS wd=270°
    rose = WindRose.from_uniform_ti(
        wind_dirs  =np.array([OUR_WD],   dtype=np.float32),
        wind_speeds=np.array([WS_TEST],  dtype=np.float32),
        freq_table =np.array([[1.0]],    dtype=np.float32),
        ti_ambient =TI_AMB,
    )
    # AEP [kWh] = farm_power [kW] × freq × 8760 h
    aep = float(cp.asnumpy(ev.evaluate(pop, rose))[0])
    return aep / 8760.0


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestFreestreamPowerAgreement:
    """
    With a single turbine (no wake), power should agree within 1 %.
    Both implementations use the same FLORIS power/Ct table, so any
    residual error comes only from interpolation method differences
    (FLORIS uses NumPy interp; ours uses CuPy interp).
    """

    def test_freestream_power_within_1pct(self):
        td = _load_floris_turbine_data()

        # FLORIS: single turbine, no wake
        fm = _floris_model([0.0], [0.0])
        p_floris_kw, _ = _floris_farm_power_kw(fm, FLORIS_WD, WS_TEST, TI_AMB)

        # Ours: single turbine, no wake
        p_ours_kw = _our_farm_power_kw([0.0], [0.0], turbine_data=td)

        rel_err = abs(p_ours_kw - p_floris_kw) / p_floris_kw
        assert rel_err < 0.01, (
            f"Freestream power mismatch: ours={p_ours_kw:.1f} kW, "
            f"FLORIS={p_floris_kw:.1f} kW, rel_err={rel_err:.2%}"
        )

    def test_freestream_power_at_multiple_speeds(self):
        """Verify power-curve agreement across several wind speeds."""
        td     = _load_floris_turbine_data()
        fm     = _floris_model([0.0], [0.0])
        speeds = [5.0, 7.0, 9.0, 11.0, 13.0]

        for ws in speeds:
            fm.set(
                wind_directions=[FLORIS_WD],
                wind_speeds=[ws],
                turbulence_intensities=[TI_AMB],
            )
            fm.run()
            p_floris = fm.get_turbine_powers()[0, 0] / 1000.0   # kW

            from wind.wind_rose import WindRose
            ev = _our_evaluator(1, td)
            pop = cp.zeros((1, 1, 3), dtype=cp.float32)
            rose = WindRose.from_uniform_ti(
                wind_dirs  =np.array([OUR_WD], dtype=np.float32),
                wind_speeds=np.array([ws],     dtype=np.float32),
                freq_table =np.array([[1.0]],  dtype=np.float32),
                ti_ambient =TI_AMB,
            )
            p_ours = float(cp.asnumpy(ev.evaluate(pop, rose))[0]) / 8760.0

            rel_err = abs(p_ours - p_floris) / max(p_floris, 1.0)
            assert rel_err < 0.01, (
                f"Power mismatch at {ws} m/s: ours={p_ours:.1f} kW, "
                f"FLORIS={p_floris:.1f} kW, rel_err={rel_err:.2%}"
            )


class TestWakeEfficiency:
    """
    Wake efficiency η = P_waked_farm / P_freestream_farm.

    Both implementations use the same FLORIS power/Ct table so differences
    reflect wake-model accuracy only.  The dominant remaining deviations are:
      - FLORIS sequential TI propagation vs our simultaneous RSS approach
      - Our TI broadcast uses source-turbine TI for sigma; FLORIS uses
        point-local TI at each x location

    Tolerance: ≤ 10 % relative error on η.
    """

    @staticmethod
    def _floris_wake_efficiency(sep_m):
        """Wake efficiency from FLORIS for 2 turbines at sep_m separation."""
        fm_waked = _floris_model([0.0, sep_m], [0.0, 0.0])
        p_waked, _ = _floris_farm_power_kw(fm_waked, FLORIS_WD, WS_TEST, TI_AMB)

        fm_free = _floris_model([0.0, 50_000.0], [0.0, 0.0])
        p_free, _ = _floris_farm_power_kw(fm_free, FLORIS_WD, WS_TEST, TI_AMB)
        return p_waked / p_free

    @staticmethod
    def _our_wake_efficiency(sep_m, turbine_data):
        p_waked = _our_farm_power_kw([0.0, sep_m], [0.0, 0.0], turbine_data)
        p_free  = _our_farm_power_kw([0.0, 50_000.0], [0.0, 0.0], turbine_data)
        return p_waked / p_free

    def test_wake_efficiency_5D(self):
        """2-turbine row at 5D: η within 10 % of FLORIS."""
        td     = _load_floris_turbine_data()
        eta_fl = self._floris_wake_efficiency(SEP_5D)
        eta_us = self._our_wake_efficiency(SEP_5D, td)
        rel_err = abs(eta_us - eta_fl) / eta_fl
        assert rel_err < 0.10, (
            f"5D wake efficiency: ours={eta_us:.4f}, FLORIS={eta_fl:.4f}, "
            f"rel_err={rel_err:.2%}"
        )

    def test_wake_efficiency_7D(self):
        """2-turbine row at 7D: η within 10 % of FLORIS."""
        td     = _load_floris_turbine_data()
        eta_fl = self._floris_wake_efficiency(SEP_7D)
        eta_us = self._our_wake_efficiency(SEP_7D, td)
        rel_err = abs(eta_us - eta_fl) / eta_fl
        assert rel_err < 0.10, (
            f"7D wake efficiency: ours={eta_us:.4f}, FLORIS={eta_fl:.4f}, "
            f"rel_err={rel_err:.2%}"
        )

    def test_wake_efficiency_10D(self):
        """2-turbine row at 10D: η within 10 % of FLORIS."""
        td     = _load_floris_turbine_data()
        eta_fl = self._floris_wake_efficiency(SEP_10D)
        eta_us = self._our_wake_efficiency(SEP_10D, td)
        rel_err = abs(eta_us - eta_fl) / eta_fl
        assert rel_err < 0.10, (
            f"10D wake efficiency: ours={eta_us:.4f}, FLORIS={eta_fl:.4f}, "
            f"rel_err={rel_err:.2%}"
        )

    def test_wake_efficiency_increases_with_distance(self):
        """η should grow (less wake loss) as separation increases, in both codes."""
        td = _load_floris_turbine_data()
        separations = [SEP_5D, SEP_7D, SEP_10D]

        eta_fl_list = [self._floris_wake_efficiency(s) for s in separations]
        eta_us_list = [self._our_wake_efficiency(s, td) for s in separations]

        for i in range(len(separations) - 1):
            assert eta_fl_list[i] < eta_fl_list[i + 1], \
                f"FLORIS η should increase with distance: {eta_fl_list}"
            assert eta_us_list[i] < eta_us_list[i + 1], \
                f"Ours η should increase with distance: {eta_us_list}"


class TestAEPWindRose:
    """
    AEP comparison over a Weibull wind speed distribution (single direction).

    Integrates farm power over wind speeds 3–24 m/s, Weibull A=9.5, k=2.
    Single direction (our 0°, FLORIS 270°) to keep comparisons clean.

    Tolerance: ≤ 12 % relative (slightly looser than single-condition to
    accommodate the TI differences over many wind-speed conditions).
    """

    @staticmethod
    def _weibull_freq(speeds, A=9.5, k=2.0):
        """Normalised Weibull PDF evaluated at speed bin centres."""
        pdf = (k / A) * (speeds / A) ** (k - 1) * np.exp(-(speeds / A) ** k)
        return pdf / pdf.sum()

    def test_aep_2turbine_aligned_single_direction(self):
        """AEP for 2-turbine aligned row over Weibull speeds, single direction."""
        from wind.wind_rose import WindRose

        speeds = np.arange(3.0, 25.0, 1.0, dtype=np.float32)   # 22 bins
        freqs  = self._weibull_freq(speeds).astype(np.float32)

        td = _load_floris_turbine_data()
        layout_x = [0.0, SEP_7D]
        layout_y = [0.0, 0.0]
        n        = len(layout_x)

        # ── FLORIS AEP ──────────────────────────────────────────────────
        fm = _floris_model(layout_x, layout_y)
        fm.set(
            wind_directions     =np.full(len(speeds), FLORIS_WD, dtype=float),
            wind_speeds         =speeds.astype(float),
            turbulence_intensities=np.full(len(speeds), TI_AMB, dtype=float),
        )
        fm.run()
        # get_farm_AEP returns total Wh; freq array must sum to 1
        aep_floris_kwh = fm.get_farm_AEP(freq=freqs.astype(float)) / 1000.0

        # ── Our AEP ─────────────────────────────────────────────────────
        ev  = _our_evaluator(n, td)
        pop = cp.zeros((1, n, 3), dtype=cp.float32)
        pop[0, :, 0] = cp.asarray(np.array(layout_x, dtype=np.float32))
        rose = WindRose.from_uniform_ti(
            wind_dirs  =np.array([OUR_WD],          dtype=np.float32),
            wind_speeds=speeds,
            freq_table =freqs.reshape(1, -1),
            ti_ambient =TI_AMB,
        )
        aep_ours_kwh = float(cp.asnumpy(ev.evaluate(pop, rose))[0])

        rel_err = abs(aep_ours_kwh - aep_floris_kwh) / aep_floris_kwh
        assert rel_err < 0.12, (
            f"AEP mismatch (2-turbine, Weibull, 1 dir): "
            f"ours={aep_ours_kwh:.0f} kWh, FLORIS={aep_floris_kwh:.0f} kWh, "
            f"rel_err={rel_err:.2%}"
        )

    def test_aep_single_turbine_matches_power_curve_integral(self):
        """
        Single turbine AEP (no wake) from our code vs FLORIS vs analytic integral.

        Since there is no wake, both should agree within 1 % of each other
        and also match the analytic integral of P(ws)*f(ws)*8760.
        """
        from wind.wind_rose import WindRose

        speeds = np.arange(3.0, 25.0, 1.0, dtype=np.float32)
        freqs  = self._weibull_freq(speeds).astype(np.float32)

        td = _load_floris_turbine_data()

        # Analytic reference: FLORIS power-curve × Weibull freq × 8760
        aep_analytic_kwh = float(
            np.sum(np.interp(speeds, td.wind_speeds, td.power_kw) * freqs) * 8760.0
        )

        # FLORIS
        fm = _floris_model([0.0], [0.0])
        fm.set(
            wind_directions      =np.full(len(speeds), FLORIS_WD, dtype=float),
            wind_speeds          =speeds.astype(float),
            turbulence_intensities=np.full(len(speeds), TI_AMB, dtype=float),
        )
        fm.run()
        aep_floris_kwh = fm.get_farm_AEP(freq=freqs.astype(float)) / 1000.0

        # Ours
        ev  = _our_evaluator(1, td)
        pop = cp.zeros((1, 1, 3), dtype=cp.float32)
        rose = WindRose.from_uniform_ti(
            wind_dirs  =np.array([OUR_WD], dtype=np.float32),
            wind_speeds=speeds,
            freq_table =freqs.reshape(1, -1),
            ti_ambient =TI_AMB,
        )
        aep_ours_kwh = float(cp.asnumpy(ev.evaluate(pop, rose))[0])

        # Both vs analytic
        assert abs(aep_floris_kwh - aep_analytic_kwh) / aep_analytic_kwh < 0.01, (
            f"FLORIS single-turbine AEP {aep_floris_kwh:.0f} vs analytic "
            f"{aep_analytic_kwh:.0f}"
        )
        assert abs(aep_ours_kwh - aep_analytic_kwh) / aep_analytic_kwh < 0.01, (
            f"Ours single-turbine AEP {aep_ours_kwh:.0f} vs analytic "
            f"{aep_analytic_kwh:.0f}"
        )


class TestMultiTurbineRow:
    """
    3-turbine and 4-turbine rows test the sequential vs simultaneous difference
    because wake interactions involve feedback between turbines 2 and 3.

    Tolerance is wider (15 %) to accommodate the known TI propagation difference
    between FLORIS's sequential solver and our simultaneous all-pairs approach.
    """

    def test_3turbine_row_farm_power(self):
        """3-turbine row at 5D spacing: farm power within 15 % of FLORIS."""
        td       = _load_floris_turbine_data()
        layout_x = [0.0, SEP_5D, 2 * SEP_5D]
        layout_y = [0.0, 0.0,    0.0]

        fm = _floris_model(layout_x, layout_y)
        p_floris, _ = _floris_farm_power_kw(fm, FLORIS_WD, WS_TEST, TI_AMB)
        p_ours      = _our_farm_power_kw(layout_x, layout_y, td)

        rel_err = abs(p_ours - p_floris) / p_floris
        assert rel_err < 0.15, (
            f"3-turbine row farm power: ours={p_ours:.1f} kW, "
            f"FLORIS={p_floris:.1f} kW, rel_err={rel_err:.2%}"
        )

    def test_3turbine_row_wake_efficiency(self):
        """3-turbine row: η within 15 % of FLORIS."""
        td       = _load_floris_turbine_data()
        layout_x = [0.0, SEP_5D, 2 * SEP_5D]
        layout_y = [0.0, 0.0,    0.0]

        # Waked farm
        fm_waked = _floris_model(layout_x, layout_y)
        p_fl_waked, _ = _floris_farm_power_kw(fm_waked, FLORIS_WD, WS_TEST, TI_AMB)
        p_us_waked    = _our_farm_power_kw(layout_x, layout_y, td)

        # Freestream (turbines far apart)
        free_x = [0.0, 50_000.0, 100_000.0]
        fm_free = _floris_model(free_x, layout_y)
        p_fl_free, _ = _floris_farm_power_kw(fm_free, FLORIS_WD, WS_TEST, TI_AMB)
        p_us_free    = _our_farm_power_kw(free_x, layout_y, td)

        eta_fl = p_fl_waked / p_fl_free
        eta_us = p_us_waked / p_us_free

        rel_err = abs(eta_us - eta_fl) / eta_fl
        assert rel_err < 0.15, (
            f"3-turbine η: ours={eta_us:.4f}, FLORIS={eta_fl:.4f}, "
            f"rel_err={rel_err:.2%}"
        )


class TestPrintSummary:
    """
    Non-asserting diagnostic test that prints a side-by-side comparison table.
    Always passes; run with -s to see the output.
    """

    def test_print_comparison_table(self, capsys):
        td = _load_floris_turbine_data()
        print("\n" + "=" * 60)
        print("GPUwfarm vs FLORIS -- wake efficiency comparison (9 m/s)")
        print("=" * 60)
        print(f"{'Sep':>8}  {'FLORIS eff':>10}  {'Ours eff':>10}  {'Rel err':>10}")
        print("-" * 60)

        for sep_D, sep_m in [(5, SEP_5D), (7, SEP_7D), (10, SEP_10D)]:
            fm = _floris_model([0.0, sep_m], [0.0, 0.0])
            p_waked, _ = _floris_farm_power_kw(fm, FLORIS_WD, WS_TEST, TI_AMB)
            fm_free = _floris_model([0.0, 50_000.0], [0.0, 0.0])
            p_free, _ = _floris_farm_power_kw(fm_free, FLORIS_WD, WS_TEST, TI_AMB)
            eta_fl = p_waked / p_free

            p_w_us = _our_farm_power_kw([0.0, sep_m], [0.0, 0.0], td)
            p_f_us = _our_farm_power_kw([0.0, 50_000.0], [0.0, 0.0], td)
            eta_us = p_w_us / p_f_us

            rel = (eta_us - eta_fl) / eta_fl
            print(f"{sep_D:>5}D    {eta_fl:>10.4f}  {eta_us:>10.4f}  {rel:>+10.2%}")

        print("=" * 60)
