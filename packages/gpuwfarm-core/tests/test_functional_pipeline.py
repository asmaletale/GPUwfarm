"""
The public pipeline stages must reproduce FarmEvaluator.evaluate() exactly.

evaluate() is a shim over upload_conditions / to_wind_frame /
pairwise_displacement / effective_ti / update_inflow / integrate_aep. This is
the check that fails if one of those extractions drifts from the orchestrator,
and it doubles as the executable form of example_core.py.
"""
import numpy as np
import pytest

try:
    import cupy as cp
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="CuPy not available")

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator, N_JACOBI_ITERS
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose


def _evaluator(combination: str = "SOSFS") -> FarmEvaluator:
    return FarmEvaluator(
        FarmConfig(n_turbines=6, area_width=2000, area_height=2000),
        TurbineConfig(),
        WakeConfig(combination=combination),
        TurbineData.nrel_5mw(),
    )


def _pop(P: int = 4, T: int = 6, yaw_deg: float = 15.0) -> "cp.ndarray":
    """Random layout with real yaw, so deflection is exercised too."""
    rng = np.random.default_rng(0)
    pop = np.zeros((P, T, 3), dtype=np.float32)
    pop[:, :, 0] = rng.uniform(0, 2000, (P, T))
    pop[:, :, 1] = rng.uniform(0, 2000, (P, T))
    pop[:, :, 2] = np.deg2rad(rng.uniform(-yaw_deg, yaw_deg, (P, T)))
    return cp.asarray(pop)


def _run_stages(ev, pop, wind_rose, n_jacobi=N_JACOBI_ITERS, per_turbine=False):
    """The pipeline written out by hand — mirrors example_core.py."""
    wd_rad, ws, freq, ti = ev.upload_conditions(wind_rose)
    xw, yw, yaw, ws_b, ti_b = ev.to_wind_frame(pop, wd_rad, ws, ti)
    dx, dy, downstream = ev.pairwise_displacement(xw, yw)

    u_src  = cp.broadcast_to(ws_b[:, None], xw.shape).copy()
    x_zero = cp.zeros_like(xw)

    for _ in range(n_jacobi):
        ct = ev.power_curve.ct_gpu(u_src)
        ai = ev.power_curve.axial_induction_from_ct(ct)
        ti_added = ev.turbulence_model.compute(
            dx=dx, axial_induction=ai,
            ambient_ti=ti_b[:, None, None], rotor_diameter=ev.D,
        )
        ti_eff = ev.effective_ti(ti_added, dx, dy, downstream, ti_b)
        delta = ev.deflection_model.compute(
            dx=dx, ct=ct, ti_eff=ti_eff, yaw=yaw, u_inf=u_src,
            rotor_diameter=ev.D, x_i=x_zero,
        )
        deficit = ev.velocity_model.compute(
            dx=dx, dy=dy, delta=delta, ct=ct, ti_eff=ti_eff, yaw=yaw,
            u_inf=u_src, rotor_diameter=ev.D, x_i=x_zero,
        )
        deficit = cp.where(downstream, deficit, cp.zeros_like(deficit))
        u_src = ev.update_inflow(ws_b, ev.combination_model.combine(deficit))

    power_kw = ev.power_curve.power_gpu(u_src, yaw)
    return ev.integrate_aep(power_kw, freq, per_turbine=per_turbine)


class TestStagesMatchEvaluate:
    def test_farm_aep_is_identical(self):
        ev, pop = _evaluator(), _pop()
        rose = WindRose.default_12sector()
        assert cp.allclose(_run_stages(ev, pop, rose), ev.evaluate(pop, rose))

    def test_per_turbine_aep_is_identical(self):
        ev, pop = _evaluator(), _pop()
        rose = WindRose.default_12sector()
        staged = _run_stages(ev, pop, rose, per_turbine=True)
        assert staged.shape == (pop.shape[0], pop.shape[1])
        assert cp.allclose(staged, ev.evaluate(pop, rose, per_turbine=True))

    def test_multispeed_rose_is_identical(self):
        """F > n_wd, so the (P, F) -> B fold is doing real work."""
        ev, pop = _evaluator(), _pop()
        rose = WindRose.default_12sector_multispeed()
        assert cp.allclose(_run_stages(ev, pop, rose), ev.evaluate(pop, rose))

    @pytest.mark.parametrize("combination", ["SOSFS", "FLS", "MAX"])
    def test_every_combination_model(self, combination):
        ev, pop = _evaluator(combination), _pop()
        rose = WindRose.default_12sector()
        assert cp.allclose(_run_stages(ev, pop, rose), ev.evaluate(pop, rose))

    def test_stages_stay_float32(self):
        """
        Same trap as test_power_curve.py::test_float32_is_not_promoted: a float64
        scalar anywhere in the chain promotes every (B, T, T) tensor.
        """
        ev, pop = _evaluator(), _pop()
        wd_rad, ws, freq, ti = ev.upload_conditions(WindRose.default_12sector())
        xw, yw, yaw, ws_b, ti_b = ev.to_wind_frame(pop, wd_rad, ws, ti)
        dx, dy, downstream = ev.pairwise_displacement(xw, yw)

        for name, arr in [("wd_rad", wd_rad), ("freq", freq), ("xw", xw),
                          ("yaw", yaw), ("ws_b", ws_b), ("dx", dx), ("dy", dy)]:
            assert arr.dtype == cp.float32, f"{name} is {arr.dtype}"
        assert downstream.dtype == cp.bool_

        ti_added = ev.turbulence_model.compute(
            dx=dx, axial_induction=cp.full(xw.shape, cp.float32(0.3)),
            ambient_ti=ti_b[:, None, None], rotor_diameter=ev.D,
        )
        ti_eff = ev.effective_ti(ti_added, dx, dy, downstream, ti_b)
        u_src = ev.update_inflow(ws_b, cp.full(xw.shape, cp.float32(0.2)))
        aep = ev.integrate_aep(ev.power_curve.power_gpu(u_src, yaw), freq)

        assert ti_eff.dtype == cp.float32
        assert u_src.dtype == cp.float32
        assert aep.dtype == cp.float32

    def test_n_jacobi_is_a_parameter(self):
        """
        evaluate(n_jacobi=1) must equal a one-pass hand-written loop, and differ
        from the converged default on a layout with a wake chain deeper than 1.
        """
        ev = _evaluator()
        rose = WindRose.default_12sector()
        # In-line row of 4: chain depth 3, so 1 pass is measurably wrong.
        pop = cp.zeros((1, 4, 3), dtype=cp.float32)
        pop[0, :, 0] = cp.asarray(np.linspace(200, 1800, 4, dtype=np.float32))
        pop[0, :, 1] = cp.float32(1000.0)

        one_pass = ev.evaluate(pop, rose, n_jacobi=1)
        assert cp.allclose(one_pass, _run_stages(ev, pop, rose, n_jacobi=1))
        assert not cp.allclose(one_pass, ev.evaluate(pop, rose))


class TestIntegrateAepInfersShapes:
    def test_p_is_inferred_from_freq_length(self):
        """No shape arguments to thread: P = B // F."""
        P, F, T = 3, 12, 5
        freq = cp.full(F, cp.float32(1.0 / F))
        power = cp.ones((P * F, T), dtype=cp.float32)
        aep = FarmEvaluator.integrate_aep(power, freq)
        assert aep.shape == (P,)
        # 1 kW per turbine, all year: T * 8760 kWh
        assert cp.allclose(aep, cp.float32(T * 8760.0))
