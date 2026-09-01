"""
FarmEvaluator.evaluate(zero_yaw=True) — the deflection-skip fast path.

With yaw ≡ 0 and ad = bd = 0 the Gauss deflection model returns an all-zero
tensor (theta_c0 ∝ yaw), so skipping it must be bit-for-bit equivalent.
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
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.physics.wake_deflection.gauss import GaussVelocityDeflection
from gpuwfarm_core.wind.wind_rose import WindRose


def _pop(yaw_deg: float = 0.0, P: int = 4, T: int = 6) -> "cp.ndarray":
    """Staggered grid so turbines genuinely wake each other."""
    rng = np.random.default_rng(0)
    pop = np.zeros((P, T, 3), dtype=np.float32)
    pop[:, :, 0] = rng.uniform(0, 2000, (P, T))
    pop[:, :, 1] = rng.uniform(0, 2000, (P, T))
    pop[:, :, 2] = np.deg2rad(yaw_deg)
    return cp.asarray(pop)


def _evaluator(**wake_kw) -> FarmEvaluator:
    return FarmEvaluator(
        FarmConfig(n_turbines=6), TurbineConfig(),
        WakeConfig(**wake_kw), TurbineData.nrel_5mw(),
    )


def test_deflection_is_zero_at_zero_yaw():
    """The premise: the model itself returns exactly zero when yaw = 0."""
    B, T = 8, 4
    m = GaussVelocityDeflection(WakeConfig())
    assert m.ad == 0.0 and m.bd == 0.0, "FLORIS default ad/bd must be 0"
    delta = m.compute(
        dx=cp.full((B, T, T), 600.0, cp.float32),
        ct=cp.full((B, T), 0.7, cp.float32),
        ti_eff=cp.full((B, T, T), 0.06, cp.float32),
        yaw=cp.zeros((B, T), cp.float32),
        u_inf=cp.full((B, T), 8.0, cp.float32),
        rotor_diameter=TurbineConfig().rotor_diameter,
        x_i=cp.zeros((B, T), cp.float32),
    )
    assert cp.all(delta == 0), f"deflection non-zero at yaw=0: max {cp.abs(delta).max()}"


def test_skip_matches_full_path():
    ev, pop, wr = _evaluator(), _pop(0.0), WindRose.default_12sector_multispeed()
    assert ev.evaluate(pop, wr).dtype == cp.float32, "pipeline promoted out of float32"
    full = cp.asnumpy(ev.evaluate(pop, wr))
    fast = cp.asnumpy(ev.evaluate(pop, wr, zero_yaw=True))
    # Bit-exact: delta is identically zero, so (dy - delta) == dy in float32.
    assert np.array_equal(full, fast), f"AEP changed: {full} vs {fast}"

    full_t = cp.asnumpy(ev.evaluate(pop, wr, per_turbine=True))
    fast_t = cp.asnumpy(ev.evaluate(pop, wr, per_turbine=True, zero_yaw=True))
    assert np.array_equal(full_t, fast_t), "per-turbine AEP changed"


def test_nonzero_ad_bd_disables_the_skip():
    """ad/bd add a yaw-independent offset, so the shortcut must not fire."""
    ev  = _evaluator(ad=5.0, bd=0.01)
    pop, wr = _pop(0.0), WindRose.default_12sector()
    full = cp.asnumpy(ev.evaluate(pop, wr))
    fast = cp.asnumpy(ev.evaluate(pop, wr, zero_yaw=True))
    assert np.array_equal(full, fast), "skip fired despite ad/bd != 0"


def test_default_is_the_full_path():
    """zero_yaw must never be inferred: a yawed pop is unaffected by the default."""
    ev, wr = _evaluator(), WindRose.default_12sector()
    yawed, straight = _pop(25.0), _pop(0.0)
    assert not np.allclose(
        cp.asnumpy(ev.evaluate(yawed, wr)), cp.asnumpy(ev.evaluate(straight, wr))
    ), "yaw had no effect on AEP — test layout has no wake interaction"
