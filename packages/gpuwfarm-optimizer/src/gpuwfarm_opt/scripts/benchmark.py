"""
GPU vs CPU speedup benchmark for the wind farm physics pipeline.

Runs the *actual* gpuwfarm_core.physics.farm_evaluator.FarmEvaluator source
twice per configuration: once bound to real CuPy (GPU) and once with `cupy`
swapped in sys.modules for a NumPy-backed facade (CPU). Both runs execute the
exact same .py files -- there is no hand-ported CPU copy of the physics to
drift out of sync with the real evaluator.

(An earlier version of this script hand-duplicated the pipeline in NumPy and
silently fell behind two physics fixes -- the Jacobi waked-source-inflow solve
and the max()-across-sources wake-added-TI rule -- because nothing forced the
copy to track the original. The facade-swap approach makes that class of bug
structurally impossible: CPU and GPU always run identical source.)

This also doubles as a portability/vectorization check: every module reached
via _CUPY_BOUND_MODULES must import cleanly and run correctly against plain
NumPy, or the swap raises/produces mismatched numbers immediately -- see
run_correctness_check().

Usage:
    python benchmark.py
"""
from __future__ import annotations
import os
import sys
import time
import types
import importlib

if sys.platform == "win32":
    _torch_lib = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib"
    )
    _torch_lib = os.path.normpath(_torch_lib)
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

import numpy as np
import cupy as cp

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose


# ─────────────────────────────────────────────────────────────────────────────
# CPU variant: re-import the real physics modules with `cupy` swapped for a
# NumPy-backed facade, instead of hand-porting cp -> np by hand.
# ─────────────────────────────────────────────────────────────────────────────

# Every module in the wake-physics call graph that does `import cupy as cp`.
# gpuwfarm_core.physics.base is intentionally excluded: it only uses `cp` in
# (lazily-evaluated, `from __future__ import annotations`) type hints, never
# at runtime, so it does not need to be re-imported against the facade.
_CUPY_BOUND_MODULES = [
    "gpuwfarm_core.physics.wake_turbulence.crespo_hernandez",
    "gpuwfarm_core.physics.wake_deflection.gauss",
    "gpuwfarm_core.physics.wake_velocity.gauss",
    "gpuwfarm_core.physics.wake_combination.sosfs",
    "gpuwfarm_core.physics.wake_combination.fls",
    "gpuwfarm_core.physics.wake_combination.max",
    "gpuwfarm_core.physics.turbine.power_curve",
    "gpuwfarm_core.physics.farm_evaluator",
]


def _numpy_cupy_facade() -> types.ModuleType:
    """A `cupy`-shaped module backed entirely by NumPy."""
    facade = types.ModuleType("cupy")
    facade.__dict__.update({k: v for k, v in vars(np).items() if not k.startswith("_")})
    return facade


def get_cpu_farm_evaluator_class():
    """
    Re-import FarmEvaluator (and its whole dependency chain) with `cupy`
    pointed at a NumPy facade, then restore the real CuPy bindings.

    Returns the CPU-backed FarmEvaluator class -- same source, same behaviour,
    running entirely on NumPy arrays. If any physics module used a CuPy-only
    API this import would fail (or the correctness check below would diverge);
    that failure is itself the vectorization/portability finding.
    """
    real_cupy = sys.modules["cupy"]
    saved = {name: sys.modules.get(name) for name in _CUPY_BOUND_MODULES}
    for name in _CUPY_BOUND_MODULES:
        sys.modules.pop(name, None)
    sys.modules["cupy"] = _numpy_cupy_facade()
    try:
        cpu_fe_mod = importlib.import_module("gpuwfarm_core.physics.farm_evaluator")
        return cpu_fe_mod.FarmEvaluator
    finally:
        sys.modules["cupy"] = real_cupy
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check: GPU and CPU must agree, since they run identical source.
# ─────────────────────────────────────────────────────────────────────────────

def run_correctness_check(cpu_fe_class) -> None:
    wake_cfg    = WakeConfig()
    farm_cfg    = FarmConfig(n_turbines=5)
    turbine_cfg = TurbineConfig()
    td          = TurbineData.nrel_5mw()
    wind_rose   = WindRose.default_12sector_multispeed()

    gpu_eval = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, td)
    cpu_eval = cpu_fe_class(farm_cfg, turbine_cfg, wake_cfg, td)

    rng = np.random.default_rng(0)
    pop_np = rng.random((16, farm_cfg.n_turbines, 3), dtype=np.float64).astype(np.float32)
    pop_np[:, :, 0] *= farm_cfg.area_width
    pop_np[:, :, 1] *= farm_cfg.area_height
    pop_np[:, :, 2]  = (pop_np[:, :, 2] - 0.5) * np.deg2rad(30)
    pop_gpu = cp.asarray(pop_np)

    aep_gpu = cp.asnumpy(gpu_eval.evaluate(pop_gpu, wind_rose))
    aep_cpu = cpu_eval.evaluate(pop_np, wind_rose)

    max_rel_err = np.max(np.abs(aep_gpu - aep_cpu) / np.maximum(np.abs(aep_gpu), 1.0))
    status = "OK" if max_rel_err < 1e-3 else "MISMATCH"
    print(f"Correctness check (GPU vs CPU, identical source): {status} "
          f"(max relative error {max_rel_err:.2e})\n")
    assert max_rel_err < 1e-3, "GPU and CPU evaluators diverged -- see max_rel_err above"


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_gpu(evaluator: FarmEvaluator, pop_gpu: cp.ndarray,
              wind_rose: WindRose, n_reps: int = 3) -> float:
    evaluator.evaluate(pop_gpu, wind_rose)   # warmup (JIT/allocator/kernel cache)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        evaluator.evaluate(pop_gpu, wind_rose)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def _time_cpu(evaluator, pop_np: np.ndarray,
              wind_rose: WindRose, n_reps: int = 2) -> float:
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        evaluator.evaluate(pop_np, wind_rose)
        times.append(time.perf_counter() - t0)
    return min(times)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark matrix
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    (32,  5),
    (64,  10),
    (128, 10),
    (256, 20),
]


def main():
    wake_cfg    = WakeConfig()
    turbine_cfg = TurbineConfig()
    td          = TurbineData.nrel_5mw()
    wind_rose   = WindRose.default_12sector()

    cpu_fe_class = get_cpu_farm_evaluator_class()
    run_correctness_check(cpu_fe_class)

    print(f"Device: {cp.cuda.Device().id}  ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
    print(f"Wind conditions per evaluate() call: {len(list(wind_rose.conditions()))} (wd x ws bins)\n")

    header = f"{'Pop':>6}  {'Turbines':>8}  {'GPU (s)':>9}  {'CPU (s)':>9}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for pop_size, n_turb in CONFIGS:
        farm_cfg_local = FarmConfig(n_turbines=n_turb)

        gpu_eval = FarmEvaluator(farm_cfg_local, turbine_cfg, wake_cfg, td)
        cpu_eval = cpu_fe_class(farm_cfg_local, turbine_cfg, wake_cfg, td)

        pop_np = np.random.rand(pop_size, n_turb, 3).astype(np.float32)
        pop_np[:, :, 0] *= farm_cfg_local.area_width
        pop_np[:, :, 1] *= farm_cfg_local.area_height
        pop_np[:, :, 2]  = (pop_np[:, :, 2] - 0.5) * np.deg2rad(30)

        pop_gpu = cp.asarray(pop_np)

        t_gpu = _time_gpu(gpu_eval, pop_gpu, wind_rose)
        t_cpu = _time_cpu(cpu_eval, pop_np, wind_rose)
        speedup = t_cpu / t_gpu

        print(f"{pop_size:>6}  {n_turb:>8}  {t_gpu:>9.4f}  {t_cpu:>9.4f}  {speedup:>7.1f}x")

    print()


if __name__ == "__main__":
    main()
