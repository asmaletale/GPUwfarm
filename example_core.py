"""
The AEP pipeline written out stage by stage — reference for gpuwfarm_core.

`FarmEvaluator.evaluate(pop, wind_rose)` is exactly the sequence below. Running
it yourself instead costs three extra lines and gives you every intermediate
tensor: the wind-frame coordinates, the pairwise geometry, the added turbulence,
the deflection, the per-pair deficit and the effective wind speed at each
turbine, on every Jacobi pass.

The contract: every call returns a real CuPy array. Nothing is lazy, nothing is
a placeholder, there is no graph to compile. Insert a line anywhere, print or
plot any intermediate, or replace a stage with your own function — the next call
only cares about the shapes.

Shape convention (see CLAUDE.md):
    P = population   T = turbines   F = wind conditions   B = P * F

Open in an IDE and step through, or run it directly:
    python example_core.py
"""
import numpy as np
import cupy as cp

from gpuwfarm_core import (
    FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData,
)

N_TURBINES = 3
POP_SIZE   = 4
N_JACOBI   = 3   # must be >= the longest downstream wake chain in the farm

farm_cfg     = FarmConfig(n_turbines=N_TURBINES, area_width=2000, area_height=2000)
wake_cfg     = WakeConfig(combination="SOSFS")
turbine_cfg  = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()

ev = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)

# A simple aligned row, repeated across the population. Yaw stays 0.
xs = np.linspace(200, 1800, N_TURBINES)
ys = np.full(N_TURBINES, 1000.0)
pop = cp.zeros((POP_SIZE, N_TURBINES, 3), dtype=cp.float32)
pop[:, :, 0] = cp.asarray(xs)
pop[:, :, 1] = cp.asarray(ys)

wind_rose = WindRose.default_12sector()

# ══════════════════════════════════════════════════════════════════════
# The pipeline
# ══════════════════════════════════════════════════════════════════════

# 1. Flatten the wind rose to a findex axis and upload it once.
wd_rad, ws, freq, ti = ev.upload_conditions(wind_rose)          # (F,) x4  float32

# 2. Rotate into every wind frame, folding (P, F) into one batch axis B.
#    Downstream is +x in this frame, which is what makes step 3 a simple
#    subtraction regardless of wind direction.
xw, yw, yaw, ws_b, ti_b = ev.to_wind_frame(pop, wd_rad, ws, ti)  # (B,T) x3, (B,) x2

# 3. All-pairs displacement. dx[b, i, j] > 0 means j sits downstream of i.
dx, dy, downstream = ev.pairwise_displacement(xw, yw)            # (B,T,T) float32 x2 + bool

# 4-8. Jacobi fixed-point solve for the waked inflow at each source turbine.
#      Pass 0 starts from freestream; because wake dependencies form a DAG this
#      converges to the sorted-solver answer in `chain_depth` passes.
u_src  = cp.broadcast_to(ws_b[:, None], xw.shape).copy()         # (B,T) float32
x_zero = cp.zeros_like(xw)   # sources sit at relative position 0: dx is relative

for pass_i in range(N_JACOBI):
    ct = ev.power_curve.ct_gpu(u_src)                            # (B,T)   float32
    ai = ev.power_curve.axial_induction_from_ct(ct)              # (B,T)   float32

    ti_added = ev.turbulence_model.compute(                      # (B,T,T) float32
        dx=dx, axial_induction=ai,
        ambient_ti=ti_b[:, None, None], rotor_diameter=ev.D,
    )
    ti_eff = ev.effective_ti(ti_added, dx, dy, downstream, ti_b)  # (B,T,T) float32

    delta = ev.deflection_model.compute(                         # (B,T,T) float32
        dx=dx, ct=ct, ti_eff=ti_eff, yaw=yaw, u_inf=u_src,
        rotor_diameter=ev.D, x_i=x_zero,
    )
    deficit = ev.velocity_model.compute(                         # (B,T,T) float32
        dx=dx, dy=dy, delta=delta, ct=ct, ti_eff=ti_eff, yaw=yaw,
        u_inf=u_src, rotor_diameter=ev.D, x_i=x_zero,
    )
    deficit = cp.where(downstream, deficit, cp.zeros_like(deficit))

    total_deficit = ev.combination_model.combine(deficit)        # (B,T)   float32
    u_src = ev.update_inflow(ws_b, total_deficit)                # (B,T)   float32

    # ─── Add your own step here ──────────────────────────────────────
    # Anything that takes and returns a (B, T) speed field, e.g. a
    # blockage correction:  u_src = my_blockage(u_src, dx, dy)
    # ─────────────────────────────────────────────────────────────────

    print(f"  Jacobi pass {pass_i}: mean U_eff = {float(u_src.mean()):.3f} m/s")

# 9. Power, then integrate over the wind rose.
power_kw = ev.power_curve.power_gpu(u_src, yaw)                  # (B,T)   float32 kW
aep      = ev.integrate_aep(power_kw, freq)                      # (P,)    float32 kWh

# ══════════════════════════════════════════════════════════════════════
# Results, and the proof that this is what evaluate() does
# ══════════════════════════════════════════════════════════════════════

aep_builtin     = ev.evaluate(pop, wind_rose)                    # (P,)
aep_per_turbine = ev.integrate_aep(power_kw, freq, per_turbine=True)  # (P,T)

P, T, _ = pop.shape
F = wd_rad.shape[0]
print()
print(f"{'quantity':<20} {'shape':<16} dtype")
print("-" * 48)
for name, arr in [
    ("pop", pop), ("xw", xw), ("dx", dx), ("ti_eff", ti_eff),
    ("deficit", deficit), ("u_src", u_src), ("power_kw", power_kw), ("aep", aep),
]:
    print(f"{name:<20} {str(arr.shape):<16} {arr.dtype}")
print("-" * 48)
print(f"P={P}  T={T}  F={F}  B=P*F={P * F}")

print()
print(f"Farm AEP per individual (kWh):        {cp.asnumpy(aep)}")
print(f"Same via evaluate() (kWh):            {cp.asnumpy(aep_builtin)}")
print(f"Max relative difference:              {float(cp.max(cp.abs(aep - aep_builtin) / aep_builtin)):.2e}")
print(f"Per-turbine AEP, individual 0 (kWh):  {cp.asnumpy(aep_per_turbine)[0]}")
