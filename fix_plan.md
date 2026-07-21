# Fix Plan — Jacobi Fixed-Point Iteration for Waked-Source Inflow

**Goal:** Eliminate the residual T3 (interior-turbine) power/AEP error against FLORIS
by recomputing each turbine's shed wake from its *local waked inflow speed* instead
of freestream — while keeping the fully-vectorized GPU batch design (no per-individual
sorting).

**Status going in:** The coordinate-frame bug (`x_i` passed as absolute `xw`) was already
fixed in commit `b71a7e2`. That removed the ~40–50% overstatement. What remains is the
*architectural* deviation documented in `CLAUDE.md` #2/#5 and the
`test_floris_comparison.py` docstring: every source turbine's Ct / axial-induction /
`u_inf` is taken from **freestream**, so a turbine that is itself waked (e.g. T2 in a
3-row) sheds a too-strong wake onto the turbine behind it (T3). Expected residual today:
~5% single-condition power, up to ~20% AEP on affected interior turbines.

---

## 1. Why this is safe (read before implementing)

The concern "won't the superposition compound endlessly to deficit saturation?" — it
**cannot**, because wind-farm flow is strictly downstream-only. Sorted by downstream
position, `u_j = F_j(u_1, …, u_{j-1})` depends **only on upstream turbines** → the
dependency graph is a **DAG**, not a cycle. There is no feedback loop to run away.

Consequence — **exact finite termination**, not asymptotic convergence:
- Front-row turbines are exact at iteration 0 (they see freestream, no upstream source).
- By induction, after iteration `k` turbines `1…k` hold their exact fixed-point values
  and never change again. Each pass "unlocks" one more layer of wake depth.
- After `N_ITERS = (longest wake chain depth)` passes, `u_src` stops moving entirely
  (`max|Δu_src| → 0`) and the result is the bit-exact fixed point.

Iteration 0 of the loop reproduces exactly today's freestream result, so this is a strict
refinement with a known baseline.

This is Jacobi (all turbines updated simultaneously from the *previous* iteration's
state) — the parallel-friendly variant, chosen precisely because it needs no sort.

---

## 2. Files to change

| File | Change |
|------|--------|
| `physics/farm_evaluator.py` | Wrap steps 3–8 of `evaluate()` in a fixed-point loop; feed `u_eff` back as the source inflow each pass. |
| `physics/wake_velocity/gauss.py` | Make `u_inf` a per-source tensor `(P, T_src)` instead of a Python `float`. |
| `physics/wake_deflection/gauss.py` | Same `u_inf` → tensor change (`compute` + `near_far_wake_boundary`). |
| `tests/test_floris_comparison.py` | Loosen/tighten tolerances after validation; update the "Known residual deviations" docstring. |
| `CLAUDE.md` | Update deviation #2 to describe the iterative solve. |

---

## 3. Implementation

### 3a. `farm_evaluator.py` — the iteration loop

Current code computes `ct`/`ai` once from `u_fs` (freestream) at lines ~113–115 and runs
steps 3–8 once. Replace the single pass with a loop. Sketch (inside the
`for wd_rad, ws_float, freq, ti_float in wind_rose.conditions():` body, after the
`dx`/`dy`/`downstream_mask`/`dx_safe` setup):

```python
N_ITERS = 3                       # covers wake chains up to depth 3; bump for deeper farms
TOL     = cp.float32(0.01)        # m/s; optional residual-based early stop

# Iteration 0 source inflow = freestream (reproduces today's result exactly)
u_src = cp.full((P, T), ws, dtype=cp.float32)   # (P, T) local inflow at each SOURCE turbine

for _it in range(N_ITERS):
    # 3. Ct / axial induction from LOCAL inflow (was: freestream u_fs)
    ct = self.power_curve.ct_gpu(u_src)                 # (P, T)
    ai = self.power_curve.axial_induction_gpu(u_src)    # (P, T)

    # 4. Crespo-Hernandez added TI (unchanged except ai now local)
    ti_added = self.turbulence_model.compute(
        dx=dx_safe, axial_induction=ai, ambient_ti=ti, rotor_diameter=self.D,
    )
    ti_added = cp.where(downstream_mask, ti_added, cp.zeros_like(ti_added))
    ti_eff_per_dst = cp.sqrt(cp.float32(ti ** 2) + cp.sum(ti_added ** 2, axis=1))
    ti_eff_pairs   = cp.broadcast_to(ti_eff_per_dst[:, :, None], (P, T, T)).copy()

    # 5. Deflection — pass u_src (per-source) instead of scalar ws
    delta = self.deflection_model.compute(
        dx=dx_safe, ct=ct, ti_eff=ti_eff_pairs, yaw=yaw,
        u_inf=u_src,                       # <-- now (P, T) tensor
        rotor_diameter=self.D, x_i=cp.zeros_like(xw),
    )

    # 6. Velocity deficit — pass u_src (per-source) instead of scalar ws
    deficit = self.velocity_model.compute(
        dx=dx_safe, dy=dy_raw, delta=delta, ct=ct, ti_eff=ti_eff_pairs, yaw=yaw,
        u_inf=u_src,                       # <-- now (P, T) tensor
        rotor_diameter=self.D, x_i=cp.zeros_like(xw),
    )
    deficit = cp.where(downstream_mask, deficit, cp.zeros_like(deficit))

    # 7. Combine → total deficit (fraction of FREESTREAM), clip
    total_deficit = self.combination_model.combine(deficit)
    total_deficit = cp.clip(total_deficit, cp.float32(0.0), cp.float32(0.95))

    # 8. New effective speed at every turbine, fed back as next iter's source inflow
    u_new = ws * (cp.float32(1.0) - total_deficit)      # (P, T)

    # optional early stop (safe to keep N_ITERS fixed instead for static kernel pattern)
    if float(cp.max(cp.abs(u_new - u_src))) < float(TOL):
        u_src = u_new
        break
    u_src = u_new

u_eff    = u_src
power_kw = self.power_curve.power_gpu(u_eff, yaw)
# ... then the existing per_turbine / farm_power AEP accumulation unchanged
```

Notes:
- **Keep `total_deficit` as a fraction of freestream** and `u_eff = ws*(1 - total_deficit)`
  — combination is relative to freestream; only the *source-side* quantities (Ct, sigma,
  `u_inf`) become local. This is the minimal, physically-correct change.
- The early-stop `cp.max(...)` forces a device→host sync each iteration. For the GA hot
  loop, prefer a **fixed `N_ITERS`** (no sync, static launch pattern) once you've
  confirmed the chain depth of your layouts. Use the early-stop only while validating.
- `u_fs` / the old one-shot `ct`,`ai` computation is removed.

### 3b. `wake_velocity/gauss.py` — `u_inf` float → tensor

`compute(..., u_inf, ...)` currently takes a `float` and passes it to `_sigma_initial`
and `_x0`. Change the signature to accept `u_inf: cp.ndarray  # (P, T_src)` and broadcast:

```python
def compute(self, dx, dy, delta, ct, ti_eff, yaw, u_inf, rotor_diameter, x_i):
    ...
    u_b = u_inf[:, :, None]      # (P, T_src, 1)  — was a scalar float
    sigma_y0, sigma_z0, _, _ = self._sigma_initial(ct_b, yaw_b, u_b, D)
    x0 = self._x0(ct_b, yaw_b, ti_src, xi_b, self.alpha, self.beta, D)   # x0 has no u_inf
```

In `_sigma_initial`, change the `u_inf: float` parameter to accept the `(P, T_src, 1)`
tensor — the arithmetic (`uR = u_inf*ct/...`, `u0 = u_inf*sqrt_1_ct`,
`sigma_z0 = D*0.5*sqrt(uR/(u_inf+u0))`) already broadcasts correctly with no code change
beyond the type. `_x0` does **not** use `u_inf`, leave it.

### 3c. `wake_deflection/gauss.py` — `u_inf` float → tensor

Both `compute` and `near_far_wake_boundary` take `u_inf: float`. Change to
`u_inf: cp.ndarray  # (P, T_src)` and broadcast inside `near_far_wake_boundary` as
`u_b = u_inf[:, :, None]`:

```python
uR = u_b * ct_ / (2.0*(1.0 - sqrt_1_ct + 1e-8))
u0 = u_b * sqrt_1_ct
sigma_z0 = D * 0.5 * cp.sqrt(uR / (u_b + u0 + 1e-8))
```

And in `compute`, the far-wake log formula uses `C0 = 1.0 - u0 / u_inf` (line ~140) —
becomes `C0 = 1.0 - u0 / u_b` (both `(P, T, 1)`, broadcasts fine).

> All these `u_inf` uses were already scalars broadcasting against `(P, T, 1)` arrays, so
> swapping in a `(P, T, 1)` tensor is arithmetically transparent — just fix the type
> annotations and the one `u_inf[:, :, None]` reshape at each call site.

---

## 4. Optional second-order correction (do AFTER 3 is validated)

To match FLORIS exactly, each emitter's deficit should be normalized by *its own* local
velocity before combining relative to freestream. Concretely, scale each source's
contribution by `u_src[:, :, None] / ws` inside the combination step. This is a smaller
effect than the Ct/sigma change in §3 — implement and measure only if a residual remains
after §3. Do not do it in the first pass; it complicates the SOSFS/FLS/MAX combine and
can mask whether §3 is correct.

---

## 5. Validation

Run against FLORIS (needs `floris` + `cupy`):

```bash
pytest tests/test_floris_comparison.py -v -s
```

Expected outcomes:
- `TestFreestreamPowerAgreement` / single-turbine AEP: **unchanged** (no wake sources).
- `TestWakeEfficiency` 2-turbine (5D/7D/10D): **unchanged** — T2's only source (T1) is
  unwaked, so iteration 0 == converged.
- `TestMultiTurbineRow` 3-turbine row power & efficiency: residual should drop from
  ~15–20% toward **low single digits**. This is the key signal.
- Print table (`TestPrintSummary`, run with `-s`) for a side-by-side sanity check.

Additional manual check — confirm termination:
- Temporarily `log`/print `float(cp.max(cp.abs(u_new - u_src)))` each iteration for a
  3-row and a 5-row layout. It must **strictly decrease and hit ~0** by iteration
  == chain depth. If it doesn't, the DAG assumption is being violated somewhere (likely a
  sign or masking bug), not a physics problem.
- Also run `plot_floris_comparison.py` for the per-turbine T1/T2/T3 breakdown.

Tighten `TestMultiTurbineRow` tolerance (currently 0.15) once the new residual is known,
and update the docstring "Known residual deviations" + `CLAUDE.md` deviation #2/#5 to say
the solver now does an N-pass Jacobi fixed-point on waked-source inflow.

---

## 6. Tuning & guardrails

- **`N_ITERS`:** 3 is safe for rows up to depth 3; use 4–5 for deep (10+ turbine) rows.
  It's the *longest wake chain*, not the turbine count — a wide sparse farm may still be
  depth 2–3. When unsure, measure with the residual print above.
- **No divergence risk:** feed-forward fixed points are stable; the existing
  `clip(total_deficit, 0, 0.95)` is a per-geometry safety ceiling, never reached through
  iteration.
- **If you ever see oscillation** (you shouldn't, on a DAG): under-relax with
  `u_src = 0.5*u_src_old + 0.5*u_new`. Not expected to be necessary.
- **Cost:** deterministic `N_ITERS×` the current per-condition work (~2–3×). No
  data-dependent branching, no sort, still one broadcast kernel per pass — GPU-friendly.

---

## 7. Commit / branch

- Develop on branch `claude/t3-floris-power-aep-bug-98x37a` (create from latest default if
  needed).
- Suggested commit message subject:
  `Add Jacobi fixed-point solve for waked-source inflow (fixes T3 residual)`
- Keep §3 and §4 as **separate commits** so the second-order correction can be measured
  and reverted independently.
- Do **not** open a PR unless explicitly asked.

---

## 8. Rollback

`N_ITERS = 1` reproduces the exact pre-change behavior (freestream sources). If the
comparison regresses unexpectedly, set `N_ITERS = 1` to confirm the loop scaffolding is
neutral before debugging the physics.
