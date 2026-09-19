"""
The SPSA gradient-descent loop written out — reference for gpuwfarm_opt.

`GradientDescent.run()` is exactly the loop below. Writing it out yourself gives
you the batch, the weight vectors, the running ideal point and the gradient as
named arrays at every step.

Two things make this different from the GA and the swarm.

**The gradient comes from evaluations, not autodiff.** The physics core is
CuPy and has no derivatives, and two of the things being optimised have no
useful ones anyway: visual impact is a sorted sweep-line union area, and the
power curve is a lookup table that is flat above rated. SPSA sidesteps all of
it — it perturbs the whole batch with a random +/-1 vector and takes a central
difference, so it needs 2 evaluations per iteration no matter how many turbines
there are, and it never differentiates anything. Both halves of the pair go to
the evaluator in a single stacked (2P, T, 3) call.

**Rows do not compete.** Each row descends its *own* scalarised objective,
built from its own weight vector on the simplex. Row 0 cares only about visual
impact, the last row only about LCOE, and the rows in between trade off. So the
batch is not a population converging to one answer — it *is* the Pareto front.

Adam's momentum is the one piece of genuinely order-dependent state, and it
lives in a caller-owned AdamState threaded through the loop, exactly the way
the swarm threads its velocity. That is not ceremony: torch keys its moment
buffers to tensor identity, so leaves reallocated inside step() would silently
reset Adam to plain SGD without raising anything.
"""
import numpy as np
import cupy as cp

from gpuwfarm_core import (
    FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData,
)
from gpuwfarm_core.config import CostConfig, VisualImpactConfig
from gpuwfarm_opt.config import GDConfig
from gpuwfarm_opt.gradient import GradientDescent
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection

N_TURBINES = 6

# Decision variables: "both", "layout" (yaw pinned to 0), or "yaw" (layout frozen).
OPTIMIZE = "both"

farm_cfg     = FarmConfig(n_turbines=N_TURBINES)
wake_cfg     = WakeConfig(combination="SOSFS")
turbine_cfg  = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()
wind_rose    = WindRose.default_12sector()

# lr and spsa_c are split into metres and degrees on purpose. x/y live on a
# 2 km farm and yaw lives in +/-30 deg, so a single step size or perturbation
# radius cannot serve both columns — this is the usual way to get SPSA wrong.
gd_cfg = GDConfig(
    pop_size=32, n_generations=40, optimize=OPTIMIZE,
    lr_xy=25.0, lr_yaw_deg=1.0,
    spsa_c_xy=8.0, spsa_c_yaw_deg=0.5,
    spsa_pairs=2,             # 2 pairs = 4 evaluations/iter, lower gradient noise
    torch_optimizer="adam",   # any torch.optim class: sgd, rmsprop, adamw, ...
)

evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
projection = CompositeProjection([
    PairwiseSpacingProjection(farm_cfg, n_passes=10),
    BoundaryProjection(farm_cfg),
])

gd = GradientDescent(
    farm_cfg, gd_cfg, evaluator, projection, wind_rose,
    cost_cfg=CostConfig(),
    vi_cfg=VisualImpactConfig(),
    objectives_mode="lcoe_vi",   # column 0 LCOE EUR/MWh, column 1 visual impact
)

with gd:

    # ══════════════════════════════════════════════════════════════════
    # Initial batch
    # ══════════════════════════════════════════════════════════════════
    pop = gd.init_population()                  # (P, T, 3) float32
    pop = gd.project(pop)                       # (P, T, 3) feasibility repair

    weights = gd.weight_vectors()               # (P, 2) rows sum to 1
    z_min, z_max = gd.init_ideal()              # (2,), (2,) running ideal/nadir
    adam = gd.init_adam(pop)                    # momentum lives here, not on gd

    history, scalar_history = [], []

    # ══════════════════════════════════════════════════════════════════
    # Iterations
    # ══════════════════════════════════════════════════════════════════
    for i in range(gd_cfg.n_generations):

        aep = gd.evaluate(pop)                       # (P,)   float32 kWh
        obj = gd.compute_objectives(pop, aep)        # (P, 2) float32 numpy

        # LCOE is ~50 EUR/MWh and VI is ~0.01, so the Chebyshev max is
        # meaningless until both are scaled to their observed ranges.
        z_min, z_max = gd.update_ideal(obj, z_min, z_max)

        gd.log(i, pop, aep, objectives=obj)          # no-op without history_file
        history.append(float(cp.max(aep).item()))
        scalar_history.append(float(gd.scalarize(obj, weights, z_min, z_max).mean()))

        # One stacked (2P, T, 3) evaluator call per pair. The perturbed points
        # are deliberately NOT projected: spacing repair could cancel the
        # perturbation outright and flatten the difference to zero.
        grad = gd.spsa_gradient(pop, weights, z_min, z_max)   # (P, T, 3)

        # ─── Add your own step here ──────────────────────────────────
        # e.g. clip or precondition the gradient before stepping:
        #   grad = my_gradient_filter(grad)
        # ─────────────────────────────────────────────────────────────

        pop = gd.step(pop, grad, adam)               # (P, T, 3) Adam step
        pop = gd.project(pop)                        # (P, T, 3) repair the step

        if i % 5 == 0:
            print(f"Iter {i:4d}  Best AEP: {history[-1]:.4e} kWh  "
                  f"Mean Chebyshev: {scalar_history[-1]:.4f}")

    aep = gd.evaluate(pop)
    obj = gd.compute_objectives(pop, aep)
    best, pareto_obj, best_vi = gd.best(pop, aep, obj)     # (T, 3), (n, 2), (T, 3)

print()
print(f"Optimised: {OPTIMIZE}")
print(f"Best AEP:  {float(cp.max(aep).item()):.4e} kWh")
print(f"Mean Chebyshev: {scalar_history[0]:.4f} -> {scalar_history[-1]:.4f}")
print(f"Pareto front size: {len(pareto_obj)}")
print(f"  LCOE range: {pareto_obj[:, 0].min():.2f} - {pareto_obj[:, 0].max():.2f} EUR/MWh")
print(f"  VI range:   {pareto_obj[:, 1].min():.4f} - {pareto_obj[:, 1].max():.4f}")
print(f"Best layout (x, y, yaw):\n{cp.asnumpy(best)}")

# Each row descends its own scalarisation, so the batch mean must come down.
# SPSA is stochastic, so compare the ends rather than demanding monotonicity.
assert scalar_history[-1] < scalar_history[0], \
    "mean Chebyshev scalarisation did not decrease: check lr / spsa_c units"
print("\nMean scalarised objective decreased (descent works).")
