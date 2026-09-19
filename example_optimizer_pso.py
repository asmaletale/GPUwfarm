"""
The MOPSO iteration loop written out — reference for gpuwfarm_opt, multi-objective.

`ParticleSwarm.run()` is exactly the loop below. Writing it out yourself gives
you the swarm, its velocity, the personal bests and the Pareto archive as named
CuPy arrays at every step, so you can inspect them, log them, or drop your own
operator in between two existing ones.

The contract is the GA's: every call takes a (P, T, 3) population and returns
one. Nothing is lazy. PSO additionally carries state the GA does not —
velocity, personal bests, and the external archive — and *none of it lives on
the object*. Each stage takes the state it reads as an argument and hands the
new value back, exactly as survive_pareto already does for the GA. So the state
is yours, in the loop, under names you chose:

    vel = pso.velocity(pop, vel, pbest, leaders)
    vel = my_velocity_damping(vel)            # drop your own step in

To replace a stage, just call your own function instead of pso.<stage> — no
subclassing, no registration, no config flag.

Compare with example_optimizer_mo.py: the NSGA-II loop solves the same problem
with the same objectives and writes the same HDF5 history.
"""
import numpy as np
import cupy as cp

from gpuwfarm_core import (
    FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData,
)
from gpuwfarm_core.config import CostConfig, VisualImpactConfig
from gpuwfarm_opt.config import PSOConfig
from gpuwfarm_opt.swarm import ParticleSwarm
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
pso_cfg      = PSOConfig(pop_size=32, n_generations=30, optimize=OPTIMIZE)
wind_rose    = WindRose.default_12sector()

evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
projection = CompositeProjection([
    PairwiseSpacingProjection(farm_cfg, n_passes=10),
    BoundaryProjection(farm_cfg),
])

# Without vi_cfg the visual-impact column is all zeros, every particle lands on
# the same front, and the archive degenerates to a single point.
pso = ParticleSwarm(
    farm_cfg, pso_cfg, evaluator, projection, wind_rose,
    cost_cfg=CostConfig(),
    vi_cfg=VisualImpactConfig(),
    objectives_mode="lcoe_vi",   # column 0 LCOE EUR/MWh, column 1 visual impact
)

with pso:

    # ══════════════════════════════════════════════════════════════════
    # Initial swarm
    # ══════════════════════════════════════════════════════════════════
    pop = pso.init_population()                     # (P, T, 3) float32
    pop = pso.project(pop)                          # (P, T, 3) feasibility repair
    vel = pso.init_swarm(pop)                       # (P, T, 3) float32
    aep = pso.evaluate(pop)                         # (P,)      float32 kWh
    obj = pso.compute_objectives(pop, aep)          # (P, 2)    float32 numpy

    # Personal bests start as the swarm itself.
    pbest, pbest_obj = pop, obj

    # The external archive: the Pareto front found so far. Variable length,
    # capped at pso.archive_size, pruned by crowding distance.
    archive, archive_obj = pso.archive_update(None, None, pop, obj)

    history, front_sizes = [], []

    # ══════════════════════════════════════════════════════════════════
    # Iterations
    # ══════════════════════════════════════════════════════════════════
    for i in range(pso_cfg.n_generations):

        # One leader per particle, drawn from the archive with probability
        # proportional to crowding distance — sparse parts of the front pull
        # hardest, which is what spreads the swarm instead of collapsing it.
        leaders = pso.select_leaders(archive, archive_obj, pop.shape[0])   # (P, T, 3)

        vel = pso.velocity(pop, vel, pbest, leaders)   # (P, T, 3)
        pop = pso.advance(pop, vel)                    # (P, T, 3) + box clip
        pop = pso.project(pop)                         # (P, T, 3) spacing repair

        # ─── Add your own step here ──────────────────────────────────
        # Takes the previous return, gives back a (P, T, 3) array:
        #   pop = my_local_search(pop, evaluator, wind_rose)
        # ─────────────────────────────────────────────────────────────

        aep = pso.evaluate(pop)                        # (P,)   float32 kWh
        obj = pso.compute_objectives(pop, aep)         # (P, 2) float32 numpy

        # Personal best moves only on Pareto dominance, not a scalar score.
        pbest, pbest_obj = pso.update_pbest(pop, obj, pbest, pbest_obj)

        # The archive is the elitism: it can only ever improve.
        archive, archive_obj = pso.archive_update(archive, archive_obj, pop, obj)

        pso.log(i, pop, aep, objectives=obj)           # no-op without history_file
        history.append(float(cp.max(aep).item()))
        front_sizes.append(len(archive_obj))

        if i % 5 == 0:
            print(f"Iter {i:4d}  Best AEP: {history[-1]:.4e} kWh  "
                  f"Archive: {front_sizes[-1]:3d}")

    # The archive is the result, not the final swarm — particles keep moving
    # after passing through good positions.
    best_idx = int(cp.argmax(pso.evaluate(archive)).item())
    best = archive[best_idx]                           # (T, 3)

print()
print(f"Optimised: {OPTIMIZE}")
print(f"Best AEP in archive: {float(pso.evaluate(archive)[best_idx]):.4e} kWh")
print(f"Pareto front size:   {len(archive_obj)}")
print(f"  LCOE range: {archive_obj[:, 0].min():.2f} - {archive_obj[:, 0].max():.2f} EUR/MWh")
print(f"  VI range:   {archive_obj[:, 1].min():.4f} - {archive_obj[:, 1].max():.4f}")
print(f"Best layout (x, y, yaw):\n{cp.asnumpy(best)}")

# The archive keeps only non-dominated members, so this must hold. If it ever
# fails, archive_update is admitting a dominated solution.
ranks, _ = pso.fast_nondominated_sort(archive_obj)
assert (ranks == 0).all(), "archive contains a dominated member"
assert len(archive_obj) <= pso.archive_size, "archive exceeded its cap"
print("\nArchive is non-dominated and within its cap (elitist archiving works).")
