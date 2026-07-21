# GPUwfarm — CuPy Wind Farm Optimizer

## Python Interpreter

Default interpreter for this project (CUDA 12.6 laptop), located under the
current user's profile directory as `venv311pytorchcuda`:

```
%USERPROFILE%\PycharmProjects\venv311pytorchcuda\Scripts\python.exe
```

This path is user-specific — resolve `%USERPROFILE%` (PowerShell: `$env:USERPROFILE`)
for the current machine rather than hardcoding a username, since this repo is used
across multiple machines/user accounts.

Run tests and scripts with this interpreter:

```powershell
& "$env:USERPROFILE\PycharmProjects\venv311pytorchcuda\Scripts\python.exe" -m pytest packages/gpuwfarm-core -v
& "$env:USERPROFILE\PycharmProjects\venv311pytorchcuda\Scripts\python.exe" -m gpuwfarm_opt.main
```

## Project Purpose

A GPU-accelerated genetic algorithm for wind farm layout and yaw optimization.
The physics layer is a faithful port of [FLORIS](https://github.com/NREL/floris),
making every equation traceable to NREL source code.

## Architecture: two packages

The repo is a uv workspace split into two installable packages so the fast
GPU evaluator can be used **without** the optimizer (e.g. inside a
reinforcement-learning loop):

- **`gpuwfarm_core`** (`packages/gpuwfarm-core/`) — the evaluation/simulation
  layer: wake physics, batched `FarmEvaluator`, power curve, wind rose,
  LCOE / visual-impact objectives, FLORIS-YAML loader, and the physics config
  dataclasses. Depends only on `numpy` + `cupy`. This layer imports nothing
  from the optimizer.
- **`gpuwfarm_opt`** (`packages/gpuwfarm-optimizer/`) — the optimization layer:
  genetic algorithm, feasibility-repair projection chain, CLI, and analysis
  scripts. Depends on `gpuwfarm_core` (injects a `FarmEvaluator` into the GA).

Standalone evaluation (no optimizer imported):

```python
from gpuwfarm_core import FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData
evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
aep = evaluator.evaluate(pop, wind_rose)   # (P,) cupy array
```

Editable install of both packages (uv workspace):

```bash
uv sync
# or with pip:
pip install -e packages/gpuwfarm-core -e packages/gpuwfarm-optimizer
```

## Repository Layout

```
GPUwfarm/
├── pyproject.toml                       # uv workspace root
├── examples/gch.yaml                    # FLORIS-YAML loader example
├── CLAUDE.md                            # This file
├── packages/gpuwfarm-core/              # EVALUATION CORE (numpy + cupy only)
│   ├── pyproject.toml
│   ├── src/gpuwfarm_core/
│   │   ├── __init__.py                  # public API
│   │   ├── config.py                    # WakeConfig, FarmConfig, TurbineConfig,
│   │   │                                #   CostConfig, VisualImpactConfig
│   │   ├── objectives.py                # ObjectiveEvaluator (LCOE + visual impact)
│   │   ├── physics/
│   │   │   ├── base.py                  # wake-model ABCs
│   │   │   ├── farm_evaluator.py        # batched pipeline orchestrator
│   │   │   ├── wake_velocity/gauss.py   # FLORIS GaussVelocityDeficit port
│   │   │   ├── wake_turbulence/crespo_hernandez.py  # FLORIS CrespoHernandez port
│   │   │   ├── wake_deflection/gauss.py # FLORIS GaussVelocityDeflection port
│   │   │   ├── wake_combination/        # SOSFS / FLS / MAX (FLORIS ports)
│   │   │   └── turbine/power_curve.py   # Tabulated power curve + cosine yaw loss
│   │   ├── wind/wind_rose.py            # WindRose (dir×speed bins + Weibull)
│   │   └── loaders/floris_yaml.py       # FLORIS v4 YAML → config objects
│   └── tests/                           # Unit tests (physics/AEP; see Validation)
└── packages/gpuwfarm-optimizer/         # OPTIMIZER (depends on gpuwfarm-core)
    ├── pyproject.toml
    ├── src/gpuwfarm_opt/
    │   ├── config.py                    # GAConfig
    │   ├── genetic.py                   # GeneticAlgorithm
    │   ├── population_logger.py         # async HDF5 history logger
    │   ├── projection/                  # Feasibility repair operators
    │   ├── main.py                      # CLI entry point (gpuwfarm-optimize)
    │   └── scripts/                     # benchmark, validate_aep, analyze_history, extract_pareto
    ├── smoke/                           # manual run-at-import smoke scripts
    └── tests/                           # optimizer unit tests
```

## FLORIS Source References

All physics are ported from FLORIS `main` branch:

| Component | FLORIS file | Class |
|-----------|------------|-------|
| Gauss wake deficit | `floris/core/wake_velocity/gauss.py` | `GaussVelocityDeficit` |
| Turbulence model | `floris/core/wake_turbulence/crespo_hernandez.py` | `CrespoHernandez` |
| Wake deflection | `floris/core/wake_deflection/gauss.py` | `GaussVelocityDeflection` |
| Combination | `floris/core/wake_combination/{sosfs,fls,max}.py` | `SOSFS/FLS/MAX` |
| Power curve | `floris/core/turbine/operation_models.py` | `SimpleTurbine` |
| Wind rose / AEP | `floris/wind_data.py` + `floris/floris_model.py` | `WindRose` |

When modifying physics code, always cross-reference the FLORIS source file listed above.

## Tensor Shape Convention

```
P = POP_SIZE (batch)     T = N_TURBINES
pop          (P, T, 3)   — x, y, yaw
dx, dy       (P, T, T)   — pairwise displacement (src i → dst j)
deficit      (P, T, T)   — velocity deficit at j from i
total_deficit (P, T)     — combined deficit at each turbine
U_eff        (P, T)      — effective wind speed
power        (P, T)      — turbine power
AEP          (P,)        — fitness value
```

## Physics Parameters (FLORIS defaults)

`WakeConfig`, `TurbineConfig`, and `FarmConfig.air_density`/`ti_ambient` in
`gpuwfarm_core/config.py` do **not** hardcode these numbers — they are
`field(default_factory=...)` values parsed at import time from
`examples/gch.yaml` (farm/wake) and `examples/nrel_5MW.yaml` (turbine,
FLORIS's own file). `gpuwfarm_core/loaders/floris_yaml.py` parses a
user-supplied YAML through the exact same `WakeConfig.from_wake_dict` /
`TurbineConfig.from_turbine_dict` classmethods, so `WakeConfig()` and loading
`examples/gch.yaml` always agree. To change a default, edit the YAML, not
config.py. The same pattern applies to `CostConfig` (from `examples/costs.yaml`)
and `VisualImpactConfig` (from `examples/visual_impact.yaml`) via
`from_costs_dict`/`from_vi_dict`. Each YAML is kept separate on purpose — one
concern per file (mirrors FLORIS's own farm-input vs. turbine-library split) —
do not merge them into a single config file.

```python
# Gauss wake / deflection
alpha = 0.58, beta = 0.077, ka = 0.38, kb = 0.004
ad = 0.0, bd = 0.0, dm = 1.0

# Crespo-Hernandez
initial = 0.1, constant = 0.9, ai = 0.8, downstream = -0.32
# Note: original paper uses 0.0325, 0.73, 0.8325; FLORIS uses 0.1, 0.9, 0.8
# See https://github.com/NREL/floris/issues/773
```

## Known Deviations from FLORIS

1. **Hub-height point evaluation only** — FLORIS evaluates on a 3D mesh. We evaluate only at hub height. Standard for layout optimization.
2. **Simultaneous all-pairs, Jacobi fixed-point solve** — FLORIS sorts turbines downstream and evaluates sequentially, recomputing each turbine's Ct/axial-induction from its true local (waked) inflow before using it as a wake source. We broadcast all pairs at once for GPU vectorization; instead of a per-individual topological sort, `FarmEvaluator.evaluate()` runs an `N_JACOBI_ITERS`-pass Jacobi loop (`physics/farm_evaluator.py`) that recomputes every source turbine's Ct/axial-induction/`u_inf` each pass from the *previous* pass's local effective speed (initialized at freestream on pass 0). Because wake dependencies are strictly downstream (a DAG, not a cycle), this converges to the exact same fixed point as FLORIS's sorted solver in exactly `chain_depth` passes — verified (`tests/test_floris_comparison.py`) to bring the 3-turbine-row residual down from ~5% power / ~20% AEP to float32 noise (<1%). `N_JACOBI_ITERS` (currently 3) must be ≥ the longest downstream wake chain in the farm; bump it for deeper/denser layouts.
3. **No wind veer** — Zero wind veer (2D model). Fully 3D wind veer can be added.
4. **CuPy instead of numexpr** — All `ne.evaluate(...)` calls replaced with CuPy broadcasting.
5. **No Cumulative Gauss Curl** — Architecture supports it via plug-in interface; implementation deferred (CGC requires iterative solver incompatible with batch GA).

## GPU Rules

- **Never** call `cp.asnumpy()` or `cp.get()` inside the fitness evaluation loop.
- **Never** move wake model parameters to GPU inside the per-generation loop — upload once at init.
- All population operations must remain on `(P, T)` or `(P, T, T)` CuPy tensors.
- Use `cp.interp` for power curve lookup (not scipy).

## Wake Combination Selection

Set `WakeConfig.combination` to one of:
- `"SOSFS"` — sum of squares (default, Katic 1986)
- `"FLS"` — linear superposition
- `"MAX"` — maximum deficit

## Adding a New Wake Model

1. Subclass the relevant base from `gpuwfarm_core/physics/base.py`
2. Implement `compute(dx, dy, ...)` returning a `(P, T, T)` tensor
3. Register in `FarmEvaluator` via `WakeConfig`

## Adding a New Projection Operator

1. Subclass `ProjectionOperator` from `gpuwfarm_opt/projection/base.py`
2. Implement `project(pop: cp.ndarray) -> cp.ndarray`
3. Add to `CompositeProjection` chain in `gpuwfarm_opt/main.py`

## Running Tests

```bash
pytest packages/gpuwfarm-core        # physics / AEP core (runs standalone)
pytest packages/gpuwfarm-optimizer   # optimizer unit tests
```

The core suite passes with only `gpuwfarm-core` installed — proof that the
evaluation layer is independent of the optimizer. Compare AEP output against
the FLORIS reference for a 2-turbine aligned case (`test_floris_comparison.py`,
requires the `floris` package).

## Dependencies

Declared per package in the respective `pyproject.toml` (no top-level
`requirements.txt`):

```
gpuwfarm-core:       numpy, cupy-cuda12x   (extras: floris → pyyaml; test → pytest, pyyaml, floris)
gpuwfarm-optimizer:  gpuwfarm-core, numpy, h5py, hdf5plugin   (extras: viz → matplotlib)
```

Note: `scipy` was listed historically but is unused — power-curve lookup uses
`cp.interp`, not scipy.

## FLORIS Equation Cheat Sheet

### Far-wake start distance
```
x0 = D * cos(yaw) * (1 + sqrt(1 - Ct)) /
     (sqrt(2) * (4*alpha*TI + 2*beta*(1 - sqrt(1-Ct)))) + x_i
```

### Sigma initial conditions
```
uR = U * Ct / (2*(1 - sqrt(1 - Ct)))
u0 = U * sqrt(1 - Ct)
sigma_z0 = D * 0.5 * sqrt(uR / (U + u0))
sigma_y0 = sigma_z0 * cos(yaw)
```

### Far-wake sigma evolution
```
ky = kz = ka * TI + kb
sigma_y = ky * (x - x0) + sigma_y0
sigma_z = kz * (x - x0) + sigma_z0
```

### Deficit amplitude C
```
C = 1 - sqrt(clip(1 - Ct*cos(yaw) / (8*sigma_y*sigma_z/D²), 0, 1))
```

### Crespo-Hernandez TI
```
TI_wake = constant * a^ai * TI_amb^initial * (dx/D)^downstream
TI_eff  = sqrt(TI_amb² + TI_wake²)
```
`TI_wake` per source is only counted toward a destination turbine if it is
downstream, laterally within `2*D`, and within `15*D` downstream (matches
FLORIS `solver.py` sequential_solver's wake-added-turbulence area-of-influence
gating). With multiple upstream sources, `TI_wake` above is the **max** across
sources, not an RSS sum across sources — FLORIS combines via
`np.maximum(sqrt(ti_added_i² + TI_amb²), running_TI)` per source, so the
strongest single wake sets TI, contributions don't stack. On an in-line row
(one active source per destination at a time) max and RSS-sum agree, which is
why this only shows up on non-single-chain layouts (see
`test_floris_comparison.py::TestFullWindRose3x3`).

### Deflection (far wake)
```
theta_c0 = dm * 0.3 * yaw_rad / cos(yaw) * (1 - sqrt(1 - Ct*cos(yaw)))
C0 = 1 - u0/U;  M0 = C0*(2 - C0);  E0 = C0² - 3*e^(1/12)*C0 + 3*e^(1/3)
delta0 = tan(theta_c0) * (x0 - x_i)
middle = theta_c0 * E0 / 5.2 * sqrt(sigma_y0*sigma_z0/(ky*kz*M0)) * log(lnNum/lnDen)
delta_fw = delta0 + middle + (ad + bd*(x - x_i))
```

### AEP integration
```
AEP = sum_{wd, ws} [ sum_t P_t(U_eff_t, yaw_t) ] * freq(wd, ws) * 8760 h/yr
```
