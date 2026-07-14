# GPUwfarm — CuPy Wind Farm Optimizer

## Python Interpreter

Default interpreter for this project (CUDA 12.6 laptop):

```
C:\Users\alari\PycharmProjects\venv311pytorchcuda\Scripts\python.exe
```

Run tests and scripts with this interpreter:

```bash
& "C:\Users\alari\PycharmProjects\venv311pytorchcuda\Scripts\python.exe" -m pytest tests/ -v
& "C:\Users\alari\PycharmProjects\venv311pytorchcuda\Scripts\python.exe" main.py
```

## Project Purpose

A GPU-accelerated genetic algorithm for wind farm layout and yaw optimization.
The physics layer is a faithful port of [FLORIS](https://github.com/NREL/floris),
making every equation traceable to NREL source code.

## Repository Layout

```
gpuwfarm/
├── config.py                    # FarmConfig, GAConfig, WakeConfig dataclasses
├── main.py                      # Entry point
├── CLAUDE.md                    # This file
├── physics/
│   ├── base.py                  # BaseWakeComponent ABC
│   ├── wake_velocity/gauss.py   # FLORIS GaussVelocityDeficit port
│   ├── wake_turbulence/crespo_hernandez.py  # FLORIS CrespoHernandez port
│   ├── wake_deflection/gauss.py # FLORIS GaussVelocityDeflection port
│   ├── wake_combination/        # SOSFS / FLS / MAX (FLORIS ports)
│   ├── turbine/power_curve.py   # Tabulated power curve + cosine yaw loss
│   └── farm_evaluator.py        # Pipeline orchestrator
├── projection/                  # Feasibility repair operators
├── wind/wind_rose.py            # WindRose (dir×speed bins + Weibull)
├── optimizer/genetic.py         # GeneticAlgorithm
└── tests/                       # Unit tests (see Validation section)
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
`config.py` do **not** hardcode these numbers — they are `field(default_factory=...)`
values parsed at import time from `examples/gch.yaml` (farm/wake) and
`examples/nrel_5MW.yaml` (turbine, FLORIS's own file). `loaders/floris_yaml.py`
parses a user-supplied YAML through the exact same `WakeConfig.from_wake_dict`
/ `TurbineConfig.from_turbine_dict` classmethods, so `WakeConfig()` and loading
`examples/gch.yaml` always agree. To change a default, edit the YAML, not
config.py. The two YAMLs are kept separate on purpose (mirrors FLORIS's own
farm-input vs. turbine-library split) — do not merge turbine data into gch.yaml.

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
2. **Simultaneous all-pairs** — FLORIS sorts turbines downstream and evaluates sequentially, recomputing each turbine's Ct/axial-induction from its true local (waked) inflow before using it as a wake source. We broadcast all pairs at once for GPU vectorization and compute every source turbine's Ct/axial-induction from freestream speed instead — verified (`tests/test_floris_comparison.py`) to cause ~5% power / ~20% AEP error on turbines that sit downstream of an already-waked turbine (e.g. the last turbine in a row of 3+); turbines whose sources are all unwaked are unaffected.
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

1. Subclass `BaseWakeComponent` from `physics/base.py`
2. Implement `prepare(config)` and `compute(dx, dy, ...)` returning `(P, T, T)` tensor
3. Register in `FarmEvaluator` via `WakeConfig`

## Adding a New Projection Operator

1. Subclass `ProjectionOperator` from `projection/base.py`
2. Implement `project(pop: cp.ndarray) -> cp.ndarray`
3. Add to `CompositeProjection` chain in `main.py`

## Running Tests

```bash
pytest tests/ -v
```

Compare AEP output against FLORIS reference for a 2-turbine aligned case.

## Dependencies

```
cupy-cuda11x   (or cupy-cuda12x)
numpy
scipy          # power curve interpolation (CPU only, run once at init)
matplotlib
pytest
```

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
