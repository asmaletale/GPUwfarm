"""
Multi-objective functions: LCOE, costs, and visual impact.

The batch paths used by the GA every generation -- compute_lcoe_batch() and
compute_vi_batch() (+ its _rectangle_union_area_batch() helper) -- are CuPy-
native: they accept and return GPU-resident (P,) / (P, T) tensors so the
optimizer never has to drop back to NumPy/CPU to score a population.

compute_visual_impact() and _rectangle_union_area() are the single-layout
NumPy reference/smoke-test path (see smoke/vi.py) -- not on the GA hot path,
kept on CPU intentionally since they process one layout at a time.

Costs use simplified LCOE model (no derating).
"""
from __future__ import annotations
import numpy as np
import cupy as cp

from gpuwfarm_core.config import FarmConfig, TurbineConfig, CostConfig, VisualImpactConfig
from gpuwfarm_core.wind.wind_rose import WindRose


class ObjectiveEvaluator:
    """
    Compute LCOE, costs, and VI for a wind farm layout.

    Simplified model: no derating, no foundation-specific mooring variation.
    Visual impact ported from legacy/AEP.get_farm_VI(); shapely replaced by
    a sweep-line skyline algorithm (no new dependency).
    """

    def __init__(
        self,
        farm_cfg:    FarmConfig,
        turbine_cfg: TurbineConfig,
        cost_cfg:    CostConfig,
        vi_cfg:      VisualImpactConfig | None = None,
    ) -> None:
        self.farm_cfg    = farm_cfg
        self.turbine_cfg = turbine_cfg
        self.cost_cfg    = cost_cfg
        self.vi_cfg      = vi_cfg

        self.D = turbine_cfg.rotor_diameter
        self.rated_power = 15.0  # MW, hardcoded for now (TODO: add to TurbineConfig)

    # ──────────────────────────────────────────────────────────────────
    # Cost calculation
    # ──────────────────────────────────────────────────────────────────

    def capex_per_turbine(self) -> float:
        """
        CAPEX per turbine excluding transmission.
        Includes: development, turbine+substructure.

        Units: millions EUR
        """
        dev_cons = self.cost_cfg.dev_consenting_1wt
        turb_substr = self.cost_cfg.turb_substructure_1wt
        return dev_cons + turb_substr

    def transmission_costs(
        self,
        n_turbines: float,
        total_cable_length_km: float,
        avg_water_depth_m: float,
    ) -> dict:
        """
        Transmission system costs: cables, substations, mooring, installation.

        Args:
            n_turbines: number of turbines
            total_cable_length_km: sum of internal cable lengths (from K-means clustering)
            avg_water_depth_m: average water depth at turbine locations

        Returns:
            dict with keys: intcab, expcab, offsub, onsub, moo, inst, total
        """
        c = self.cost_cfg

        # Internal array cables
        intcab = total_cable_length_km * c.c_intcab

        # Export cable (single long cable, distance from farm center to shore)
        len_expcab = 9.0  # km, hardcoded for now (TODO: make configurable)
        if len_expcab < c.ac_dc_threshold:
            c_expcable = c.c_expcable_ac
            n_expcables = c.n_expcables_ac
            offsub = c.c_offsub_ac
            onsub = 0.0
        else:
            c_expcable = c.c_expcable_dc
            n_expcables = c.n_expcables_dc
            offsub = c.c_offsub_dc
            onsub = c.c_onsub_dc

        k = max(1, int(np.ceil(n_turbines * self.rated_power / 330)))
        expcab = len_expcab * n_expcables * k * c_expcable

        # Mooring (simplified: constant per turbine, no water depth variation)
        # Full model: Moo_1WT = (n_lines * ((0.0591*MBL_chain - 87.69)*H_av + 10.198*MBL_DEA) * f_USD_E) / 1e6
        # Simplified: use fixed cost per turbine
        moo_per_turb = 2.0  # millions/turbine (empirical average)
        moo = moo_per_turb * n_turbines

        # Installation
        d_port = c.d_port
        v_ahts = c.v_ahts
        v_psv = c.v_psv
        n_turtrip = c.n_turtrip
        n_fltrip = c.n_fltrip

        n_trips_tur = int(np.ceil(n_turbines / n_turtrip))
        n_trips_fl = int(np.ceil(n_turbines / n_fltrip))

        inst_tur = c.c_boat * (
            n_turbines * c.t_inst
            + 2 * d_port / v_ahts * n_trips_fl
            + 2 * d_port / v_psv * n_trips_tur
        )
        inst_intcab = c.c_inst_intcab * total_cable_length_km
        inst_expcab = c.c_inst_expcab * len_expcab * n_expcables * k
        inst_offsub = c.c_inst_offsub
        inst_moo = c.c_inst_moo_per_turb * n_turbines

        inst = inst_tur + inst_intcab + inst_expcab + inst_offsub + inst_moo

        return {
            "intcab": float(intcab),
            "expcab": float(expcab),
            "offsub": float(offsub),
            "onsub": float(onsub),
            "moo": float(moo),
            "inst": float(inst),
            "total": float(intcab + expcab + offsub + onsub + moo + inst),
        }

    def compute_costs(
        self,
        n_turbines: float,
        aep_gwh: float,
        total_cable_length_km: float,
        avg_water_depth_m: float = 100.0,
    ) -> dict:
        """
        Compute total costs and LCOE.

        Args:
            n_turbines: number of turbines
            aep_gwh: annual energy production (GWh)
            total_cable_length_km: internal cable length
            avg_water_depth_m: average water depth (unused in simplified model)

        Returns:
            dict with keys: capex_turb, capex_trans, capex_total, opex_annual,
                           opex_discounted, costs_total, lcoe
        """
        c = self.cost_cfg

        # CAPEX
        capex_turb = self.capex_per_turbine() * n_turbines
        trans_dict = self.transmission_costs(
            n_turbines, total_cable_length_km, avg_water_depth_m
        )
        capex_trans = trans_dict["total"]
        capex_total = capex_turb + capex_trans

        # OPEX (simplified: constant annual, discounted over lifetime)
        opex_annual = c.opex_1wt * n_turbines
        r = c.discount_rate
        lifetime = c.lifetime
        opex_discounted = sum(
            opex_annual * ((1 + r) ** (-i)) for i in range(1, int(lifetime) + 1)
        )

        # Total costs
        costs_total = capex_total + opex_discounted

        # LCOE (simplified: no derating)
        # LCOE = total costs / total energy = costs * 1e6 EUR / (AEP * 1e9 Wh)
        # = costs (M€) / AEP (GWh) * 1000 EUR/MWh
        if aep_gwh > 0:
            lcoe_eur_mwh = costs_total / aep_gwh
        else:
            lcoe_eur_mwh = np.inf

        return {
            "capex_turb": float(capex_turb),
            "capex_trans": float(capex_trans),
            "capex_total": float(capex_total),
            "opex_annual": float(opex_annual),
            "opex_discounted": float(opex_discounted),
            "costs_total": float(costs_total),
            "lcoe": float(lcoe_eur_mwh),
        }

    # ──────────────────────────────────────────────────────────────────
    # Vectorized batch LCOE
    # ──────────────────────────────────────────────────────────────────

    def _fixed_costs(self, n_turbines: int) -> tuple[float, float]:
        """
        Split total costs into a fixed part (constant for given n_turbines)
        and a per-km cable coefficient.

        Returns:
            fixed_total_M_eur:  sum of all cost components except intcab (M EUR)
            cost_per_cable_km:  c_intcab + c_inst_intcab  (M EUR / km)
        """
        c = self.cost_cfg
        T = n_turbines

        capex_turb = self.capex_per_turbine() * T

        # Export cable and substation (fixed for given T)
        len_expcab = 9.0  # km — see transmission_costs()
        if len_expcab < c.ac_dc_threshold:
            c_expcable, n_expcables = c.c_expcable_ac, c.n_expcables_ac
            offsub, onsub = c.c_offsub_ac, 0.0
        else:
            c_expcable, n_expcables = c.c_expcable_dc, c.n_expcables_dc
            offsub, onsub = c.c_offsub_dc, c.c_onsub_dc

        k = max(1, int(np.ceil(T * self.rated_power / 330)))
        expcab = len_expcab * n_expcables * k * c_expcable

        moo = 2.0 * T  # simplified fixed mooring cost

        n_trips_tur = int(np.ceil(T / c.n_turtrip))
        n_trips_fl  = int(np.ceil(T / c.n_fltrip))
        inst_tur    = c.c_boat * (
            T * c.t_inst
            + 2 * c.d_port / c.v_ahts * n_trips_fl
            + 2 * c.d_port / c.v_psv  * n_trips_tur
        )
        inst_expcab = c.c_inst_expcab * len_expcab * n_expcables * k
        inst_offsub = c.c_inst_offsub
        inst_moo    = c.c_inst_moo_per_turb * T

        fixed_trans = expcab + offsub + onsub + moo + inst_tur + inst_expcab + inst_offsub + inst_moo

        opex_annual     = c.opex_1wt * T
        r, lt           = c.discount_rate, c.lifetime
        opex_discounted = opex_annual * (1.0 - (1.0 + r) ** (-lt)) / r  # annuity

        fixed_total      = capex_turb + fixed_trans + opex_discounted
        cost_per_cable_km = c.c_intcab + c.c_inst_intcab

        return float(fixed_total), float(cost_per_cable_km)

    def compute_lcoe_batch(
        self,
        n_turbines: int,
        aep_gwh: cp.ndarray,
        cable_length_km: cp.ndarray,
    ) -> cp.ndarray:
        """
        Vectorized LCOE over a population batch. GPU-resident: aep_gwh and
        cable_length_km are expected to already be CuPy arrays (as produced by
        FarmEvaluator.evaluate() / GeneticAlgorithm.compute_objectives()), so
        the GA never has to drop back to NumPy to score a population.

        Args:
            n_turbines:       number of turbines (scalar, same for all individuals)
            aep_gwh:          (P,) annual energy production in GWh
            cable_length_km:  (P,) internal cable length in km

        Returns:
            lcoe: (P,) LCOE in EUR/MWh as float32
        """
        fixed, per_km = self._fixed_costs(n_turbines)
        total_costs = fixed + cable_length_km * per_km        # (P,) M EUR
        # Avoid the division-by-zero warning path entirely rather than relying on
        # a NumPy-only errstate context (CuPy has no direct equivalent).
        safe_aep = cp.where(aep_gwh > 0, aep_gwh, cp.float32(1.0))
        lcoe = cp.where(aep_gwh > 0, total_costs / safe_aep, cp.float32(np.inf))
        return lcoe.astype(cp.float32)

    # ──────────────────────────────────────────────────────────────────
    # Visual impact
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _rectangle_union_area(
        xiL: np.ndarray,
        xiR: np.ndarray,
        ziTot: np.ndarray,
    ) -> float:
        """
        Area of the union of axis-aligned rectangles [xiL[i], xiR[i]] × [0, ziTot[i]].

        Sweep-line skyline: replaces shapely.unary_union from legacy code.
        O(T²), T ≤ 50 so this is fast.

        Legacy source: shapely box + unary_union in AEP.get_farm_VI()
        """
        valid = (xiR > xiL) & (ziTot > 0.0)
        if not valid.any():
            return 0.0
        L, R, H = xiL[valid], xiR[valid], ziTot[valid]
        x_events = np.unique(np.concatenate([L, R]))
        if len(x_events) < 2:
            return 0.0
        area = 0.0
        for k in range(len(x_events) - 1):
            x_lo, x_hi = x_events[k], x_events[k + 1]
            active = (L <= 0.5 * (x_lo + x_hi)) & (0.5 * (x_lo + x_hi) < R)
            if active.any():
                area += float(H[active].max()) * (x_hi - x_lo)
        return area

    @staticmethod
    def _rectangle_union_area_batch(
        xiL: cp.ndarray,
        xiR: cp.ndarray,
        ziTot: cp.ndarray,
    ) -> cp.ndarray:
        """
        Exact skyline union area for P individuals simultaneously. GPU-resident
        (CuPy) — this is the batch path called from compute_vi_batch(), on the
        GA hot loop.

        Vectorized equivalent of _rectangle_union_area: sorts event x-values per row,
        then checks coverage with a (P, T, 2T-1) broadcast — no Python loop over P.

        Args:
            xiL:   (P, T) left angular edge of each turbine rectangle
            xiR:   (P, T) right angular edge
            ziTot: (P, T) height of each turbine rectangle

        Returns:
            (P,) union area per individual
        """
        # Zero-out invalid turbines so they never win the max
        H = cp.where((xiR > xiL) & (ziTot > 0.0), ziTot, cp.float32(0.0))

        # Sweep-line events: sorted union of left and right edges per individual
        all_events = cp.sort(cp.concatenate([xiL, xiR], axis=1), axis=1)  # (P, 2T)

        x_lo  = all_events[:, :-1]              # (P, 2T-1)
        x_hi  = all_events[:, 1:]
        x_mid = 0.5 * (x_lo + x_hi)            # (P, 2T-1)

        # covers[p, t, k]: turbine t covers midpoint of interval k
        # xiL (P,T,1), x_mid (P,1,2T-1) → broadcast (P, T, 2T-1)
        covers = (xiL[:, :, None] <= x_mid[:, None, :]) & \
                 (x_mid[:, None, :] < xiR[:, :, None])

        max_h = cp.where(covers, H[:, :, None], cp.float32(0.0)).max(axis=1)  # (P, 2T-1)
        return (max_h * (x_hi - x_lo)).sum(axis=1)                 # (P,)

    def compute_visual_impact(
        self,
        x: np.ndarray,
        y: np.ndarray,
        wind_rose: WindRose,
    ) -> float:
        """
        Visual impact score for a single farm layout.

        Ported from legacy/AEP.get_farm_VI().  Each observer sees a set of
        angular rectangles (horizontal × vertical subtense) whose union,
        normalised by the horizontal/vertical FOV and weighted by direction
        frequency and observer weight, gives the VI score.

        Args:
            x:         (T,) turbine x coordinates (m)
            y:         (T,) turbine y coordinates (m)
            wind_rose: provides wind directions and marginal direction frequencies

        Returns:
            VI score (dimensionless, ≥ 0)
        """
        if self.vi_cfg is None:
            return 0.0

        cfg  = self.vi_cfg
        R    = cfg.earth_radius
        ht   = self.turbine_cfg.hub_height
        D    = self.turbine_cfg.rotor_diameter
        xfov = np.deg2rad(cfg.xfov_deg)
        zfov = np.deg2rad(cfg.zfov_deg)

        # Marginal direction frequency: (n_wd,)
        freq_dir = wind_rose.freq_table.sum(axis=1)
        wd_array = wind_rose.wind_dirs

        VI_total = 0.0
        for obs_xy, weight, obs_h in zip(
            cfg.obs_coords, cfg.obs_weights, cfg.obs_heights
        ):
            x_obs, y_obs = float(obs_xy[0]), float(obs_xy[1])

            # Angle and arc-length to geometric horizon from observer height
            gamma_hor    = np.arccos(np.clip(R / (R + obs_h), -1.0, 1.0))
            horizon_dist = gamma_hor * R

            # Distance observer → each turbine: (T,)
            li = np.sqrt((x - x_obs) ** 2 + (y - y_obs) ** 2)
            li = np.maximum(li, 1.0)    # avoid division by zero

            # Earth-curvature height drop for turbines beyond the horizon: (T,)
            gamma_i = li / R
            denom   = np.cos(gamma_i - gamma_hor)
            denom   = np.where(np.abs(denom) < 1e-9, np.float64(1e-9), denom)
            hdi     = np.where(li <= horizon_dist, 0.0, R / denom - R)

            # Vertical angular subtenses: (T,)
            zi    = (ht - hdi) / li         # hub angular height (rad)
            ziTot = zi + D / (2.0 * li)     # add top half of rotor

            # Bearing (in-plane angle) from observer to each turbine: (T,)
            theta_i = np.arctan2(x - x_obs, y - y_obs)

            # Bearing to farm centroid — used as horizontal reference zero
            xc       = 0.5 * (x.max() + x.min())
            yc       = 0.5 * (y.max() + y.min())
            theta_fv = np.arctan2(xc - x_obs, yc - y_obs)

            # Horizontal angular offset from farm centre direction: (T,)
            xi = theta_i - theta_fv   # dsop = 1 (legacy always)

            VI_obs = 0.0
            for wd_deg, freq_d in zip(wd_array, freq_dir):
                # Projected horizontal width of each rotor as seen from observer
                phi  = np.deg2rad(float(wd_deg)) - theta_i   # (T,)
                xiD  = D / li * np.abs(np.cos(phi))            # (T,)
                xiL  = xi - xiD * 0.5
                xiR  = xi + xiD * 0.5

                union_area = ObjectiveEvaluator._rectangle_union_area(xiL, xiR, ziTot)
                VI_obs += (union_area / (xfov * zfov)) * float(freq_d)

            VI_total += VI_obs * float(weight)

        return float(VI_total)

    def compute_vi_batch(
        self,
        x_batch: cp.ndarray,
        y_batch: cp.ndarray,
        wind_rose: WindRose,
    ) -> cp.ndarray:
        """
        Compute VI for an entire population batch — fully vectorized over P.
        GPU-resident: x_batch/y_batch are expected to already be CuPy arrays
        (as sliced straight out of the GA population tensor), so the optimizer
        never has to transfer the population to NumPy to score VI.

        Replaces the O(P × n_obs × n_wd × T²) Python loop with CuPy broadcasts
        over (P, T) tensors.  The only remaining loops are over observers (n_obs,
        usually ≤ 5) and wind directions (n_wd, usually 36) -- each iteration is
        one small Python-level step driving a (P, T) GPU op, not a P-sized loop.

        Args:
            x_batch:   (P, T) turbine x coordinates
            y_batch:   (P, T) turbine y coordinates
            wind_rose: wind directions and direction-marginal frequencies

        Returns:
            (P,) VI scores as float32
        """
        P, T = x_batch.shape
        vi = cp.zeros(P, dtype=cp.float32)
        if self.vi_cfg is None:
            return vi

        cfg  = self.vi_cfg
        R    = cfg.earth_radius
        ht   = self.turbine_cfg.hub_height
        D    = self.turbine_cfg.rotor_diameter
        xfov = np.deg2rad(cfg.xfov_deg)
        zfov = np.deg2rad(cfg.zfov_deg)

        # Small (n_wd,) host-side arrays -- cheap to keep as NumPy/Python scalars
        # driving the outer Python loop below.
        freq_dir = wind_rose.freq_table.sum(axis=1)   # (n_wd,)
        wd_rads  = np.deg2rad(wind_rose.wind_dirs)    # (n_wd,)

        VI = cp.zeros(P, dtype=cp.float64)

        for obs_xy, weight, obs_h in zip(cfg.obs_coords, cfg.obs_weights, cfg.obs_heights):
            x_obs, y_obs = float(obs_xy[0]), float(obs_xy[1])

            gamma_hor    = np.arccos(np.clip(R / (R + obs_h), -1.0, 1.0))
            horizon_dist = gamma_hor * R

            # All (P, T) geometry — computed once per observer
            li = cp.sqrt((x_batch - x_obs) ** 2 + (y_batch - y_obs) ** 2)
            li = cp.maximum(li, cp.float32(1.0))

            gamma_i = li / R
            denom   = cp.cos(gamma_i - gamma_hor)
            denom   = cp.where(cp.abs(denom) < 1e-9, cp.float32(1e-9), denom)
            hdi     = cp.where(li <= horizon_dist, cp.float32(0.0), R / denom - R)

            zi    = (ht - hdi) / li
            ziTot = zi + D / (2.0 * li)                           # (P, T)

            theta_i = cp.arctan2(x_batch - x_obs, y_batch - y_obs)  # (P, T)

            xc = 0.5 * (x_batch.max(axis=1, keepdims=True) + x_batch.min(axis=1, keepdims=True))
            yc = 0.5 * (y_batch.max(axis=1, keepdims=True) + y_batch.min(axis=1, keepdims=True))
            theta_fv = cp.arctan2(xc - x_obs, yc - y_obs)          # (P, 1)

            xi = theta_i - theta_fv                                # (P, T)

            VI_obs = cp.zeros(P, dtype=cp.float64)

            for wd_rad, freq_d in zip(wd_rads, freq_dir):
                phi  = float(wd_rad) - theta_i                     # (P, T)
                xiD  = D / li * cp.abs(cp.cos(phi))                # (P, T)
                xiL  = xi - 0.5 * xiD
                xiR  = xi + 0.5 * xiD

                areas = ObjectiveEvaluator._rectangle_union_area_batch(xiL, xiR, ziTot)
                VI_obs += areas * float(freq_d)

            VI += VI_obs * float(weight)

        VI /= (xfov * zfov)
        return VI.astype(cp.float32)
