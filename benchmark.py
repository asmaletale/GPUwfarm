"""
GPU vs CPU speedup benchmark for the wind farm physics pipeline.

Runs the core FarmEvaluator.evaluate() logic in both CuPy (GPU) and NumPy
(CPU) across several (pop_size, n_turbines) configurations and reports
wall-clock speedup.

Usage:
    python benchmark.py
"""
from __future__ import annotations
import os
import sys
import time

if sys.platform == "win32":
    _torch_lib = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib"
    )
    _torch_lib = os.path.normpath(_torch_lib)
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

import numpy as np
import cupy as cp

from config import WakeConfig, FarmConfig, TurbineConfig
from physics.farm_evaluator import FarmEvaluator
from physics.turbine.power_curve import TurbineData
from wind.wind_rose import WindRose


# ─────────────────────────────────────────────────────────────────────────────
# NumPy CPU implementation of the full physics pipeline
# (mirrors FarmEvaluator.evaluate() with cp → np)
# ─────────────────────────────────────────────────────────────────────────────

class _CPUEvaluator:
    """Pure-NumPy replica of FarmEvaluator for CPU timing."""

    def __init__(self, wake_cfg: WakeConfig, turbine_cfg: TurbineConfig,
                 turbine_data: TurbineData, air_density: float = 1.225):
        self.alpha = wake_cfg.alpha
        self.beta  = wake_cfg.beta
        self.ka    = wake_cfg.ka
        self.kb    = wake_cfg.kb
        self.ad    = wake_cfg.ad
        self.bd    = wake_cfg.bd
        self.dm    = wake_cfg.dm
        self.ch_initial    = wake_cfg.ch_initial
        self.ch_constant   = wake_cfg.ch_constant
        self.ch_ai         = wake_cfg.ch_ai
        self.ch_downstream = wake_cfg.ch_downstream

        self.D   = turbine_cfg.rotor_diameter
        self.td  = turbine_data
        self.rho = air_density

        self.ws_np  = turbine_data.wind_speeds
        self.pow_np = turbine_data.power_kw
        self.ct_np  = turbine_data.ct_values

    # ── power curve helpers ───────────────────────────────────────────────

    def _u_corr(self, u: np.ndarray) -> np.ndarray:
        return u * (self.rho / self.td.ref_air_density) ** (1.0 / 3.0)

    def _ct(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(self._u_corr(u), self.ws_np, self.ct_np), 1e-4, 0.9999)

    def _ai(self, u: np.ndarray) -> np.ndarray:
        ct = self._ct(u)
        return (1.0 - np.sqrt(np.clip(1.0 - ct, 0.0, 1.0))) / 2.0

    def _power(self, u: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        p_base = np.interp(self._u_corr(u), self.ws_np, self.pow_np,
                           left=0.0, right=0.0)
        return p_base * np.cos(yaw) ** self.td.cosine_loss_exponent_yaw

    # ── Crespo-Hernandez ──────────────────────────────────────────────────

    def _ch(self, dx: np.ndarray, ai: np.ndarray, amb_ti: float) -> np.ndarray:
        mask = dx > 0.1
        dx_s = np.where(mask, dx, 1.0)
        ai_b = ai[:, :, None]
        ti_w = (self.ch_constant * ai_b ** self.ch_ai
                * amb_ti ** self.ch_initial
                * (dx_s / self.D) ** self.ch_downstream)
        return ti_w * mask

    # ── near/far wake boundary (shared by deflection + deficit) ──────────

    def _boundary(self, ct: np.ndarray, ti: np.ndarray, yaw_int: np.ndarray,
                  u_inf: float, x_i: np.ndarray):
        ct_   = ct[:, :, None]
        yaw_  = yaw_int[:, :, None]
        xi_   = x_i[:, :, None]
        ti_s  = ti[:, :, 0:1]
        s1ct  = np.sqrt(np.clip(1.0 - ct_, 0.0, 1.0))
        uR = u_inf * ct_ / (2.0 * (1.0 - s1ct + 1e-8))
        u0 = u_inf * s1ct
        sz0 = self.D * 0.5 * np.sqrt(uR / (u_inf + u0 + 1e-8))
        sy0 = sz0 * np.cos(yaw_)
        x0 = (self.D * np.cos(yaw_) * (1.0 + s1ct)
               / (2.0**0.5 * (4.0*self.alpha*ti_s + 2.0*self.beta*(1.0-s1ct) + 1e-8))
               ) + xi_
        return x0, sy0, sz0, uR, u0

    # ── wake deflection ───────────────────────────────────────────────────

    def _deflection(self, dx: np.ndarray, ct: np.ndarray, ti_eff: np.ndarray,
                    yaw: np.ndarray, u_inf: float, x_i: np.ndarray) -> np.ndarray:
        yaw_int = -yaw
        x0, sy0, sz0, uR, u0 = self._boundary(ct, ti_eff, yaw_int, u_inf, x_i)

        ct_  = ct[:, :, None]
        yaw_ = yaw_int[:, :, None]
        xi_  = x_i[:, :, None]

        ky = self.ka * ti_eff + self.kb
        kz = ky
        sig_y = np.where(dx >= x0, ky*(dx - x0) + sy0, sy0)
        sig_z = np.where(dx >= x0, kz*(dx - x0) + sz0, sz0)

        C0 = 1.0 - u0 / u_inf
        M0 = C0 * (2.0 - C0)
        E0 = C0**2 - 3.0*np.exp(1.0/12.0)*C0 + 3.0*np.exp(1.0/3.0)
        sqM0 = np.sqrt(np.clip(M0, 1e-12, None))

        s1ct_cos = np.sqrt(np.clip(1.0 - ct_*np.cos(yaw_), 0.0, 1.0))
        theta = self.dm * (0.3*yaw_ / (np.cos(yaw_) + 1e-8)) * (1.0 - s1ct_cos)
        delta0 = np.tan(theta) * (x0 - xi_)

        # near-wake
        xR = xi_
        rd = np.where(np.abs(x0 - xR) > 1e-3, x0 - xR, 1e-3 * np.ones_like(x0))
        d_nw = ((dx - xR)/rd)*delta0 + (self.ad + self.bd*(dx - xi_))
        d_nw = d_nw * ((dx >= xR) & (dx <= x0))

        # far-wake log formula
        ratio = np.sqrt(np.clip(sig_y*sig_z / (sy0*sz0 + 1e-12), 1e-12, None))
        ln_num = (1.6 + sqM0) * (1.6*ratio - sqM0)
        ln_den = (1.6 - sqM0) * (1.6*ratio + sqM0)
        log_arg = np.clip(ln_num / (ln_den + 1e-12), 1e-12, None)
        kzkm = np.clip(ky*kz*M0, 1e-12, None)
        mid = (theta*E0/5.2 * np.sqrt(np.clip(sy0*sz0/kzkm, 0.0, None))
               * np.log(log_arg))
        d_fw = (delta0 + mid + (self.ad + self.bd*(dx - xi_))) * (dx > x0)

        return d_nw + d_fw

    # ── Gaussian velocity deficit ─────────────────────────────────────────

    def _deficit(self, dx: np.ndarray, dy: np.ndarray, delta: np.ndarray,
                 ct: np.ndarray, ti_eff: np.ndarray, yaw: np.ndarray,
                 u_inf: float, x_i: np.ndarray) -> np.ndarray:
        D = self.D
        yaw_int = -yaw
        ct_b  = ct[:, :, None]
        yaw_b = yaw_int[:, :, None]
        xi_b  = x_i[:, :, None]
        ti_s  = ti_eff[:, :, 0:1]

        s1ct = np.sqrt(np.clip(1.0 - ct_b, 0.0, 1.0))
        uR = u_inf * ct_b / (2.0*(1.0 - s1ct + 1e-8))
        u0 = u_inf * s1ct
        sz0 = D * 0.5 * np.sqrt(uR / (u_inf + u0 + 1e-8))
        sy0 = sz0 * np.cos(yaw_b)
        x0 = (D * np.cos(yaw_b) * (1.0 + s1ct)
               / (2.0**0.5 * (4.0*self.alpha*ti_s + 2.0*self.beta*(1.0-s1ct) + 1e-8))
               ) + xi_b

        ky = self.ka * ti_eff + self.kb
        kz = ky
        xR = xi_b

        # far-wake sigma
        sy_fw = np.where(dx >= x0, ky*(dx - x0) + sy0, sy0)
        sz_fw = np.where(dx >= x0, kz*(dx - x0) + sz0, sz0)

        # near-wake sigma
        rd = np.where(np.abs(x0 - xR) > 1.0, x0 - xR, np.ones_like(x0))
        ramp_up = (dx - xR) / rd
        ramp_dn = (x0 - dx) / rd
        nw_inner = 0.501 * D * np.sqrt(np.clip(ct_b / 2.0, 0.0, None))
        sy_nw = ramp_dn*nw_inner + ramp_up*sy0
        sz_nw = ramp_dn*nw_inner + ramp_up*sz0
        sy_nw = np.where(dx >= xR, sy_nw, 0.5*D)
        sz_nw = np.where(dx >= xR, sz_nw, 0.5*D)

        nw_mask = (dx > xR + 0.1) & (dx < x0)
        fw_mask = dx >= x0
        sy = np.where(nw_mask, sy_nw, np.where(fw_mask, sy_fw, 0.5*D))
        sz = np.where(nw_mask, sz_nw, np.where(fw_mask, sz_fw, 0.5*D))
        sy = np.maximum(sy, 1e-3)
        sz = np.maximum(sz, 1e-3)

        inner = 1.0 - ct_b*np.cos(yaw_b) / (8.0*sy*sz / (D*D) + 1e-12)
        C = 1.0 - np.sqrt(np.clip(inner, 0.0, 1.0))

        r_sq = (dy - delta)**2 / (2.0*sy**2)
        vel_def = C * np.exp(-r_sq)

        ds_mask = dx > (xR + 0.1)
        return np.where(ds_mask, vel_def, 0.0)

    # ── Main evaluate ─────────────────────────────────────────────────────

    def evaluate(self, pop_np: np.ndarray, wind_rose: WindRose) -> np.ndarray:
        P, T, _ = pop_np.shape
        x   = pop_np[:, :, 0]
        y   = pop_np[:, :, 1]
        yaw = pop_np[:, :, 2]

        total_AEP = np.zeros(P, dtype=np.float32)

        for wd_rad, ws_float, freq, ti_float in wind_rose.conditions():
            if freq < 1e-9:
                continue
            cos_w = float(np.cos(wd_rad))
            sin_w = float(np.sin(wd_rad))
            ws = np.float32(ws_float)
            ti = float(ti_float)

            xw = x*cos_w + y*sin_w
            yw = -x*sin_w + y*cos_w

            dx_raw = xw[:, None, :] - xw[:, :, None]
            dy_raw = yw[:, None, :] - yw[:, :, None]
            downstream_mask = dx_raw > 0.1
            dx_safe = np.maximum(dx_raw, np.float32(1.0))

            u_fs = np.full((P, T), ws, dtype=np.float32)
            ct   = self._ct(u_fs)
            ai   = self._ai(u_fs)

            ti_added = self._ch(dx_safe, ai, ti)
            ti_added = np.where(downstream_mask, ti_added, 0.0)

            ti_eff_dst = np.sqrt(ti**2 + np.sum(ti_added**2, axis=1))
            ti_eff_pairs = np.broadcast_to(
                ti_eff_dst[:, :, None], (P, T, T)
            ).copy()

            delta = self._deflection(dx_safe, ct, ti_eff_pairs, yaw, float(ws), xw)
            deficit = self._deficit(dx_safe, dy_raw, delta, ct, ti_eff_pairs,
                                    yaw, float(ws), xw)
            deficit = np.where(downstream_mask, deficit, 0.0)

            # SOSFS combination
            total_deficit = np.sqrt(np.sum(deficit**2, axis=1))
            total_deficit = np.clip(total_deficit, 0.0, 0.95)

            u_eff    = ws * (1.0 - total_deficit)
            power_kw = self._power(u_eff, yaw)
            farm_power = np.sum(power_kw, axis=1)
            total_AEP += farm_power * np.float32(freq) * np.float32(8760.0)

        return total_AEP


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_gpu(evaluator: FarmEvaluator, pop_gpu: cp.ndarray,
              wind_rose: WindRose, n_reps: int = 3) -> float:
    # warmup
    evaluator.evaluate(pop_gpu, wind_rose)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        evaluator.evaluate(pop_gpu, wind_rose)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def _time_cpu(cpu_eval: _CPUEvaluator, pop_np: np.ndarray,
              wind_rose: WindRose, n_reps: int = 3) -> float:
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        cpu_eval.evaluate(pop_np, wind_rose)
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
    (512, 20),
    (256, 50),
]

def main():
    wake_cfg    = WakeConfig()
    turbine_cfg = TurbineConfig()
    farm_cfg    = FarmConfig()
    td          = TurbineData.nrel_5mw()
    wind_rose   = WindRose.default_12sector()

    print(f"Device: {cp.cuda.Device().id}  ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
    print(f"Wind conditions: {len(list(wind_rose.conditions()))} (wd × ws bins)\n")

    header = f"{'Pop':>6}  {'Turbines':>8}  {'GPU (s)':>9}  {'CPU (s)':>9}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for pop_size, n_turb in CONFIGS:
        farm_cfg_local = FarmConfig(n_turbines=n_turb)
        turbine_cfg_local = TurbineConfig()

        gpu_eval = FarmEvaluator(farm_cfg_local, turbine_cfg_local, wake_cfg, td)
        cpu_eval = _CPUEvaluator(wake_cfg, turbine_cfg_local, td)

        pop_np  = np.random.rand(pop_size, n_turb, 3).astype(np.float32)
        pop_np[:, :, 0] *= farm_cfg_local.area_width
        pop_np[:, :, 1] *= farm_cfg_local.area_height
        pop_np[:, :, 2]  = (pop_np[:, :, 2] - 0.5) * np.deg2rad(30)

        pop_gpu = cp.asarray(pop_np)

        t_gpu = _time_gpu(gpu_eval, pop_gpu, wind_rose)
        t_cpu = _time_cpu(cpu_eval, pop_np, wind_rose)
        speedup = t_cpu / t_gpu

        print(f"{pop_size:>6}  {n_turb:>8}  {t_gpu:>9.3f}  {t_cpu:>9.3f}  {speedup:>7.1f}x")

    print()


if __name__ == "__main__":
    main()
