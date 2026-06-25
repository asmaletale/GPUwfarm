"""
Wind rose and AEP integration.

FLORIS source: floris/wind_data.py (WindRose class)
               floris/floris_model.py (get_farm_AEP)

Structure mirrors FLORIS WindRose:
  - freq_table: (n_wd, n_ws) probability mass, must sum to 1.0
  - wind_dirs:  (n_wd,) degrees
  - wind_speeds:(n_ws,) m/s
  - ti_table:   (n_wd, n_ws) turbulence intensity

AEP integration (exact FLORIS pattern):
    AEP = sum_{wd, ws} [ P_farm(wd, ws) * freq(wd, ws) ] * 8760

Extensions beyond FLORIS:
  - from_weibull(): Weibull-based frequency generation per sector
  - Uniform TI assignment if ti_table not provided
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindRose:
    """
    Wind rose with (n_wd × n_ws) frequency table.

    FLORIS source: floris/wind_data.py — WindRose class

    Attributes:
        wind_dirs:   (n_wd,) degrees, 0–360
        wind_speeds: (n_ws,) m/s, sorted ascending
        freq_table:  (n_wd, n_ws) probability mass; must sum to 1.0
        ti_table:    (n_wd, n_ws) turbulence intensity
    """
    wind_dirs:   np.ndarray   # degrees
    wind_speeds: np.ndarray   # m/s
    freq_table:  np.ndarray   # (n_wd, n_ws), sums to 1
    ti_table:    np.ndarray   # (n_wd, n_ws)

    def __post_init__(self) -> None:
        assert self.freq_table.shape == (len(self.wind_dirs), len(self.wind_speeds)), \
            "freq_table must be (n_wd, n_ws)"
        # Normalise (mirrors FLORIS WindRose.__init__)
        total = np.sum(self.freq_table)
        if not np.isclose(total, 1.0, atol=1e-4):
            self.freq_table = self.freq_table / total

    # ── Iteration helpers ──────────────────────────────────────────────

    @property
    def n_conditions(self) -> int:
        return len(self.wind_dirs) * len(self.wind_speeds)

    def conditions(self):
        """
        Iterate over all (wd_rad, ws, freq, ti) combinations.
        Mirrors FLORIS WindRose.unpack() flattening.
        """
        for i, wd in enumerate(self.wind_dirs):
            for j, ws in enumerate(self.wind_speeds):
                yield np.deg2rad(wd), ws, self.freq_table[i, j], self.ti_table[i, j]

    # ── Factory: uniform TI per-sector ────────────────────────────────

    @classmethod
    def from_uniform_ti(
        cls,
        wind_dirs: np.ndarray,
        wind_speeds: np.ndarray,
        freq_table: np.ndarray,
        ti_ambient: float = 0.06,
    ) -> "WindRose":
        ti = np.full((len(wind_dirs), len(wind_speeds)), ti_ambient, dtype=np.float32)
        return cls(
            wind_dirs=np.asarray(wind_dirs, dtype=np.float32),
            wind_speeds=np.asarray(wind_speeds, dtype=np.float32),
            freq_table=np.asarray(freq_table, dtype=np.float32),
            ti_table=ti,
        )

    # ── Factory: Weibull-based wind rose ──────────────────────────────

    @classmethod
    def from_weibull(
        cls,
        wind_dirs: np.ndarray,           # (n_wd,) degrees
        sector_freqs: np.ndarray,        # (n_wd,) fraction of time in each sector, sums to 1
        weibull_A: np.ndarray,           # (n_wd,) Weibull scale parameter (m/s)
        weibull_k: np.ndarray,           # (n_wd,) Weibull shape parameter
        wind_speeds: np.ndarray,         # (n_ws,) speed bin centres (m/s)
        ti_ambient: float = 0.06,
    ) -> "WindRose":
        """
        Build a WindRose from per-sector Weibull distributions.

        The Weibull PDF is:
            f(u) = (k/A) * (u/A)^(k-1) * exp(-(u/A)^k)

        Each row of freq_table is the Weibull PDF evaluated at wind_speeds,
        scaled by the sector frequency and normalised so the entire table
        sums to 1.

        This extends FLORIS (which takes freq_table directly) with parametric
        generation from Weibull statistics.
        """
        n_wd = len(wind_dirs)
        n_ws = len(wind_speeds)
        freq_table = np.zeros((n_wd, n_ws), dtype=np.float64)

        for i in range(n_wd):
            A, k = weibull_A[i], weibull_k[i]
            # Weibull CDF integral over each speed bin (trapezoidal)
            pdf = (k / A) * (wind_speeds / A) ** (k - 1) * np.exp(-(wind_speeds / A) ** k)
            # Approximate bin probability via PDF × bin_width
            dw = np.diff(wind_speeds)
            bin_widths = np.concatenate([[dw[0]], (dw[:-1] + dw[1:]) / 2, [dw[-1]]])
            prob = pdf * bin_widths
            prob = prob / (prob.sum() + 1e-12)   # normalise within sector
            freq_table[i] = prob * sector_freqs[i]

        return cls.from_uniform_ti(
            wind_dirs=wind_dirs,
            wind_speeds=wind_speeds.astype(np.float32),
            freq_table=freq_table.astype(np.float32),
            ti_ambient=ti_ambient,
        )

    # ── Default rose (matches the original main.py wind rose) ─────────

    @classmethod
    def default_12sector(cls) -> "WindRose":
        """
        Reproduces the original main.py wind rose as a 12×1 single-speed rose.
        """
        dirs  = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                         dtype=np.float32)
        # Original main.py used one speed per direction; reproduce as 12×1 table
        speeds = np.array([9.0], dtype=np.float32)   # representative single speed
        freq_1d = np.array([0.05,0.06,0.07,0.10,0.12,0.11,0.10,0.09,0.08,0.08,0.07,0.07],
                           dtype=np.float32)
        freq_table = freq_1d[:, None]   # (12, 1)
        ti = np.full((12, 1), 0.06, dtype=np.float32)
        return cls(wind_dirs=dirs, wind_speeds=speeds,
                   freq_table=freq_table, ti_table=ti)

    @classmethod
    def default_12sector_multispeed(cls) -> "WindRose":
        """
        12 direction sectors × 9 speed bins (4–14 m/s) with Weibull-like distribution.
        More physically realistic than a single speed per sector.
        """
        dirs  = np.arange(0, 360, 30, dtype=np.float32)
        speeds = np.arange(4.0, 15.0, 1.0, dtype=np.float32)   # 11 bins
        dir_freqs = np.array([0.05,0.06,0.07,0.10,0.12,0.11,
                               0.10,0.09,0.08,0.08,0.07,0.07], dtype=np.float32)
        A_arr = np.full(12, 9.5, dtype=np.float32)
        k_arr = np.full(12, 2.0, dtype=np.float32)
        return cls.from_weibull(dirs, dir_freqs, A_arr, k_arr, speeds)
