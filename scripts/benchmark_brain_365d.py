"""Strictly walk-forward 365-day Brain calibration benchmark (<= 2026-11-24 OOS quarantine).

Evaluates Contour 1 probabilistic accuracy across realistic Polymarket 1°C (non-US) and
2°F with Remark Tenths (US) buckets using strictly chronological walk-forward updates:
  - M0_hist_emos_45d:   Current histogram + 45d hierarchical Empirical Bayes EMOS + 0.001 floor
  - M1a_normal_cdf_45d: Analytical Normal CDF integration over exact bucket bounds
  - M1b_student_t_cdf:  Variance-matched Student-t CDF sweep (nu in 5, 7, 10, 15)
  - M1c_weighted_het_t: Regional walk-forward NWP weights (ECMWF/GFS/ICON) + heteroscedastic Student-t
  - M2_ewma_ridge:      EWMA half-life sweep (14d..45d) + trailing L2 Ridge residual correction

Strictly excludes local_date > --max-date (default 2026-11-24) so 2026-11-25+ live Parquet
remains 100% uncontaminated Out-of-Sample.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from weather_quant.discovery.stations import get_station
from weather_quant.domain.models import Bucket, Polarity, Station
from weather_quant.ml.dataset_builder import MosDatasetSample
from weather_quant.probability.calibration import build_synthetic_degree_buckets, resolve_bucket_index
from weather_quant.probability.engine import (
    DEFAULT_BIAS_SHRINKAGE_DAYS,
    DEFAULT_SCALE_SHRINKAGE_DAYS,
    DEFAULT_SPREAD_INFLATION,
    Z_P90,
    bucket_effective_bounds_c,
    map_extremes_c_to_probabilities,
    round_half_away,
    to_event_unit,
)
from weather_quant.replay.metrics import compute_rps

_EPS = 1e-9
_PROB_CLIP_LL = 1e-4
_CDF_FLOOR = 1e-6
_Z_GRID_121 = np.linspace(-2.5, 2.5, 121)
_CRPS_U_101 = (np.arange(1, 102, dtype=np.float64) - 0.5) / 101.0
_CRPS_WEIGHTS_101 = 2.0 * np.arange(101, dtype=np.float64) - 101.0 + 1.0  # sorted O(N) Gini half-diff
_T_PPF_GRID_101 = {float(nu): stats.t.ppf(_CRPS_U_101, df=float(nu)) for nu in (5, 7, 10, 15)}
_T_Q90 = {float(nu): float(stats.t.ppf(0.90, df=float(nu))) for nu in (5, 7, 10, 15)}
_CRPS_Z_STD_101 = {
    nu: float(0.5 * np.mean(np.abs(g[:, None] - g[None, :])))
    for nu, g in _T_PPF_GRID_101.items()
}

_RIDGE_FEATURES: tuple[str, ...] = (
    "pressure_msl_tendency_24h",
    "prior_diurnal_range_c",
    "delta_obs_temp_24h",
    "cloud_cover_mean",
    "wind_speed_mean",
    "relative_humidity_mean",
    "doy_sin",
    "doy_cos",
)


def _season_of(d: date) -> str:
    if d.month in (12, 1, 2):
        return "DJF"
    if d.month in (3, 4, 5):
        return "MAM"
    if d.month in (6, 7, 8):
        return "JJA"
    return "SON"


def _region_of(station: Station) -> str:
    c = station.country.strip().upper()
    if c == "US":
        return "US"
    if c in {"FR", "DE", "GB", "NL", "ES", "IT", "AT", "CH", "RU", "TR"}:
        return "EU"
    return "ASIA_OTHER"


def _build_polymarket_buckets(
    station: Station,
    center_c: float,
    event_id: str,
) -> list[Bucket]:
    """Build 11-bucket 1°C (non-US) or 2°F (US) partition matching Polymarket geometry."""
    if station.country.strip().upper() != "US":
        return build_synthetic_degree_buckets(round_half_away(center_c), event_id=event_id, half_width=5)

    center_f = round_half_away(to_event_unit(center_c, "F"))
    if center_f % 2 != 0:
        center_f += 1
    low_edge = center_f - 10
    high_edge = center_f + 10
    buckets: list[Bucket] = [
        Bucket(
            bucket_id=f"{event_id}-b0",
            market_id=f"{event_id}-m0",
            token_id=f"{event_id}-t0",
            label=f"{low_edge}°F or below",
            lower_bound=None,
            upper_bound=float(low_edge),
            unit="F",
            outcome_index=0,
        )
    ]
    idx = 1
    for deg_f in range(low_edge + 2, high_edge, 2):
        buckets.append(
            Bucket(
                bucket_id=f"{event_id}-b{idx}",
                market_id=f"{event_id}-m{idx}",
                token_id=f"{event_id}-t{idx}",
                label=f"{deg_f}-{deg_f + 1}°F",
                lower_bound=float(deg_f),
                upper_bound=float(deg_f + 1),
                unit="F",
                outcome_index=idx,
            )
        )
        idx += 1
    buckets.append(
        Bucket(
            bucket_id=f"{event_id}-b{idx}",
            market_id=f"{event_id}-m{idx}",
            token_id=f"{event_id}-t{idx}",
            label=f"{high_edge}°F or higher",
            lower_bound=float(high_edge),
            upper_bound=None,
            unit="F",
            outcome_index=idx,
        )
    )
    return buckets


def _cdf_probs_normal(
    mu_c: float,
    sigma_c: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
) -> list[float]:
    s = max(0.20, sigma_c)
    c_lo = np.where(np.isfinite(bounds_lo), 0.5 * (1.0 + special.erf((bounds_lo - mu_c) / (s * math.sqrt(2.0)))), 0.0)
    c_hi = np.where(np.isfinite(bounds_hi), 0.5 * (1.0 + special.erf((bounds_hi - mu_c) / (s * math.sqrt(2.0)))), 1.0)
    raw = np.maximum(_CDF_FLOOR, c_hi - c_lo)
    return (raw / float(np.sum(raw))).tolist()


def _cdf_probs_student_t(
    mu_c: float,
    sigma_c: float,
    nu: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
) -> list[float]:
    eff_sigma = max(0.20, sigma_c)
    scale = eff_sigma * math.sqrt((nu - 2.0) / nu) if nu > 2.0 else eff_sigma
    c_lo = np.where(np.isfinite(bounds_lo), special.stdtr(nu, (bounds_lo - mu_c) / scale), 0.0)
    c_hi = np.where(np.isfinite(bounds_hi), special.stdtr(nu, (bounds_hi - mu_c) / scale), 1.0)
    raw = np.maximum(_CDF_FLOOR, c_hi - c_lo)
    return (raw / float(np.sum(raw))).tolist()


def _crps_normal(y: float, mu: float, sigma: float) -> float:
    s = max(0.20, sigma)
    z = (y - mu) / s
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return s * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def _crps_student_t(y: float, mu: float, sigma: float, nu: float) -> float:
    s = max(0.20, sigma)
    scale = s * math.sqrt((nu - 2.0) / nu) if nu > 2.0 else s
    qs = mu + scale * _T_PPF_GRID_101[float(nu)]
    return float(np.mean(np.abs(qs - y))) - scale * _CRPS_Z_STD_101[float(nu)]


def _crps_empirical(y: float, draws: np.ndarray) -> float:
    n = len(draws)
    s_draws = np.sort(draws)
    spread = float(np.dot(2.0 * np.arange(1, n + 1, dtype=np.float64) - n - 1.0, s_draws)) / (n * n)
    return float(np.mean(np.abs(draws - y))) - spread


def _compute_global_prior_cached(
    global_hist: Sequence[tuple[date, float, float]],
    current_date: date,
    window_days: int = 45,
) -> tuple[float, float]:
    """Compute cross-station hierarchical prior once per (date, polarity, window_days)."""
    g_valid: list[tuple[float, float]] = []
    # global_hist is appended chronologically, so stop early once age > window_days
    for d, err, std in reversed(global_hist):
        age = (current_date - d).days
        if age > window_days:
            break
        if age > 0:
            g_valid.append((err, std))
    n_g = len(g_valid)
    if n_g < 5:
        return 0.0, DEFAULT_SPREAD_INFLATION
    g_errs = sorted(e for e, _ in g_valid)
    mid = n_g // 2
    med_g = g_errs[mid] if n_g % 2 == 1 else 0.5 * (g_errs[mid - 1] + g_errs[mid])
    prior_bias_c = (n_g / (n_g + 10.0)) * med_g
    gz = sorted(abs(e - prior_bias_c) / std for e, std in g_valid if std > 1e-6)
    if len(gz) < 5:
        return prior_bias_c, DEFAULT_SPREAD_INFLATION
    q80_pos = 0.80 * (len(gz) - 1)
    lo_i = int(math.floor(q80_pos))
    hi_i = min(len(gz) - 1, lo_i + 1)
    q80_z = gz[lo_i] + (q80_pos - lo_i) * (gz[hi_i] - gz[lo_i])
    raw_gs = max(1.0, min(2.8, q80_z / Z_P90))
    prior_scale = max(
        1.0,
        min(2.8, (n_g * raw_gs + 15.0 * DEFAULT_SPREAD_INFLATION) / (n_g + 15.0)),
    )
    return prior_bias_c, prior_scale


def _compute_hierarchical_emos_params(
    station_hist: Sequence[tuple[date, float, float]],
    prior_bias_c: float,
    prior_scale: float,
    current_date: date,
    window_days: int = 45,
    half_life_days: float | None = None,
) -> tuple[float, float]:
    """Match EnsembleProbabilityEngine._compute_station_bias using precomputed global prior."""
    s_valid: list[tuple[date, float, float]] = []
    for d, err, std in reversed(station_hist):
        age = (current_date - d).days
        if age > window_days:
            break
        if age > 0:
            s_valid.append((d, err, std))
    if not s_valid:
        return prior_bias_c, prior_scale

    n = len(s_valid)
    if half_life_days is not None and half_life_days > 0:
        weights = [0.5 ** (((current_date - d).days - 1) / half_life_days) for d, _, _ in s_valid]
        w_sum = sum(weights)
        raw_bias_c = sum(w * e for w, (_, e, _) in zip(weights, s_valid)) / max(w_sum, _EPS)
        n_eff = (w_sum * w_sum) / max(sum(w * w for w in weights), _EPS)
    else:
        weights = [1.0] * n
        w_sum = float(n)
        raw_bias_c = sum(e for _, e, _ in s_valid) / n
        n_eff = float(n)

    bias_c = (n_eff * raw_bias_c + DEFAULT_BIAS_SHRINKAGE_DAYS * prior_bias_c) / (
        n_eff + DEFAULT_BIAS_SHRINKAGE_DAYS
    )
    if n < 2:
        return bias_c, prior_scale

    residuals = [e - raw_bias_c for _, e, _ in s_valid]
    if half_life_days is not None and half_life_days > 0:
        err_var = sum(w * r * r for w, r in zip(weights, residuals)) / max(w_sum - 1e-6, _EPS)
        err_std = math.sqrt(max(0.0, err_var))
        ref_std = sum(w * std for w, (_, _, std) in zip(weights, s_valid)) / max(w_sum, _EPS)
    else:
        err_std = math.sqrt(sum(r * r for r in residuals) / max(1, n - 1))
        ref_std = sum(std for _, _, std in s_valid) / n

    if ref_std <= 1e-6:
        return bias_c, prior_scale

    scale_var = err_std / ref_std
    scale_cov = scale_var
    if n >= 5:
        z_scores = sorted(abs(r) / max(std, 1e-6) for r, (_, _, std) in zip(residuals, s_valid))
        q80_pos = 0.80 * (len(z_scores) - 1)
        lo_i = int(math.floor(q80_pos))
        hi_i = min(len(z_scores) - 1, lo_i + 1)
        q80_z = z_scores[lo_i] + (q80_pos - lo_i) * (z_scores[hi_i] - z_scores[lo_i])
        scale_cov = q80_z / Z_P90

    raw_scale = max(scale_var, scale_cov)
    shrunk_scale = (
        n_eff * raw_scale + DEFAULT_SCALE_SHRINKAGE_DAYS * prior_scale
    ) / (n_eff + DEFAULT_SCALE_SHRINKAGE_DAYS)
    min_scale = 1.0 if n < 10 else 0.85
    return bias_c, max(min_scale, min(3.0, shrunk_scale))


def _fit_simplex_nwp_weights(
    rows: Sequence[tuple[float, float, float, float]],
    shrink_n: float = 30.0,
) -> tuple[float, float, float]:
    """Fit constrained simplex weights (ECMWF, GFS, ICON) >= 0, sum = 1 shrunk toward (1/3,1/3,1/3)."""
    if len(rows) < 20:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    arr = np.asarray(rows, dtype=np.float64)
    X = arr[:, :3]
    y = arr[:, 3]
    # Center both X and y so station bias doesn't distort relative NWP weights
    Xc = X - np.mean(X, axis=0, keepdims=True)
    yc = y - np.mean(y)
    eq_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    def loss(w: np.ndarray) -> float:
        pred = Xc @ w
        mse = float(np.mean((pred - yc) ** 2))
        # Mild L2 shrinkage toward equal weights
        reg = (shrink_n / max(1.0, float(len(rows)))) * float(np.sum((w - eq_w) ** 2))
        return mse + reg

    res = optimize.minimize(
        loss,
        eq_w,
        method="SLSQP",
        bounds=[(0.05, 0.85), (0.05, 0.85), (0.05, 0.85)],
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    if not res.success:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    w = np.clip(res.x, 0.05, 0.85)
    w = w / np.sum(w)
    return float(w[0]), float(w[1]), float(w[2])


def _fit_debiased_het_var_params(
    rows: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Fit de-biased sigma^2 = c0 + c1 * inter_model_var via NNLS with shrinkage toward (1.0, 0.5)."""
    if len(rows) < 20:
        return (1.0, 0.5)
    arr = np.asarray(rows, dtype=np.float64)
    v = arr[:, 0]
    r2 = arr[:, 1]
    cap = float(np.percentile(r2, 98))
    r2_w = np.minimum(r2, cap)
    A = np.column_stack([np.ones_like(v), v])
    coef, _ = optimize.nnls(A, r2_w)
    c0 = max(0.25, min(3.5, float(coef[0])))
    c1 = max(0.0, min(2.5, float(coef[1])))
    n = float(len(rows))
    c0_s = (n * c0 + 30.0 * 1.0) / (n + 30.0)
    c1_s = (n * c1 + 30.0 * 0.5) / (n + 30.0)
    return c0_s, c1_s


def _compute_station_ewma_bias(
    station_hist: Sequence[tuple[date, float, float]],
    prior_bias_c: float,
    current_date: date,
    window_days: int = 60,
    half_life_days: float = 14.0,
) -> tuple[float, float]:
    """Compute EWMA station bias shrunk toward global prior bias, decoupled from scale."""
    s_valid: list[tuple[date, float]] = []
    for d, err, _ in reversed(station_hist):
        age = (current_date - d).days
        if age > window_days:
            break
        if age > 0:
            s_valid.append((d, err))
    if not s_valid:
        return prior_bias_c, 0.0
    weights = [0.5 ** (((current_date - d).days - 1) / half_life_days) for d, _ in s_valid]
    w_sum = sum(weights)
    raw_bias_c = sum(w * e for w, (_, e) in zip(weights, s_valid)) / max(w_sum, _EPS)
    n_eff = (w_sum * w_sum) / max(sum(w * w for w in weights), _EPS)
    bias_c = (n_eff * raw_bias_c + DEFAULT_BIAS_SHRINKAGE_DAYS * prior_bias_c) / (
        n_eff + DEFAULT_BIAS_SHRINKAGE_DAYS
    )
    return bias_c, n_eff


def _solve_weighted_t_mle_sigma_mult(
    z_vals: Sequence[float],
    weights: Sequence[float],
    nu: float,
    n_iters: int = 6,
) -> float:
    """Solve 1D Student-t(nu) weighted MLE for variance-matched sigma multiplier via IRLS."""
    if not z_vals:
        return 1.0
    w_sum = sum(weights)
    if w_sum <= _EPS:
        return 1.0
    var_to_scale = math.sqrt((nu - 2.0) / nu) if nu > 2.0 else 1.0
    s_t = var_to_scale  # initial guess sigma_mult = 1.0
    for _ in range(n_iters):
        s2 = max(0.04, s_t * s_t)
        num = 0.0
        for z, w in zip(z_vals, weights):
            u = (nu + 1.0) / (nu + (z * z) / s2)
            num += w * u * z * z
        s_t = math.sqrt(max(0.04, num / w_sum))
    return s_t / max(var_to_scale, _EPS)


def _compute_global_t_mle_prior(
    global_z_hist: Sequence[tuple[date, float]],
    current_date: date,
    nu: float,
    window_days: int = 60,
    half_life_days: float = 21.0,
) -> float:
    """Compute global Student-t MLE multiplier shrunk toward 1.0 once per (date, pol, nu)."""
    z_vals: list[float] = []
    weights: list[float] = []
    for d, z in reversed(global_z_hist):
        age = (current_date - d).days
        if age > window_days:
            break
        if age > 0:
            z_vals.append(z)
            weights.append(0.5 ** ((age - 1) / half_life_days))
    if len(z_vals) < 15:
        return 1.0
    raw_mult = _solve_weighted_t_mle_sigma_mult(z_vals, weights, nu=nu)
    w_sum = sum(weights)
    n_eff = (w_sum * w_sum) / max(sum(w * w for w in weights), _EPS)
    return max(0.75, min(1.35, (n_eff * raw_mult + 30.0 * 1.0) / (n_eff + 30.0)))


def _compute_station_ewma_t_mle_scale(
    station_z_hist: Sequence[tuple[date, float]],
    global_z_prior: float,
    current_date: date,
    nu: float,
    window_days: int = 60,
    half_life_days: float = 21.0,
    prior_strength: float = 10.0,
) -> float:
    """Per-station EWMA Student-t(nu) MLE scale multiplier shrunk toward global_z_prior (~1.0)."""
    z_vals: list[float] = []
    weights: list[float] = []
    for d, z in reversed(station_z_hist):
        age = (current_date - d).days
        if age > window_days:
            break
        if age > 0:
            z_vals.append(z)
            weights.append(0.5 ** ((age - 1) / half_life_days))
    if len(z_vals) < 3:
        return global_z_prior
    raw_mult = _solve_weighted_t_mle_sigma_mult(z_vals, weights, nu=nu)
    w_sum = sum(weights)
    n_eff = (w_sum * w_sum) / max(sum(w * w for w in weights), _EPS)
    shrunk = (n_eff * raw_mult + prior_strength * global_z_prior) / (n_eff + prior_strength)
    return max(0.55, min(2.0, shrunk))


def _compute_global_rms_prior(
    global_z_hist: Sequence[tuple[date, float]],
    current_date: date,
    window_days: int = 60,
    half_life_days: float = 21.0,
) -> float:
    """Compute global EWMA RMS multiplier shrunk toward 1.0."""
    z_vals: list[float] = []
    weights: list[float] = []
    for d, z in reversed(global_z_hist):
        age = (current_date - d).days
        if age > window_days:
            break
        if age > 0:
            z_vals.append(z)
            weights.append(0.5 ** ((age - 1) / half_life_days))
    if len(z_vals) < 15:
        return 1.0
    w_sum = sum(weights)
    raw_rms = math.sqrt(sum(w * z * z for w, z in zip(weights, z_vals)) / max(w_sum, _EPS))
    n_eff = (w_sum * w_sum) / max(sum(w * w for w in weights), _EPS)
    return max(0.75, min(1.35, (n_eff * raw_rms + 30.0 * 1.0) / (n_eff + 30.0)))


def _compute_station_ewma_rms_scale(
    station_z_hist: Sequence[tuple[date, float]],
    global_z_prior: float,
    current_date: date,
    window_days: int = 60,
    half_life_days: float = 21.0,
    prior_strength: float = 10.0,
) -> float:
    """Per-station EWMA RMS scale multiplier shrunk toward global_z_prior (~1.0)."""
    z_vals: list[float] = []
    weights: list[float] = []
    for d, z in reversed(station_z_hist):
        age = (current_date - d).days
        if age > window_days:
            break
        if age > 0:
            z_vals.append(z)
            weights.append(0.5 ** ((age - 1) / half_life_days))
    if len(z_vals) < 3:
        return global_z_prior
    w_sum = sum(weights)
    raw_rms = math.sqrt(sum(w * z * z for w, z in zip(weights, z_vals)) / max(w_sum, _EPS))
    n_eff = (w_sum * w_sum) / max(sum(w * w for w in weights), _EPS)
    shrunk = (n_eff * raw_rms + prior_strength * global_z_prior) / (n_eff + prior_strength)
    return max(0.55, min(2.0, shrunk))


def _fit_heteroscedastic_var_params(
    rows: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Fit sigma^2 = c0 + c1 * inter_model_var via non-negative least squares with shrinkage."""
    if len(rows) < 20:
        return (1.2, 0.8)
    arr = np.asarray(rows, dtype=np.float64)
    v = arr[:, 0]  # inter-model variance
    r2 = arr[:, 1]  # squared error
    # Robust Winsorization at 98th percentile so single storm outliers don't blow up variance
    cap = float(np.percentile(r2, 98))
    r2_w = np.minimum(r2, cap)
    A = np.column_stack([np.ones_like(v), v])
    coef, _ = optimize.nnls(A, r2_w)
    c0 = max(0.35, min(4.0, float(coef[0])))
    c1 = max(0.0, min(3.0, float(coef[1])))
    # Shrink toward prior (1.1, 0.7) for stability
    n = float(len(rows))
    c0_s = (n * c0 + 30.0 * 1.1) / (n + 30.0)
    c1_s = (n * c1 + 30.0 * 0.7) / (n + 30.0)
    return c0_s, c1_s


def _ridge_vector(s: MosDatasetSample) -> np.ndarray:
    f = s.features
    vals = [float(f.get(k, 0.0)) for k in _RIDGE_FEATURES]
    vals.append(float(f["ecmwf_extreme_c"]) - float(f["gfs_extreme_c"]))
    vals.append(float(f["ecmwf_extreme_c"]) - float(f["icon_extreme_c"]))
    return np.asarray(vals, dtype=np.float64)


def _fit_ridge_model(
    X: np.ndarray,
    y_res: np.ndarray,
    alpha: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu_x = np.mean(X, axis=0)
    sd_x = np.maximum(np.std(X, axis=0), 1e-6)
    Xs = (X - mu_x) / sd_x
    mu_y = float(np.mean(y_res))
    yc = y_res - mu_y
    d = Xs.shape[1]
    coef = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(d), Xs.T @ yc)
    return mu_x, sd_x, coef, mu_y


class _MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.log_losses: list[float] = []
        self.rpss: list[float] = []
        self.briers: list[float] = []
        self.crpss: list[float] = []
        self.cov_hits: int = 0
        self.abs_errs: list[float] = []
        self.bias_errs: list[float] = []
        self.tail_sq_errs: list[float] = []
        self.tail_bins: dict[str, list[tuple[float, float]]] = {
            "1-5%": [],
            "5-15%": [],
            "85-99%": [],
        }
        self.pit_mids: list[float] = []
        self.pit_hist_10: list[float] = [0.0] * 10

    def add(
        self,
        *,
        probs: Sequence[float],
        win_idx: int,
        actual_c: float,
        p10_c: float,
        p50_c: float,
        p90_c: float,
        crps_val: float,
    ) -> None:
        self.n += 1
        clipped = [max(_PROB_CLIP_LL, min(1.0 - _PROB_CLIP_LL, float(p))) for p in probs]
        tot = sum(clipped)
        p_norm = [x / tot for x in clipped]

        self.log_losses.append(-math.log(p_norm[win_idx]))
        self.rpss.append(compute_rps(p_norm, win_idx))
        self.briers.append(
            sum((p_i - (1.0 if i == win_idx else 0.0)) ** 2 for i, p_i in enumerate(p_norm))
        )
        self.crpss.append(crps_val)
        if p10_c - 1e-9 <= actual_c <= p90_c + 1e-9:
            self.cov_hits += 1
        err = actual_c - p50_c
        self.abs_errs.append(abs(err))
        self.bias_errs.append(err)

        cdf_lo = sum(p_norm[:win_idx])
        p_win = p_norm[win_idx]
        cdf_hi = min(1.0, cdf_lo + p_win)
        mid_pit = cdf_lo + 0.5 * p_win
        self.pit_mids.append(mid_pit)
        width = cdf_hi - cdf_lo
        if width > _EPS:
            for b_idx in range(10):
                b_lo = b_idx / 10.0
                b_hi = (b_idx + 1) / 10.0
                overlap = max(0.0, min(cdf_hi, b_hi) - max(cdf_lo, b_lo))
                if overlap > 0.0:
                    self.pit_hist_10[b_idx] += overlap / width
        else:
            b_idx = min(9, max(0, int(mid_pit * 10.0)))
            self.pit_hist_10[b_idx] += 1.0

        for i, p_i in enumerate(probs):
            y_i = 1.0 if i == win_idx else 0.0
            if 0.01 <= p_i < 0.05:
                self.tail_sq_errs.append((p_i - y_i) ** 2)
                self.tail_bins["1-5%"].append((p_i, y_i))
            elif 0.05 <= p_i < 0.15:
                self.tail_sq_errs.append((p_i - y_i) ** 2)
                self.tail_bins["5-15%"].append((p_i, y_i))
            elif 0.85 < p_i <= 0.99:
                self.tail_sq_errs.append((p_i - y_i) ** 2)
                self.tail_bins["85-99%"].append((p_i, y_i))

    def summarize(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0}
        tail_cnt = sum(len(v) for v in self.tail_bins.values())
        tail_ece = None
        tail_bins_detail: dict[str, dict[str, Any]] = {}
        if tail_cnt > 0:
            w_err = 0.0
            for lbl in ("1-5%", "5-15%", "85-99%"):
                v = self.tail_bins[lbl]
                if not v:
                    continue
                mp = sum(x[0] for x in v) / len(v)
                my = sum(x[1] for x in v) / len(v)
                err = abs(mp - my)
                w_err += len(v) * err
                tail_bins_detail[lbl] = {
                    "n": len(v),
                    "mean_p": round(mp, 5),
                    "emp_y": round(my, 5),
                    "err": round(err, 5),
                }
            tail_ece = round(w_err / tail_cnt, 5)
        pit_mean = sum(self.pit_mids) / self.n
        pit_std = math.sqrt(sum((u - pit_mean) ** 2 for u in self.pit_mids) / self.n)
        pit_ratios = [round((x / self.n) * 10.0, 3) for x in self.pit_hist_10]
        return {
            "n": self.n,
            "log_loss": round(sum(self.log_losses) / self.n, 4),
            "rps": round(sum(self.rpss) / self.n, 5),
            "brier": round(sum(self.briers) / self.n, 4),
            "crps": round(sum(self.crpss) / self.n, 4),
            "cov_p10_p90": round(self.cov_hits / self.n, 4),
            "tail_ece": tail_ece,
            "tail_n": tail_cnt,
            "tail_bins_detail": tail_bins_detail,
            "pit_mean": round(pit_mean, 3),
            "pit_std": round(pit_std, 3),
            "pit_hist_10": pit_ratios,
            "mae_c": round(sum(self.abs_errs) / self.n, 3),
            "bias_c": round(sum(self.bias_errs) / self.n, 3),
        }


def run_benchmark(
    samples: list[MosDatasetSample],
    *,
    max_date: date,
    warmup_days: int,
) -> dict[str, Any]:
    # Strict OOS quarantine: reject anything > max_date
    clean: list[MosDatasetSample] = []
    for s in samples:
        if s.target_temp_c is None:
            continue
        if s.local_date > max_date:
            continue
        clean.append(s)

    if not clean:
        raise RuntimeError(f"No labeled samples found with local_date <= {max_date.isoformat()}")

    assert max(s.local_date for s in clean) <= max_date
    min_date = min(s.local_date for s in clean)
    score_start_date = min_date + timedelta(days=warmup_days)

    models_to_eval = (
        "M0_hist_emos_45d",
        "M2_old_ewma14_t7",
        "M2a_debiased_rms_t7",
        "M2b_tmle_ewma14_nu5",
        "M2b_tmle_ewma14_nu7",
        "M2b_tmle_ewma14_nu10",
        "M2b_tmle_ewma14_nu15",
        "M2b_tmle_ewma21_nu7",
        "M2c_debiased_rms_norm",
    )

    acc_all: dict[str, _MetricAccumulator] = {m: _MetricAccumulator() for m in models_to_eval}
    acc_by_pol: dict[str, dict[str, _MetricAccumulator]] = {
        pol: {m: _MetricAccumulator() for m in models_to_eval}
        for pol in ("HIGHEST", "LOWEST")
    }
    acc_by_season: dict[str, dict[str, _MetricAccumulator]] = {
        sec: {m: _MetricAccumulator() for m in models_to_eval}
        for sec in ("DJF", "MAM", "JJA", "SON")
    }
    acc_by_region: dict[str, dict[str, _MetricAccumulator]] = {
        reg: {m: _MetricAccumulator() for m in models_to_eval}
        for reg in ("US", "NON_US")
    }

    # Walk-forward state (strictly <= d - 1)
    station_hist_eq: dict[tuple[str, str], list[tuple[date, float, float]]] = defaultdict(list)
    global_hist_eq: dict[str, list[tuple[date, float, float]]] = defaultdict(list)

    # Old uncalibrated M2 state (for apples-to-apples before/after comparison)
    reg_nwp_rows: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    reg_var_rows: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    station_hist_w: dict[tuple[str, str], list[tuple[date, float, float]]] = defaultdict(list)
    global_hist_w: dict[str, list[tuple[date, float, float]]] = defaultdict(list)

    # Calibrated de-biased heteroscedastic + standardized residual history
    reg_debiased_var_rows: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    station_z_hist_14: dict[tuple[str, str], list[tuple[date, float]]] = defaultdict(list)
    global_z_hist_14: dict[str, list[tuple[date, float]]] = defaultdict(list)
    station_z_hist_21: dict[tuple[str, str], list[tuple[date, float]]] = defaultdict(list)
    global_z_hist_21: dict[str, list[tuple[date, float]]] = defaultdict(list)

    reg_weights_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
    reg_var_cache: dict[tuple[str, str], tuple[float, float]] = {}
    reg_debiased_var_cache: dict[tuple[str, str], tuple[float, float]] = {}
    last_reg_fit_day: date | None = None

    by_date: dict[date, list[MosDatasetSample]] = defaultdict(list)
    for s in clean:
        by_date[s.local_date].append(s)

    station_cache: dict[str, Station] = {}
    for icao in {s.station_icao for s in clean}:
        st = get_station(icao)
        if st is not None:
            station_cache[icao] = st

    sorted_dates = sorted(by_date)
    for cur_date in sorted_dates:
        day_samples = by_date[cur_date]

        # Refit regional NWP weights & raw/debiased heteroscedastic variance weekly on strictly prior days (< cur_date)
        if last_reg_fit_day is None or (cur_date - last_reg_fit_day).days >= 7:
            last_reg_fit_day = cur_date
            for reg in ("US", "EU", "ASIA_OTHER"):
                for pol in ("HIGHEST", "LOWEST"):
                    r_rows = reg_nwp_rows[(reg, pol)][-3000:]
                    reg_weights_cache[(reg, pol)] = _fit_simplex_nwp_weights(r_rows)
                    v_rows = reg_var_rows[(reg, pol)][-3000:]
                    reg_var_cache[(reg, pol)] = _fit_heteroscedastic_var_params(v_rows)
                    dv_rows = reg_debiased_var_rows[(reg, pol)][-3000:]
                    reg_debiased_var_cache[(reg, pol)] = _fit_debiased_het_var_params(dv_rows)

        # Compute global hierarchical priors once per (cur_date, pol)
        prior_eq_45 = {
            pol: _compute_global_prior_cached(global_hist_eq[pol], cur_date, window_days=45)
            for pol in ("HIGHEST", "LOWEST")
        }
        prior_w_60 = {
            pol: _compute_global_prior_cached(global_hist_w[pol], cur_date, window_days=60)
            for pol in ("HIGHEST", "LOWEST")
        }
        prior_rms_14 = {
            pol: _compute_global_rms_prior(global_z_hist_14[pol], cur_date, window_days=60, half_life_days=21.0)
            for pol in ("HIGHEST", "LOWEST")
        }
        prior_tmle_14: dict[tuple[str, int], float] = {
            (pol, nu): _compute_global_t_mle_prior(
                global_z_hist_14[pol], cur_date, nu=float(nu), window_days=60, half_life_days=21.0
            )
            for pol in ("HIGHEST", "LOWEST")
            for nu in (5, 7, 10, 15)
        }
        prior_tmle_21_7 = {
            pol: _compute_global_t_mle_prior(
                global_z_hist_21[pol], cur_date, nu=7.0, window_days=60, half_life_days=21.0
            )
            for pol in ("HIGHEST", "LOWEST")
        }

        day_updates_eq: list[tuple[str, str, date, float, float]] = []
        day_updates_w: list[tuple[str, str, str, date, float, float, float, float, float, float, float, float, float]] = []

        for s in day_samples:
            st = station_cache.get(s.station_icao)
            if st is None or s.target_temp_c is None:
                continue
            pol = s.polarity.value
            actual_c = float(s.target_temp_c)
            e_t = float(s.features["ecmwf_extreme_c"])
            g_t = float(s.features["gfs_extreme_c"])
            i_t = float(s.features["icon_extreme_c"])
            eq_mean = (e_t + g_t + i_t) / 3.0
            eq_var = ((e_t - eq_mean) ** 2 + (g_t - eq_mean) ** 2 + (i_t - eq_mean) ** 2) / 3.0
            eq_std = max(0.35, math.sqrt(eq_var))

            reg = _region_of(st)
            reg_bucket = "US" if reg == "US" else "NON_US"
            season = _season_of(cur_date)

            w_e, w_g, w_i = reg_weights_cache.get((reg, pol), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
            w_mean = w_e * e_t + w_g * g_t + w_i * i_t
            c0_old, c1_old = reg_var_cache.get((reg, pol), (1.1, 0.7))
            het_std_old = max(0.35, math.sqrt(c0_old + c1_old * eq_var))

            c0_deb, c1_deb = reg_debiased_var_cache.get((reg, pol), (1.0, 0.5))
            het_std_deb = max(0.35, math.sqrt(c0_deb + c1_deb * eq_var))

            # Old M0 & M2_old params strictly from < cur_date
            pb_eq, ps_eq = prior_eq_45[pol]
            bias_45, scale_45 = _compute_hierarchical_emos_params(
                station_hist_eq[(s.station_icao, pol)],
                pb_eq,
                ps_eq,
                cur_date,
                window_days=45,
                half_life_days=None,
            )
            mu_emos_45 = eq_mean + bias_45
            sig_emos_45 = eq_std * scale_45

            pb_w60, ps_w60 = prior_w_60[pol]
            w_bias_14_old, w_scale_14_old = _compute_hierarchical_emos_params(
                station_hist_w[(s.station_icao, pol)],
                pb_w60,
                ps_w60,
                cur_date,
                window_days=60,
                half_life_days=14.0,
            )

            # Decoupled EWMA-14 and EWMA-21 station bias (strictly < cur_date)
            bias_14, _ = _compute_station_ewma_bias(
                station_hist_w[(s.station_icao, pol)],
                pb_w60,
                cur_date,
                window_days=60,
                half_life_days=14.0,
            )
            bias_21, _ = _compute_station_ewma_bias(
                station_hist_w[(s.station_icao, pol)],
                pb_w60,
                cur_date,
                window_days=60,
                half_life_days=21.0,
            )
            mu_14 = w_mean + bias_14
            mu_21 = w_mean + bias_21

            # Standardized de-biased residuals from strictly prior-day bias
            deb_res_14 = actual_c - mu_14
            deb_res_21 = actual_c - mu_21
            z_14 = deb_res_14 / het_std_deb
            z_21 = deb_res_21 / het_std_deb

            day_updates_eq.append((s.station_icao, pol, cur_date, actual_c - eq_mean, eq_std))
            day_updates_w.append(
                (s.station_icao, reg, pol, cur_date, e_t, g_t, i_t, actual_c, actual_c - w_mean, het_std_old, deb_res_14, z_14, z_21)
            )

            if cur_date < score_start_date:
                continue

            buckets = _build_polymarket_buckets(
                st,
                mu_emos_45,
                event_id=f"b-{s.station_icao}-{cur_date.isoformat()}-{pol}",
            )
            bounds_c = [bucket_effective_bounds_c(b, remark_tenths_enabled=True) for b in buckets]
            bounds_lo = np.asarray([b[0] for b in bounds_c], dtype=np.float64)
            bounds_hi = np.asarray([b[1] for b in bounds_c], dtype=np.float64)
            no_imp = [False] * len(buckets)
            win_idx = resolve_bucket_index(to_event_unit(actual_c, buckets[0].unit), buckets)

            def _record(m_name: str, probs: list[float], p10: float, p50: float, p90: float, crps_v: float) -> None:
                acc_all[m_name].add(
                    probs=probs, win_idx=win_idx, actual_c=actual_c, p10_c=p10, p50_c=p50, p90_c=p90, crps_val=crps_v
                )
                acc_by_pol[pol][m_name].add(
                    probs=probs, win_idx=win_idx, actual_c=actual_c, p10_c=p10, p50_c=p50, p90_c=p90, crps_val=crps_v
                )
                if pol == "HIGHEST":
                    acc_by_season[season][m_name].add(
                        probs=probs, win_idx=win_idx, actual_c=actual_c, p10_c=p10, p50_c=p50, p90_c=p90, crps_val=crps_v
                    )
                    acc_by_region[reg_bucket][m_name].add(
                        probs=probs, win_idx=win_idx, actual_c=actual_c, p10_c=p10, p50_c=p50, p90_c=p90, crps_val=crps_v
                    )

            # 1) M0_hist_emos_45d (baseline)
            draws_m0 = mu_emos_45 + _Z_GRID_121 * sig_emos_45
            probs_m0 = map_extremes_c_to_probabilities(
                draws_m0.tolist(), buckets, no_imp, remark_tenths_enabled=True, smoothing_alpha=0.001
            )
            _record(
                "M0_hist_emos_45d",
                probs_m0,
                mu_emos_45 - Z_P90 * sig_emos_45,
                mu_emos_45,
                mu_emos_45 + Z_P90 * sig_emos_45,
                _crps_empirical(actual_c, draws_m0),
            )

            # 2) M2_old_ewma14_t7 (previous uncalibrated M2)
            mu_old = w_mean + w_bias_14_old
            sig_old = het_std_old * max(0.80, min(2.0, w_scale_14_old))
            t7_factor = _T_Q90[7.0] * math.sqrt(5.0 / 7.0)
            _record(
                "M2_old_ewma14_t7",
                _cdf_probs_student_t(mu_old, sig_old, 7.0, bounds_lo, bounds_hi),
                mu_old - t7_factor * sig_old,
                mu_old,
                mu_old + t7_factor * sig_old,
                _crps_student_t(actual_c, mu_old, sig_old, 7.0),
            )

            # 3) M2a_debiased_rms_t7 & M2c_debiased_rms_norm
            rms_mult = _compute_station_ewma_rms_scale(
                station_z_hist_14[(s.station_icao, pol)],
                prior_rms_14[pol],
                cur_date,
                window_days=60,
                half_life_days=21.0,
                prior_strength=10.0,
            )
            sig_rms = het_std_deb * rms_mult
            _record(
                "M2a_debiased_rms_t7",
                _cdf_probs_student_t(mu_14, sig_rms, 7.0, bounds_lo, bounds_hi),
                mu_14 - t7_factor * sig_rms,
                mu_14,
                mu_14 + t7_factor * sig_rms,
                _crps_student_t(actual_c, mu_14, sig_rms, 7.0),
            )
            _record(
                "M2c_debiased_rms_norm",
                _cdf_probs_normal(mu_14, sig_rms, bounds_lo, bounds_hi),
                mu_14 - Z_P90 * sig_rms,
                mu_14,
                mu_14 + Z_P90 * sig_rms,
                _crps_normal(actual_c, mu_14, sig_rms),
            )

            # 4) M2b_tmle_ewma14_nu{5,7,10,15}
            for nu in (5, 7, 10, 15):
                tmle_mult = _compute_station_ewma_t_mle_scale(
                    station_z_hist_14[(s.station_icao, pol)],
                    prior_tmle_14[(pol, nu)],
                    cur_date,
                    nu=float(nu),
                    window_days=60,
                    half_life_days=21.0,
                    prior_strength=10.0,
                )
                sig_tmle = het_std_deb * tmle_mult
                t_scale_f = _T_Q90[float(nu)] * math.sqrt((nu - 2.0) / nu)
                _record(
                    f"M2b_tmle_ewma14_nu{nu}",
                    _cdf_probs_student_t(mu_14, sig_tmle, float(nu), bounds_lo, bounds_hi),
                    mu_14 - t_scale_f * sig_tmle,
                    mu_14,
                    mu_14 + t_scale_f * sig_tmle,
                    _crps_student_t(actual_c, mu_14, sig_tmle, float(nu)),
                )

            # 5) M2b_tmle_ewma21_nu7
            tmle_mult_21 = _compute_station_ewma_t_mle_scale(
                station_z_hist_21[(s.station_icao, pol)],
                prior_tmle_21_7[pol],
                cur_date,
                nu=7.0,
                window_days=60,
                half_life_days=21.0,
                prior_strength=10.0,
            )
            sig_tmle_21 = het_std_deb * tmle_mult_21
            _record(
                "M2b_tmle_ewma21_nu7",
                _cdf_probs_student_t(mu_21, sig_tmle_21, 7.0, bounds_lo, bounds_hi),
                mu_21 - t7_factor * sig_tmle_21,
                mu_21,
                mu_21 + t7_factor * sig_tmle_21,
                _crps_student_t(actual_c, mu_21, sig_tmle_21, 7.0),
            )

        # Update walk-forward history strictly AFTER scoring cur_date
        for icao, pol, d, err_eq, std_eq in day_updates_eq:
            station_hist_eq[(icao, pol)].append((d, err_eq, std_eq))
            global_hist_eq[pol].append((d, err_eq, std_eq))
        for icao, reg, pol, d, e_t, g_t, i_t, act_c, err_w, std_w, deb_res_14, z_14, z_21 in day_updates_w:
            reg_nwp_rows[(reg, pol)].append((e_t, g_t, i_t, act_c))
            eq_m = (e_t + g_t + i_t) / 3.0
            im_var = ((e_t - eq_m) ** 2 + (g_t - eq_m) ** 2 + (i_t - eq_m) ** 2) / 3.0
            reg_var_rows[(reg, pol)].append((im_var, err_w * err_w))
            reg_debiased_var_rows[(reg, pol)].append((im_var, deb_res_14 * deb_res_14))
            station_hist_w[(icao, pol)].append((d, err_w, std_w))
            global_hist_w[pol].append((d, err_w, std_w))
            station_z_hist_14[(icao, pol)].append((d, z_14))
            global_z_hist_14[pol].append((d, z_14))
            station_z_hist_21[(icao, pol)].append((d, z_21))
            global_z_hist_21[pol].append((d, z_21))

    return {
        "max_date_quarantine": max_date.isoformat(),
        "score_start_date": score_start_date.isoformat(),
        "score_end_date": sorted_dates[-1].isoformat(),
        "overall": {m: acc_all[m].summarize() for m in models_to_eval},
        "by_polarity": {
            pol: {m: acc_by_pol[pol][m].summarize() for m in models_to_eval}
            for pol in ("HIGHEST", "LOWEST")
        },
        "highest_by_season": {
            sec: {m: acc_by_season[sec][m].summarize() for m in models_to_eval}
            for sec in ("DJF", "MAM", "JJA", "SON")
        },
        "highest_by_region": {
            reg: {m: acc_by_region[reg][m].summarize() for m in models_to_eval}
            for reg in ("US", "NON_US")
        },
        "final_regional_nwp_weights": {
            f"{reg}_{pol}": [round(x, 3) for x in reg_weights_cache.get((reg, pol), (0.333, 0.333, 0.333))]
            for reg in ("US", "EU", "ASIA_OTHER")
            for pol in ("HIGHEST", "LOWEST")
        },
        "final_regional_het_var": {
            f"{reg}_{pol}": [round(x, 3) for x in reg_var_cache.get((reg, pol), (1.1, 0.7))]
            for reg in ("US", "EU", "ASIA_OTHER")
            for pol in ("HIGHEST", "LOWEST")
        },
        "final_regional_debiased_het_var": {
            f"{reg}_{pol}": [round(x, 3) for x in reg_debiased_var_cache.get((reg, pol), (1.0, 0.5))]
            for reg in ("US", "EU", "ASIA_OTHER")
            for pol in ("HIGHEST", "LOWEST")
        },
    }


def _print_table(title: str, data: dict[str, dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    hdr = f"{'Model':23s} | {'N':>5s} | {'LogLoss':>7s} | {'RPS':>7s} | {'Brier':>6s} | {'CRPS':>6s} | {'P10-90':>6s} | {'TailECE':>7s} | {'PIT(m/s)':>11s} | {'MAE':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in data.items():
        if m.get("n", 0) == 0:
            continue
        te = f"{m['tail_ece']*100:6.2f}%" if m.get("tail_ece") is not None else "    n/a"
        pit = f"{m['pit_mean']:.3f}/{m['pit_std']:.3f}"
        print(
            f"{name:23s} | {m['n']:5d} | {m['log_loss']:7.4f} | {m['rps']:7.5f} | "
            f"{m['brier']:6.4f} | {m['crps']:6.4f} | {m['cov_p10_p90']*100:5.1f}% | {te} | {pit:>11s} | {m['mae_c']:5.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward 365d Brain calibration benchmark.")
    parser.add_argument("--input-path", type=str, default="data/ml/mos_history_365d.json")
    parser.add_argument("--output-path", type=str, default="data/ml/brain_benchmark_365d.json")
    parser.add_argument("--max-date", type=str, default="2026-11-24")
    parser.add_argument("--warmup-days", type=int, default=30)
    args = parser.parse_args()

    t0 = time.perf_counter()
    raw = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    samples = [MosDatasetSample.model_validate(item) for item in raw]
    max_d = date.fromisoformat(args.max_date)

    report = run_benchmark(samples, max_date=max_d, warmup_days=args.warmup_days)
    Path(args.output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"365d Walk-Forward Brain Benchmark (quarantine <= {report['max_date_quarantine']}, "
        f"scored {report['score_start_date']} .. {report['score_end_date']}, elapsed {time.perf_counter()-t0:.1f}s)"
    )
    _print_table("OVERALL (HIGHEST + LOWEST)", report["overall"])
    _print_table("HIGHEST ONLY", report["by_polarity"]["HIGHEST"])
    _print_table("LOWEST ONLY", report["by_polarity"]["LOWEST"])
    _print_table("HIGHEST: US (2°F RMK tenths)", report["highest_by_region"]["US"])
    _print_table("HIGHEST: NON-US (1°C)", report["highest_by_region"]["NON_US"])

    print("\n=== HIGHEST BY SEASON (Log-Loss / P10-90 / TailECE) ===")
    for sec in ("DJF", "MAM", "JJA", "SON"):
        s_map = report["highest_by_season"][sec]
        parts = []
        for k, short in (
            ("M0_hist_emos_45d", "M0"),
            ("M2_old_ewma14_t7", "M2_old"),
            ("M2a_debiased_rms_t7", "M2a_rms"),
            ("M2b_tmle_ewma14_nu7", "M2b_t7"),
            ("M2b_tmle_ewma14_nu10", "M2b_t10"),
            ("M2c_debiased_rms_norm", "M2c_N"),
        ):
            m = s_map[k]
            parts.append(
                f"{short}={m['log_loss']:.4f}(cov={m['cov_p10_p90']*100:.1f}%,te={m['tail_ece']*100:.2f}%)"
            )
        print(f"  {sec} (n={s_map['M0_hist_emos_45d']['n']:4d}): " + " | ".join(parts))

    print("\n=== TAIL CALIBRATION BIN BREAKDOWN (OVERALL: mean_p vs emp_y) ===")
    for k in (
        "M0_hist_emos_45d",
        "M2_old_ewma14_t7",
        "M2a_debiased_rms_t7",
        "M2b_tmle_ewma14_nu5",
        "M2b_tmle_ewma14_nu7",
        "M2b_tmle_ewma14_nu10",
        "M2b_tmle_ewma14_nu15",
        "M2c_debiased_rms_norm",
    ):
        m = report["overall"][k]
        det = m.get("tail_bins_detail", {})
        bins_str = " | ".join(
            f"{lbl}: p={d['mean_p']*100:.2f}% y={d['emp_y']*100:.2f}% (n={d['n']})"
            for lbl, d in det.items()
        )
        print(f"  {k:23s} [ECE={m['tail_ece']*100:.2f}%]: {bins_str}")
        print(f"    PIT-10: {m.get('pit_hist_10', [])}")

    print("\n=== LEARNED REGIONAL NWP WEIGHTS & RAW vs DEBIASED HET-VAR (c0, c1) ===")
    for k, w in report["final_regional_nwp_weights"].items():
        hv_old = report["final_regional_het_var"][k]
        hv_deb = report["final_regional_debiased_het_var"][k]
        print(
            f"  {k:18s}: w={w} | old_var={hv_old[0]:.3f}+{hv_old[1]:.3f}*s^2 -> debiased_var={hv_deb[0]:.3f}+{hv_deb[1]:.3f}*s^2"
        )


if __name__ == "__main__":
    main()
