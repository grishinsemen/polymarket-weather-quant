"""D0 Censored-Max intraday Kalman-Cholesky conditioning for daily temperature extremes.

Models the intraday diurnal trajectory as a Gaussian process conditioned on all
Routine METAR observations recorded so far on the Station's local date, then
simulates remaining routine timestamps via Cholesky decomposition to preserve
the exact point-mass peak-lock probability ``P_lock(t) = P(T_max = M_t | F_t)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from weather_quant.domain.models import Polarity

DEFAULT_CENSORED_MAX_SIMULATIONS = 400
DEFAULT_AR1_PHI = 0.85
DEFAULT_OBS_NOISE_C = 0.25
DEFAULT_SYNOPTIC_SHARE = 0.45
DEFAULT_MIN_PRIOR_STD_C = 0.35


@dataclass(frozen=True)
class CensoredMaxResult:
    """Result of D0 Censored-Max conditional simulation for one Weather Event."""

    simulated_extremes_c: list[float]
    realized_extreme_c: float
    lock_probability: float
    conditional_future_mean_c: list[float]


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _deterministic_seed(seed_key: str) -> int:
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _interp_curve(
    grid_hours: np.ndarray,
    grid_values: np.ndarray,
    target_hours: np.ndarray,
) -> np.ndarray:
    return np.interp(target_hours, grid_hours, grid_values)


def _build_covariance_matrix(
    hours: np.ndarray,
    stds: np.ndarray,
    *,
    phi: float,
    obs_noise_c: float,
    synoptic_share: float,
) -> np.ndarray:
    """Construct joint error covariance: Sigma_ij = syn_i*syn_j + u_i*u_j*phi^|t_i-t_j| + eps^2*delta_ij."""
    syn_scale = np.sqrt(max(0.0, min(1.0, synoptic_share))) * stds
    meso_scale = np.sqrt(max(0.0, 1.0 - max(0.0, min(1.0, synoptic_share)))) * stds
    lag_matrix = np.abs(hours[:, None] - hours[None, :])
    cov = np.outer(syn_scale, syn_scale) + np.outer(meso_scale, meso_scale) * (phi ** lag_matrix)
    cov += (obs_noise_c ** 2) * np.eye(len(hours), dtype=np.float64)
    return cov


def _safe_cholesky(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(sym + 1e-8 * np.eye(sym.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sym)
        clipped = np.maximum(eigvals, 1e-8)
        psd = (eigvecs * clipped) @ eigvecs.T
        return np.linalg.cholesky(0.5 * (psd + psd.T) + 1e-8 * np.eye(sym.shape[0], dtype=np.float64))


def simulate_censored_extreme(
    *,
    day_timestamps: Sequence[datetime],
    prior_mean_curve_c: Sequence[float],
    prior_std_curve_c: Sequence[float],
    obs_points: Sequence[tuple[datetime, float]],
    polarity: Polarity,
    seed_key: str,
    n_simulations: int = DEFAULT_CENSORED_MAX_SIMULATIONS,
    phi: float = DEFAULT_AR1_PHI,
    obs_noise_c: float = DEFAULT_OBS_NOISE_C,
    synoptic_share: float = DEFAULT_SYNOPTIC_SHARE,
    min_prior_std_c: float = DEFAULT_MIN_PRIOR_STD_C,
) -> CensoredMaxResult:
    """Simulate conditional daily extreme T_max = max(M_t, R_t) (or min for LOWEST).

    Uses exact multivariate Gaussian Kalman conditioning across all intraday
    Routine METAR observations ``obs_points`` and Cholesky sampling over the
    remaining timestamps of the local day.
    """
    if not obs_points:
        raise ValueError("obs_points must be non-empty for D0 Censored-Max conditioning.")
    if len(day_timestamps) != len(prior_mean_curve_c) or len(day_timestamps) != len(prior_std_curve_c):
        raise ValueError("day_timestamps, prior_mean_curve_c, and prior_std_curve_c must have equal length.")

    sorted_obs = sorted(((_ensure_utc(ts), float(temp)) for ts, temp in obs_points), key=lambda x: x[0])
    obs_temps = [temp for _, temp in sorted_obs]
    realized_extreme_c = max(obs_temps) if polarity == Polarity.HIGHEST else min(obs_temps)
    latest_obs_utc = sorted_obs[-1][0]

    day_utc = [_ensure_utc(ts) for ts in day_timestamps]
    future_indices = [idx for idx, ts in enumerate(day_utc) if ts > latest_obs_utc]
    if not future_indices:
        return CensoredMaxResult(
            simulated_extremes_c=[realized_extreme_c] * n_simulations,
            realized_extreme_c=realized_extreme_c,
            lock_probability=1.0,
            conditional_future_mean_c=[],
        )

    origin = day_utc[0]
    grid_hours = np.asarray([(ts - origin).total_seconds() / 3600.0 for ts in day_utc], dtype=np.float64)
    grid_means = np.asarray(prior_mean_curve_c, dtype=np.float64)
    grid_stds = np.maximum(np.asarray(prior_std_curve_c, dtype=np.float64), min_prior_std_c)

    obs_hours = np.asarray([(ts - origin).total_seconds() / 3600.0 for ts, _ in sorted_obs], dtype=np.float64)
    future_hours = grid_hours[future_indices]

    mu_o = _interp_curve(grid_hours, grid_means, obs_hours)
    std_o = _interp_curve(grid_hours, grid_stds, obs_hours)
    mu_r = grid_means[future_indices]
    std_r = grid_stds[future_indices]

    all_hours = np.concatenate([obs_hours, future_hours])
    all_stds = np.concatenate([std_o, std_r])
    cov = _build_covariance_matrix(
        all_hours,
        all_stds,
        phi=phi,
        obs_noise_c=obs_noise_c,
        synoptic_share=synoptic_share,
    )

    n_o = len(obs_hours)
    sigma_oo = cov[:n_o, :n_o] + 1e-6 * np.eye(n_o, dtype=np.float64)
    sigma_ro = cov[n_o:, :n_o]
    sigma_rr = cov[n_o:, n_o:]

    innovation = np.asarray(obs_temps, dtype=np.float64) - mu_o
    gain_t = np.linalg.solve(sigma_oo, sigma_ro.T)
    mu_r_given_o = mu_r + gain_t.T @ innovation
    sigma_r_given_o = sigma_rr - sigma_ro @ gain_t

    chol = _safe_cholesky(sigma_r_given_o)
    rng = np.random.default_rng(_deterministic_seed(seed_key))
    draws = rng.standard_normal(size=(n_simulations, len(future_indices)))
    future_paths = mu_r_given_o[None, :] + draws @ chol.T

    if polarity == Polarity.HIGHEST:
        future_ext = np.max(future_paths, axis=1)
        locked = future_ext <= realized_extreme_c
        simulated = np.maximum(realized_extreme_c, future_ext)
    else:
        future_ext = np.min(future_paths, axis=1)
        locked = future_ext >= realized_extreme_c
        simulated = np.minimum(realized_extreme_c, future_ext)

    return CensoredMaxResult(
        simulated_extremes_c=[float(x) for x in simulated],
        realized_extreme_c=realized_extreme_c,
        lock_probability=float(np.mean(locked)),
        conditional_future_mean_c=[float(x) for x in mu_r_given_o],
    )
