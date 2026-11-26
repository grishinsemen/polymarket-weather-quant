"""Residual & PIT concept-drift detector using two-sample KS test and 14-day CUSUM.

Monitors forecast residuals (``observed - predicted_p50``) and Probability Integral
Transform (PIT) values per Station so seasonal temperature swings never trigger false
drift alerts while NWP model upgrades or sensor bias shifts are caught early.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pydantic import BaseModel
from scipy import stats


class DriftReport(BaseModel):
    """Structured residual and calibration drift diagnostic for a single ICAO Station."""

    station_icao: str
    sample_count: int
    recent_window: int
    baseline_mean_residual_c: float = 0.0
    recent_mean_residual_c: float = 0.0
    mean_shift_c: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    cusum_pos_c: float = 0.0
    cusum_neg_c: float = 0.0
    ks_triggered: bool = False
    cusum_triggered: bool = False
    pit_p_value: float | None = None
    pit_triggered: bool = False
    alert: bool = False
    recommended_sigma_multiplier: float = 1.0


def evaluate_residual_drift(
    station_icao: str,
    residuals_c: Sequence[float],
    *,
    pit_values: Sequence[float] | None = None,
    recent_window: int = 14,
    ks_alpha: float = 0.05,
    mean_shift_threshold_c: float = 0.5,
    cusum_slack_c: float = 0.25,
    cusum_threshold_c: float = 2.5,
    sigma_inflation_on_alert: float = 1.20,
) -> DriftReport:
    """Run two-sample KS + two-sided CUSUM on chronologically ordered forecast residuals."""
    icao = station_icao.strip().upper()
    n = len(residuals_c)
    if n < max(6, recent_window + 4):
        return DriftReport(
            station_icao=icao,
            sample_count=n,
            recent_window=recent_window,
        )

    arr = np.asarray(residuals_c, dtype=np.float64)
    baseline = arr[:-recent_window]
    recent = arr[-recent_window:]

    base_mean = float(np.mean(baseline))
    recent_mean = float(np.mean(recent))
    mean_shift = recent_mean - base_mean

    ks_res = stats.ks_2samp(baseline, recent)
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)
    ks_triggered = (ks_pval < ks_alpha) or (abs(mean_shift) >= mean_shift_threshold_c)

    # Two-sided CUSUM on centered recent residuals
    c_pos = 0.0
    c_neg = 0.0
    max_pos = 0.0
    max_neg = 0.0
    for val in recent:
        centered = float(val) - base_mean
        c_pos = max(0.0, c_pos + centered - cusum_slack_c)
        c_neg = max(0.0, c_neg - centered - cusum_slack_c)
        if c_pos > max_pos:
            max_pos = c_pos
        if c_neg > max_neg:
            max_neg = c_neg
    cusum_triggered = max(max_pos, max_neg) >= cusum_threshold_c

    pit_pval: float | None = None
    pit_triggered = False
    if pit_values is not None and len(pit_values) >= recent_window:
        recent_pit = np.clip(np.asarray(pit_values[-recent_window:], dtype=np.float64), 0.0, 1.0)
        pit_res = stats.kstest(recent_pit, "uniform")
        pit_pval = round(float(pit_res.pvalue), 6)
        pit_triggered = pit_pval < ks_alpha

    alert = ks_triggered or cusum_triggered or pit_triggered
    sigma_mult = float(sigma_inflation_on_alert) if alert else 1.0

    return DriftReport(
        station_icao=icao,
        sample_count=n,
        recent_window=recent_window,
        baseline_mean_residual_c=round(base_mean, 4),
        recent_mean_residual_c=round(recent_mean, 4),
        mean_shift_c=round(mean_shift, 4),
        ks_statistic=round(ks_stat, 6),
        ks_p_value=round(ks_pval, 6),
        cusum_pos_c=round(max_pos, 4),
        cusum_neg_c=round(max_neg, 4),
        ks_triggered=ks_triggered,
        cusum_triggered=cusum_triggered,
        pit_p_value=pit_pval,
        pit_triggered=pit_triggered,
        alert=alert,
        recommended_sigma_multiplier=sigma_mult,
    )
