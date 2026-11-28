"""Analytical Student-t EMOS primitives: regional NWP weights, EWMA bias, t-MLE scale & warm-start.

Validated on the 365-day quarantined walk-forward benchmark (<= 2026-11-24):
- Analytical Student-t(df=7) CDF integration over exact bucket_effective_bounds_c
- Regional simplex NWP weights (ECMWF / GFS / ICON) for US, EU, and ASIA_OTHER
- De-biased heteroscedastic variance sigma^2 = c0 + c1 * inter_model_var
- EWMA-14 station bias + outlier-resistant Student-t weighted MLE station scale shrunk to 1.0
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import special

from weather_quant.domain.clock import TemporalIntegrityError, assert_before_decision
from weather_quant.domain.models import Bucket, EnsembleSnapshot, Polarity, Station

_EPS = 1e-9
_CDF_FLOOR = 1e-6

# Learned strictly on 365-day walk-forward history <= 2026-11-24 (zero OOS leakage into 2026-11-25+)
DEFAULT_REGIONAL_NWP_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "US_HIGHEST": (0.050, 0.401, 0.549),
    "US_LOWEST": (0.219, 0.133, 0.647),
    "EU_HIGHEST": (0.258, 0.148, 0.594),
    "EU_LOWEST": (0.230, 0.050, 0.720),
    "ASIA_OTHER_HIGHEST": (0.443, 0.050, 0.507),
    "ASIA_OTHER_LOWEST": (0.406, 0.185, 0.410),
}

DEFAULT_REGIONAL_DEBIASED_HET_VAR: dict[str, tuple[float, float]] = {
    "US_HIGHEST": (1.227, 0.137),
    "US_LOWEST": (1.247, 0.215),
    "EU_HIGHEST": (0.767, 0.249),
    "EU_LOWEST": (1.082, 0.195),
    "ASIA_OTHER_HIGHEST": (1.157, 0.138),
    "ASIA_OTHER_LOWEST": (1.258, 0.059),
}


def region_of_station(station: Station) -> Literal["US", "EU", "ASIA_OTHER"]:
    """Map ICAO Station country into the 3 regional EMOS calibration pools."""
    c = station.country.strip().upper()
    if c == "US":
        return "US"
    if c in {"FR", "DE", "GB", "NL", "ES", "IT", "AT", "CH", "RU", "TR"}:
        return "EU"
    return "ASIA_OTHER"


def region_polarity_key(station: Station, polarity: Polarity) -> str:
    return f"{region_of_station(station)}_{polarity.value}"


def _extract_model_extremes_c(
    ensemble: EnsembleSnapshot,
    day_indices: Sequence[int],
    polarity: Polarity,
) -> dict[str, float]:
    meta = ensemble.metadata if isinstance(ensemble.metadata, dict) else {}
    mhm = meta.get("model_hourly_means_c")
    model_extremes: dict[str, float] = {}
    if isinstance(mhm, dict) and day_indices:
        max_idx = max(day_indices)
        for m_name, series in mhm.items():
            if isinstance(series, list) and len(series) > max_idx:
                day_vals = [float(series[i]) for i in day_indices]
                model_extremes[str(m_name)] = (
                    max(day_vals) if polarity == Polarity.HIGHEST else min(day_vals)
                )
    return model_extremes


def has_weighted_nwp_means(
    ensemble: EnsembleSnapshot,
    day_indices: Sequence[int],
    polarity: Polarity,
) -> bool:
    """Return True when all 3 core NWP models exist in ``model_hourly_means_c``."""
    model_extremes = _extract_model_extremes_c(ensemble, day_indices, polarity)
    return all(m in model_extremes for m in ("ecmwf_ifs025", "gfs_seamless", "icon_seamless"))


def compute_weighted_nwp_mean_and_var(
    *,
    ensemble: EnsembleSnapshot,
    day_indices: Sequence[int],
    polarity: Polarity,
    raw_extremes_c: Sequence[float],
    weights: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute (weighted_mean_c, inter_model_var_c, raw_mean_c) from per-model hourly means.

    Falls back cleanly to equal-weight ensemble member mean/variance when per-model
    hourly means are unavailable in older archived snapshots.
    """
    raw_mean = sum(raw_extremes_c) / len(raw_extremes_c) if raw_extremes_c else 0.0
    raw_var = (
        sum((x - raw_mean) ** 2 for x in raw_extremes_c) / len(raw_extremes_c)
        if raw_extremes_c
        else 0.25
    )

    model_extremes = _extract_model_extremes_c(ensemble, day_indices, polarity)
    e_val = model_extremes.get("ecmwf_ifs025")
    g_val = model_extremes.get("gfs_seamless")
    i_val = model_extremes.get("icon_seamless")

    if e_val is not None and g_val is not None and i_val is not None:
        w_e, w_g, w_i = weights
        w_sum = w_e + w_g + w_i
        if w_sum > _EPS:
            w_mean = (w_e * e_val + w_g * g_val + w_i * i_val) / w_sum
        else:
            w_mean = (e_val + g_val + i_val) / 3.0
        eq_m = (e_val + g_val + i_val) / 3.0
        im_var = ((e_val - eq_m) ** 2 + (g_val - eq_m) ** 2 + (i_val - eq_m) ** 2) / 3.0
        return w_mean, im_var, raw_mean

    return raw_mean, raw_var, raw_mean


def _obs_forecast_extreme_c(
    obs: Any,
    obs_field: Literal["forecast_extreme_c", "raw_forecast_extreme_c"] = "forecast_extreme_c",
) -> float:
    if obs_field == "raw_forecast_extreme_c":
        raw_val = getattr(obs, "raw_forecast_extreme_c", None)
        if raw_val is not None:
            return float(raw_val)
    return float(obs.forecast_extreme_c)


def solve_weighted_t_mle_location(
    errors: Sequence[float],
    weights: Sequence[float],
    nu: float = 7.0,
    n_iters: int = 8,
) -> float:
    """Solve weighted 1D Student-t(nu) MLE location via IRLS so storm outliers don't pull station bias."""
    if not errors:
        return 0.0
    w_sum = sum(weights)
    if w_sum <= _EPS:
        return 0.0
    mu = sum(w * x for w, x in zip(weights, errors)) / w_sum
    s2 = max(0.09, sum(w * (x - mu) ** 2 for w, x in zip(weights, errors)) / w_sum)
    for _ in range(n_iters):
        wu_sum = 0.0
        wu_x_sum = 0.0
        for x, w in zip(errors, weights):
            diff = x - mu
            u = (nu + 1.0) / (nu + (diff * diff) / s2)
            wu = w * u
            wu_sum += wu
            wu_x_sum += wu * x
        if wu_sum <= _EPS:
            break
        mu = wu_x_sum / wu_sum
        s2_num = 0.0
        for x, w in zip(errors, weights):
            diff = x - mu
            u = (nu + 1.0) / (nu + (diff * diff) / s2)
            s2_num += w * u * diff * diff
        s2 = max(0.09, s2_num / w_sum)
    return mu


def _filter_valid_bias_obs(
    obs_list: Sequence[Any],
    event_local_date: date,
    decision_time: datetime,
    window_days: int,
    strict_temporal: bool,
    label: str,
) -> list[Any]:
    valid: list[Any] = []
    seen_dates: set[date] = set()
    for obs in obs_list:
        if not obs.is_routine:
            continue
        age_days = (event_local_date - obs.local_date).days
        if age_days <= 0 or age_days > window_days:
            continue
        if obs.observed_at_utc is not None:
            try:
                assert_before_decision(obs.observed_at_utc, decision_time, label=label)
            except TemporalIntegrityError:
                if strict_temporal:
                    raise
                continue
        if obs.local_date in seen_dates:
            continue
        seen_dates.add(obs.local_date)
        valid.append(obs)
    return valid


def compute_ewma_station_bias(
    *,
    recent_bias_obs: Sequence[Any],
    global_bias_obs: Sequence[Any],
    event_local_date: date,
    decision_time: datetime,
    window_days: int = 60,
    half_life_days: float = 14.0,
    shrinkage_days: float = 6.0,
    strict_temporal: bool = False,
    use_robust_t_location: bool = False,
    location_nu: float = 7.0,
    obs_field: Literal["forecast_extreme_c", "raw_forecast_extreme_c"] = "forecast_extreme_c",
) -> float:
    """Compute leak-free EWMA station bias shrunk toward global robust median prior."""
    g_valid: list[float] = []
    for obs in global_bias_obs:
        if not obs.is_routine:
            continue
        age = (event_local_date - obs.local_date).days
        if age <= 0 or age > window_days:
            continue
        if obs.observed_at_utc is not None:
            try:
                assert_before_decision(obs.observed_at_utc, decision_time, label="global_bias_observation")
            except TemporalIntegrityError:
                if strict_temporal:
                    raise
                continue
        g_valid.append(float(obs.observed_extreme_c) - _obs_forecast_extreme_c(obs, obs_field))

    if len(g_valid) >= 5:
        g_valid.sort()
        mid = len(g_valid) // 2
        med_g = g_valid[mid] if len(g_valid) % 2 == 1 else 0.5 * (g_valid[mid - 1] + g_valid[mid])
        prior_bias_c = (len(g_valid) / (len(g_valid) + 10.0)) * med_g
    else:
        prior_bias_c = 0.0

    s_valid = _filter_valid_bias_obs(
        recent_bias_obs,
        event_local_date=event_local_date,
        decision_time=decision_time,
        window_days=window_days,
        strict_temporal=strict_temporal,
        label="bias_observation",
    )
    if not s_valid:
        return prior_bias_c

    hl = max(1.0, float(half_life_days))
    weights = [0.5 ** (((event_local_date - obs.local_date).days - 1) / hl) for obs in s_valid]
    errs = [float(obs.observed_extreme_c) - _obs_forecast_extreme_c(obs, obs_field) for obs in s_valid]
    w_sum = sum(weights)
    if use_robust_t_location:
        raw_bias_c = solve_weighted_t_mle_location(errs, weights, nu=location_nu)
    else:
        raw_bias_c = sum(w * e for w, e in zip(weights, errs)) / max(w_sum, _EPS)
    n_eff = (w_sum * w_sum) / max(sum(w * w for w in weights), _EPS)
    return (n_eff * raw_bias_c + shrinkage_days * prior_bias_c) / (n_eff + shrinkage_days)


def solve_weighted_t_mle_sigma_mult(
    z_vals: Sequence[float],
    weights: Sequence[float],
    nu: float = 7.0,
    n_iters: int = 6,
) -> float:
    """Solve 1D weighted Student-t(nu) MLE for variance-matched sigma multiplier via IRLS."""
    if not z_vals:
        return 1.0
    w_sum = sum(weights)
    if w_sum <= _EPS:
        return 1.0
    var_to_scale = math.sqrt((nu - 2.0) / nu) if nu > 2.0 else 1.0
    s_t = var_to_scale
    for _ in range(n_iters):
        s2 = max(0.04, s_t * s_t)
        num = 0.0
        for z, w in zip(z_vals, weights):
            u = (nu + 1.0) / (nu + (z * z) / s2)
            num += w * u * z * z
        s_t = math.sqrt(max(0.04, num / w_sum))
    return s_t / max(var_to_scale, _EPS)


def compute_ewma_t_mle_scale(
    *,
    recent_bias_obs: Sequence[Any],
    global_bias_obs: Sequence[Any],
    station_bias_c: float,
    het_std_fallback_c: float,
    event_local_date: date,
    decision_time: datetime,
    nu: float = 7.0,
    window_days: int = 60,
    half_life_days: float = 21.0,
    prior_strength: float = 10.0,
    strict_temporal: bool = False,
    obs_field: Literal["forecast_extreme_c", "raw_forecast_extreme_c"] = "forecast_extreme_c",
) -> float:
    """Compute per-station EWMA Student-t MLE scale multiplier shrunk toward 1.0."""
    hl = max(1.0, float(half_life_days))
    fallback_std = max(0.35, float(het_std_fallback_c))

    # Global standardized residual prior centered at 1.0
    g_z: list[float] = []
    g_w: list[float] = []
    g_errs: list[tuple[int, float, float]] = []
    for obs in global_bias_obs:
        if not obs.is_routine:
            continue
        age = (event_local_date - obs.local_date).days
        if age <= 0 or age > window_days:
            continue
        if obs.observed_at_utc is not None:
            try:
                assert_before_decision(obs.observed_at_utc, decision_time, label="global_bias_observation")
            except TemporalIntegrityError:
                if strict_temporal:
                    raise
                continue
        ref_s = (
            float(obs.forecast_std_c)
            if (obs.forecast_std_c is not None and obs.forecast_std_c > 0.25)
            else fallback_std
        )
        g_errs.append((age, float(obs.observed_extreme_c) - _obs_forecast_extreme_c(obs, obs_field), ref_s))

    if len(g_errs) >= 15:
        g_med = sorted(e for _, e, _ in g_errs)[len(g_errs) // 2]
        for age, err, ref_s in g_errs:
            g_z.append((err - g_med) / ref_s)
            g_w.append(0.5 ** ((age - 1) / hl))
        g_raw = solve_weighted_t_mle_sigma_mult(g_z, g_w, nu=nu)
        gw_sum = sum(g_w)
        gn_eff = (gw_sum * gw_sum) / max(sum(w * w for w in g_w), _EPS)
        global_prior = max(0.75, min(1.35, (gn_eff * g_raw + 30.0 * 1.0) / (gn_eff + 30.0)))
    else:
        global_prior = 1.0

    s_valid = _filter_valid_bias_obs(
        recent_bias_obs,
        event_local_date=event_local_date,
        decision_time=decision_time,
        window_days=window_days,
        strict_temporal=strict_temporal,
        label="bias_observation",
    )
    if len(s_valid) < 3:
        return global_prior

    s_z: list[float] = []
    s_w: list[float] = []
    for obs in s_valid:
        age = (event_local_date - obs.local_date).days
        ref_s = (
            float(obs.forecast_std_c)
            if (obs.forecast_std_c is not None and obs.forecast_std_c > 0.25)
            else fallback_std
        )
        res = (float(obs.observed_extreme_c) - _obs_forecast_extreme_c(obs, obs_field)) - station_bias_c
        s_z.append(res / ref_s)
        s_w.append(0.5 ** ((age - 1) / hl))

    raw_mult = solve_weighted_t_mle_sigma_mult(s_z, s_w, nu=nu)
    w_sum = sum(s_w)
    n_eff = (w_sum * w_sum) / max(sum(w * w for w in s_w), _EPS)
    shrunk = (n_eff * raw_mult + prior_strength * global_prior) / (n_eff + prior_strength)
    return max(0.55, min(2.0, shrunk))


def map_student_t_cdf_to_probabilities(
    *,
    mu_c: float,
    sigma_c: float,
    nu: float,
    bounds_c: Sequence[tuple[float, float, bool]],
    impossible_mask: Sequence[bool],
    cdf_floor: float = _CDF_FLOOR,
) -> list[float]:
    """Integrate variance-matched Student-t(nu) CDF over exact °C bucket bounds."""
    n_buckets = len(bounds_c)
    if n_buckets == 0:
        return []
    possible_indices = [i for i in range(n_buckets) if not impossible_mask[i]]
    if not possible_indices:
        return [0.0] * n_buckets

    eff_sigma = max(0.20, float(sigma_c))
    scale = eff_sigma * math.sqrt((nu - 2.0) / nu) if nu > 2.0 else eff_sigma

    lo_arr = np.asarray([bounds_c[i][0] for i in possible_indices], dtype=np.float64)
    hi_arr = np.asarray([bounds_c[i][1] for i in possible_indices], dtype=np.float64)
    c_lo = np.where(np.isfinite(lo_arr), special.stdtr(nu, (lo_arr - mu_c) / scale), 0.0)
    c_hi = np.where(np.isfinite(hi_arr), special.stdtr(nu, (hi_arr - mu_c) / scale), 1.0)
    raw = np.maximum(cdf_floor, c_hi - c_lo)
    total = float(np.sum(raw))
    if total <= _EPS:
        u = 1.0 / len(possible_indices)
        return [u if not impossible_mask[i] else 0.0 for i in range(n_buckets)]

    out = [0.0] * n_buckets
    for pos, b_idx in enumerate(possible_indices):
        out[b_idx] = float(raw[pos]) / total
    drift = 1.0 - sum(out)
    if drift != 0.0:
        anchor_idx = max(possible_indices, key=lambda i: out[i])
        out[anchor_idx] = max(0.0, out[anchor_idx] + drift)
    return out


@lru_cache(maxsize=8)
def load_warmstart_bias_cache(path_str: str) -> dict[str, list[dict[str, Any]]]:
    """Load cached pre-OOS DailyBiasObservation dicts keyed by '{ICAO}_{POLARITY}'."""
    p = Path(path_str)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def compute_external_mos_posterior(
    *,
    ensemble: EnsembleSnapshot,
    polarity: Polarity,
    emos_mean_c: float,
    emos_std_c: float | None = None,
    mos_anchor_weight: float = 0.30,
    sigma_shrink_strength: float = 0.0,
) -> tuple[float, float]:
    """Blend external $0 MOS anchors and contract sigma_c smoothly on MOS/EMOS agreement.

    Returns ``(anchor_shift_c, sigma_scale_factor)``.
    """
    if mos_anchor_weight <= 0.0 or not isinstance(ensemble.metadata, dict):
        return 0.0, 1.0
    meta = ensemble.metadata
    suffix = "high_c" if polarity == Polarity.HIGHEST else "low_c"
    prefixes = ("mosmix", "nbm", "nws_point")

    entries: list[tuple[float, float | None]] = []
    for pref in prefixes:
        raw_v = meta.get(f"{pref}_{suffix}")
        if raw_v is None and polarity == Polarity.HIGHEST:
            raw_v = meta.get(f"{pref}_extreme_c")
        if raw_v is None:
            continue
        try:
            val = float(raw_v)
            if not math.isfinite(val):
                continue
        except (TypeError, ValueError):
            continue

        unc_val: float | None = None
        raw_unc = meta.get(f"{pref}_uncertainty_c")
        if raw_unc is not None:
            try:
                u = float(raw_unc)
                if math.isfinite(u) and u > 0.1:
                    unc_val = u
            except (TypeError, ValueError):
                unc_val = None
        entries.append((val, unc_val))

    if not entries:
        return 0.0, 1.0

    eff_emos_std = max(0.35, float(emos_std_c)) if emos_std_c is not None else 1.0
    has_uncertainty = any(u is not None for _, u in entries)
    if not has_uncertainty:
        anchor_mean = sum(v for v, _ in entries) / len(entries)
        eff_weight = mos_anchor_weight
    else:
        emos_var = eff_emos_std * eff_emos_std
        base_w = max(0.05, min(0.55, mos_anchor_weight))
        fallback_mos_var = emos_var * ((1.0 - base_w) / base_w)

        prec_sum = 0.0
        weighted_val_sum = 0.0
        for val, unc in entries:
            # Inflate reported E_TTT by 1.4x to account for shared NWP forecast error correlation
            var_k = ((max(0.35, unc) * 1.4) ** 2) if unc is not None else fallback_mos_var
            prec_k = 1.0 / max(0.05, var_k)
            prec_sum += prec_k
            weighted_val_sum += prec_k * val

        anchor_mean = weighted_val_sum / prec_sum
        bayes_w = prec_sum / (prec_sum + (1.0 / emos_var))
        eff_weight = max(0.10, min(0.45, bayes_w))

    raw_delta = max(-2.5, min(2.5, anchor_mean - emos_mean_c))
    shift_c = eff_weight * raw_delta

    sigma_scale = 1.0
    strength = max(0.0, min(1.0, float(sigma_shrink_strength)))
    if strength > _EPS:
        z_diff = abs(anchor_mean - emos_mean_c) / eff_emos_std
        # Smooth Gaussian agreement gate (<= 1 sigma agrees strongly; large disagreement preserves uncertainty)
        agreement = math.exp(-0.5 * (z_diff / 1.5) ** 2)
        var_reduction = min(0.45, strength * eff_weight * agreement)
        sigma_scale = math.sqrt(max(0.55, 1.0 - var_reduction))
    return shift_c, sigma_scale
