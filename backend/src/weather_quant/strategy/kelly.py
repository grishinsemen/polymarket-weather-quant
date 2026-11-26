"""Level 4 execution math: market shrinkage, convex Multi-Outcome Kelly, Baker-McHale & blackouts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-9

# Key UTC publication windows for major NWP cycles (GFS/ECMWF/AIFS/ICON)
_NWP_BLACKOUT_WINDOWS_UTC: tuple[tuple[int, int, int, int], ...] = (
    (4, 0, 4, 15),    # ICON-EU / AIGEFS 00Z
    (5, 15, 5, 30),   # AIFS 00Z / GFS 00Z full
    (7, 45, 8, 5),    # ECMWF IFS ENS 00Z
    (16, 0, 16, 15),  # ICON-EU / AIGEFS 12Z
    (19, 45, 20, 5),  # ECMWF IFS ENS 12Z
)


def shrink_probabilities_to_market(
    model_probs: Sequence[float],
    market_mids: Sequence[float | None],
    spreads: Sequence[float | None],
    *,
    lambda_mkt: float = 0.20,
    max_spread: float = 0.06,
    min_mid_sum: float = 0.95,
    max_mid_sum: float = 1.08,
) -> list[float]:
    """Bayesian shrinkage toward de-vigged two-sided CLOB midpoints.

    Returns unchanged ``model_probs`` if books are incomplete, spreads exceed
    ``max_spread``, or the sum of midpoints falls outside ``[min_mid_sum, max_mid_sum]``.
    """
    n = len(model_probs)
    if n == 0 or lambda_mkt <= 0.0:
        return [float(p) for p in model_probs]

    lam = max(0.0, min(0.85, float(lambda_mkt)))
    if len(market_mids) != n or len(spreads) != n:
        return [float(p) for p in model_probs]

    if any(m is None or s is None or s > max_spread + _EPS or m < 0.0 for m, s in zip(market_mids, spreads)):
        return [float(p) for p in model_probs]

    raw_mids = [float(m) for m in market_mids]  # type: ignore[arg-type]
    mid_sum = sum(raw_mids)
    if mid_sum < min_mid_sum or mid_sum > max_mid_sum or mid_sum <= _EPS:
        return [float(p) for p in model_probs]

    devigged = [m / mid_sum for m in raw_mids]
    blended = [(1.0 - lam) * float(p) + lam * q for p, q in zip(model_probs, devigged)]
    total = sum(blended)
    if total <= _EPS:
        return [float(p) for p in model_probs]
    return [b / total for b in blended]


def baker_mchale_shrinkage(edge: float, model_prob_std: float) -> float:
    """Epistemic uncertainty shrinkage factor ``edge^2 / (edge^2 + sigma_p^2)``."""
    e2 = float(edge) ** 2
    s2 = max(0.0, float(model_prob_std)) ** 2
    if e2 + s2 <= _EPS:
        return 0.0
    return e2 / (e2 + s2)


def solve_multi_outcome_kelly(
    posterior_probs: Sequence[float],
    yes_prices: Sequence[float | None],
    no_prices: Sequence[float | None] | None = None,
    *,
    kelly_fraction: float = 0.20,
    max_total_fraction: float = 0.25,
    model_prob_std: float = 0.03,
) -> tuple[list[float], list[float]]:
    """Solve convex Multi-Outcome Kelly for mutually exclusive YES/NO buckets via SLSQP.

    Returns ``(yes_fractions, no_fractions)`` scaled by ``min(kelly_fraction, k_BM)``
    and capped so ``sum(yes_fractions) + sum(no_fractions) <= max_total_fraction``.
    """
    k = len(posterior_probs)
    if k == 0:
        return [], []

    p = np.clip(np.asarray(posterior_probs, dtype=np.float64), 0.0, 1.0)
    p_sum = float(np.sum(p))
    if p_sum <= _EPS:
        return [0.0] * k, [0.0] * k
    p = p / p_sum

    q_yes = [
        float(q) if (q is not None and 0.01 <= float(q) <= 0.99) else None
        for q in yes_prices
    ]
    q_no = (
        [
            float(q) if (q is not None and 0.01 <= float(q) <= 0.99) else None
            for q in no_prices
        ]
        if no_prices is not None
        else [None] * k
    )

    yes_active = [i for i in range(k) if q_yes[i] is not None and p[i] > q_yes[i] + _EPS]  # type: ignore[operator]
    no_active = [i for i in range(k) if q_no[i] is not None and (1.0 - p[i]) > q_no[i] + _EPS]  # type: ignore[operator]

    if not yes_active and not no_active:
        return [0.0] * k, [0.0] * k

    n_yes = len(yes_active)
    n_no = len(no_active)
    n_vars = n_yes + n_no

    def neg_expected_log_wealth(x: np.ndarray) -> float:
        w = x[:n_yes]
        v = x[n_yes:]
        total_spent = float(np.sum(x))
        if total_spent >= 0.999:
            return 1e6
        wealth = np.full(k, 1.0 - total_spent, dtype=np.float64)
        for idx_pos, bucket_idx in enumerate(yes_active):
            wealth[bucket_idx] += w[idx_pos] / q_yes[bucket_idx]  # type: ignore[operator]
        for idx_pos, bucket_idx in enumerate(no_active):
            payout = v[idx_pos] / q_no[bucket_idx]  # type: ignore[operator]
            wealth += payout
            wealth[bucket_idx] -= payout
        if np.any(wealth <= 1e-9):
            return 1e6
        return -float(np.dot(p, np.log(wealth)))

    x0 = np.full(n_vars, min(0.02, 0.5 / max(1, n_vars)), dtype=np.float64)
    bounds = [(0.0, 0.95) for _ in range(n_vars)]
    constraints = [{"type": "ineq", "fun": lambda x: 0.95 - float(np.sum(x))}]

    res = minimize(
        neg_expected_log_wealth,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-8, "maxiter": 100},
    )

    sol = np.clip(res.x, 0.0, 0.95) if res.success else np.zeros(n_vars, dtype=np.float64)

    yes_out = [0.0] * k
    no_out = [0.0] * k

    for idx_pos, bucket_idx in enumerate(yes_active):
        edge = float(p[bucket_idx] - q_yes[bucket_idx])  # type: ignore[operator]
        mult = min(kelly_fraction, baker_mchale_shrinkage(edge, model_prob_std))
        yes_out[bucket_idx] = float(sol[idx_pos]) * mult

    for idx_pos, bucket_idx in enumerate(no_active):
        edge = float((1.0 - p[bucket_idx]) - q_no[bucket_idx])  # type: ignore[operator]
        mult = min(kelly_fraction, baker_mchale_shrinkage(edge, model_prob_std))
        no_out[bucket_idx] = float(sol[n_yes + idx_pos]) * mult

    total_alloc = sum(yes_out) + sum(no_out)
    if total_alloc > max_total_fraction > 0.0:
        scale = max_total_fraction / total_alloc
        yes_out = [x * scale for x in yes_out]
        no_out = [x * scale for x in no_out]

    return yes_out, no_out


def is_metar_or_nwp_blackout(
    now_utc: datetime,
    station_country: str,
    *,
    metar_blackout_enabled: bool = True,
    nwp_blackout_enabled: bool = True,
) -> tuple[bool, str | None]:
    """Return ``(in_blackout, reason)`` for scheduled METAR release & NWP publication windows."""
    utc = now_utc.replace(tzinfo=timezone.utc) if now_utc.tzinfo is None else now_utc.astimezone(timezone.utc)
    minute = utc.minute
    hour = utc.hour

    if metar_blackout_enabled:
        if station_country.strip().upper() == "US":
            # US routine METARs release around :51-:54 and hit AWC within ~6 min -> blackout :49..:59, :00..:01
            if minute >= 49 or minute <= 1:
                return True, "BLACKOUT_METAR_RELEASE"
        else:
            # International stations (EASA/Europe/Asia) commonly release routine METARs at :20 and :50
            if (18 <= minute <= 27) or (48 <= minute <= 57):
                return True, "BLACKOUT_METAR_RELEASE"

    if nwp_blackout_enabled:
        now_min = hour * 60 + minute
        for sh, sm, eh, em in _NWP_BLACKOUT_WINDOWS_UTC:
            if (sh * 60 + sm) <= now_min <= (eh * 60 + em):
                return True, "BLACKOUT_NWP_RELEASE"

    return False, None
