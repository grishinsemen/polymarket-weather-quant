"""Two-contour Experiment Replay evaluation metrics (Brain vs Market & Strategy Execution).

Contour 1 (Brain / Model):
- Evaluates probabilistic forecasts against live archived Polymarket CLOB midpoints on identical
  timestamps using Multi-Class Log-Loss, Market-Relative ΔLogLoss, Ranked Probability Score (RPS),
  Brier Score / Brier Skill Score (BSS), and Tail Calibration (<15c / >85c).
- Separates honest Premarket (pre-METAR) and Decision/Entry-time slices, plus ML-covered station
  subsets and strict out-of-sample (OOS > train_end_utc) filtering.

Contour 2 (Strategy / Execution):
- Separates post-fill Markout / CLV by horizon (5m, 30m, 60m, Pre-Close) with $-volume weighting.
- Computes Realized Log-Growth, peak-to-trough Max Drawdown %, Sortino, Calmar, PnL concentration
  (top-3 share), station stability, and hard pass/fail promotion gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, Field

from weather_quant.domain.models import (
    ClvSnapshot,
    EventProbabilitySnapshot,
    FillRecord,
    OrderBookSnapshot,
    PositionRecord,
    WeatherEvent,
)

_EPS = 1e-9
_PROB_CLIP_EPS = 1e-4
_TAIL_LOW_THRESHOLD = 0.15
_TAIL_HIGH_THRESHOLD = 0.85
_MAX_GATE_DD_PCT = 20.0
_MIN_GATE_MARKOUT = -0.02
_MIN_GATE_FILL_RATE = 0.10


class BrainSliceMetrics(BaseModel):
    """Contour 1 probabilistic accuracy vs Market Baseline on one evaluation slice."""

    label: str
    sample_count: int = 0
    market_sample_count: int = 0
    oos_sample_count: int = 0
    in_sample_excluded_count: int = 0

    model_brier: float | None = None
    model_log_loss: float | None = None
    model_rps: float | None = None
    model_tail_brier: float | None = None
    model_tail_ece: float | None = None
    tail_bucket_count: int = 0

    market_brier: float | None = None
    market_log_loss: float | None = None
    market_rps: float | None = None

    delta_log_loss: float | None = None
    brier_skill_score: float | None = None
    delta_rps: float | None = None

    gate_beats_market: bool | None = None


class MarkoutHorizonSummary(BaseModel):
    """Post-fill Markout / CLV summary for a single horizon (or -1 for pre-settlement)."""

    horizon_minutes: int
    label: str
    sample_count: int = 0
    mean_clv: float | None = None
    volume_weighted_clv: float | None = None


class StationPnlSummary(BaseModel):
    """Per-station realized PnL summary for cross-city stability checks."""

    station_icao: str
    trade_count: int = 0
    net_pnl_usd: float = 0.0


class GateEvaluation(BaseModel):
    """Hard pass/fail gates separating Brain vs Strategy contours."""

    brain_beats_market: bool | None = None
    max_dd_ok: bool = True
    markout_ok: bool | None = None
    fill_rate_ok: bool = True
    overall_pass: bool = False
    reasons: list[str] = Field(default_factory=list)


class StrategyContourMetrics(BaseModel):
    """Contour 2 execution, adverse selection, risk-adjusted growth, and stability diagnostics."""

    markouts: list[MarkoutHorizonSummary] = Field(default_factory=list)
    primary_clv_30m: float | None = None
    volume_weighted_clv: float | None = None
    realized_log_growth: float | None = None
    max_drawdown_pct: float = 0.0
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    calendar_days: int = 0
    low_sample_warning: bool = True
    top3_pnl_share_pct: float | None = None
    profitable_stations: int = 0
    total_traded_stations: int = 0
    best_station: StationPnlSummary | None = None
    worst_station: StationPnlSummary | None = None


class ReplayDiagnostics(BaseModel):
    """Full two-contour evaluation report persisted alongside ExperimentRunSummary."""

    schema_version: int = 1
    primary_brain_slice: Literal["premarket", "decision_entry"] = "premarket"
    is_ml_model: bool = False
    train_end_utc: datetime | None = None
    premarket: BrainSliceMetrics
    decision_entry: BrainSliceMetrics
    ml_covered_premarket: BrainSliceMetrics | None = None
    ml_covered_decision_entry: BrainSliceMetrics | None = None
    strategy: StrategyContourMetrics
    gates: GateEvaluation


@dataclass(frozen=True)
class ForecastEvalObservation:
    """Single point-in-time (model probs, CLOB books) snapshot captured during replay."""

    event: WeatherEvent
    prob_snapshot: EventProbabilitySnapshot
    books: Mapping[str, OrderBookSnapshot]
    captured_at_utc: datetime
    metar_count_local_day: int = 0


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _ordered_buckets(event: WeatherEvent):
    return sorted(event.buckets, key=lambda b: (b.outcome_index, b.bucket_id))


def _clip_and_normalize(probs: Sequence[float], eps: float = _PROB_CLIP_EPS) -> list[float]:
    if not probs:
        return []
    clipped = [max(eps, min(1.0 - eps, float(p))) for p in probs]
    total = sum(clipped)
    if total <= _EPS:
        u = 1.0 / len(clipped)
        return [u for _ in clipped]
    return [x / total for x in clipped]


def extract_market_probs(
    event: WeatherEvent,
    books: Mapping[str, OrderBookSnapshot],
    *,
    max_two_sided_spread: float = 0.25,
    max_one_sided_tail_ask: float = 0.12,
    min_mid_sum: float = 0.75,
    max_mid_sum: float = 1.35,
) -> list[float] | None:
    """Extract normalized market implied probability vector aligned with ordered event buckets.

    Requires every bucket of ``event`` to have a fresh (non-stale) quote so partially loaded
    initial WebSocket frames never assign artificial zero probabilities to missing buckets.
    """
    ordered = _ordered_buckets(event)
    n = len(ordered)
    if n < 2 or len(books) < n:
        return None

    raw_mids: list[float] = []
    two_sided_count = 0

    for bucket in ordered:
        book = books.get(bucket.bucket_id)
        if book is None or book.is_stale:
            return None

        bid = book.best_bid
        ask = book.best_ask
        has_bid = bid is not None and math.isfinite(bid) and bid > 0.0
        has_ask = ask is not None and math.isfinite(ask) and 0.0 < ask < 1.0

        if has_bid and has_ask:
            assert bid is not None and ask is not None
            spread = ask - bid
            if spread > max_two_sided_spread + _EPS:
                return None
            raw_mids.append(0.5 * (bid + ask))
            two_sided_count += 1
        elif has_ask:
            assert ask is not None
            if ask <= max_one_sided_tail_ask + _EPS:
                raw_mids.append(0.5 * ask)
            else:
                return None
        elif has_bid:
            assert bid is not None
            raw_mids.append(bid)
        elif book.midpoint is not None and math.isfinite(book.midpoint) and 0.0 < book.midpoint < 1.0:
            raw_mids.append(float(book.midpoint))
        else:
            return None

    if two_sided_count == 0:
        return None

    total = sum(raw_mids)
    if total < min_mid_sum or total > max_mid_sum:
        return None

    return [m / total for m in raw_mids]


def compute_rps(probs: Sequence[float], true_idx: int) -> float:
    """Compute Ranked Probability Score over ordered outcome buckets (0 = perfect, lower = better)."""
    k = len(probs)
    if k <= 1:
        return 0.0
    cdf_p = 0.0
    total = 0.0
    for m in range(k - 1):
        cdf_p += float(probs[m])
        cdf_y = 1.0 if m >= true_idx else 0.0
        total += (cdf_p - cdf_y) ** 2
    return total / (k - 1)


def compute_brain_slice_metrics(
    *,
    label: str,
    observations: Sequence[ForecastEvalObservation],
    winning_idx_by_event: Mapping[str, int],
    train_end_utc: datetime | None = None,
    allowed_stations: set[str] | None = None,
) -> BrainSliceMetrics:
    """Evaluate Contour 1 probabilistic metrics vs Market Baseline over a set of observations."""
    train_end_date = _ensure_utc(train_end_utc).date() if train_end_utc is not None else None

    filtered: list[ForecastEvalObservation] = []
    in_sample_excluded = 0

    for obs in observations:
        icao = obs.event.station.icao.strip().upper()
        if allowed_stations is not None and icao not in allowed_stations:
            continue
        if obs.event.event_id not in winning_idx_by_event:
            continue
        if train_end_date is not None and obs.event.local_date <= train_end_date:
            in_sample_excluded += 1
            continue
        filtered.append(obs)

    if not filtered:
        return BrainSliceMetrics(
            label=label,
            sample_count=0,
            market_sample_count=0,
            oos_sample_count=0,
            in_sample_excluded_count=in_sample_excluded,
        )

    model_briers: list[float] = []
    model_lls: list[float] = []
    model_rpss: list[float] = []

    mkt_briers: list[float] = []
    mkt_lls: list[float] = []
    mkt_rpss: list[float] = []
    paired_model_briers: list[float] = []
    paired_model_lls: list[float] = []
    paired_model_rpss: list[float] = []

    tail_sq_errors: list[float] = []
    low_tail_pairs: list[tuple[float, float]] = []
    high_tail_pairs: list[tuple[float, float]] = []

    for obs in filtered:
        ordered = _ordered_buckets(obs.event)
        raw_win_idx = winning_idx_by_event[obs.event.event_id]
        if raw_win_idx < 0 or raw_win_idx >= len(obs.event.buckets):
            continue
        win_bucket_id = obs.event.buckets[raw_win_idx].bucket_id
        ordered_win_idx = next(
            (i for i, b in enumerate(ordered) if b.bucket_id == win_bucket_id),
            None,
        )
        if ordered_win_idx is None:
            continue

        prob_map = {b.bucket_id: float(b.fair_probability) for b in obs.prob_snapshot.buckets}
        raw_p = [prob_map.get(b.bucket_id, 0.0) for b in ordered]
        p = _clip_and_normalize(raw_p, eps=_PROB_CLIP_EPS)

        m_brier = sum((p_i - (1.0 if i == ordered_win_idx else 0.0)) ** 2 for i, p_i in enumerate(p))
        m_ll = -math.log(p[ordered_win_idx])
        m_rps = compute_rps(p, ordered_win_idx)

        model_briers.append(m_brier)
        model_lls.append(m_ll)
        model_rpss.append(m_rps)

        for i, p_i in enumerate(raw_p):
            y_i = 1.0 if i == ordered_win_idx else 0.0
            if p_i < _TAIL_LOW_THRESHOLD:
                tail_sq_errors.append((p_i - y_i) ** 2)
                low_tail_pairs.append((p_i, y_i))
            elif p_i > _TAIL_HIGH_THRESHOLD:
                tail_sq_errors.append((p_i - y_i) ** 2)
                high_tail_pairs.append((p_i, y_i))

        raw_q = extract_market_probs(obs.event, obs.books)
        if raw_q is not None and len(raw_q) == len(p):
            q = _clip_and_normalize(raw_q, eps=_PROB_CLIP_EPS)
            q_brier = sum((q_i - (1.0 if i == ordered_win_idx else 0.0)) ** 2 for i, q_i in enumerate(q))
            q_ll = -math.log(q[ordered_win_idx])
            q_rps = compute_rps(q, ordered_win_idx)

            mkt_briers.append(q_brier)
            mkt_lls.append(q_ll)
            mkt_rpss.append(q_rps)
            paired_model_briers.append(m_brier)
            paired_model_lls.append(m_ll)
            paired_model_rpss.append(m_rps)

    n = len(model_briers)
    if n == 0:
        return BrainSliceMetrics(
            label=label,
            sample_count=0,
            market_sample_count=0,
            oos_sample_count=0,
            in_sample_excluded_count=in_sample_excluded,
        )

    avg_m_brier = sum(model_briers) / n
    avg_m_ll = sum(model_lls) / n
    avg_m_rps = sum(model_rpss) / n

    tail_brier = (sum(tail_sq_errors) / len(tail_sq_errors)) if tail_sq_errors else None
    tail_ece: float | None = None
    tail_total = len(low_tail_pairs) + len(high_tail_pairs)
    if tail_total > 0:
        weighted_err = 0.0
        for grp in (low_tail_pairs, high_tail_pairs):
            if not grp:
                continue
            mean_p = sum(x[0] for x in grp) / len(grp)
            mean_y = sum(x[1] for x in grp) / len(grp)
            weighted_err += len(grp) * abs(mean_p - mean_y)
        tail_ece = weighted_err / tail_total

    m_cnt = len(mkt_briers)
    avg_q_brier: float | None = None
    avg_q_ll: float | None = None
    avg_q_rps: float | None = None
    delta_ll: float | None = None
    bss: float | None = None
    delta_rps: float | None = None
    beats_mkt: bool | None = None

    if m_cnt > 0:
        avg_q_brier = sum(mkt_briers) / m_cnt
        avg_q_ll = sum(mkt_lls) / m_cnt
        avg_q_rps = sum(mkt_rpss) / m_cnt

        # Report paired model scores alongside market scores so Δ = market - model holds identically
        avg_m_brier = sum(paired_model_briers) / m_cnt
        avg_m_ll = sum(paired_model_lls) / m_cnt
        avg_m_rps = sum(paired_model_rpss) / m_cnt

        delta_ll = avg_q_ll - avg_m_ll
        bss = (1.0 - (avg_m_brier / avg_q_brier)) if avg_q_brier > _EPS else None
        delta_rps = avg_q_rps - avg_m_rps
        beats_mkt = delta_ll > 0.0

    return BrainSliceMetrics(
        label=label,
        sample_count=n,
        market_sample_count=m_cnt,
        oos_sample_count=n if train_end_date is not None else 0,
        in_sample_excluded_count=in_sample_excluded,
        model_brier=round(avg_m_brier, 6),
        model_log_loss=round(avg_m_ll, 6),
        model_rps=round(avg_m_rps, 6),
        model_tail_brier=round(tail_brier, 6) if tail_brier is not None else None,
        model_tail_ece=round(tail_ece, 6) if tail_ece is not None else None,
        tail_bucket_count=tail_total,
        market_brier=round(avg_q_brier, 6) if avg_q_brier is not None else None,
        market_log_loss=round(avg_q_ll, 6) if avg_q_ll is not None else None,
        market_rps=round(avg_q_rps, 6) if avg_q_rps is not None else None,
        delta_log_loss=round(delta_ll, 6) if delta_ll is not None else None,
        brier_skill_score=round(bss, 6) if bss is not None else None,
        delta_rps=round(delta_rps, 6) if delta_rps is not None else None,
        gate_beats_market=beats_mkt,
    )


def _horizon_label(horizon_minutes: int) -> str:
    if horizon_minutes < 0:
        return "Pre-Close"
    if horizon_minutes >= 60 and horizon_minutes % 60 == 0:
        return f"{horizon_minutes // 60}h"
    return f"{horizon_minutes}m"


def compute_strategy_contour_metrics(
    *,
    clvs: Sequence[ClvSnapshot],
    fills: Sequence[FillRecord],
    positions: Sequence[PositionRecord],
    equity_points: Sequence[tuple[datetime, float]],
    initial_bankroll: float,
    final_equity: float,
    max_drawdown_usd: float,
    net_return_pct: float,
) -> StrategyContourMetrics:
    """Compute Contour 2 execution, adverse selection, risk, and stability diagnostics."""
    fill_usd_by_id = {f.fill_id: max(0.0, float(f.fill_usd)) for f in fills}

    by_horizon: dict[int, list[ClvSnapshot]] = {}
    for c in clvs:
        by_horizon.setdefault(int(c.horizon_minutes), []).append(c)

    # Order: positive minute horizons ascending (5m, 30m, 60m...), then pre-settlement (-1) last
    ordered_horizons = sorted(
        by_horizon.keys(),
        key=lambda h: (1 if h < 0 else 0, h),
    )

    markouts: list[MarkoutHorizonSummary] = []
    by_h_summary: dict[int, MarkoutHorizonSummary] = {}
    for h in ordered_horizons:
        items = by_horizon[h]
        mean_val = sum(c.clv for c in items) / len(items)
        w_sum = sum(fill_usd_by_id.get(c.fill_id, 0.0) for c in items)
        if w_sum > _EPS:
            vw_val = sum(c.clv * fill_usd_by_id.get(c.fill_id, 0.0) for c in items) / w_sum
        else:
            vw_val = mean_val
        summary = MarkoutHorizonSummary(
            horizon_minutes=h,
            label=_horizon_label(h),
            sample_count=len(items),
            mean_clv=round(mean_val, 6),
            volume_weighted_clv=round(vw_val, 6),
        )
        markouts.append(summary)
        by_h_summary[h] = summary

    # Primary markout prefers 30m -> 60m -> 5m -> Pre-Close
    primary_clv_30m: float | None = None
    for preferred_h in (30, 60, 5, -1):
        if preferred_h in by_h_summary:
            primary_clv_30m = by_h_summary[preferred_h].volume_weighted_clv
            break
    if primary_clv_30m is None and markouts:
        primary_clv_30m = markouts[0].volume_weighted_clv

    # One CLV per fill (picking the longest/closing available horizon per fill) weighted by fill_usd
    best_clv_per_fill: dict[str, ClvSnapshot] = {}
    for c in clvs:
        prev = best_clv_per_fill.get(c.fill_id)
        if prev is None:
            best_clv_per_fill[c.fill_id] = c
        else:
            # Prefer pre-settlement (-1), else largest minute horizon
            prev_rank = (1, 0) if prev.horizon_minutes < 0 else (0, prev.horizon_minutes)
            curr_rank = (1, 0) if c.horizon_minutes < 0 else (0, c.horizon_minutes)
            if curr_rank >= prev_rank:
                best_clv_per_fill[c.fill_id] = c

    overall_vw_clv: float | None = None
    if best_clv_per_fill:
        total_w = sum(fill_usd_by_id.get(fid, 0.0) for fid in best_clv_per_fill)
        if total_w > _EPS:
            overall_vw_clv = round(
                sum(snap.clv * fill_usd_by_id.get(fid, 0.0) for fid, snap in best_clv_per_fill.items())
                / total_w,
                6,
            )
        else:
            overall_vw_clv = round(
                sum(snap.clv for snap in best_clv_per_fill.values()) / len(best_clv_per_fill),
                6,
            )

    # Realized log-growth
    log_growth: float | None = None
    if initial_bankroll > _EPS and final_equity > _EPS:
        log_growth = round(math.log(final_equity / initial_bankroll), 6)

    # Peak-to-trough Max Drawdown %
    peak = initial_bankroll
    max_dd_pct = 0.0
    by_day: dict[str, float] = {}
    for ts, eq in equity_points:
        day_key = _ensure_utc(ts).strftime("%Y-%m-%d")
        by_day[day_key] = eq
        if eq > peak:
            peak = eq
        if peak > _EPS:
            dd_pct = ((peak - eq) / peak) * 100.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    if max_dd_pct <= _EPS and initial_bankroll > _EPS and max_drawdown_usd > _EPS:
        max_dd_pct = (max_drawdown_usd / initial_bankroll) * 100.0

    calendar_days = len(by_day)

    # Sortino ratio (annualized daily returns when >= 7 days, else non-zero step returns)
    sortino_ratio: float | None = None
    if len(by_day) >= 7:
        ordered_eq = [by_day[k] for k in sorted(by_day)]
        returns = [
            (ordered_eq[i] - ordered_eq[i - 1]) / ordered_eq[i - 1]
            for i in range(1, len(ordered_eq))
            if ordered_eq[i - 1] > _EPS
        ]
        scale = math.sqrt(365.0)
    else:
        returns = [
            (equity_points[i][1] - equity_points[i - 1][1]) / equity_points[i - 1][1]
            for i in range(1, len(equity_points))
            if equity_points[i - 1][1] > _EPS
            and abs(equity_points[i][1] - equity_points[i - 1][1]) > _EPS
        ]
        scale = math.sqrt(float(len(returns))) if returns else 0.0

    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        downside_sq = [min(0.0, r) ** 2 for r in returns]
        downside_dev = math.sqrt(sum(downside_sq) / len(returns))
        if downside_dev > 1e-12:
            sortino_ratio = round((mean_r / downside_dev) * scale, 4)

    calmar_ratio: float | None = None
    if max_dd_pct > 1e-6:
        calmar_ratio = round(net_return_pct / max_dd_pct, 4)

    # PnL concentration & station stability
    station_pnl: dict[str, float] = {}
    station_trades: dict[str, int] = {}
    for f in fills:
        icao = f.station_icao.strip().upper()
        station_trades[icao] = station_trades.get(icao, 0) + 1
        station_pnl.setdefault(icao, 0.0)

    pos_pnls: list[float] = []
    for p in positions:
        pnl = float(p.realized_pnl_usd) + float(p.unrealized_pnl_usd)
        if abs(pnl) > _EPS or p.status in ("CLOSED", "SETTLED"):
            pos_pnls.append(pnl)
        icao = p.station_icao.strip().upper()
        station_pnl[icao] = station_pnl.get(icao, 0.0) + pnl

    total_pos_pnl = sum(pos_pnls)
    top3_share: float | None = None
    positive_pnls = sorted((x for x in pos_pnls if x > _EPS), reverse=True)
    if positive_pnls and total_pos_pnl > _EPS:
        top3_share = round((sum(positive_pnls[:3]) / total_pos_pnl) * 100.0, 2)

    station_summaries = [
        StationPnlSummary(
            station_icao=icao,
            trade_count=station_trades.get(icao, 0),
            net_pnl_usd=round(pnl, 4),
        )
        for icao, pnl in station_pnl.items()
    ]
    station_summaries.sort(key=lambda s: (-s.net_pnl_usd, s.station_icao))

    profitable_stations = sum(1 for s in station_summaries if s.net_pnl_usd > _EPS)
    best_station = station_summaries[0] if station_summaries else None
    worst_station = station_summaries[-1] if station_summaries else None

    return StrategyContourMetrics(
        markouts=markouts,
        primary_clv_30m=round(primary_clv_30m, 6) if primary_clv_30m is not None else None,
        volume_weighted_clv=overall_vw_clv,
        realized_log_growth=log_growth,
        max_drawdown_pct=round(max_dd_pct, 2),
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        calendar_days=calendar_days,
        low_sample_warning=(calendar_days < 30 or len(fills) < 100),
        top3_pnl_share_pct=top3_share,
        profitable_stations=profitable_stations,
        total_traded_stations=len(station_summaries),
        best_station=best_station,
        worst_station=worst_station,
    )


def evaluate_replay_gates(
    *,
    primary_brain: BrainSliceMetrics,
    strategy: StrategyContourMetrics,
    fill_rate: float,
    trade_count: int,
) -> GateEvaluation:
    """Evaluate Contour 1 & Contour 2 hard pass/fail promotion gates."""
    reasons: list[str] = []

    brain_ok = primary_brain.gate_beats_market
    if brain_ok is False:
        reasons.append("BRAIN_LOSES_TO_MARKET")
    elif brain_ok is None:
        reasons.append("NO_MARKET_BASELINE")

    max_dd_ok = strategy.max_drawdown_pct <= _MAX_GATE_DD_PCT + _EPS
    if not max_dd_ok:
        reasons.append(f"MAX_DD_EXCEEDS_{int(_MAX_GATE_DD_PCT)}PCT")

    markout_ok: bool | None = None
    if strategy.primary_clv_30m is not None:
        markout_ok = strategy.primary_clv_30m >= _MIN_GATE_MARKOUT - _EPS
        if not markout_ok:
            reasons.append("ADVERSE_MARKOUT")

    fill_rate_ok = trade_count > 0 and fill_rate >= _MIN_GATE_FILL_RATE - _EPS
    if not fill_rate_ok:
        reasons.append("LOW_FILL_ACTIVITY")

    overall_pass = (
        brain_ok is True
        and max_dd_ok
        and (markout_ok is not False)
        and fill_rate_ok
    )
    return GateEvaluation(
        brain_beats_market=brain_ok,
        max_dd_ok=max_dd_ok,
        markout_ok=markout_ok,
        fill_rate_ok=fill_rate_ok,
        overall_pass=overall_pass,
        reasons=reasons,
    )
