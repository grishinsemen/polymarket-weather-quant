"""Deep 3-layer Ensemble Probability Engine: Physical Ensemble + Station Bias + Intraday METAR."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Sequence
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from weather_quant.domain.clock import Clock, SystemClock, TemporalIntegrityError, assert_before_decision
from weather_quant.domain.models import (
    Bucket,
    BucketFairProbability,
    EnsembleSnapshot,
    EventProbabilitySnapshot,
    MetarObservation,
    Polarity,
    Station,
    WeatherEvent,
)

if TYPE_CHECKING:
    from weather_quant.storage.database import Database

DEFAULT_MODEL_VERSION = "ensemble-3layer-v1"
DEFAULT_SPREAD_INFLATION = 1.45
DEFAULT_BIAS_SHRINKAGE_DAYS = 4.0
DEFAULT_SCALE_SHRINKAGE_DAYS = 6.0
Z_P90 = 1.2815515655446004


class DailyBiasObservation(BaseModel):
    """Historical forecast vs verified Routine METAR extreme pair for Station Bias MOS."""

    local_date: date
    forecast_extreme_c: float
    observed_extreme_c: float
    forecast_std_c: float | None = None
    observed_at_utc: datetime | None = None
    is_routine: bool = True


def to_event_unit(temp_c: float, unit: Literal["C", "F"]) -> float:
    return temp_c * 9.0 / 5.0 + 32.0 if unit == "F" else temp_c


def round_half_away(value: float) -> int:
    """Round to nearest integer matching aviation METAR whole-degree rounding."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def bucket_effective_bounds(bucket: Bucket) -> tuple[float, float, bool]:
    """Return (low, high, is_integer_grid) continuous interval for a Bucket."""
    bounds = [b for b in (bucket.lower_bound, bucket.upper_bound) if b is not None]
    is_integer = all(float(b).is_integer() for b in bounds)
    if is_integer:
        low = bucket.lower_bound - 0.5 if bucket.lower_bound is not None else -math.inf
        high = bucket.upper_bound + 0.5 if bucket.upper_bound is not None else math.inf
    else:
        low = bucket.lower_bound if bucket.lower_bound is not None else -math.inf
        high = bucket.upper_bound if bucket.upper_bound is not None else math.inf
    return low, high, is_integer


def bucket_distance(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value >= high:
        return value - high
    return 0.0


def map_extremes_to_probabilities(
    extremes_unit: Sequence[float],
    buckets: Sequence[Bucket],
    impossible_mask: Sequence[bool],
    smoothing_alpha: float = 0.001,
) -> list[float]:
    """Map continuous extremes onto Bucket slices, apply calibrated smoothing, normalize to 1.0."""
    n_buckets = len(buckets)
    if n_buckets == 0:
        return []

    bounds = [bucket_effective_bounds(b) for b in buckets]
    possible_indices = [i for i in range(n_buckets) if not impossible_mask[i]]

    if not possible_indices:
        # Degenerate safeguard: concentrate on bucket closest to median extreme
        anchor = sorted(extremes_unit)[len(extremes_unit) // 2] if extremes_unit else 0.0
        best_idx = min(range(n_buckets), key=lambda i: bucket_distance(anchor, bounds[i][0], bounds[i][1]))
        return [1.0 if i == best_idx else 0.0 for i in range(n_buckets)]

    counts = [0.0] * n_buckets
    if extremes_unit:
        for val in extremes_unit:
            matching = [
                i
                for i in possible_indices
                if bounds[i][0] <= val < bounds[i][1]
            ]
            if matching:
                share = 1.0 / len(matching)
                for idx in matching:
                    counts[idx] += share
            else:
                nearest = min(
                    possible_indices,
                    key=lambda i: bucket_distance(val, bounds[i][0], bounds[i][1]),
                )
                counts[nearest] += 1.0
        total_members = float(len(extremes_unit))
    else:
        total_members = 1.0
        uniform = 1.0 / len(possible_indices)
        for idx in possible_indices:
            counts[idx] = uniform

    probs = [0.0] * n_buckets
    for idx in possible_indices:
        probs[idx] = (counts[idx] / total_members) + smoothing_alpha

    total_prob = sum(probs)
    if total_prob <= 0.0:
        uniform = 1.0 / len(possible_indices)
        return [uniform if not impossible_mask[i] else 0.0 for i in range(n_buckets)]

    normalized = [p / total_prob if not impossible_mask[i] else 0.0 for i, p in enumerate(probs)]
    # Fix any IEEE-754 rounding drift on the highest-probability possible bucket
    drift = 1.0 - sum(normalized)
    if drift != 0.0:
        anchor_idx = max(possible_indices, key=lambda i: normalized[i])
        normalized[anchor_idx] = max(0.0, normalized[anchor_idx] + drift)
    return normalized


class EnsembleProbabilityEngine:
    """Deep 3-layer calibrated Fair Probability engine.

    Layer 1: Local-date physical extreme extraction from ~120 ensemble trajectories.
    Layer 2: Rolling 30-45 day Station Bias mean offset and dispersion scaling.
    Layer 3: Intraday Routine METAR impossible-bucket zeroing and residual drift conditioning.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.001,
        bias_window_days: int = 45,
        intraday_decay_hours: float = 6.0,
        obs_noise_c: float = 0.4,
        default_spread_inflation: float = DEFAULT_SPREAD_INFLATION,
    ) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.bias_window_days = bias_window_days
        self.intraday_decay_hours = intraday_decay_hours
        self.obs_noise_c = obs_noise_c
        self.default_spread_inflation = max(1.0, float(default_spread_inflation))

    def _compute_station_bias(
        self,
        recent_bias_obs: Sequence[DailyBiasObservation],
        event_local_date: date,
        decision_time: datetime,
        raw_extremes_c: Sequence[float],
        strict_temporal: bool,
    ) -> tuple[float, float]:
        """Compute rolling EMOS Station Bias (mean offset bias_c, P10-P90 calibrated scale)."""
        valid: list[DailyBiasObservation] = []
        for obs in recent_bias_obs:
            if not obs.is_routine:
                continue
            age_days = (event_local_date - obs.local_date).days
            if age_days <= 0 or age_days > self.bias_window_days:
                continue
            if obs.observed_at_utc is not None:
                try:
                    assert_before_decision(obs.observed_at_utc, decision_time, label="bias_observation")
                except TemporalIntegrityError:
                    if strict_temporal:
                        raise
                    continue
            valid.append(obs)

        if not valid:
            return 0.0, self.default_spread_inflation

        n = len(valid)
        errors = [obs.observed_extreme_c - obs.forecast_extreme_c for obs in valid]
        raw_bias_c = sum(errors) / n
        # Empirical Bayes shrinkage prevents small-sample (1-3 day) noise overfitting
        bias_c = (n / (n + DEFAULT_BIAS_SHRINKAGE_DAYS)) * raw_bias_c

        if n < 2:
            return bias_c, self.default_spread_inflation

        # Bessel-corrected residual std for unbiased small-sample dispersion estimation
        residuals = [e - raw_bias_c for e in errors]
        err_std = math.sqrt(sum(r * r for r in residuals) / max(1, n - 1))
        hist_stds = [
            obs.forecast_std_c
            for obs in valid
            if obs.forecast_std_c is not None and obs.forecast_std_c > 1e-6
        ]
        if hist_stds:
            ref_std = sum(hist_stds) / len(hist_stds)
        elif raw_extremes_c:
            raw_mean = sum(raw_extremes_c) / len(raw_extremes_c)
            ref_std = math.sqrt(sum((x - raw_mean) ** 2 for x in raw_extremes_c) / len(raw_extremes_c))
        else:
            ref_std = 0.0

        if ref_std <= 1e-6:
            return bias_c, self.default_spread_inflation

        scale_var = err_std / ref_std
        scale_cov = scale_var
        if n >= 5:
            # Target ~80% empirical coverage for the P10-P90 interval (|z| <= Z_P90)
            z_scores = sorted(
                abs(r)
                / (
                    obs.forecast_std_c
                    if obs.forecast_std_c is not None and obs.forecast_std_c > 1e-6
                    else ref_std
                )
                for r, obs in zip(residuals, valid)
            )
            q80_pos = 0.80 * (len(z_scores) - 1)
            lo_idx = int(math.floor(q80_pos))
            hi_idx = min(len(z_scores) - 1, lo_idx + 1)
            frac = q80_pos - lo_idx
            q80_z = z_scores[lo_idx] + frac * (z_scores[hi_idx] - z_scores[lo_idx])
            scale_cov = q80_z / Z_P90

        raw_scale = max(scale_var, scale_cov)
        shrunk_scale = (
            n * raw_scale + DEFAULT_SCALE_SHRINKAGE_DAYS * self.default_spread_inflation
        ) / (n + DEFAULT_SCALE_SHRINKAGE_DAYS)
        min_scale = 1.0 if n < 10 else 0.85
        scale = max(min_scale, min(3.0, shrunk_scale))
        return bias_c, scale

    compute_station_bias = _compute_station_bias

    def _filter_valid_intraday_metars(
        self,
        intraday_metars: Sequence[MetarObservation],
        station_icao: str,
        local_date: date,
        station_tz: ZoneInfo,
        decision_time: datetime,
        strict_temporal: bool,
    ) -> list[MetarObservation]:
        valid: list[MetarObservation] = []
        target_icao = station_icao.strip().upper()
        for obs in intraday_metars:
            if not obs.is_routine or obs.raw_report.strip().upper().startswith("SPECI"):
                continue
            if obs.station_icao.strip().upper() != target_icao:
                continue
            try:
                assert_before_decision(obs.observed_at_utc, decision_time, label="intraday_metar")
            except TemporalIntegrityError:
                if strict_temporal:
                    raise
                continue
            obs_utc = (
                obs.observed_at_utc
                if obs.observed_at_utc.tzinfo
                else obs.observed_at_utc.replace(tzinfo=timezone.utc)
            ).astimezone(timezone.utc)
            if obs_utc.astimezone(station_tz).date() != local_date:
                continue
            valid.append(obs)
        valid.sort(key=lambda o: o.observed_at_utc)
        return valid

    @staticmethod
    def obs_settlement_temp_c(obs: MetarObservation, remark_tenths_enabled: bool) -> float:
        if remark_tenths_enabled and obs.has_remark_tenths and obs.remark_tenths_c is not None:
            return float(obs.remark_tenths_c)
        return float(obs.whole_temp_c)

    @staticmethod
    def compute_impossible_mask(
        buckets: Sequence[Bucket],
        polarity: Polarity,
        realized_extreme_unit: float,
    ) -> list[bool]:
        mask: list[bool] = []
        for bucket in buckets:
            _, _, is_integer = bucket_effective_bounds(bucket)
            threshold = float(round_half_away(realized_extreme_unit)) if is_integer else realized_extreme_unit
            if polarity == Polarity.HIGHEST:
                impossible = bucket.upper_bound is not None and bucket.upper_bound < threshold
            else:
                impossible = bucket.lower_bound is not None and bucket.lower_bound > threshold
            mask.append(impossible)
        return mask

    _obs_settlement_temp_c = obs_settlement_temp_c
    _compute_impossible_mask = compute_impossible_mask

    @staticmethod
    def _ensemble_stats_at_time(
        day_timestamps: Sequence[datetime],
        bias_trajectories: Sequence[Sequence[float]],
        target_utc: datetime,
    ) -> tuple[float, float]:
        """Linearly interpolate bias-adjusted ensemble P50 (median) and variance at target_utc."""
        target = (
            target_utc if target_utc.tzinfo else target_utc.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        n_hours = len(day_timestamps)
        n_members = len(bias_trajectories)

        if target <= day_timestamps[0]:
            vals = [bias_trajectories[m][0] for m in range(n_members)]
        elif target >= day_timestamps[-1]:
            vals = [bias_trajectories[m][-1] for m in range(n_members)]
        else:
            hi = next(i for i in range(1, n_hours) if day_timestamps[i] >= target)
            lo = hi - 1
            span = (day_timestamps[hi] - day_timestamps[lo]).total_seconds()
            alpha = (target - day_timestamps[lo]).total_seconds() / span if span > 0 else 0.0
            vals = [
                bias_trajectories[m][lo] + alpha * (bias_trajectories[m][hi] - bias_trajectories[m][lo])
                for m in range(n_members)
            ]

        sorted_vals = sorted(vals)
        mid = n_members // 2
        if n_members % 2 == 1:
            p50_val = sorted_vals[mid]
        else:
            p50_val = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
        mean_val = sum(vals) / n_members
        var_val = sum((v - mean_val) ** 2 for v in vals) / n_members
        return p50_val, var_val

    def compute_intraday_shifted_trajectories(
        self,
        day_timestamps: Sequence[datetime],
        bias_trajectories: Sequence[Sequence[float]],
        latest_obs_utc: datetime,
        latest_temp_c: float,
    ) -> tuple[list[list[float]], float, float]:
        """Apply decaying Intraday Residual Shift ``alpha * (METAR_now - Model_P50(now))``.

        Returns ``(shifted_trajectories, residual_c, alpha)`` where hours at or after
        ``latest_obs_utc`` are shifted by ``alpha * residual_c * exp(-dt / decay_hours)``.
        """
        obs_utc = (
            latest_obs_utc if latest_obs_utc.tzinfo else latest_obs_utc.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        model_p50_c, ens_var_c = self._ensemble_stats_at_time(
            day_timestamps,
            bias_trajectories,
            obs_utc,
        )
        residual_c = latest_temp_c - model_p50_c
        prior_var = max(ens_var_c, 0.25)
        alpha = prior_var / (prior_var + self.obs_noise_c ** 2)

        shifts: list[float] = []
        for ts in day_timestamps:
            ts_utc = (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
            if ts_utc < obs_utc:
                shifts.append(0.0)
            else:
                dt_hours = (ts_utc - obs_utc).total_seconds() / 3600.0
                shifts.append(alpha * residual_c * math.exp(-dt_hours / self.intraday_decay_hours))

        shifted = [
            [traj[idx] + shifts[idx] for idx in range(len(day_timestamps))]
            for traj in bias_trajectories
        ]
        return shifted, residual_c, alpha

    def calibrate_day_trajectories(
        self,
        day_trajectories: Sequence[Sequence[float]],
        *,
        bias_c: float,
        scale: float,
    ) -> list[list[float]]:
        """Apply Layer 2 Station Bias offset and EMOS spread scaling to hourly member trajectories."""
        n_members = len(day_trajectories)
        if n_members == 0:
            return []
        n_hours = len(day_trajectories[0])
        hourly_means = [
            sum(day_trajectories[m][h] for m in range(n_members)) / n_members
            for h in range(n_hours)
        ]
        return [
            [
                hourly_means[h] + bias_c + scale * (day_trajectories[m][h] - hourly_means[h])
                for h in range(n_hours)
            ]
            for m in range(n_members)
        ]

    def _apply_intraday_layer(
        self,
        station: Station,
        event: WeatherEvent,
        day_timestamps: Sequence[datetime],
        bias_trajectories: Sequence[Sequence[float]],
        bias_extremes_c: Sequence[float],
        intraday_metars: Sequence[MetarObservation],
        station_tz: ZoneInfo,
        decision_time: datetime,
        strict_temporal: bool,
    ) -> tuple[list[float], list[bool]]:
        valid_obs = self._filter_valid_intraday_metars(
            intraday_metars=intraday_metars,
            station_icao=station.icao,
            local_date=event.local_date,
            station_tz=station_tz,
            decision_time=decision_time,
            strict_temporal=strict_temporal,
        )
        if not valid_obs:
            no_impossible = [False] * len(event.buckets)
            probs = map_extremes_to_probabilities(
                [to_event_unit(x, event.unit) for x in bias_extremes_c],
                event.buckets,
                no_impossible,
                smoothing_alpha=self.smoothing_alpha,
            )
            return probs, no_impossible

        obs_temps_c = [
            self.obs_settlement_temp_c(obs, event.remark_tenths_enabled)
            for obs in valid_obs
        ]
        realized_extreme_c = (
            max(obs_temps_c) if event.polarity == Polarity.HIGHEST else min(obs_temps_c)
        )
        realized_extreme_unit = to_event_unit(realized_extreme_c, event.unit)
        impossible_mask = self.compute_impossible_mask(
            event.buckets,
            event.polarity,
            realized_extreme_unit,
        )

        latest_obs = valid_obs[-1]
        latest_obs_utc = (
            latest_obs.observed_at_utc
            if latest_obs.observed_at_utc.tzinfo
            else latest_obs.observed_at_utc.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        latest_temp_c = obs_temps_c[-1]
        shifted_trajectories, _, _ = self.compute_intraday_shifted_trajectories(
            day_timestamps,
            bias_trajectories,
            latest_obs_utc,
            latest_temp_c,
        )

        future_indices = [
            idx for idx, ts in enumerate(day_timestamps) if ts > latest_obs_utc
        ]

        intraday_extremes_c: list[float] = []
        for shifted_traj in shifted_trajectories:
            future_vals = [shifted_traj[idx] for idx in future_indices]
            if event.polarity == Polarity.HIGHEST:
                intraday_extremes_c.append(max([realized_extreme_c, *future_vals]))
            else:
                intraday_extremes_c.append(min([realized_extreme_c, *future_vals]))

        intraday_probs = map_extremes_to_probabilities(
            [to_event_unit(x, event.unit) for x in intraday_extremes_c],
            event.buckets,
            impossible_mask,
            smoothing_alpha=self.smoothing_alpha,
        )
        return intraday_probs, impossible_mask

    def compute_probabilities(
        self,
        station: Station,
        event: WeatherEvent,
        ensemble: EnsembleSnapshot,
        recent_bias_obs: Sequence[DailyBiasObservation] = (),
        intraday_metars: Sequence[MetarObservation] = (),
        clock: Clock | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        strict_temporal: bool = False,
    ) -> EventProbabilitySnapshot:
        """Compute normalized 3-layer Bucket Fair Probabilities strictly prior to decision_time."""
        active_clock: Clock = clock or SystemClock()
        decision_time = active_clock.now()
        if decision_time.tzinfo is None:
            decision_time = decision_time.replace(tzinfo=timezone.utc)
        else:
            decision_time = decision_time.astimezone(timezone.utc)

        station_tz = ZoneInfo(station.timezone)
        day_indices = [
            idx
            for idx, ts in enumerate(ensemble.hourly_timestamps_utc)
            if (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(station_tz).date()
            == event.local_date
        ]
        if not day_indices:
            raise ValueError(
                f"EnsembleSnapshot {ensemble.event_id} has no hourly timestamps in local_date "
                f"{event.local_date.isoformat()} for timezone {station.timezone}."
            )

        day_timestamps = [
            (
                ensemble.hourly_timestamps_utc[idx]
                if ensemble.hourly_timestamps_utc[idx].tzinfo
                else ensemble.hourly_timestamps_utc[idx].replace(tzinfo=timezone.utc)
            ).astimezone(timezone.utc)
            for idx in day_indices
        ]
        day_trajectories = [
            [float(traj[idx]) for idx in day_indices]
            for traj in ensemble.member_temps_c
            if len(traj) > max(day_indices)
        ]
        if not day_trajectories:
            raise ValueError(f"EnsembleSnapshot {ensemble.event_id} contains zero valid member trajectories.")

        # Layer 1: Raw physical ensemble daily extremes
        raw_extremes_c = [
            max(traj) if event.polarity == Polarity.HIGHEST else min(traj)
            for traj in day_trajectories
        ]
        raw_extremes_unit = [to_event_unit(x, event.unit) for x in raw_extremes_c]
        no_impossible = [False] * len(event.buckets)
        raw_probs = map_extremes_to_probabilities(
            raw_extremes_unit,
            event.buckets,
            no_impossible,
            smoothing_alpha=self.smoothing_alpha,
        )

        # Layer 2: Rolling Station Bias mean offset & dispersion scaling
        bias_c, scale = self._compute_station_bias(
            recent_bias_obs=recent_bias_obs,
            event_local_date=event.local_date,
            decision_time=decision_time,
            raw_extremes_c=raw_extremes_c,
            strict_temporal=strict_temporal,
        )
        bias_trajectories = self.calibrate_day_trajectories(
            day_trajectories,
            bias_c=bias_c,
            scale=scale,
        )
        bias_extremes_c = [
            max(traj) if event.polarity == Polarity.HIGHEST else min(traj)
            for traj in bias_trajectories
        ]
        bias_extremes_unit = [to_event_unit(x, event.unit) for x in bias_extremes_c]
        bias_probs = map_extremes_to_probabilities(
            bias_extremes_unit,
            event.buckets,
            no_impossible,
            smoothing_alpha=self.smoothing_alpha,
        )

        # Layer 3: Intraday Routine METAR conditioning
        intraday_probs, impossible_mask = self._apply_intraday_layer(
            station=station,
            event=event,
            day_timestamps=day_timestamps,
            bias_trajectories=bias_trajectories,
            bias_extremes_c=bias_extremes_c,
            intraday_metars=intraday_metars,
            station_tz=station_tz,
            decision_time=decision_time,
            strict_temporal=strict_temporal,
        )

        bucket_records = [
            BucketFairProbability(
                event_id=event.event_id,
                bucket_id=bucket.bucket_id,
                label=bucket.label,
                fair_probability=intraday_probs[i],
                raw_ensemble_probability=raw_probs[i],
                bias_adjusted_probability=bias_probs[i],
                intraday_adjusted_probability=intraday_probs[i],
                is_impossible=impossible_mask[i],
                model_version=model_version,
            )
            for i, bucket in enumerate(event.buckets)
        ]

        return EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao=station.icao,
            model_version=model_version,
            computed_at_utc=decision_time,
            buckets=bucket_records,
            probability_sum=sum(b.fair_probability for b in bucket_records),
        )


async def _ensure_model_version_exists(db: Database, model_version: str, now_utc: datetime) -> None:
    """Ensure model_version row exists so tagged predictions satisfy FK constraint."""
    await db.conn.execute(
        """
        INSERT OR IGNORE INTO model_versions (
            model_version, model_type, description, feature_schema_hash,
            config_json, is_active, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_version,
            "ensemble_3layer",
            f"Registered model version {model_version}",
            "baseline-v1",
            json.dumps({}),
            0,
            now_utc.isoformat(),
        ),
    )


async def save_event_probability_snapshot(
    db: Database,
    snapshot: EventProbabilitySnapshot,
) -> None:
    """Persist all Bucket Fair Probability estimates in SQLite tagged with model_version."""
    await _ensure_model_version_exists(db, snapshot.model_version, snapshot.computed_at_utc)
    computed_iso = snapshot.computed_at_utc.isoformat()
    for bucket in snapshot.buckets:
        prediction_id = f"{snapshot.event_id}:{snapshot.model_version}:{computed_iso}:{bucket.bucket_id}"
        await db.conn.execute(
            """
            INSERT INTO predictions (
                prediction_id, event_id, station_icao, bucket_id, fair_probability,
                raw_ensemble_probability, bias_adjusted_probability,
                intraday_adjusted_probability, is_impossible, model_version, computed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                fair_probability = excluded.fair_probability,
                raw_ensemble_probability = excluded.raw_ensemble_probability,
                bias_adjusted_probability = excluded.bias_adjusted_probability,
                intraday_adjusted_probability = excluded.intraday_adjusted_probability,
                is_impossible = excluded.is_impossible
            """,
            (
                prediction_id,
                snapshot.event_id,
                snapshot.station_icao,
                bucket.bucket_id,
                bucket.fair_probability,
                bucket.raw_ensemble_probability,
                bucket.bias_adjusted_probability,
                bucket.intraday_adjusted_probability,
                int(bucket.is_impossible),
                snapshot.model_version,
                computed_iso,
            ),
        )
    await db.conn.commit()


async def _load_bucket_metadata(db: Database) -> dict[tuple[str, str], tuple[str, int]]:
    """Map (event_id, bucket_id) -> (label, outcome_index) from weather_events table."""
    mapping: dict[tuple[str, str], tuple[str, int]] = {}
    for event in await db.list_weather_events(active_only=False):
        for bucket in event.buckets:
            mapping[(event.event_id, bucket.bucket_id)] = (bucket.label, bucket.outcome_index)
    return mapping


def _rows_to_snapshot(
    rows: Sequence[Any],
    bucket_meta: dict[tuple[str, str], tuple[str, int]],
) -> EventProbabilitySnapshot:
    first = rows[0]
    event_id = str(first["event_id"])
    station_icao = str(first["station_icao"])
    model_version = str(first["model_version"])
    computed_at = datetime.fromisoformat(str(first["computed_at_utc"]))

    ordered_rows = sorted(
        rows,
        key=lambda r: bucket_meta.get((event_id, str(r["bucket_id"])), ("", 0))[1],
    )
    buckets = [
        BucketFairProbability(
            event_id=event_id,
            bucket_id=str(row["bucket_id"]),
            label=bucket_meta.get((event_id, str(row["bucket_id"])), (str(row["bucket_id"]), 0))[0],
            fair_probability=float(row["fair_probability"]),
            raw_ensemble_probability=(
                float(row["raw_ensemble_probability"])
                if row["raw_ensemble_probability"] is not None
                else None
            ),
            bias_adjusted_probability=(
                float(row["bias_adjusted_probability"])
                if row["bias_adjusted_probability"] is not None
                else None
            ),
            intraday_adjusted_probability=(
                float(row["intraday_adjusted_probability"])
                if row["intraday_adjusted_probability"] is not None
                else None
            ),
            is_impossible=bool(row["is_impossible"]),
            model_version=model_version,
        )
        for row in ordered_rows
    ]
    return EventProbabilitySnapshot(
        event_id=event_id,
        station_icao=station_icao,
        model_version=model_version,
        computed_at_utc=computed_at,
        buckets=buckets,
        probability_sum=sum(b.fair_probability for b in buckets),
    )


async def get_latest_event_probability_snapshot(
    db: Database,
    event_id: str,
    model_version: str | None = None,
) -> EventProbabilitySnapshot | None:
    """Fetch the latest persisted EventProbabilitySnapshot for an event."""
    if model_version is None:
        active = await db.get_active_model_version()
        target_version = active.model_version
    else:
        target_version = model_version

    async with db.conn.execute(
        """
        SELECT MAX(computed_at_utc) AS latest_ts
        FROM predictions
        WHERE event_id = ? AND model_version = ?
        """,
        (event_id, target_version),
    ) as cursor:
        latest_row = await cursor.fetchone()
    if latest_row is None or latest_row["latest_ts"] is None:
        return None

    latest_ts = str(latest_row["latest_ts"])
    async with db.conn.execute(
        """
        SELECT * FROM predictions
        WHERE event_id = ? AND model_version = ? AND computed_at_utc = ?
        """,
        (event_id, target_version, latest_ts),
    ) as cursor:
        rows = await cursor.fetchall()
    if not rows:
        return None

    bucket_meta = await _load_bucket_metadata(db)
    return _rows_to_snapshot(rows, bucket_meta)


async def list_latest_event_probability_snapshots(
    db: Database,
    model_version: str | None = None,
) -> list[EventProbabilitySnapshot]:
    """Fetch the latest persisted EventProbabilitySnapshot for every predicted event."""
    if model_version is None:
        active = await db.get_active_model_version()
        target_version = active.model_version
    else:
        target_version = model_version

    async with db.conn.execute(
        """
        SELECT p.*
        FROM predictions p
        INNER JOIN (
            SELECT event_id, MAX(computed_at_utc) AS latest_ts
            FROM predictions
            WHERE model_version = ?
            GROUP BY event_id
        ) latest
        ON p.event_id = latest.event_id AND p.computed_at_utc = latest.latest_ts
        WHERE p.model_version = ?
        ORDER BY p.event_id
        """,
        (target_version, target_version),
    ) as cursor:
        rows = await cursor.fetchall()
    if not rows:
        return []

    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row["event_id"]), []).append(row)

    bucket_meta = await _load_bucket_metadata(db)
    return [_rows_to_snapshot(group, bucket_meta) for group in grouped.values()]


async def list_station_routine_metars(
    db: Database,
    station_icao: str,
) -> list[MetarObservation]:
    """Load persisted Routine METAR observations for a Station ordered chronologically."""
    async with db.conn.execute(
        """
        SELECT * FROM metar_observations
        WHERE station_icao = ? AND is_routine = 1
        ORDER BY observed_at_utc ASC
        """,
        (station_icao.strip().upper(),),
    ) as cursor:
        rows = await cursor.fetchall()
    return [db._row_to_metar(row) for row in rows]


def build_daily_bias_observations_from_history(
    *,
    station: Station,
    polarity: Polarity,
    remark_tenths_enabled: bool,
    ensembles: Sequence[EnsembleSnapshot],
    metars: Sequence[MetarObservation],
    before_local_date: date,
    window_days: int = 45,
    decision_time_utc: datetime | None = None,
) -> list[DailyBiasObservation]:
    """Build leak-free historical DailyBiasObservation pairs strictly prior to *before_local_date*.

    When *decision_time_utc* is provided, also excludes the in-progress current local day
    so partial-day Routine METAR observations (e.g. morning readings before the afternoon peak)
    never corrupt Station Bias for tomorrow's Weather Event.
    """
    tz = ZoneInfo(station.timezone)
    target_icao = station.icao.strip().upper()
    decision_utc: datetime | None = None
    current_local_date: date | None = None
    if decision_time_utc is not None:
        decision_utc = (
            decision_time_utc
            if decision_time_utc.tzinfo
            else decision_time_utc.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        current_local_date = decision_utc.astimezone(tz).date()

    metars_by_day: dict[date, list[MetarObservation]] = {}
    for obs in metars:
        if not obs.is_routine or obs.raw_report.strip().upper().startswith("SPECI"):
            continue
        if obs.station_icao.strip().upper() != target_icao:
            continue
        obs_utc = (
            obs.observed_at_utc
            if obs.observed_at_utc.tzinfo
            else obs.observed_at_utc.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        if decision_utc is not None and obs_utc >= decision_utc:
            continue
        obs_day = obs_utc.astimezone(tz).date()
        if current_local_date is not None and obs_day >= current_local_date:
            continue
        age = (before_local_date - obs_day).days
        if age <= 0 or age > window_days:
            continue
        metars_by_day.setdefault(obs_day, []).append(obs)

    if not metars_by_day:
        return []

    # Use the earliest archived ensemble snapshot covering each past day (strict pre-day forecast)
    ens_by_day: dict[date, EnsembleSnapshot] = {}
    sorted_ens = sorted(
        (e for e in ensembles if e.station_icao.strip().upper() == target_icao),
        key=lambda e: (
            e.ingested_at_utc
            if e.ingested_at_utc.tzinfo
            else e.ingested_at_utc.replace(tzinfo=timezone.utc)
        ),
    )
    for snap in sorted_ens:
        if not snap.hourly_timestamps_utc or not snap.member_temps_c:
            continue
        covered_days = {
            (
                ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            ).astimezone(tz).date()
            for ts in snap.hourly_timestamps_utc
        }
        for d in covered_days:
            if d in metars_by_day and d not in ens_by_day:
                ens_by_day[d] = snap

    out: list[DailyBiasObservation] = []
    for d in sorted(ens_by_day):
        snap = ens_by_day[d]
        day_metars = metars_by_day[d]
        if len(day_metars) < 6:
            # Require sufficient hourly Routine METAR coverage before trusting the daily extreme
            continue
        day_indices = [
            idx
            for idx, ts in enumerate(snap.hourly_timestamps_utc)
            if (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(tz).date() == d
        ]
        if not day_indices:
            continue
        extremes = [
            (
                max(float(traj[i]) for i in day_indices)
                if polarity == Polarity.HIGHEST
                else min(float(traj[i]) for i in day_indices)
            )
            for traj in snap.member_temps_c
            if len(traj) > max(day_indices)
        ]
        if not extremes:
            continue
        f_mean = sum(extremes) / len(extremes)
        f_std = math.sqrt(sum((x - f_mean) ** 2 for x in extremes) / len(extremes))
        obs_temps = [
            EnsembleProbabilityEngine.obs_settlement_temp_c(m, remark_tenths_enabled)
            for m in day_metars
        ]
        obs_ext = max(obs_temps) if polarity == Polarity.HIGHEST else min(obs_temps)
        last_obs_utc = max(
            (
                m.observed_at_utc
                if m.observed_at_utc.tzinfo
                else m.observed_at_utc.replace(tzinfo=timezone.utc)
            ).astimezone(timezone.utc)
            for m in day_metars
        )
        out.append(
            DailyBiasObservation(
                local_date=d,
                forecast_extreme_c=f_mean,
                observed_extreme_c=obs_ext,
                forecast_std_c=f_std if f_std > 1e-6 else None,
                observed_at_utc=last_obs_utc,
                is_routine=True,
            )
        )
    return out


