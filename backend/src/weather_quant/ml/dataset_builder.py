"""Leak-free historical forecast + Routine METAR dataset builder for Quantile ML MOS."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol, Sequence
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, Field

from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import Clock, SystemClock, assert_before_decision
from weather_quant.domain.models import (
    EnsembleSnapshot,
    MetarObservation,
    Polarity,
    Station,
)
from weather_quant.metar.parser import parse_metar_line
from weather_quant.probability.engine import DailyBiasObservation

TOP_LIQUIDITY_STATIONS: tuple[str, ...] = (
    "LFPB",
    "RKSI",
    "EDDM",
    "EGLL",
    "KMIA",
    "KATL",
    "RKPK",
    "VILK",
)

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
IEM_ASOS_METAR_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

DEFAULT_FORECAST_MODELS: tuple[str, ...] = (
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
)

HISTORICAL_HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "relative_humidity_2m",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "ecmwf_extreme_c",
    "gfs_extreme_c",
    "icon_extreme_c",
    "ensemble_mean_c",
    "ensemble_std_c",
    "ensemble_min_c",
    "ensemble_max_c",
    "model_spread_c",
    "wind_speed_mean",
    "wind_dir_sin",
    "wind_dir_cos",
    "cloud_cover_mean",
    "relative_humidity_mean",
    "doy_sin",
    "doy_cos",
    "peak_hour_sin",
    "peak_hour_cos",
    "polarity_is_highest",
    "station_lat",
    "station_lon",
    "station_elevation_m",
    "rolling_bias_c",
    "prior_obs_temp_c",
)

FEATURE_SCHEMA_HASH: str = hashlib.sha256(",".join(FEATURE_COLUMNS).encode("utf-8")).hexdigest()[:16]


def is_top_liquidity_station(icao: str) -> bool:
    """Return True if the 4-letter ICAO Station belongs to the top liquidity tier."""
    return icao.strip().upper() in TOP_LIQUIDITY_STATIONS


def get_top_liquidity_stations() -> list[Station]:
    """Return Station catalog objects for the top liquidity ICAO Stations."""
    stations: list[Station] = []
    for icao in TOP_LIQUIDITY_STATIONS:
        station = get_station(icao)
        if station is not None:
            stations.append(station)
    return stations


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _parse_iso_utc(raw_ts: str) -> datetime:
    cleaned = raw_ts.strip().replace(" ", "T")
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return _ensure_utc(dt)


class HistoricalForecastPoint(BaseModel):
    """Single archived hourly forecast point with explicit availability timestamp."""

    station_icao: str
    model: str
    valid_at_utc: datetime
    available_at_utc: datetime
    temperature_2m_c: float
    wind_speed_10m: float | None = None
    wind_direction_10m: float | None = None
    cloud_cover: float | None = None
    relative_humidity_2m: float | None = None


class MosDatasetSample(BaseModel):
    """Leak-free feature row + optional realized Routine METAR extreme target."""

    station_icao: str
    local_date: date
    polarity: Polarity
    decision_time_utc: datetime
    forecast_available_at_utc: datetime
    features: dict[str, float]
    baseline_mean_c: float
    baseline_std_c: float
    baseline_extremes_c: list[float] = Field(default_factory=list)
    target_temp_c: float | None = None
    target_observed_at_utc: datetime | None = None
    remark_tenths_enabled: bool = True


class HistoricalArchiveFetcherPort(Protocol):
    """Async seam for fetching historical archived forecasts and Routine METARs."""

    async def fetch_historical_forecasts(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> list[HistoricalForecastPoint]:
        ...

    async def fetch_historical_routine_metars(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> list[MetarObservation]:
        ...


class HistoricalArchiveFetcher:
    """Async client querying Open-Meteo Historical Forecast API and IEM Routine METAR archives."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        models: Sequence[str] = DEFAULT_FORECAST_MODELS,
        forecast_lead_hours: int = 12,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._external_client = client
        self._clock: Clock = clock or SystemClock()
        self._models = list(models)
        self._forecast_lead_hours = forecast_lead_hours
        self._timeout_seconds = timeout_seconds

    async def _get_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._external_client is not None:
            return self._external_client, False
        return httpx.AsyncClient(timeout=self._timeout_seconds), True

    async def fetch_historical_forecasts(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> list[HistoricalForecastPoint]:
        """Fetch archived NWP hourly forecasts at exact Station runway coordinates."""
        params: dict[str, Any] = {
            "latitude": station.lat,
            "longitude": station.lon,
            "elevation": station.elevation_m,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(HISTORICAL_HOURLY_VARIABLES),
            "models": ",".join(self._models),
            "timezone": "UTC",
        }
        client, owns_client = await self._get_client()
        try:
            resp = await client.get(HISTORICAL_FORECAST_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
        finally:
            if owns_client:
                await client.aclose()

        blocks = payload if isinstance(payload, list) else [payload]
        points: list[HistoricalForecastPoint] = []
        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            model_name = str(
                block.get("model")
                or (self._models[idx] if idx < len(self._models) else "unknown")
            )
            hourly = block.get("hourly")
            if not isinstance(hourly, dict):
                continue
            raw_times = hourly.get("time") or []
            temps = hourly.get("temperature_2m") or []
            winds = hourly.get("wind_speed_10m") or []
            wind_dirs = hourly.get("wind_direction_10m") or []
            clouds = hourly.get("cloud_cover") or []
            rhs = hourly.get("relative_humidity_2m") or []

            for pos, raw_ts in enumerate(raw_times):
                if pos >= len(temps) or temps[pos] is None:
                    continue
                valid_utc = _parse_iso_utc(str(raw_ts))
                # Archived forecast run was issued strictly before valid time (e.g. previous day 12Z)
                run_day = (valid_utc - timedelta(hours=self._forecast_lead_hours)).date()
                available_utc = datetime(run_day.year, run_day.month, run_day.day, 12, 0, tzinfo=timezone.utc)
                if available_utc >= valid_utc:
                    available_utc = valid_utc - timedelta(hours=self._forecast_lead_hours)

                points.append(
                    HistoricalForecastPoint(
                        station_icao=station.icao,
                        model=model_name,
                        valid_at_utc=valid_utc,
                        available_at_utc=available_utc,
                        temperature_2m_c=float(temps[pos]),
                        wind_speed_10m=float(winds[pos]) if pos < len(winds) and winds[pos] is not None else None,
                        wind_direction_10m=(
                            float(wind_dirs[pos]) if pos < len(wind_dirs) and wind_dirs[pos] is not None else None
                        ),
                        cloud_cover=float(clouds[pos]) if pos < len(clouds) and clouds[pos] is not None else None,
                        relative_humidity_2m=float(rhs[pos]) if pos < len(rhs) and rhs[pos] is not None else None,
                    )
                )
        return points

    async def fetch_historical_routine_metars(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> list[MetarObservation]:
        """Fetch historical Routine METAR reports (report_type=3) from Iowa State Mesonet, rejecting SPECI."""
        end_plus = end_date + timedelta(days=1)
        params: dict[str, Any] = {
            "station": station.icao,
            "data": "metar",
            "report_type": "3",
            "year1": start_date.year,
            "month1": start_date.month,
            "day1": start_date.day,
            "year2": end_plus.year,
            "month2": end_plus.month,
            "day2": end_plus.day,
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "no",
            "elev": "no",
            "missing": "M",
            "trace": "T",
            "direct": "no",
        }
        client, owns_client = await self._get_client()
        try:
            resp = await client.get(IEM_ASOS_METAR_URL, params=params)
            resp.raise_for_status()
            raw_text = resp.text
        finally:
            if owns_client:
                await client.aclose()

        ingested_at = self._clock.now()
        observations: list[MetarObservation] = []
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() and not ln.startswith("#")]
        if not lines:
            return observations

        if "," in lines[0] and "metar" in lines[0].lower():
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            for row in reader:
                metar_raw = (row.get("metar") or "").strip()
                if not metar_raw or metar_raw == "M":
                    continue
                valid_raw = (row.get("valid") or "").strip()
                ref_time = _parse_iso_utc(valid_raw) if valid_raw else datetime(
                    end_date.year, end_date.month, end_date.day, 23, 59, tzinfo=timezone.utc
                )
                obs = parse_metar_line(
                    metar_raw,
                    ingested_at_utc=ingested_at,
                    reference_time_utc=ref_time,
                )
                if obs is not None:
                    observations.append(obs)
        else:
            ref_time = datetime(end_date.year, end_date.month, end_date.day, 23, 59, tzinfo=timezone.utc)
            for line in lines:
                obs = parse_metar_line(line, ingested_at_utc=ingested_at, reference_time_utc=ref_time)
                if obs is not None:
                    observations.append(obs)

        observations.sort(key=lambda o: o.observed_at_utc)
        return observations


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _std(values: Sequence[float], mean_val: float) -> float:
    if len(values) < 2:
        return 0.5
    return math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))


def _circular_wind_stats(directions_deg: Sequence[float]) -> tuple[float, float]:
    if not directions_deg:
        return 0.0, 0.0
    sins = [math.sin(math.radians(d)) for d in directions_deg]
    coss = [math.cos(math.radians(d)) for d in directions_deg]
    return sum(sins) / len(sins), sum(coss) / len(coss)


def _compute_feature_dict(
    *,
    station: Station,
    local_date: date,
    polarity: Polarity,
    model_extremes_c: dict[str, float],
    all_extremes_c: Sequence[float],
    peak_local_hour: float,
    wind_speeds: Sequence[float],
    wind_dirs: Sequence[float],
    clouds: Sequence[float],
    rhs: Sequence[float],
    rolling_bias_c: float,
    prior_obs_temp_c: float,
) -> tuple[dict[str, float], float, float]:
    ens_mean = _mean(all_extremes_c)
    ens_std = _std(all_extremes_c, ens_mean)
    ens_min = min(all_extremes_c) if all_extremes_c else ens_mean
    ens_max = max(all_extremes_c) if all_extremes_c else ens_mean

    ecmwf_ext = model_extremes_c.get("ecmwf_ifs025", ens_mean)
    gfs_ext = model_extremes_c.get("gfs_seamless", ens_mean)
    icon_ext = model_extremes_c.get("icon_seamless", ens_mean)
    model_spread = max(ecmwf_ext, gfs_ext, icon_ext) - min(ecmwf_ext, gfs_ext, icon_ext)

    wind_sin, wind_cos = _circular_wind_stats(wind_dirs)
    doy = local_date.timetuple().tm_yday
    doy_angle = 2.0 * math.pi * doy / 365.25
    hour_angle = 2.0 * math.pi * peak_local_hour / 24.0

    features = {
        "ecmwf_extreme_c": float(ecmwf_ext),
        "gfs_extreme_c": float(gfs_ext),
        "icon_extreme_c": float(icon_ext),
        "ensemble_mean_c": float(ens_mean),
        "ensemble_std_c": float(ens_std),
        "ensemble_min_c": float(ens_min),
        "ensemble_max_c": float(ens_max),
        "model_spread_c": float(model_spread),
        "wind_speed_mean": float(_mean(wind_speeds, 8.0)),
        "wind_dir_sin": float(wind_sin),
        "wind_dir_cos": float(wind_cos),
        "cloud_cover_mean": float(_mean(clouds, 40.0)),
        "relative_humidity_mean": float(_mean(rhs, 65.0)),
        "doy_sin": float(math.sin(doy_angle)),
        "doy_cos": float(math.cos(doy_angle)),
        "peak_hour_sin": float(math.sin(hour_angle)),
        "peak_hour_cos": float(math.cos(hour_angle)),
        "polarity_is_highest": 1.0 if polarity == Polarity.HIGHEST else 0.0,
        "station_lat": float(station.lat),
        "station_lon": float(station.lon),
        "station_elevation_m": float(station.elevation_m),
        "rolling_bias_c": float(rolling_bias_c),
        "prior_obs_temp_c": float(prior_obs_temp_c),
    }
    return features, ens_mean, ens_std


def build_mos_sample(
    *,
    station: Station,
    local_date: date,
    polarity: Polarity,
    decision_time_utc: datetime,
    forecasts: Sequence[HistoricalForecastPoint],
    prior_observations: Sequence[MetarObservation] = (),
    recent_bias_obs: Sequence[DailyBiasObservation] = (),
    target_temp_c: float | None = None,
    target_observed_at_utc: datetime | None = None,
    remark_tenths_enabled: bool = True,
) -> MosDatasetSample:
    """Build a strictly leak-free MosDatasetSample validating ts < decision_time_utc."""
    decision_utc = _ensure_utc(decision_time_utc)
    station_tz = ZoneInfo(station.timezone)
    target_icao = station.icao.strip().upper()

    day_forecasts: list[HistoricalForecastPoint] = []
    for point in forecasts:
        if point.station_icao.strip().upper() != target_icao:
            continue
        assert_before_decision(point.available_at_utc, decision_utc, label="forecast_availability")
        if _ensure_utc(point.valid_at_utc).astimezone(station_tz).date() == local_date:
            day_forecasts.append(point)

    if not day_forecasts:
        raise ValueError(
            f"No historical forecast points available for {target_icao} on {local_date.isoformat()} "
            f"prior to {decision_utc.isoformat()}."
        )

    valid_obs: list[MetarObservation] = []
    for obs in prior_observations:
        assert_before_decision(obs.observed_at_utc, decision_utc, label="prior_observation")
        if not obs.is_routine or obs.raw_report.strip().upper().startswith("SPECI"):
            continue
        if obs.station_icao.strip().upper() == target_icao:
            valid_obs.append(obs)
    valid_obs.sort(key=lambda o: o.observed_at_utc)

    bias_errors: list[float] = []
    for bias in recent_bias_obs:
        if bias.observed_at_utc is not None:
            assert_before_decision(bias.observed_at_utc, decision_utc, label="bias_observation")
        if not bias.is_routine:
            continue
        age_days = (local_date - bias.local_date).days
        if 0 < age_days <= 45:
            bias_errors.append(bias.observed_extreme_c - bias.forecast_extreme_c)

    by_model: dict[str, list[HistoricalForecastPoint]] = {}
    for pt in day_forecasts:
        by_model.setdefault(pt.model, []).append(pt)

    model_extremes_c: dict[str, float] = {}
    peak_hours: list[float] = []
    for model_name, pts in by_model.items():
        chosen = (
            max(pts, key=lambda p: p.temperature_2m_c)
            if polarity == Polarity.HIGHEST
            else min(pts, key=lambda p: p.temperature_2m_c)
        )
        model_extremes_c[model_name] = chosen.temperature_2m_c
        local_dt = _ensure_utc(chosen.valid_at_utc).astimezone(station_tz)
        peak_hours.append(local_dt.hour + local_dt.minute / 60.0)

    all_extremes = list(model_extremes_c.values())
    wind_speeds = [p.wind_speed_10m for p in day_forecasts if p.wind_speed_10m is not None]
    wind_dirs = [p.wind_direction_10m for p in day_forecasts if p.wind_direction_10m is not None]
    clouds = [p.cloud_cover for p in day_forecasts if p.cloud_cover is not None]
    rhs = [p.relative_humidity_2m for p in day_forecasts if p.relative_humidity_2m is not None]

    rolling_bias_c = _mean(bias_errors, 0.0)
    prior_obs_temp_c = valid_obs[-1].temp_c if valid_obs else _mean(all_extremes)
    latest_avail = max(_ensure_utc(p.available_at_utc) for p in day_forecasts)

    features, ens_mean, ens_std = _compute_feature_dict(
        station=station,
        local_date=local_date,
        polarity=polarity,
        model_extremes_c=model_extremes_c,
        all_extremes_c=all_extremes,
        peak_local_hour=_mean(peak_hours, 14.0 if polarity == Polarity.HIGHEST else 6.0),
        wind_speeds=wind_speeds,
        wind_dirs=wind_dirs,
        clouds=clouds,
        rhs=rhs,
        rolling_bias_c=rolling_bias_c,
        prior_obs_temp_c=prior_obs_temp_c,
    )

    return MosDatasetSample(
        station_icao=target_icao,
        local_date=local_date,
        polarity=polarity,
        decision_time_utc=decision_utc,
        forecast_available_at_utc=latest_avail,
        features=features,
        baseline_mean_c=ens_mean + rolling_bias_c,
        baseline_std_c=max(ens_std, 0.5),
        baseline_extremes_c=all_extremes,
        target_temp_c=target_temp_c,
        target_observed_at_utc=_ensure_utc(target_observed_at_utc) if target_observed_at_utc else None,
        remark_tenths_enabled=remark_tenths_enabled,
    )


def build_mos_samples_from_history(
    *,
    station: Station,
    forecasts: Sequence[HistoricalForecastPoint],
    routine_metars: Sequence[MetarObservation],
    polarity: Polarity = Polarity.HIGHEST,
    remark_tenths_enabled: bool = True,
    decision_hour_utc_offset_hours: int = -6,
) -> list[MosDatasetSample]:
    """Assemble chronological supervised MosDatasetSamples from historical forecasts and Routine METARs."""
    station_tz = ZoneInfo(station.timezone)
    target_icao = station.icao.strip().upper()

    obs_by_date: dict[date, list[MetarObservation]] = {}
    for obs in routine_metars:
        if not obs.is_routine or obs.raw_report.strip().upper().startswith("SPECI"):
            continue
        if obs.station_icao.strip().upper() != target_icao:
            continue
        local_d = _ensure_utc(obs.observed_at_utc).astimezone(station_tz).date()
        obs_by_date.setdefault(local_d, []).append(obs)

    samples: list[MosDatasetSample] = []
    history_bias: list[DailyBiasObservation] = []
    sorted_dates = sorted(obs_by_date.keys())

    for local_d in sorted_dates:
        # Decision time strictly prior to local day start (premarket decision point)
        local_midnight = datetime(local_d.year, local_d.month, local_d.day, 0, 0, tzinfo=station_tz)
        decision_utc = local_midnight.astimezone(timezone.utc) + timedelta(hours=decision_hour_utc_offset_hours)

        day_obs = obs_by_date[local_d]
        if remark_tenths_enabled:
            target_obs = (
                max(day_obs, key=lambda o: o.temp_c)
                if polarity == Polarity.HIGHEST
                else min(day_obs, key=lambda o: o.temp_c)
            )
            target_val = target_obs.temp_c
        else:
            target_obs = (
                max(day_obs, key=lambda o: float(o.whole_temp_c))
                if polarity == Polarity.HIGHEST
                else min(day_obs, key=lambda o: float(o.whole_temp_c))
            )
            target_val = float(target_obs.whole_temp_c)

        eligible_forecasts = [
            f for f in forecasts
            if _ensure_utc(f.available_at_utc) < decision_utc
        ]
        prior_obs = [
            o for o in routine_metars
            if _ensure_utc(o.observed_at_utc) < decision_utc
        ]
        try:
            sample = build_mos_sample(
                station=station,
                local_date=local_d,
                polarity=polarity,
                decision_time_utc=decision_utc,
                forecasts=eligible_forecasts,
                prior_observations=prior_obs,
                recent_bias_obs=history_bias,
                target_temp_c=target_val,
                target_observed_at_utc=target_obs.observed_at_utc,
                remark_tenths_enabled=remark_tenths_enabled,
            )
        except ValueError:
            continue

        samples.append(sample)
        history_bias.append(
            DailyBiasObservation(
                local_date=local_d,
                forecast_extreme_c=sample.features["ensemble_mean_c"],
                observed_extreme_c=target_val,
                forecast_std_c=sample.features["ensemble_std_c"],
                observed_at_utc=target_obs.observed_at_utc,
                is_routine=True,
            )
        )

    samples.sort(key=lambda s: s.decision_time_utc)
    return samples


def extract_features_from_ensemble(
    *,
    station: Station,
    local_date: date,
    polarity: Polarity,
    ensemble: EnsembleSnapshot,
    decision_time_utc: datetime,
    recent_bias_obs: Sequence[DailyBiasObservation] = (),
    prior_observations: Sequence[MetarObservation] = (),
) -> dict[str, float]:
    """Extract aligned MOS feature vector from a live EnsembleSnapshot prior to decision_time_utc."""
    decision_utc = _ensure_utc(decision_time_utc)
    assert_before_decision(ensemble.fetched_at_utc, decision_utc, label="ensemble_fetched_at")

    station_tz = ZoneInfo(station.timezone)
    day_indices = [
        idx
        for idx, ts in enumerate(ensemble.hourly_timestamps_utc)
        if _ensure_utc(ts).astimezone(station_tz).date() == local_date
    ]
    if not day_indices:
        raise ValueError(
            f"EnsembleSnapshot {ensemble.event_id} has no timestamps in local_date {local_date.isoformat()}."
        )

    day_timestamps = [_ensure_utc(ensemble.hourly_timestamps_utc[idx]) for idx in day_indices]
    trajectories = [
        [float(traj[idx]) for idx in day_indices]
        for traj in ensemble.member_temps_c
        if len(traj) > max(day_indices)
    ]
    if not trajectories:
        raise ValueError(f"EnsembleSnapshot {ensemble.event_id} has zero valid trajectories.")

    all_extremes: list[float] = []
    peak_hours: list[float] = []
    for traj in trajectories:
        if polarity == Polarity.HIGHEST:
            best_idx = max(range(len(traj)), key=lambda i: traj[i])
        else:
            best_idx = min(range(len(traj)), key=lambda i: traj[i])
        all_extremes.append(traj[best_idx])
        local_dt = day_timestamps[best_idx].astimezone(station_tz)
        peak_hours.append(local_dt.hour + local_dt.minute / 60.0)

    # Approximate per-model extremes if model_member_counts metadata exists; otherwise ensemble mean
    ens_mean = _mean(all_extremes)
    model_extremes: dict[str, float] = {}
    model_means = ensemble.metadata.get("model_extremes_c")
    if isinstance(model_means, dict):
        for k, v in model_means.items():
            model_extremes[str(k)] = float(v)
    else:
        for m_name in ensemble.models:
            model_extremes[m_name] = ens_mean

    bias_errors: list[float] = []
    for bias in recent_bias_obs:
        if bias.observed_at_utc is not None:
            assert_before_decision(bias.observed_at_utc, decision_utc, label="bias_observation")
        if not bias.is_routine:
            continue
        age_days = (local_date - bias.local_date).days
        if 0 < age_days <= 45:
            bias_errors.append(bias.observed_extreme_c - bias.forecast_extreme_c)

    valid_obs: list[MetarObservation] = []
    target_icao = station.icao.strip().upper()
    for obs in prior_observations:
        assert_before_decision(obs.observed_at_utc, decision_utc, label="prior_observation")
        if obs.is_routine and not obs.raw_report.strip().upper().startswith("SPECI"):
            if obs.station_icao.strip().upper() == target_icao:
                valid_obs.append(obs)
    valid_obs.sort(key=lambda o: o.observed_at_utc)

    rolling_bias_c = _mean(bias_errors, 0.0)
    prior_obs_temp_c = valid_obs[-1].temp_c if valid_obs else ens_mean

    meta = ensemble.metadata
    wind_speeds = [float(x) for x in meta.get("wind_speed_10m", [])] if isinstance(meta.get("wind_speed_10m"), list) else []
    wind_dirs = [float(x) for x in meta.get("wind_direction_10m", [])] if isinstance(meta.get("wind_direction_10m"), list) else []
    clouds = [float(x) for x in meta.get("cloud_cover", [])] if isinstance(meta.get("cloud_cover"), list) else []
    rhs = [float(x) for x in meta.get("relative_humidity_2m", [])] if isinstance(meta.get("relative_humidity_2m"), list) else []

    features, _, _ = _compute_feature_dict(
        station=station,
        local_date=local_date,
        polarity=polarity,
        model_extremes_c=model_extremes,
        all_extremes_c=all_extremes,
        peak_local_hour=_mean(peak_hours, 14.0 if polarity == Polarity.HIGHEST else 6.0),
        wind_speeds=wind_speeds,
        wind_dirs=wind_dirs,
        clouds=clouds,
        rhs=rhs,
        rolling_bias_c=rolling_bias_c,
        prior_obs_temp_c=prior_obs_temp_c,
    )
    return features


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dictionary into the canonical FEATURE_COLUMNS vector."""
    return [float(features[col]) for col in FEATURE_COLUMNS]
