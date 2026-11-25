"""FastAPI endpoints for Quantile ML MOS training, versions, covered stations, and benchmarks."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import Clock, TemporalIntegrityError
from weather_quant.domain.models import ModelVersionRecord, Polarity, Station
from weather_quant.ml.dataset_builder import (
    TOP_LIQUIDITY_STATIONS,
    HistoricalArchiveFetcher,
    HistoricalArchiveFetcherPort,
    MosDatasetSample,
    build_mos_samples_from_history,
    get_top_liquidity_stations,
)
from weather_quant.ml.quantile_mos import (
    DEFAULT_QUANTILES,
    ChronologicalBenchmarkReport,
    MosProbabilityProvider,
    train_and_register_mos,
)
from weather_quant.probability.engine import DEFAULT_MODEL_VERSION
from weather_quant.storage.database import Database

router = APIRouter(prefix="/api/ml", tags=["ml"])


class MlVersionsResponse(BaseModel):
    """Registered immutable Model Versions ordered chronologically."""

    active_model_version: str
    count: int
    versions: list[ModelVersionRecord] = Field(default_factory=list)


class CoveredStationsResponse(BaseModel):
    """Top liquidity ICAO Stations eligible for Quantile ML MOS."""

    icaos: list[str]
    stations: list[Station]
    mos_active: bool
    active_model_version: str


class MlBenchmarksResponse(BaseModel):
    """Chronological validation and test calibration benchmarks."""

    count: int
    benchmarks: list[ChronologicalBenchmarkReport] = Field(default_factory=list)


class MlTrainRequest(BaseModel):
    """Request payload to trigger out-of-band Quantile LightGBM MOS training."""

    model_version: str | None = None
    samples: list[MosDatasetSample] | None = None
    station_icaos: list[str] = Field(default_factory=lambda: list(TOP_LIQUIDITY_STATIONS))
    start_date: date | None = None
    end_date: date | None = None
    train_end_utc: datetime | None = None
    val_start_utc: datetime | None = None
    test_start_utc: datetime | None = None
    quantiles: list[float] = Field(default_factory=lambda: list(DEFAULT_QUANTILES))
    lgbm_params: dict[str, Any] | None = None
    activate: bool = True


class MlTrainResponse(BaseModel):
    """Result of an out-of-band MOS training and registration run."""

    model_version: ModelVersionRecord
    benchmark: ChronologicalBenchmarkReport
    covered_stations: list[str]


def _get_db(request: Request) -> Database:
    return request.app.state.db


def _get_clock(request: Request) -> Clock:
    return request.app.state.clock


def get_mos_provider(request: Request) -> MosProbabilityProvider:
    provider = getattr(request.app.state, "mos_provider", None)
    if isinstance(provider, MosProbabilityProvider):
        return provider
    provider = MosProbabilityProvider()
    request.app.state.mos_provider = provider
    return provider


_get_provider = get_mos_provider


def _get_historical_fetcher(request: Request, clock: Clock) -> HistoricalArchiveFetcherPort:
    fetcher = getattr(request.app.state, "historical_fetcher", None)
    return fetcher if fetcher is not None else HistoricalArchiveFetcher(clock=clock)


def _derive_default_split(
    samples: list[MosDatasetSample],
) -> tuple[datetime, datetime, datetime]:
    sorted_samples = sorted(samples, key=lambda s: s.decision_time_utc)
    n = len(sorted_samples)
    if n < 6:
        raise ValueError("At least 6 labeled samples are required for chronological train/val/test splitting.")
    val_idx = max(2, int(n * 0.6))
    test_idx = max(val_idx + 1, int(n * 0.8))
    if test_idx >= n:
        test_idx = n - 1
    train_end = sorted_samples[val_idx].decision_time_utc
    val_start = sorted_samples[val_idx].decision_time_utc
    test_start = sorted_samples[test_idx].decision_time_utc
    if train_end >= val_start:
        train_end = val_start - timedelta(seconds=1)
    return train_end, val_start, test_start


@router.get("/versions", response_model=MlVersionsResponse)
async def list_ml_versions(request: Request) -> MlVersionsResponse:
    """List all registered ModelVersionRecord entries from SQLite."""
    db = _get_db(request)
    versions = await db.list_model_versions()
    active = await db.get_active_model_version()
    return MlVersionsResponse(
        active_model_version=active.model_version,
        count=len(versions),
        versions=versions,
    )


@router.get("/covered-stations", response_model=CoveredStationsResponse)
async def get_covered_stations(request: Request) -> CoveredStationsResponse:
    """Return top-liquidity Stations covered by Quantile ML MOS and provider status."""
    provider = get_mos_provider(request)
    stations = get_top_liquidity_stations()
    mos_active = provider.mos_model is not None
    active_version = provider.mos_model.model_version if provider.mos_model is not None else DEFAULT_MODEL_VERSION
    return CoveredStationsResponse(
        icaos=list(TOP_LIQUIDITY_STATIONS),
        stations=stations,
        mos_active=mos_active,
        active_model_version=active_version,
    )


@router.get("/benchmarks", response_model=MlBenchmarksResponse)
async def list_ml_benchmarks(request: Request) -> MlBenchmarksResponse:
    """Return chronological calibration benchmarks stored in registered MOS Model Versions."""
    db = _get_db(request)
    versions = await db.list_model_versions()
    benchmarks: list[ChronologicalBenchmarkReport] = []
    for rec in versions:
        raw_bench = rec.config_json.get("benchmark")
        if isinstance(raw_bench, dict):
            benchmarks.append(ChronologicalBenchmarkReport.model_validate(raw_bench))
    return MlBenchmarksResponse(count=len(benchmarks), benchmarks=benchmarks)


@router.post("/train", response_model=MlTrainResponse)
async def train_mos_model(
    request: Request,
    body: MlTrainRequest | None = None,
) -> MlTrainResponse:
    """Train Quantile LightGBM MOS out-of-band, evaluate chronologically, and register Model Version."""
    db = _get_db(request)
    clock = _get_clock(request)
    provider = get_mos_provider(request)
    req = body or MlTrainRequest()

    now_utc = clock.now()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    model_version = req.model_version or f"mos-lgbm-{now_utc:%Y%m%dT%H%M%SZ}"

    if req.samples is not None:
        samples = sorted(req.samples, key=lambda s: s.decision_time_utc)
    else:
        fetcher = _get_historical_fetcher(request, clock)
        end_d = req.end_date or (now_utc.date() - timedelta(days=1))
        start_d = req.start_date or (end_d - timedelta(days=90))
        samples = []
        for icao in req.station_icaos:
            station = get_station(icao)
            if station is None:
                continue
            forecasts = await fetcher.fetch_historical_forecasts(station, start_d, end_d)
            metars = await fetcher.fetch_historical_routine_metars(station, start_d, end_d)
            samples.extend(
                build_mos_samples_from_history(
                    station=station,
                    forecasts=forecasts,
                    routine_metars=metars,
                    polarity=Polarity.HIGHEST,
                )
            )
        samples.sort(key=lambda s: s.decision_time_utc)

    try:
        if req.train_end_utc and req.val_start_utc and req.test_start_utc:
            t_train_end = req.train_end_utc
            t_val_start = req.val_start_utc
            t_test_start = req.test_start_utc
        else:
            t_train_end, t_val_start, t_test_start = _derive_default_split(samples)

        mos_model, record, benchmark = await train_and_register_mos(
            db=db,
            samples=samples,
            model_version=model_version,
            train_end_utc=t_train_end,
            val_start_utc=t_val_start,
            test_start_utc=t_test_start,
            quantiles=req.quantiles,
            covered_stations=req.station_icaos,
            lgbm_params=req.lgbm_params,
            activate=req.activate,
            clock=clock,
        )
    except TemporalIntegrityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        msg = str(exc)
        status = 409 if "already exists" in msg else 400
        raise HTTPException(status_code=status, detail=msg) from exc

    provider.set_mos_model(mos_model)
    return MlTrainResponse(
        model_version=record,
        benchmark=benchmark,
        covered_stations=list(mos_model.covered_stations),
    )
