"""FastAPI endpoints for 3-layer Ensemble Fair Probability inspection and recomputation."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from weather_quant.domain.clock import Clock, ensure_utc
from weather_quant.domain.models import EnsembleSnapshot, EventProbabilitySnapshot
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.probability.engine import (
    DailyBiasObservation,
    EnsembleProbabilityEngine,
    build_daily_bias_observations_from_history,
    get_latest_event_probability_snapshot,
    list_latest_event_probability_snapshots,
    list_station_routine_metars,
    save_event_probability_snapshot,
)
from weather_quant.probability.ensemble_fetcher import EnsembleFetcherPort, OpenMeteoEnsembleFetcher
from weather_quant.storage.database import Database

router = APIRouter(prefix="/api/probability", tags=["probability"])


class ProbabilityEventsResponse(BaseModel):
    """List of latest Fair Probability snapshots grouped by model_version."""

    model_version: str
    count: int
    events: list[EventProbabilitySnapshot] = Field(default_factory=list)


class ProbabilityRecomputeRequest(BaseModel):
    """Optional overrides for triggering a probability recomputation cycle."""

    event_ids: list[str] | None = None
    model_version: str | None = None
    ensembles_by_event: dict[str, EnsembleSnapshot] | None = None
    recent_bias_by_station: dict[str, list[DailyBiasObservation]] | None = None


def _get_db(request: Request) -> Database:
    return request.app.state.db


def _get_clock(request: Request) -> Clock:
    return request.app.state.clock


def _get_engine(request: Request) -> EnsembleProbabilityEngine:
    engine = getattr(request.app.state, "probability_engine", None)
    return engine if isinstance(engine, EnsembleProbabilityEngine) else EnsembleProbabilityEngine()


def _get_fetcher(request: Request, clock: Clock) -> EnsembleFetcherPort:
    fetcher = getattr(request.app.state, "ensemble_fetcher", None)
    return fetcher if fetcher is not None else OpenMeteoEnsembleFetcher(clock=clock)


@router.get("/events", response_model=ProbabilityEventsResponse)
async def list_probability_events(
    request: Request,
    model_version: str | None = Query(default=None),
) -> ProbabilityEventsResponse:
    """Return latest persisted Fair Probability distributions across active Weather Events."""
    db = _get_db(request)
    target_version = model_version or (await db.get_active_model_version()).model_version
    snapshots = await list_latest_event_probability_snapshots(db, model_version=target_version)
    return ProbabilityEventsResponse(
        model_version=target_version,
        count=len(snapshots),
        events=snapshots,
    )


@router.get("/events/{event_id}", response_model=EventProbabilitySnapshot)
async def get_probability_event(
    request: Request,
    event_id: str,
    model_version: str | None = Query(default=None),
) -> EventProbabilitySnapshot:
    """Return the latest Fair Probability snapshot for a single Weather Event."""
    db = _get_db(request)
    target_version = model_version or (await db.get_active_model_version()).model_version
    snapshot = await get_latest_event_probability_snapshot(
        db,
        event_id=event_id,
        model_version=target_version,
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No probability snapshot found for event {event_id}")
    return snapshot


@router.post("/recompute", response_model=ProbabilityEventsResponse)
async def recompute_probabilities(
    request: Request,
    body: ProbabilityRecomputeRequest | None = None,
) -> ProbabilityEventsResponse:
    """Fetch/consume ensembles + persisted Routine METARs and persist 3-layer Fair Probabilities."""
    db = _get_db(request)
    clock = _get_clock(request)
    engine = _get_engine(request)
    fetcher = _get_fetcher(request, clock)

    req = body or ProbabilityRecomputeRequest()
    target_version = req.model_version or (await db.get_active_model_version()).model_version
    mos_provider = await load_mos_provider_for_version(
        db,
        target_version,
        existing_provider=getattr(request.app.state, "mos_provider", None),
    )
    active_events = await db.list_weather_events(active_only=True)
    if req.event_ids is not None:
        wanted = set(req.event_ids)
        active_events = [e for e in active_events if e.event_id in wanted]

    archive = getattr(request.app.state, "parquet_archive", None)
    hist_ensembles: list[EnsembleSnapshot] = []
    if archive is not None and not req.recent_bias_by_station:
        try:
            hist_ensembles = await archive.aread_ensemble_snapshots(
                start_utc=ensure_utc(clock.now()) - timedelta(days=engine.bias_window_days),
            )
        except Exception:
            hist_ensembles = []

    snapshots: list[EventProbabilitySnapshot] = []
    for event in active_events:
        if req.ensembles_by_event and event.event_id in req.ensembles_by_event:
            ensemble = req.ensembles_by_event[event.event_id]
        else:
            ensemble = await fetcher.fetch_ensemble(
                station=event.station,
                event_id=event.event_id,
                local_date=event.local_date,
            )

        intraday_metars = await list_station_routine_metars(db, event.station.icao)
        if req.recent_bias_by_station:
            bias_obs = req.recent_bias_by_station.get(event.station.icao, [])
        elif hist_ensembles:
            bias_obs = build_daily_bias_observations_from_history(
                station=event.station,
                polarity=event.polarity,
                remark_tenths_enabled=event.remark_tenths_enabled,
                ensembles=hist_ensembles,
                metars=intraday_metars,
                before_local_date=event.local_date,
                window_days=engine.bias_window_days,
                decision_time_utc=ensure_utc(clock.now()),
            )
        else:
            bias_obs = []
        if mos_provider is not None and mos_provider.is_station_covered(event.station.icao):
            snapshot = mos_provider.compute_probabilities(
                station=event.station,
                event=event,
                ensemble=ensemble,
                recent_bias_obs=bias_obs,
                intraday_metars=intraday_metars,
                clock=clock,
            )
        else:
            snapshot = engine.compute_probabilities(
                station=event.station,
                event=event,
                ensemble=ensemble,
                recent_bias_obs=bias_obs,
                intraday_metars=intraday_metars,
                clock=clock,
                model_version=target_version,
            )
        await save_event_probability_snapshot(db, snapshot)
        snapshots.append(snapshot)

    return ProbabilityEventsResponse(
        model_version=target_version,
        count=len(snapshots),
        events=snapshots,
    )
