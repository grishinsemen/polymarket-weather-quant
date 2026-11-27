"""TDD tests for Phase B: $0 MOS feeds (MOSMIX, NOAA NBS/LAV, NWS, Previous Runs) & new features."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import EnsembleSnapshot, MetarObservation, Polarity
from weather_quant.feeds import (
    DwdMosmixClient,
    NoaaTextMosClient,
    NwsPointForecastClient,
    PreviousRunsArchiveFetcher,
    parse_mosmix_kml,
    parse_noaa_text_bulletin,
    parse_nws_forecast_periods,
)
from weather_quant.ml.dataset_builder import (
    FEATURE_COLUMNS,
    build_mos_samples_from_parquet_ensembles,
    features_to_vector,
)


def _sample_mosmix_kmz() -> bytes:
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml:kml xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:dwd="https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd">
  <kml:Document>
    <kml:ExtendedData>
      <dwd:ProductDefinition>
        <dwd:IssueTime>2026-11-25T12:00:00.000Z</dwd:IssueTime>
        <dwd:ForecastTimeSteps>
          <dwd:TimeStep>2026-11-26T06:00:00.000Z</dwd:TimeStep>
          <dwd:TimeStep>2026-11-26T14:00:00.000Z</dwd:TimeStep>
          <dwd:TimeStep>2026-11-26T20:00:00.000Z</dwd:TimeStep>
        </dwd:ForecastTimeSteps>
      </dwd:ProductDefinition>
    </kml:ExtendedData>
    <kml:Placemark>
      <kml:ExtendedData>
        <dwd:Forecast dwd:elementName="TTT">
          <dwd:value>280.15 288.65 283.15</dwd:value>
        </dwd:Forecast>
        <dwd:Forecast dwd:elementName="E_TTT">
          <dwd:value>1.1 1.3 1.2</dwd:value>
        </dwd:Forecast>
      </kml:ExtendedData>
    </kml:Placemark>
  </kml:Document>
</kml:kml>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MOSMIX_L_LATEST_EDDM.kml", kml.encode("utf-8"))
    return buf.getvalue()


def test_parse_mosmix_kmz_extracts_celsius_extremes_and_uncertainty() -> None:
    station = get_station("EDDM")
    assert station is not None
    res = parse_mosmix_kml(
        _sample_mosmix_kmz(),
        station=station,
        target_local_date=date(2026, 11, 26),
        ingested_at_utc=datetime(2026, 11, 25, 13, 15, tzinfo=timezone.utc),
    )
    assert res is not None
    assert res.source == "dwd_mosmix"
    assert res.predicted_high_c == pytest.approx(15.5, abs=1e-2)
    assert res.predicted_low_c == pytest.approx(7.0, abs=1e-2)
    assert res.uncertainty_c == pytest.approx(1.2, abs=1e-2)
    assert res.available_at_utc == datetime(2026, 11, 25, 13, 15, tzinfo=timezone.utc)


def test_parse_noaa_nbm_text_bulletin_extracts_us_station_extremes() -> None:
    station = get_station("KLGA")
    assert station is not None
    bulletin = """
 KLGA   NBM V5.0 NBS GUIDANCE   11/25/2026  1200 UTC
 UTC  18  00  06  12  18  21  00
 TMP  50  44  41  45  59  57  48
"""
    res = parse_noaa_text_bulletin(
        bulletin,
        station=station,
        target_local_date=date(2026, 11, 26),
        source="noaa_nbm_nbs",
    )
    assert res is not None
    assert res.station_icao == "KLGA"
    # 59F = 15.0C, 41F = 5.0C
    assert res.predicted_high_c == pytest.approx(15.0, abs=0.1)
    assert res.predicted_low_c == pytest.approx(5.0, abs=0.1)
    assert res.available_at_utc == datetime(2026, 11, 25, 13, 15, tzinfo=timezone.utc)


def test_parse_nws_point_forecast_periods() -> None:
    station = get_station("KMIA")
    assert station is not None
    payload = {
        "properties": {
            "updateTime": "2026-11-25T15:00:00+00:00",
            "periods": [
                {"startTime": "2026-11-26T07:00:00-05:00", "isDaytime": True, "temperature": 82, "temperatureUnit": "F"},
                {"startTime": "2026-11-26T18:00:00-05:00", "isDaytime": False, "temperature": 68, "temperatureUnit": "F"},
            ],
        }
    }
    res = parse_nws_forecast_periods(
        payload,
        station=station,
        target_local_date=date(2026, 11, 26),
    )
    assert res is not None
    assert res.predicted_high_c == pytest.approx((82.0 - 32.0) * 5.0 / 9.0, abs=0.05)
    assert res.predicted_low_c == pytest.approx((68.0 - 32.0) * 5.0 / 9.0, abs=0.05)


@pytest.mark.asyncio
async def test_previous_runs_fetcher_applies_p95_pub_latency_and_reads_pressure() -> None:
    station = get_station("LFPB")
    assert station is not None
    payload = [
        {
            "model": "ecmwf_ifs025",
            "hourly": {
                "time": ["2026-11-26T06:00", "2026-11-26T14:00"],
                "temperature_2m_previous_day1": [8.0, 14.5],
                "wind_speed_10m_previous_day1": [10.0, 12.0],
                "wind_direction_10m_previous_day1": [220.0, 230.0],
                "cloud_cover_previous_day1": [40.0, 20.0],
                "relative_humidity_2m_previous_day1": [80.0, 60.0],
                "pressure_msl_previous_day1": [1018.0, 1012.0],
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "previous-runs-api.open-meteo.com" in str(request.url)
        assert "temperature_2m_previous_day1" in request.url.params["hourly"]
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = PreviousRunsArchiveFetcher(client=client, models=("ecmwf_ifs025",), previous_day=1)
        pts = await fetcher.fetch_historical_forecasts(station, date(2026, 11, 26), date(2026, 11, 26))

    assert len(pts) == 2
    # D-1 00Z init (2026-11-25 00:00Z) + 8.0h p95 ECMWF pub latency = 2026-11-25 08:00Z
    assert pts[0].available_at_utc == datetime(2026, 11, 25, 8, 0, tzinfo=timezone.utc)
    assert pts[0].pressure_msl == pytest.approx(1018.0)
    assert pts[1].pressure_msl == pytest.approx(1012.0)


@pytest.mark.asyncio
async def test_previous_runs_fetcher_parses_multi_model_suffixed_keys() -> None:
    station = get_station("KLGA")
    assert station is not None
    payload = {
        "hourly": {
            "time": ["2026-11-26T14:00"],
            "temperature_2m_previous_day1_ecmwf_ifs025": [10.2],
            "temperature_2m_previous_day1_gfs_seamless": [9.8],
            "temperature_2m_previous_day1_icon_seamless": [12.0],
            "pressure_msl_previous_day1_ecmwf_ifs025": [1015.0],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = PreviousRunsArchiveFetcher(
            client=client,
            models=("ecmwf_ifs025", "gfs_seamless", "icon_seamless"),
            previous_day=1,
        )
        pts = await fetcher.fetch_historical_forecasts(station, date(2026, 11, 26), date(2026, 11, 26))

    by_model = {p.model: p for p in pts}
    assert set(by_model) == {"ecmwf_ifs025", "gfs_seamless", "icon_seamless"}
    assert by_model["ecmwf_ifs025"].temperature_2m_c == pytest.approx(10.2)
    assert by_model["gfs_seamless"].temperature_2m_c == pytest.approx(9.8)
    assert by_model["icon_seamless"].temperature_2m_c == pytest.approx(12.0)
    assert by_model["ecmwf_ifs025"].pressure_msl == pytest.approx(1015.0)


@pytest.mark.asyncio
async def test_clients_work_with_mock_transport() -> None:
    eddm = get_station("EDDM")
    kmia = get_station("KMIA")
    assert eddm is not None and kmia is not None

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "opendata.dwd.de" in url:
            return httpx.Response(200, content=_sample_mosmix_kmz())
        if "api.weather.gov/points" in url:
            return httpx.Response(200, json={"properties": {"forecast": "https://api.weather.gov/gridpoints/MFL/109,50/forecast"}})
        if "api.weather.gov/gridpoints" in url:
            return httpx.Response(
                200,
                json={
                    "properties": {
                        "updateTime": "2026-11-25T15:00:00Z",
                        "periods": [{"startTime": "2026-11-26T08:00:00-05:00", "isDaytime": True, "temperature": 80, "temperatureUnit": "F"}],
                    }
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        clock = VirtualClock(datetime(2026, 11, 25, 16, 0, tzinfo=timezone.utc))
        mosmix = await DwdMosmixClient(client=client, clock=clock).fetch_forecast(eddm, date(2026, 11, 26))
        nws = await NwsPointForecastClient(client=client, clock=clock).fetch_forecast(kmia, date(2026, 11, 26))
        nws_intl = await NwsPointForecastClient(client=client, clock=clock).fetch_forecast(eddm, date(2026, 11, 26))

    assert mosmix is not None and mosmix.predicted_high_c == pytest.approx(15.5, abs=0.05)
    assert nws is not None and nws.predicted_high_c == pytest.approx(26.667, abs=0.05)
    assert nws_intl is None


def test_parquet_dataset_builder_populates_24h_dynamics_and_mos_anchors() -> None:
    station = get_station("LFPB")
    assert station is not None
    # Build 3 consecutive days of routine METARs so day 3 has valid d-1 diurnal range & d-1 - d-2 delta
    metars: list[MetarObservation] = []
    for day_num, (lo, hi) in [(24, (6.0, 12.0)), (25, (7.0, 16.0)), (26, (8.0, 15.0))]:
        for hr, temp in [(6, lo), (14, hi)]:
            ts = datetime(2026, 11, day_num, hr, 0, tzinfo=timezone.utc)
            metars.append(
                MetarObservation(
                    station_icao="LFPB",
                    raw_report=f"METAR LFPB {day_num:02d}{hr:02d}00Z {int(temp):02d}/05",
                    observed_at_utc=ts,
                    ingested_at_utc=ts + timedelta(minutes=5),
                    temp_c=temp,
                    whole_temp_c=int(temp),
                    is_routine=True,
                )
            )

    ens_ts = [datetime(2026, 11, 26, h, 0, tzinfo=timezone.utc) for h in range(24)]
    ens = EnsembleSnapshot(
        event_id="evt-lfpb-26",
        station_icao="LFPB",
        fetched_at_utc=datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 25, 12, 5, tzinfo=timezone.utc),
        models=["ecmwf_ifs025"],
        member_count=10,
        hourly_timestamps_utc=ens_ts,
        member_temps_c=[[10.0 + (5.0 if h == 14 else 0.0) + m * 0.1 for h in range(24)] for m in range(10)],
        metadata={"pressure_msl": [1020.0, 1014.0], "mosmix_extreme_c": 15.8},
    )

    samples = build_mos_samples_from_parquet_ensembles(
        station=station,
        ensembles=[ens],
        routine_metars=metars,
        polarity=Polarity.HIGHEST,
    )
    assert len(samples) == 1
    feats = samples[0].features
    assert set(FEATURE_COLUMNS).issubset(feats.keys())
    assert len(features_to_vector(feats)) == len(FEATURE_COLUMNS)
    # Day 25 max (16) - Day 24 max (12) = +4.0C; Day 25 diurnal range (16 - 7) = 9.0C
    assert feats["delta_obs_temp_24h"] == pytest.approx(4.0)
    assert feats["prior_diurnal_range_c"] == pytest.approx(9.0)
    assert feats["pressure_msl_tendency_24h"] == pytest.approx(-6.0)
    assert feats["mosmix_extreme_c"] == pytest.approx(15.8)


def test_previous_runs_utc_boundary_preserves_full_24h_local_day_for_americas_and_nzwn() -> None:
    """KDEN (UTC-6/7) afternoon/evening crosses UTC midnight and NZWN (UTC+13) precedes 08Z ECMWF;
    both must stitch day1+day2 from strictly pre-decision runs without truncating daytime peaks."""
    import math
    from weather_quant.feeds.previous_runs import parse_previous_runs_payload
    from weather_quant.ml.dataset_builder import build_mos_samples_from_history

    # 1) KDEN local day 2026-10-17 = 2026-10-17 06:00Z .. 2026-10-18 05:00Z
    kden = get_station("KDEN")
    assert kden is not None
    start_kden = datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc)
    times_kden = [(start_kden + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(24)]
    # Diurnal curve peaking at local 15:00 (h=15 -> 21:00Z) and warm at 18:00 (h=18 -> 00:00Z next UTC day!)
    # Put peak at h=18 (local 18:00, which is 2026-10-18T00:00Z!) to prove cross-midnight hours survive
    temps_d1 = [5.0 + (20.0 if h == 18 else (12.0 if h == 14 else 0.0)) for h in range(24)]
    temps_d2 = [5.0 + (19.0 if h == 18 else (11.0 if h == 14 else 0.0)) for h in range(24)]
    payload_kden = {
        "hourly": {
            "time": times_kden,
            "temperature_2m_previous_day1_ecmwf_ifs025": temps_d1,
            "temperature_2m_previous_day2_ecmwf_ifs025": temps_d2,
        }
    }
    pts_kden = parse_previous_runs_payload(
        payload_kden, station=kden, models=("ecmwf_ifs025",), previous_day=1
    )
    assert len(pts_kden) == 24
    # Hours 0..17 (2026-10-17 UTC) use day1 (init 2026-10-16 00Z -> avail 08Z)
    # Hours 18..23 (2026-10-18 UTC) switch to day2 (init 2026-10-16 00Z -> avail 08Z < cutoff 2026-10-17 00Z)
    assert all(p.available_at_utc == datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc) for p in pts_kden)
    metars_kden = [
        MetarObservation(
            station_icao="KDEN",
            raw_report="METAR KDEN 172153Z 18008KT 10SM FEW080 24/02 A3000 RMK T02400020",
            observed_at_utc=datetime(2026, 10, 17, 21, 53, tzinfo=timezone.utc),
            ingested_at_utc=datetime(2026, 10, 17, 21, 55, tzinfo=timezone.utc),
            temp_c=24.0,
            whole_temp_c=24,
            remark_tenths_c=24.0,
            has_remark_tenths=True,
            is_routine=True,
        )
    ]
    samples_kden = build_mos_samples_from_history(
        station=kden,
        forecasts=pts_kden,
        routine_metars=metars_kden,
        polarity=Polarity.HIGHEST,
        min_forecast_hours_per_model=20,
    )
    assert len(samples_kden) == 1
    # Peak must come from h=18 (cross-midnight day2 value = 24.0C), NOT truncated to 00:00..05:00!
    assert samples_kden[0].features["ecmwf_extreme_c"] == pytest.approx(24.0)
    peak_hr = (math.degrees(math.atan2(samples_kden[0].features["peak_hour_sin"], samples_kden[0].features["peak_hour_cos"])) % 360.0) / 15.0
    assert peak_hr == pytest.approx(18.0, abs=0.1)

    # 2) NZWN (UTC+13) local day 2026-11-26 = 2026-11-25 11:00Z .. 2026-11-26 10:00Z
    # Premarket cutoff = 2026-11-25 05:00Z; ECMWF day1 for 2026-11-26 02:00Z (15:00 local) publishes at
    # 2026-11-25 08:00Z (> 05:00Z), so parser must automatically select day2 (avail 2026-11-24 08:00Z).
    nzwn = get_station("NZWN")
    assert nzwn is not None
    start_nz = datetime(2026, 11, 25, 11, 0, tzinfo=timezone.utc)
    times_nz = [(start_nz + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(24)]
    temps_nz = [12.0 + (7.0 if h == 14 else 0.0) for h in range(24)]
    payload_nz = {
        "hourly": {
            "time": times_nz,
            "temperature_2m_previous_day1_ecmwf_ifs025": [t + 5.0 for t in temps_nz],
            "temperature_2m_previous_day2_ecmwf_ifs025": temps_nz,
        }
    }
    pts_nz = parse_previous_runs_payload(
        payload_nz, station=nzwn, models=("ecmwf_ifs025",), previous_day=1
    )
    assert len(pts_nz) == 24
    cutoff_nz = datetime(2026, 11, 25, 5, 0, tzinfo=timezone.utc)
    assert all(p.available_at_utc < cutoff_nz for p in pts_nz)
    # Local 14:00 (h=14 -> 2026-11-26 01:00Z) must come from day2 (19.0C), not future day1 (24.0C)
    assert pts_nz[14].temperature_2m_c == pytest.approx(19.0)
