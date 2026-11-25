"""Curated catalog of ICAO airport Stations with exact runway coordinates and elevation."""

from __future__ import annotations

from weather_quant.domain.models import Station

_STATIONS: list[Station] = [
    Station(icao="LFPB", name="Paris-Le Bourget Airport", city="Paris", country="FR", lat=48.9694, lon=2.4414, elevation_m=67.0, timezone="Europe/Paris"),
    Station(icao="LFPG", name="Paris Charles de Gaulle Airport", city="Paris CDG", country="FR", lat=49.0097, lon=2.5479, elevation_m=119.0, timezone="Europe/Paris"),
    Station(icao="RKSI", name="Incheon International Airport", city="Seoul", country="KR", lat=37.4602, lon=126.4407, elevation_m=7.0, timezone="Asia/Seoul"),
    Station(icao="RKPK", name="Gimhae International Airport", city="Busan", country="KR", lat=35.1796, lon=128.9382, elevation_m=2.0, timezone="Asia/Seoul"),
    Station(icao="EDDM", name="Munich Airport", city="Munich", country="DE", lat=48.3538, lon=11.7861, elevation_m=453.0, timezone="Europe/Berlin"),
    Station(icao="EGLL", name="London Heathrow Airport", city="London Heathrow", country="GB", lat=51.4700, lon=-0.4543, elevation_m=25.0, timezone="Europe/London"),
    Station(icao="EGLC", name="London City Airport", city="London", country="GB", lat=51.5053, lon=0.0553, elevation_m=6.0, timezone="Europe/London"),
    Station(icao="EHAM", name="Amsterdam Airport Schiphol", city="Amsterdam", country="NL", lat=52.3086, lon=4.7639, elevation_m=0.0, timezone="Europe/Amsterdam"),
    Station(icao="VILK", name="Chaudhary Charan Singh Airport", city="Lucknow", country="IN", lat=26.7606, lon=80.8893, elevation_m=123.0, timezone="Asia/Kolkata"),
    Station(icao="KMIA", name="Miami International Airport", city="Miami", country="US", lat=25.7959, lon=-80.2870, elevation_m=3.0, timezone="America/New_York"),
    Station(icao="KATL", name="Hartsfield-Jackson Atlanta Airport", city="Atlanta", country="US", lat=33.6407, lon=-84.4277, elevation_m=313.0, timezone="America/New_York"),
    Station(icao="KLGA", name="LaGuardia Airport", city="New York", country="US", lat=40.7769, lon=-73.8740, elevation_m=7.0, timezone="America/New_York"),
    Station(icao="KORD", name="Chicago O'Hare International Airport", city="Chicago", country="US", lat=41.9742, lon=-87.9073, elevation_m=205.0, timezone="America/Chicago"),
    Station(icao="KDFW", name="Dallas/Fort Worth International Airport", city="Dallas", country="US", lat=32.8998, lon=-97.0403, elevation_m=185.0, timezone="America/Chicago"),
    Station(icao="KLAX", name="Los Angeles International Airport", city="Los Angeles", country="US", lat=33.9416, lon=-118.4085, elevation_m=38.0, timezone="America/Los_Angeles"),
    Station(icao="KSEA", name="Seattle-Tacoma International Airport", city="Seattle", country="US", lat=47.4502, lon=-122.3088, elevation_m=132.0, timezone="America/Los_Angeles"),
    Station(icao="KDEN", name="Denver International Airport", city="Denver", country="US", lat=39.8561, lon=-104.6737, elevation_m=1655.0, timezone="America/Denver"),
    Station(icao="MMMX", name="Mexico City International Airport", city="Mexico City", country="MX", lat=19.4363, lon=-99.0721, elevation_m=2230.0, timezone="America/Mexico_City"),
    Station(icao="SBGR", name="Sao Paulo/Guarulhos Airport", city="Sao Paulo", country="BR", lat=-23.4356, lon=-46.4731, elevation_m=750.0, timezone="America/Sao_Paulo"),
    Station(icao="SAEZ", name="Ministro Pistarini Airport", city="Buenos Aires", country="AR", lat=-34.8222, lon=-58.5358, elevation_m=20.0, timezone="America/Argentina/Buenos_Aires"),
    Station(icao="UUEE", name="Sheremetyevo International Airport", city="Moscow", country="RU", lat=55.9726, lon=37.4146, elevation_m=192.0, timezone="Europe/Moscow"),
    Station(icao="RJTT", name="Tokyo Haneda Airport", city="Tokyo", country="JP", lat=35.5494, lon=139.7798, elevation_m=11.0, timezone="Asia/Tokyo"),
    Station(icao="WSSS", name="Singapore Changi Airport", city="Singapore", country="SG", lat=1.3644, lon=103.9915, elevation_m=7.0, timezone="Asia/Singapore"),
    Station(icao="VIDP", name="Indira Gandhi International Airport", city="Delhi", country="IN", lat=28.5562, lon=77.1000, elevation_m=237.0, timezone="Asia/Kolkata"),
    Station(icao="LEMD", name="Adolfo Suarez Madrid-Barajas Airport", city="Madrid", country="ES", lat=40.4983, lon=-3.5676, elevation_m=610.0, timezone="Europe/Madrid"),
    Station(icao="LIRF", name="Rome Fiumicino Airport", city="Rome", country="IT", lat=41.8003, lon=12.2389, elevation_m=4.0, timezone="Europe/Rome"),
    Station(icao="LOWW", name="Vienna International Airport", city="Vienna", country="AT", lat=48.1103, lon=16.5697, elevation_m=183.0, timezone="Europe/Vienna"),
    Station(icao="LSZH", name="Zurich Airport", city="Zurich", country="CH", lat=47.4647, lon=8.5492, elevation_m=432.0, timezone="Europe/Zurich"),
    Station(icao="CYYZ", name="Toronto Pearson International Airport", city="Toronto", country="CA", lat=43.6777, lon=-79.6248, elevation_m=173.0, timezone="America/Toronto"),
    Station(icao="LTAC", name="Ankara Esenboga Airport", city="Ankara", country="TR", lat=40.1281, lon=32.9951, elevation_m=953.0, timezone="Europe/Istanbul"),
    Station(icao="NZWN", name="Wellington International Airport", city="Wellington", country="NZ", lat=-41.3272, lon=174.8053, elevation_m=12.0, timezone="Pacific/Auckland"),
]

STATION_CATALOG: dict[str, Station] = {s.icao: s for s in _STATIONS}


def get_station(icao: str) -> Station | None:
    """Look up an eligible ICAO Station by 4-letter code."""
    return STATION_CATALOG.get(icao.strip().upper())


def list_stations() -> list[Station]:
    """Return all cataloged ICAO Stations sorted by ICAO code."""
    return [STATION_CATALOG[k] for k in sorted(STATION_CATALOG)]
