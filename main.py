from __future__ import annotations

import os
from datetime import datetime, date, time
from typing import Optional, Dict, List, Tuple, Literal

from dateutil import tz
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import swisseph as swe


# -----------------------
# Config
# -----------------------
DEFAULT_TZ = "Asia/Seoul"
DEFAULT_LAT = 37.5665
DEFAULT_LON = 126.9780

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Saturn": swe.SATURN,
}

TRANSIT_PLANETS = ["Moon", "Venus", "Mars", "Saturn"]

ASPECTS = {
    "conjunct": 0,
    "sextile": 60,
    "square": 90,
    "trine": 120,
    "opposite": 180,
}

DEFAULT_ORB = 6.0  # natal aspects
SYN_ORB = 5.0      # synastry aspects

# natal priority aspects (MVP)
PRIORITY_PAIRS = {
    ("Moon", "Saturn"),
    ("Venus", "Mars"),
    ("Mercury", "Mars"),
    ("Mercury", "Saturn"),
}

# synastry pairs (MVP)
SYN_PAIRS = [
    ("Venus", "Mars"),
    ("Mars", "Venus"),
    ("Moon", "Saturn"),
    ("Saturn", "Moon"),
    ("Mercury", "Saturn"),
    ("Saturn", "Mercury"),
    ("Sun", "Moon"),
    ("Moon", "Sun"),
]

HOUSE_SYSTEM = Literal["placidus", "whole"]
HOUSE_SYSTEM_TO_CODE: Dict[str, bytes] = {
    "placidus": b"P",
    "whole": b"W",
}


# -----------------------
# Auth
# - Accept BOTH:
#   1) x-api-key: <key>
#   2) Authorization: Bearer <key>
# -----------------------
def require_api_key(authorization: str | None, x_api_key: str | None) -> None:
    expected = os.environ.get("ASTRO_BRIDGE_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server missing ASTRO_BRIDGE_API_KEY")

    token: Optional[str] = None

    if x_api_key:
        token = x_api_key.strip()
    elif authorization:
        token = authorization.removeprefix("Bearer ").strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# -----------------------
# Models
# -----------------------
class PersonIn(BaseModel):
    label: str = Field(default="A")
    birth_date: date
    birth_time: Optional[time] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class SummaryRequest(BaseModel):
    personA: PersonIn
    personB: Optional[PersonIn] = None  # B는 선택
    basis_datetime: Optional[datetime] = None
    timezone: str = Field(default=DEFAULT_TZ)
    house_system: HOUSE_SYSTEM = Field(default="placidus")  # ✅ 추가


class Placement(BaseModel):
    sign: str
    degree: float


class PersonOut(BaseModel):
    label: str
    placements: Dict[str, Placement]
    asc: Optional[Placement] = None
    aspects: List[str]
    houses: Optional[Dict[str, int]] = None  # ✅ 추가: 행성별 하우스 번호
    house_system: Optional[str] = None       # ✅ 추가


class TransitOut(BaseModel):
    date: str
    placements: Dict[str, Placement]


class SummaryResponse(BaseModel):
    text: str
    json: Dict


# -----------------------
# Helpers
# -----------------------
def normalize_timezone(tz_name: str | None) -> str:
    if not tz_name:
        return DEFAULT_TZ
    if tz.gettz(tz_name) is None:
        return DEFAULT_TZ
    return tz_name


def lon_to_sign(lon: float) -> Tuple[str, float]:
    lon = lon % 360.0
    sign_index = int(lon // 30)
    degree_in_sign = lon - (sign_index * 30)
    return SIGNS[sign_index], degree_in_sign


def to_utc_jd(dt_local: datetime, tz_name: str) -> float:
    tzinfo = tz.gettz(tz_name)
    if tzinfo is None:
        raise ValueError(f"Unknown timezone: {tz_name}")

    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=tzinfo)

    dt_utc = dt_local.astimezone(tz.UTC)

    return swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600,
    )


def calc_planet_positions(jd_ut: float) -> Dict[str, float]:
    positions: Dict[str, float] = {}
    for name, pid in PLANETS.items():
        res = swe.calc_ut(jd_ut, pid)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            xx, _retflag = res
            lon = float(xx[0])
        else:
            lon = float(res[0])
        positions[name] = lon
    return positions


def calc_cusps_and_angles(jd_ut: float, lat: float, lon: float, hsys: bytes) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Returns (cusps, ascmc)
    cusps: Swiss Ephemeris 스타일로 index 1..12 사용 (0은 비어있을 수 있음)
    ascmc: ascmc[0] = ASC
    """
    try:
        res = swe.houses_ex(jd_ut, lat, lon, hsys)
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            cusps = list(res[0])
            ascmc = list(res[1])
            return cusps, ascmc
        return None, None
    except Exception:
        return None, None


def angle_diff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def compute_aspects(positions: Dict[str, float]) -> List[str]:
    found: List[str] = []
    names = list(PLANETS.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]

            if (p1, p2) not in PRIORITY_PAIRS and (p2, p1) not in PRIORITY_PAIRS:
                continue

            d = angle_diff(positions[p1], positions[p2])
            for asp_name, asp_deg in ASPECTS.items():
                if abs(d - asp_deg) <= DEFAULT_ORB:
                    found.append(f"{p1} {asp_name} {p2}")
                    break

    return found


def unwrap_cusps(cusps_raw: List[float]) -> List[float]:
    """
    cusps_raw가 [0..12] 형태(1..12 사용)일 수도 있고, [0..11]일 수도 있어서 안전 처리.
    결과는 길이 13, 엄격히 증가하도록 360 보정.
    """
    if len(cusps_raw) >= 13:
        raw = [float(cusps_raw[i]) for i in range(1, 13)]
    else:
        raw = [float(cusps_raw[i]) for i in range(12)]

    out = [raw[0]]
    for i in range(1, 12):
        v = raw[i]
        while v < out[-1]:
            v += 360.0
        out.append(v)

    out.append(out[0] + 360.0)  # endpoint
    return out


def house_of_lon_quadrant(cusps_u: List[float], lon: float) -> int:
    lon = lon % 360.0
    base = cusps_u[0] % 360.0
    lon_u = lon
    while lon_u < base:
        lon_u += 360.0

    for i in range(12):
        if cusps_u[i] <= lon_u < cusps_u[i + 1]:
            return i + 1
    return 12


def build_houses_mapping(
    house_system: str,
    asc_lon: float,
    planet_lons: Dict[str, float],
    cusps_raw: Optional[List[float]],
) -> Dict[str, int]:
    # Whole Sign: ASC가 속한 "별자리"를 1H로 두고, 행성의 별자리 차이로 하우스 계산
    if house_system == "whole":
        asc_sign = int((asc_lon % 360.0) // 30)
        out: Dict[str, int] = {}
        for pname, plon in planet_lons.items():
            p_sign = int((plon % 360.0) // 30)
            house = ((p_sign - asc_sign) % 12) + 1
            out[pname] = house
        return out

    # Placidus(기본): cusp 구간에 따라 하우스 계산
    if not cusps_raw:
        return {}
