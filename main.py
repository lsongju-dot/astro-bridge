from __future__ import annotations

import os
from datetime import datetime, date, time
from typing import Optional, Dict, List, Tuple

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

    # Prefer x-api-key if present
    if x_api_key:
        token = x_api_key.strip()

    # Fallback to Authorization: Bearer <token>
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
    personB: Optional[PersonIn] = None  # ✅ B는 선택
    basis_datetime: Optional[datetime] = None
    timezone: str = Field(default=DEFAULT_TZ)


class Placement(BaseModel):
    sign: str
    degree: float


class PersonOut(BaseModel):
    label: str
    placements: Dict[str, Placement]
    asc: Optional[Placement] = None
    aspects: List[str]


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
    """If invalid timezone provided, fallback to Asia/Seoul."""
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

    # treat naive as local time in tz_name
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
    """
    pyswisseph swe.calc_ut typically returns (xx, retflag)
    where xx[0] is ecliptic longitude.
    """
    positions: Dict[str, float] = {}
    for name, pid in PLANETS.items():
        res = swe.calc_ut(jd_ut, pid)

        # common shape: (xx, retflag)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            xx, _retflag = res
            lon = float(xx[0])
        else:
            # fallback (rare)
            lon = float(res[0])

        positions[name] = lon
    return positions


def calc_ascendant(jd_ut: float, lat: float, lon: float) -> Optional[float]:
    """
    pyswisseph swe.houses_ex return shape differs by version.
    We safely extract ascmc[0] (ASC). If anything fails, return None.
    """
    try:
        res = swe.houses_ex(jd_ut, lat, lon, b"P")

        # typical: (cusps, ascmc) or (cusps, ascmc, ...)
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            ascmc = res[1]
            return float(ascmc[0])

        return None
    except Exception:
        return None


def angle_diff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def compute_aspects(positions: Dict[str, float]) -> List[str]:
    """MVP natal aspects within priority pairs only."""
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


def build_person_out(person: PersonIn, tz_name: str) -> PersonOut:
    # If birth_time unknown, use noon; ASC will be unknown
    bt = person.birth_time
    dt_local = datetime.combine(person.birth_date, bt or time(12, 0, 0))

    jd_ut = to_utc_jd(dt_local, tz_name)
    pos = calc_planet_positions(jd_ut)

    placements: Dict[str, Placement] = {}
    for pname, lon in pos.items():
        sign, deg = lon_to_sign(lon)
        placements[pname] = Placement(sign=sign, degree=round(deg, 2))

    aspects = compute_aspects(pos)

    asc: Optional[Placement] = None
    if bt is not None:
        lat = person.lat if person.lat is not None else DEFAULT_LAT
        lon = person.lon if person.lon is not None else DEFAULT_LON
        asc_lon = calc_ascendant(jd_ut, lat, lon)
        if asc_lon is not None:
            s, d = lon_to_sign(asc_lon)
            asc = Placement(sign=s, degree=round(d, 2))

    return PersonOut(label=person.label, placements=placements, asc=asc, aspects=aspects)


def build_transit_out(basis_dt: datetime, tz_name: str) -> TransitOut:
    jd_ut = to_utc_jd(basis_dt, tz_name)
    pos = calc_planet_positions(jd_ut)

    placements: Dict[str, Placement] = {}
    for pname in TRANSIT_PLANETS:
        lon = pos[pname]
        s, d = lon_to_sign(lon)
        placements[pname] = Placement(sign=s, degree=round(d, 2))

    return TransitOut(date=basis_dt.date().isoformat(), placements=placements)


def format_text(person: PersonOut) -> str:
    def fmt(pname: str) -> str:
        pl = person.placements[pname]
        return f"{pname}: {pl.sign} {pl.degree}°"

    parts = [fmt("Sun"), fmt("Moon"), fmt("Mercury"), fmt("Venus"), fmt("Mars"), fmt("Saturn")]
    asc_txt = "ASC: unknown" if person.asc is None else f"ASC: {person.asc.sign} {person.asc.degree}°"
    aspects_txt = " ; ".join(person.aspects) if person.aspects else "none"

    return f"[{person.label}] " + ", ".join(parts) + f", {asc_txt}\nAspects: {aspects_txt}"


def format_transit_text(transit: TransitOut) -> str:
    parts = []
    for pname in TRANSIT_PLANETS:
        pl = transit.placements[pname]
        parts.append(f"{pname}: {pl.sign} {pl.degree}°")
    return f"[TransitToday {transit.date}] " + ", ".join(parts)


def placement_to_lon(p: Placement) -> float:
    sign_index = SIGNS.index(p.sign)
    return sign_index * 30.0 + float(p.degree)


def compute_synastry(a: PersonOut, b: PersonOut) -> List[str]:
    """
    MVP synastry: check selected A-planet vs B-planet pairs.
    Uses placements (rounded) so it's stable & fast.
    """
    a_lon = {k: placement_to_lon(v) for k, v in a.placements.items()}
    b_lon = {k: placement_to_lon(v) for k, v in b.placements.items()}

    found: List[str] = []
    for pa, pb in SYN_PAIRS:
        d = angle_diff(a_lon[pa], b_lon[pb])
        for asp_name, asp_deg in ASPECTS.items():
            if abs(d - asp_deg) <= SYN_ORB:
                found.append(f"A {pa} {asp_name} B {pb}")
                break
    return found


# -----------------------
# App
# -----------------------
app = FastAPI(title="Astro Bridge API", version="1.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/astro/summary", response_model=SummaryResponse)
def astro_summary(
    req: SummaryRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    require_api_key(authorization, x_api_key)

    tz_name = normalize_timezone(req.timezone)

    # basis_datetime: if omitted, server uses "now" in tz_name
    basis_dt = req.basis_datetime or datetime.now(tz.gettz(tz_name))

    a_out = build_person_out(req.personA, tz_name)
    t_out = build_transit_out(basis_dt, tz_name)

    # ✅ SOLO mode (only A)
    if req.personB is None:
        text = "\n\n".join([format_text(a_out), format_transit_text(t_out)])
        json_payload = {
            "mode": "solo",
            "personA": a_out.model_dump(),
            "transitToday": t_out.model_dump(),
            "basis": basis_dt.isoformat(),
            "timezone": tz_name,
        }
        return SummaryResponse(text=text, json=json_payload)

    # ✅ COUPLE mode (A + B + synastry)
    b_out = build_person_out(req.personB, tz_name)
    syn = compute_synastry(a_out, b_out)

    syn_txt = " ; ".join(syn) if syn else "none"

    text = "\n\n".join([
        format_text(a_out),
        format_text(b_out),
        f"[Synastry] {syn_txt}",
        format_transit_text(t_out),
    ])

    json_payload = {
        "mode": "couple",
        "personA": a_out.model_dump(),
        "personB": b_out.model_dump(),
        "synastryAspects": syn,
        "transitToday": t_out.model_dump(),
        "basis": basis_dt.isoformat(),
        "timezone": tz_name,
    }
    return SummaryResponse(text=text, json=json_payload)

from routers.saju import router as saju_router

app.include_router(saju_router)
