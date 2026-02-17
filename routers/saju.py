from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Literal, Optional

import swisseph as swe
from fastapi import APIRouter
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

router = APIRouter(tags=["saju"])

# =========================================================
# 고정 규칙
# =========================================================
KST = ZoneInfo("Asia/Seoul")

# KST 표준 자오선(한국 표준시 기준): 135E
KST_STANDARD_MERIDIAN_LON = 135.0

# 기본 위치(서울)
SEOUL_LAT = 37.5665
SEOUL_LON = 126.9780

SWE_FLAGS = swe.FLG_MOSEPH  # ephemeris 파일 없이도 동작

STEMS = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
BRANCHES = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]

STEM_ELEM = ["wood","wood","fire","fire","earth","earth","metal","metal","water","water"]
STEM_YY   = ["yang","yin","yang","yin","yang","yin","yang","yin","yang","yin"]

BRANCH_ELEM = ["water","earth","wood","wood","earth","fire","fire","earth","metal","metal","earth","water"]
BRANCH_YY   = ["yang","yin","yang","yin","yang","yin","yang","yin","yang","yin","yang","yin"]

ELEM_KO = {"wood":"목","fire":"화","earth":"토","metal":"금","water":"수"}

# 12개의 '절'(입절 시각) 기준 경계 황경
TERM_LON_12 = {
    "Xiaohan": 285.0,  # 小寒
    "Lichun": 315.0,   # 立春 (입춘)
    "Jingzhe": 345.0,  # 惊蛰
    "Qingming": 15.0,  # 清明
    "Lixia": 45.0,     # 立夏
    "Mangzhong": 75.0, # 芒种
    "Xiaoshu": 105.0,  # 小暑
    "Liqiu": 135.0,    # 立秋
    "Bailu": 165.0,    # 白露
    "Hanlu": 195.0,    # 寒露
    "Lidong": 225.0,   # 立冬
    "Daxue": 255.0,    # 大雪
}

# 寅월 월간 시작 매핑(연간 기준)
YIN_MONTH_STEM_START = {
    0: 2,  # 甲 -> 丙
    5: 2,  # 己 -> 丙
    1: 4,  # 乙 -> 戊
    6: 4,  # 庚 -> 戊
    2: 6,  # 丙 -> 庚
    7: 6,  # 辛 -> 庚
    3: 8,  # 丁 -> 壬
    8: 8,  # 壬 -> 壬
    4: 0,  # 戊 -> 甲
    9: 0,  # 癸 -> 甲
}

MONTH_BRANCHES = ["寅","卯","辰","巳","午","未","申","酉","戌","亥","子","丑"]
BRANCH_TO_MONTH_INDEX = {b:i for i,b in enumerate(MONTH_BRANCHES)}

# =========================================================
# 입력 스키마
# =========================================================
class SajuRequest(BaseModel):
    name: str = Field(..., example="송주")
    gender: Literal["male", "female"] = Field(..., example="female")

    # ✅ 호환: birth_datetime / birth_datetime_kst 둘 다 받음
    # - "1998-07-15 19:10"
    # - "1998-07-15T19:10:00+09:00"
    birth_datetime: str = Field(
        ...,
        validation_alias=AliasChoices("birth_datetime", "birth_datetime_kst"),
        description="KST 기준 'YYYY-MM-DD HH:MM' 또는 ISO8601('1998-07-15T19:10:00+09:00')",
        example="1998-07-15 19:10",
    )

    # ✅ 호환: lat/lon 플랫도 받고, location 객체도 받음
    lat: Optional[float] = Field(None, example=37.5665)
    lon: Optional[float] = Field(None, example=126.9780)
    location: Optional[Dict[str, float]] = Field(
        default=None,
        description="대안 입력: {'lat': 37.56, 'lon': 126.97}",
        example={"lat": 37.5665, "lon": 126.9780},
    )

    birth_time_unknown: bool = Field(False, example=False)
    use_solar_time_correction: bool = Field(True, example=True)
    include_luck: bool = Field(False, example=False)
    target_year: Optional[int] = Field(None, example=2026)
    include_debug: bool = Field(False, example=False)

    # ✅ location이 있으면 lat/lon 자동 채우기
    @model_validator(mode="after")
    def _fill_location(self):
        if self.location:
            if self.lat is None and "lat" in self.location:
                self.lat = float(self.location["lat"])
            if self.lon is None and "lon" in self.location:
                self.lon = float(self.location["lon"])
        return self


# =========================================================
# Swiss Ephemeris helpers
# =========================================================
def jd_from_utc(dt_utc: datetime) -> float:
    dt_utc = dt_utc.astimezone(timezone.utc)
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    h = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600 + dt_utc.microsecond/3.6e9
    return swe.julday(y, m, d, h, swe.GREG_CAL)

def utc_from_jd(jd: float) -> datetime:
    y, m, d, h = swe.revjul(jd, swe.GREG_CAL)
    hour = int(h)
    minute = int((h-hour)*60)
    sec = int((((h-hour)*60)-minute)*60)
    micro = int(round(((((h-hour)*60)-minute)*60 - sec)*1e6))
    base = datetime(y, m, d, tzinfo=timezone.utc)
    return base + timedelta(hours=hour, minutes=minute, seconds=sec, microseconds=micro)

@lru_cache(maxsize=512)
def solcross_kst(year: int, lon_deg: float) -> datetime:
    start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    jd_start = jd_from_utc(start)
    jd_cross = swe.solcross_ut(float(lon_deg), jd_start, SWE_FLAGS)
    return utc_from_jd(jd_cross).astimezone(KST)

def apply_longitude_correction(dt_kst: datetime, lon: float) -> datetime:
    # (lon - 135) * 4분 보정
    delta_minutes = (lon - KST_STANDARD_MERIDIAN_LON) * 4.0
    return dt_kst + timedelta(minutes=delta_minutes)

# =========================================================
# Ganzhi core
# =========================================================
def jdn_gregorian(y: int, m: int, d: int) -> int:
    a = (14 - m)//12
    y2 = y + 4800 - a
    m2 = m + 12*a - 3
    return d + (153*m2 + 2)//5 + 365*y2 + y2//4 - y2//100 + y2//400 - 32045

def sexagenary_index(stem_idx: int, branch_idx: int) -> int:
    for n in range(60):
        if n % 10 == stem_idx and n % 12 == branch_idx:
            return n
    raise ValueError("Invalid stem/branch")

def year_pillar(dt_kst: datetime) -> tuple[int,int,int,datetime]:
    y = dt_kst.year
    ipchun = solcross_kst(y, TERM_LON_12["Lichun"])
    adj_y = y-1 if dt_kst < ipchun else y
    stem = (adj_y - 4) % 10
    branch = (adj_y - 4) % 12
    return stem, branch, adj_y, ipchun

def month_pillar(dt_kst: datetime, year_stem_idx: int) -> tuple[int,int]:
    y = dt_kst.year

    daxue_prev = solcross_kst(y-1, TERM_LON_12["Daxue"])
    xiaohan = solcross_kst(y, TERM_LON_12["Xiaohan"])
    lichun = solcross_kst(y, TERM_LON_12["Lichun"])
    jingzhe = solcross_kst(y, TERM_LON_12["Jingzhe"])
    qingming = solcross_kst(y, TERM_LON_12["Qingming"])
    lixia = solcross_kst(y, TERM_LON_12["Lixia"])
    mangzhong = solcross_kst(y, TERM_LON_12["Mangzhong"])
    xiaoshu = solcross_kst(y, TERM_LON_12["Xiaoshu"])
    liqiu = solcross_kst(y, TERM_LON_12["Liqiu"])
    bailu = solcross_kst(y, TERM_LON_12["Bailu"])
    hanlu = solcross_kst(y, TERM_LON_12["Hanlu"])
    lidong = solcross_kst(y, TERM_LON_12["Lidong"])
    daxue = solcross_kst(y, TERM_LON_12["Daxue"])
    xiaohan_next = solcross_kst(y+1, TERM_LON_12["Xiaohan"])

    boundaries = sorted([
        (daxue_prev, "子"),
        (xiaohan, "丑"),
        (lichun, "寅"),
        (jingzhe, "卯"),
        (qingming, "辰"),
        (lixia, "巳"),
        (mangzhong, "午"),
        (xiaoshu, "未"),
        (liqiu, "申"),
        (bailu, "酉"),
        (hanlu, "戌"),
        (lidong, "亥"),
        (daxue, "子"),
        (xiaohan_next, "丑"),
    ], key=lambda x: x[0])

    month_branch = None
    for i in range(len(boundaries)-1):
        start_dt, br = boundaries[i]
        end_dt, _ = boundaries[i+1]
        if start_dt <= dt_kst < end_dt:
            month_branch = br
            break
    if month_branch is None:
        month_branch = boundaries[0][1]

    start_stem = YIN_MONTH_STEM_START[year_stem_idx]
    m_index = BRANCH_TO_MONTH_INDEX[month_branch]  # 寅=0 ... 丑=11
    m_stem = (start_stem + m_index) % 10
    m_branch = BRANCHES.index(month_branch)
    return m_stem, m_branch

def day_pillar(dt_kst_solar: datetime) -> tuple[int,int]:
    # (JDN + 49) % 60 오프셋은 제공된 승인 케이스 2개와 맞도록 고정
    y, m, d = dt_kst_solar.year, dt_kst_solar.month, dt_kst_solar.day
    jdn = jdn_gregorian(y, m, d)
    idx = (jdn + 49) % 60
    return idx % 10, idx % 12

def hour_pillar(dt_kst_solar: datetime, day_stem_idx: int) -> tuple[int,int]:
    # 자시: 23:00~00:59 -> branchIndex = ((hour+1)//2)%12
    h = dt_kst_solar.hour
    hour_branch = ((h + 1)//2) % 12
    hour_stem = (day_stem_idx * 2 + hour_branch) % 10
    return hour_stem, hour_branch

# =========================================================
# Ten gods / elements / yin-yang
# =========================================================
def ten_god(day_stem_idx: int, target_elem: str, target_yy: str) -> str:
    day_elem = STEM_ELEM[day_stem_idx]
    day_yy = STEM_YY[day_stem_idx]
    same_yy = (day_yy == target_yy)

    gen = {"wood":"fire","fire":"earth","earth":"metal","metal":"water","water":"wood"}
    ctrl = {"wood":"earth","earth":"water","water":"fire","fire":"metal","metal":"wood"}

    if target_elem == day_elem:
        return "비견" if same_yy else "겁재"
    if gen[day_elem] == target_elem:
        return "식신" if same_yy else "상관"
    if ctrl[day_elem] == target_elem:
        return "편재" if same_yy else "정재"
    if ctrl[target_elem] == day_elem:
        return "편관" if same_yy else "정관"
    if gen[target_elem] == day_elem:
        return "편인" if same_yy else "정인"
    return "미정"

def pct(x: float) -> float:
    return round(x, 2)

# =========================================================
# Luck (옵션) — include_luck=True일 때만 계산
# =========================================================
def _sun_lon_deg(dt_kst: datetime) -> float:
    jd = jd_from_utc(dt_kst.astimezone(timezone.utc))
    pos, _ = swe.calc_ut(jd, swe.SUN, SWE_FLAGS | swe.FLG_SPEED)
    return float(pos[0] % 360.0)

def _next_prev_solar_term(dt_kst: datetime) -> tuple[float, datetime, float, datetime]:
    lon = _sun_lon_deg(dt_kst)
    k = int(lon // 15)
    prev_lon = (k * 15) % 360
    next_lon = ((k + 1) * 15) % 360

    jd = jd_from_utc(dt_kst.astimezone(timezone.utc))
    jd_next = swe.solcross_ut(next_lon, jd, SWE_FLAGS)
    dt_next = utc_from_jd(jd_next).astimezone(KST)

    jd_prev = swe.solcross_ut(prev_lon, jd - 370, SWE_FLAGS)
    dt_prev = utc_from_jd(jd_prev).astimezone(KST)
    if dt_prev > dt_kst:
        jd_prev = swe.solcross_ut(prev_lon, jd - 730, SWE_FLAGS)
        dt_prev = utc_from_jd(jd_prev).astimezone(KST)

    return prev_lon, dt_prev, next_lon, dt_next

def build_luck(
    birth_dt_kst: datetime,
    gender: Literal["male","female"],
    year_stem_idx: int,
    month_stem_idx: int,
    month_branch_idx: int,
    target_year: int,
) -> Dict[str, Any]:
    is_year_yang = (STEM_YY[year_stem_idx] == "yang")
    forward = (gender == "male" and is_year_yang) or (gender == "female" and not is_year_yang)

    prev_lon, dt_prev, next_lon, dt_next = _next_prev_solar_term(birth_dt_kst)
    diff_days = (dt_next - birth_dt_kst).total_seconds()/86400 if forward else (birth_dt_kst - dt_prev).total_seconds()/86400
    start_age = diff_days / 3.0

    month_idx = sexagenary_index(month_stem_idx, month_branch_idx)
    step = 1 if forward else -1

    daewoon = []
    for i in range(10):
        idx = (month_idx + step*(i+1)) % 60
        daewoon.append({
            "ganzhi": STEMS[idx%10] + BRANCHES[idx%12],
            "start_age": round(start_age + i*10, 2),
            "end_age": round(start_age + (i+1)*10, 2),
        })

    saewoon = []
    for y in range(target_year-9, target_year+10):
        s = (y - 4) % 10
        b = (y - 4) % 12
        saewoon.append({"year": y, "ganzhi": STEMS[s] + BRANCHES[b]})

    # 월운(절기 기반) — 구간 정보 포함
    ystem = (target_year - 4) % 10
    start_stem = YIN_MONTH_STEM_START[ystem]

    t_lichun = solcross_kst(target_year, TERM_LON_12["Lichun"])
    t_jingzhe = solcross_kst(target_year, TERM_LON_12["Jingzhe"])
    t_qingming = solcross_kst(target_year, TERM_LON_12["Qingming"])
    t_lixia = solcross_kst(target_year, TERM_LON_12["Lixia"])
    t_mangzhong = solcross_kst(target_year, TERM_LON_12["Mangzhong"])
    t_xiaoshu = solcross_kst(target_year, TERM_LON_12["Xiaoshu"])
    t_liqiu = solcross_kst(target_year, TERM_LON_12["Liqiu"])
    t_bailu = solcross_kst(target_year, TERM_LON_12["Bailu"])
    t_hanlu = solcross_kst(target_year, TERM_LON_12["Hanlu"])
    t_lidong = solcross_kst(target_year, TERM_LON_12["Lidong"])
    t_daxue = solcross_kst(target_year, TERM_LON_12["Daxue"])
    t_xiaohan_next = solcross_kst(target_year+1, TERM_LON_12["Xiaohan"])

    month_starts = [
        t_lichun, t_jingzhe, t_qingming, t_lixia, t_mangzhong, t_xiaoshu,
        t_liqiu, t_bailu, t_hanlu, t_lidong, t_daxue, t_xiaohan_next
    ]

    wolwoon = []
    for i, br in enumerate(MONTH_BRANCHES):
        ms = (start_stem + i) % 10
        mb = BRANCHES.index(br)
        start_dt = month_starts[i]
        end_dt = month_starts[i+1] if i+1 < len(month_starts) else None
        wolwoon.append({
            "month_index": i+1,
            "ganzhi": STEMS[ms] + BRANCHES[mb],
            "start_kst": start_dt.isoformat(),
            "end_kst": end_dt.isoformat() if end_dt else None,
        })

    return {
        "direction": "forward" if forward else "backward",
        "start_age_years": round(start_age, 2),
        "start_age_basis": {
            "prev_term_lon": prev_lon,
            "prev_term_kst": dt_prev.isoformat(),
            "next_term_lon": next_lon,
            "next_term_kst": dt_next.isoformat(),
        },
        "daewoon": daewoon,
        "saewoon": saewoon,
        "wolwoon": wolwoon,
    }

# =========================================================
# API endpoint
# =========================================================
@router.post("/v1/saju/calc")
def saju_calc(req: SajuRequest) -> Dict[str, Any]:
    lat = req.lat if req.lat is not None else SEOUL_LAT
    lon = req.lon if req.lon is not None else SEOUL_LON

    dt_kst = datetime.strptime(req.birth_datetime, "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    if req.birth_time_unknown:
        dt_kst = dt_kst.replace(hour=12, minute=0)

    dt_solar = apply_longitude_correction(dt_kst, lon) if req.use_solar_time_correction else dt_kst

    y_stem, y_branch, adj_year, ipchun = year_pillar(dt_kst)
    m_stem, m_branch = month_pillar(dt_kst, y_stem)
    d_stem, d_branch = day_pillar(dt_solar)
    h_stem, h_branch = hour_pillar(dt_solar, d_stem)

    pillars = {"year": (y_stem,y_branch), "month": (m_stem,m_branch), "day": (d_stem,d_branch), "hour": (h_stem,h_branch)}

    def pillar_obj(s: int, b: int) -> Dict[str, Any]:
        return {
            "stem": {
                "hanja": STEMS[s],
                "element": ELEM_KO[STEM_ELEM[s]],
                "yin_yang": "양" if STEM_YY[s] == "yang" else "음",
            },
            "branch": {
                "hanja": BRANCHES[b],
                "element": ELEM_KO[BRANCH_ELEM[b]],
                "yin_yang": "양" if BRANCH_YY[b] == "yang" else "음",
            },
            "ganzhi": STEMS[s] + BRANCHES[b],
        }

    pillars_out = {k: pillar_obj(v[0], v[1]) for k, v in pillars.items()}

    ten_gods = {}
    for k, (s, b) in pillars.items():
        ten_gods[k] = {
            "stem": ten_god(d_stem, STEM_ELEM[s], STEM_YY[s]),
            "branch": ten_god(d_stem, BRANCH_ELEM[b], BRANCH_YY[b]),
        }

    elems = [STEM_ELEM[x] for x in [y_stem, m_stem, d_stem, h_stem]] + [BRANCH_ELEM[x] for x in [y_branch, m_branch, d_branch, h_branch]]
    counts = {e: 0 for e in ["wood","fire","earth","metal","water"]}
    for e in elems:
        counts[e] += 1

    elements_distribution = [
        {"element": ELEM_KO[e], "count": counts[e], "percent": pct(counts[e]/8*100)}
        for e in ["wood","fire","earth","metal","water"]
    ]

    yy_counts = {"yang": 0, "yin": 0}
    for s in [y_stem, m_stem, d_stem, h_stem]:
        yy_counts[STEM_YY[s]] += 1
    for b in [y_branch, m_branch, d_branch, h_branch]:
        yy_counts[BRANCH_YY[b]] += 1

    yin_yang_ratio = {
        "yang": {"count": yy_counts["yang"], "percent": pct(yy_counts["yang"]/8*100)},
        "yin": {"count": yy_counts["yin"], "percent": pct(yy_counts["yin"]/8*100)},
    }

    if req.target_year is None:
        target_year = datetime.now(tz=KST).year
    else:
        target_year = req.target_year

    luck = None
    if req.include_luck:
        luck = build_luck(
            birth_dt_kst=dt_kst,
            gender=req.gender,
            year_stem_idx=y_stem,
            month_stem_idx=m_stem,
            month_branch_idx=m_branch,
            target_year=target_year,
        )

    out: Dict[str, Any] = {
        "name": req.name,
        "gender": req.gender,
        "timezone": "Asia/Seoul",
        "birth_datetime_kst": dt_kst.isoformat(),
        "birth_datetime_solar_kst": dt_solar.isoformat(),
        "location": {"lat": lat, "lon": lon},
        "pillars": pillars_out,
        "ten_gods": ten_gods,
        "elements_distribution": elements_distribution,
        "yin_yang_ratio": yin_yang_ratio,
        "luck": luck,
        "assumptions": {
            "birth_time_unknown_assumed_noon": bool(req.birth_time_unknown),
            "use_solar_time_correction": bool(req.use_solar_time_correction),
            "solar_time_correction_rule": "KST 표준경도 135E 기준, (lon-135)*4분 보정",
            "year_boundary": "입춘(立春) 기준",
            "month_boundary": "절기(입절) 기준",
            "hour_boundary": "자시 23:00 시작",
            "elements_count_rule": "천간4+지지4=8(지장간 제외)",
        }
    }

    if req.include_debug:
        out["debug"] = {
            "ipchun_kst": ipchun.isoformat(),
            "adjusted_year_for_year_pillar": adj_year,
        }

    return out

# =========================================================
# Summary endpoints (GPT에 붙이기 좋은 짧은 출력)
# =========================================================

def _fmt_elements_line(elements_distribution):
    # elements_distribution: [{"element":"목","count":1,"percent":12.5}, ...]
    parts = []
    parts_pct = []
    for e in elements_distribution:
        parts.append(f'{e["element"]}{e["count"]}')
        parts_pct.append(f'{e["element"]}{e["percent"]}%')
    return " / ".join(parts), " / ".join(parts_pct)

def _fmt_tengods_line(ten_gods):
    # ten_gods: {"year":{"stem":"정관","branch":"상관"}, ...}
    def one(p):
        return f'{p["stem"]}/{p["branch"]}'
    return (
        f'year {one(ten_gods["year"])}, '
        f'month {one(ten_gods["month"])}, '
        f'day {one(ten_gods["day"])}, '
        f'hour {one(ten_gods["hour"])}'
    )

def _fmt_pillars_line(pillars):
    # pillars: {"year":{"ganzhi":"戊寅"}, ...}
    return f'year {pillars["year"]["ganzhi"]} | month {pillars["month"]["ganzhi"]} | day {pillars["day"]["ganzhi"]} | hour {pillars["hour"]["ganzhi"]}'

def _fmt_day_master(pillars):
    # day stem info
    ds = pillars["day"]["stem"]["hanja"]
    de = pillars["day"]["stem"]["element"]
    yy = pillars["day"]["stem"]["yin_yang"]
    return f'{ds}({de}/{yy})'

def _fmt_yinyang_line(yin_yang_ratio):
    yang = yin_yang_ratio["yang"]
    yin = yin_yang_ratio["yin"]
    return f'양 {yang["count"]}({yang["percent"]}%) / 음 {yin["count"]}({yin["percent"]}%)'

def _fmt_luck_preview(luck):
    # 너무 길어지지 않게 맛보기만
    if not luck:
        return None
    daewoon = luck.get("daewoon") or []
    preview = daewoon[:3]
    preview_str = ", ".join([f'{x["ganzhi"]}({x["start_age"]}~{x["end_age"]})' for x in preview])
    return (
        f'direction={luck.get("direction")}, '
        f'start_age={luck.get("start_age_years")}, '
        f'daewoon_top3=[{preview_str}]'
    )

@router.post("/v1/saju/summary")
def saju_summary(req: SajuRequest) -> Dict[str, Any]:
    """
    - /v1/saju/calc 결과를 기반으로
    - GPT 프롬프트에 바로 붙이기 좋은 짧은 text + 압축 json을 반환
    """
    base = saju_calc(req)

    pillars = base["pillars"]
    ten_gods = base["ten_gods"]
    elements_distribution = base["elements_distribution"]
    yin_yang_ratio = base["yin_yang_ratio"]

    elem_counts, elem_pcts = _fmt_elements_line(elements_distribution)
    tg_line = _fmt_tengods_line(ten_gods)
    pillars_line = _fmt_pillars_line(pillars)
    day_master = _fmt_day_master(pillars)
    yy_line = _fmt_yinyang_line(yin_yang_ratio)

    luck_preview = _fmt_luck_preview(base.get("luck"))

    # 텍스트는 "짧고 일정한 포맷" 고정
    lines = [
        "[SAJU_SUMMARY]",
        f'name: {base["name"]} ({base["gender"]})',
        f'birth_kst: {base["birth_datetime_kst"]}',
        f'pillars: {pillars_line}',
        f'day_master: {day_master}',
        f'ten_gods: {tg_line}',
        f'elements(8): {elem_counts}',
        f'elements(%): {elem_pcts}',
        f'yin_yang(8): {yy_line}',
        "rules: 입춘(연) / 절기(월) / 자시23(시) / 지장간 미포함(오행8)",
    ]
    if base.get("assumptions", {}).get("use_solar_time_correction"):
        lines.append("solar_time: ON (KST 135E 기준 경도보정)")
    else:
        lines.append("solar_time: OFF")

    if luck_preview:
        lines.append(f'luck: {luck_preview}')

    text = "\n".join(lines)

    # 압축 JSON (프롬프트/저장용)
    compact_json = {
        "name": base["name"],
        "gender": base["gender"],
        "birth_datetime_kst": base["birth_datetime_kst"],
        "pillars": {
            "year": pillars["year"]["ganzhi"],
            "month": pillars["month"]["ganzhi"],
            "day": pillars["day"]["ganzhi"],
            "hour": pillars["hour"]["ganzhi"],
        },
        "day_master": pillars["day"]["stem"]["hanja"],
        "ten_gods": ten_gods,
        "elements_distribution": elements_distribution,
        "yin_yang_ratio": yin_yang_ratio,
        "luck": base.get("luck"),
        "assumptions": base.get("assumptions"),
    }

    return {"text": text, "json": compact_json}
