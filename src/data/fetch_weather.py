"""
Weather fetcher via meteostat v2 (interpolated point data).

For each of 7 German cities, queries nearby weather stations and
interpolates hourly observations to that exact lat/lon. Fetches in
2-year chunks (meteostat's default block is 3 years), saves all
cities as a single parquet file in data/raw/weather.parquet.

Usage (from project root):
    python -m src.data.fetch_weather
"""

from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import meteostat as ms
from tqdm import tqdm


# ---------- Configuration ----------
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()
CHUNK_YEARS = 2  # split into <=2-year chunks to stay under meteostat's limit

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RAW_DIR / "weather.parquet"

# Allow long requests as a belt-and-suspenders safeguard
ms.config.block_large_requests = False

# 7 cities covering demand centers + renewable generation regions.
CITIES = [
    ("berlin",    52.52,  13.41,  37),
    ("hamburg",   53.55,   9.99,   8),
    ("munich",    48.14,  11.58, 519),
    ("frankfurt", 50.11,   8.68, 112),
    ("essen",     51.45,   7.01, 116),
    ("leipzig",   51.34,  12.37, 113),
    ("stuttgart", 48.78,   9.18, 245),
]


def chunked_dates(start: datetime, end: datetime, years: int):
    """Yield (chunk_start, chunk_end) datetimes splitting [start, end] into year-length pieces."""
    cur = start
    while cur < end:
        nxt = datetime(cur.year + years, cur.month, cur.day)
        if nxt > end:
            nxt = end
        yield cur, nxt
        cur = nxt


def fetch_city(name: str, lat: float, lon: float, elev: int) -> pd.DataFrame:
    """Fetch interpolated hourly weather for one city across multiple chunks."""
    point = ms.Point(lat, lon, elev)
    stations = ms.stations.nearby(point, limit=4)
    if stations.empty:
        raise RuntimeError(f"No stations found near {name}")

    parts = []
    for chunk_start, chunk_end in chunked_dates(START_DATE, END_DATE, CHUNK_YEARS):
        try:
            ts = ms.hourly(stations, chunk_start, chunk_end)
            df = ms.interpolate(ts, point).fetch()
            if df is not None and not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"    chunk {chunk_start.date()}-{chunk_end.date()} failed: {e}")
        time.sleep(0.5)

    if not parts:
        raise RuntimeError(f"All chunks failed for {name}")

    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full = full.add_prefix(f"{name}_")
    return full


def main():
    print(f"Weather fetch via meteostat {ms.__version__}")
    print(f"Range: {START_DATE.date()} -> {END_DATE.date()}")
    print(f"Cities: {len(CITIES)}")

    if OUT_PATH.exists():
        print(f"\n[SKIP] {OUT_PATH} already exists. Delete it to refetch.")
        return

    frames = []
    for name, lat, lon, elev in tqdm(CITIES, desc="Cities"):
        try:
            df = fetch_city(name, lat, lon, elev)
            print(f"  {name}: {len(df):,} rows, {len(df.columns)} columns")
            frames.append(df)
        except Exception as e:
            print(f"  ! Failed to fetch {name}: {e}")

    if not frames:
        print("[ERROR] No weather data fetched.")
        return

    weather = pd.concat(frames, axis=1).sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]

    # Convert to Berlin time so it joins cleanly with ENTSO-E data
    if weather.index.tz is None:
        weather.index = weather.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    else:
        weather.index = weather.index.tz_convert("Europe/Berlin")

    weather.columns = [str(c) for c in weather.columns]

    weather.to_parquet(OUT_PATH, engine="fastparquet")
    print(f"\n[DONE] Saved {len(weather):,} rows x {len(weather.columns)} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()