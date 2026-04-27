"""
ENTSO-E data fetcher for German electricity market.

Pulls hourly time series from the ENTSO-E Transparency Platform
for the DE_LU bidding zone, saves as parquet files in data/raw/.

Usage (from project root):
    python -m src.data.fetch_entsoe
"""

import os
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
from tqdm import tqdm


# ---------- Configuration ----------
ZONE = "DE_LU"
START_YEAR = 2020
END_DATE = pd.Timestamp.now(tz="Europe/Berlin").normalize()
TIMEZONE = "Europe/Berlin"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def get_client() -> EntsoePandasClient:
    """Load API key from .env and return a client."""
    load_dotenv()
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("ENTSOE_API_KEY not found in .env file")
    return EntsoePandasClient(api_key=key)


def yearly_chunks(start_year: int, end_date: pd.Timestamp):
    """Yield (start, end) tuples for each year, last chunk ending at end_date."""
    for year in range(start_year, end_date.year + 1):
        chunk_start = pd.Timestamp(f"{year}-01-01", tz=TIMEZONE)
        chunk_end = pd.Timestamp(f"{year + 1}-01-01", tz=TIMEZONE)
        if chunk_end > end_date:
            chunk_end = end_date
        if chunk_start >= end_date:
            break
        yield chunk_start, chunk_end


def to_hourly(series_or_df):
    """Aggregate sub-hourly data to hourly mean (handles 15-min DE prices)."""
    return series_or_df.resample("1h").mean()


def fetch_with_retry(fn, *args, max_retries=5, retry_delay=30, **kwargs):
    """Wrap an API call with retry on transient errors. Exponential backoff for 503/504."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e)
            # Longer wait for server overload errors specifically
            if "503" in err_str or "504" in err_str or "502" in err_str:
                wait = retry_delay * (2 ** attempt)  # 30s, 60s, 120s, 240s, 480s
                print(f"  Attempt {attempt + 1} failed (server error): waiting {wait}s")
            else:
                wait = retry_delay
                print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait)
    raise last_err


# ---------- Fetch routines ----------
def fetch_series(name: str, fetch_fn, client, *fn_args, **fn_kwargs):
    """
    Fetch a time series in yearly chunks, aggregate to hourly,
    save as a single parquet file under data/raw/<name>.parquet.

    Skips fetching if the file already exists.
    """
    out_path = RAW_DIR / f"{name}.parquet"
    if out_path.exists():
        print(f"[SKIP] {name} (already at {out_path})")
        return

    print(f"\n[FETCH] {name}")
    chunks = list(yearly_chunks(START_YEAR, END_DATE))
    parts = []

    for start, end in tqdm(chunks, desc=f"  {name}"):
        try:
            data = fetch_with_retry(
                fetch_fn, client, *fn_args, start=start, end=end, **fn_kwargs
            )
            if data is not None and len(data) > 0:
                parts.append(data)
        except Exception as e:
            print(f"  Skipping {start.year}: {e}")

    if not parts:
        print(f"  ! No data fetched for {name}")
        return

    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full = to_hourly(full)

    # Convert Series to DataFrame so to_parquet works (pandas 3.0+ requirement)
    if isinstance(full, pd.Series):
        full = full.to_frame(name=name)

    # Sanitize column names for parquet (no None, no tuples, all strings)
    if isinstance(full.columns, pd.MultiIndex):
        full.columns = [
            "_".join(str(level) for level in col if level)
            for col in full.columns
        ]
    full.columns = [
        str(col) if col is not None else f"col_{i}"
        for i, col in enumerate(full.columns)
    ]

    full.to_parquet(out_path, engine="fastparquet")
    print(f"  Saved {len(full):,} hourly rows to {out_path}")


# ---------- Wrappers around entsoe-py methods ----------
def _prices(client, start, end):
    return client.query_day_ahead_prices(ZONE, start=start, end=end)


def _load(client, start, end):
    return client.query_load(ZONE, start=start, end=end)


def _load_forecast(client, start, end):
    return client.query_load_forecast(ZONE, start=start, end=end)


def _generation(client, start, end):
    return client.query_generation(ZONE, start=start, end=end, psr_type=None)


def _wind_solar_forecast(client, start, end):
    return client.query_wind_and_solar_forecast(ZONE, start=start, end=end)


# ---------- Main ----------
def main():
    print(f"ENTSO-E fetch for {ZONE}")
    print(f"Range: {START_YEAR}-01-01 → {END_DATE.date()}")
    print(f"Output: {RAW_DIR.resolve()}")

    client = get_client()

    fetch_series("prices", _prices, client)
    fetch_series("load_actual", _load, client)
    fetch_series("load_forecast", _load_forecast, client)
    fetch_series("generation_actual", _generation, client)
    fetch_series("wind_solar_forecast", _wind_solar_forecast, client)

    print("\n[DONE] Fetched all series.")


if __name__ == "__main__":
    main()