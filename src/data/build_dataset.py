"""
Master dataset assembly.

Loads all 6 raw parquet files (5 ENTSO-E + weather), joins them on the
timestamp index, engineers a clean feature set, and saves a single
analysis-ready parquet to data/processed/dataset.parquet.

Usage (from project root):
    python -m src.data.build_dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------- Paths ----------
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "dataset.parquet"

# ---------- City weights for weather aggregation ----------
# Rough population-based weights for a national weather average
# (proxies for where electricity demand actually concentrates).
CITY_WEIGHTS = {
    "berlin":    0.18,
    "hamburg":   0.10,
    "munich":    0.12,
    "frankfurt": 0.10,
    "essen":     0.20,   # Ruhr metro area is the biggest demand cluster
    "leipzig":   0.07,
    "stuttgart": 0.10,
}
# Renormalize so weights sum to 1 (they should, but just in case):
_total = sum(CITY_WEIGHTS.values())
CITY_WEIGHTS = {k: v / _total for k, v in CITY_WEIGHTS.items()}


# ---------- Loaders ----------
def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / f"{name}.parquet", engine="fastparquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    else:
        df.index = df.index.tz_convert("Europe/Berlin")
    return df


# ---------- Generation cleanup ----------
def clean_generation(gen: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the 25 raw generation columns into a clean per-fuel feature set.

    For each fuel: net_<fuel> = aggregated - consumption (where applicable).
    Drops tiny categories. Renames to short snake_case.
    """
    out = pd.DataFrame(index=gen.index)

    # Mapping: short_name -> (aggregated_col, optional_consumption_col)
    fuels = {
        "biomass":      ("Biomass_Actual Aggregated", None),
        "lignite":      ("Fossil Brown coal/Lignite_Actual Aggregated", None),
        "gas":          ("Fossil Gas_Actual Aggregated", "Fossil Gas_Actual Consumption"),
        "hard_coal":    ("Fossil Hard coal_Actual Aggregated", None),
        "oil":          ("Fossil Oil_Actual Aggregated", "Fossil Oil_Actual Consumption"),
        "hydro_runof":  ("Hydro Run-of-river and poundage_Actual Aggregated", None),
        "hydro_pump":   ("Hydro Pumped Storage_Actual Aggregated",
                         "Hydro Pumped Storage_Actual Consumption"),
        "hydro_res":    ("Hydro Water Reservoir_Actual Aggregated",
                         "Hydro Water Reservoir_Actual Consumption"),
        "nuclear":      ("Nuclear_Actual Aggregated", "Nuclear_Actual Consumption"),
        "solar":        ("Solar_Actual Aggregated", "Solar_Actual Consumption"),
        "wind_on":      ("Wind Onshore_Actual Aggregated", "Wind Onshore_Actual Consumption"),
        "wind_off":     ("Wind Offshore_Actual Aggregated", None),
        "waste":        ("Waste_Actual Aggregated", None),
        "other_renew":  ("Other renewable_Actual Aggregated", "Other renewable_Actual Consumption"),
    }

    for short, (agg_col, cons_col) in fuels.items():
        if agg_col not in gen.columns:
            continue
        agg = gen[agg_col].fillna(0)
        if cons_col and cons_col in gen.columns:
            cons = gen[cons_col].fillna(0)
            out[f"gen_{short}"] = agg - cons
        else:
            out[f"gen_{short}"] = agg

    # Total renewable & total generation aggregates
    renew_cols = [c for c in out.columns
                  if c in ("gen_solar", "gen_wind_on", "gen_wind_off",
                           "gen_hydro_runof", "gen_hydro_res", "gen_biomass",
                           "gen_other_renew")]
    out["gen_renewable_total"] = out[renew_cols].sum(axis=1)

    fossil_cols = [c for c in out.columns
                   if c in ("gen_lignite", "gen_gas", "gen_hard_coal", "gen_oil")]
    out["gen_fossil_total"] = out[fossil_cols].sum(axis=1)

    out["gen_total"] = out[[c for c in out.columns if c.startswith("gen_")
                            and c not in ("gen_renewable_total", "gen_fossil_total")]].sum(axis=1)

    return out


# ---------- Weather aggregation ----------
def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Take 7-city x 11-variable weather and produce:
      - weighted national means for key variables (temp, wind, radiation, humidity, pressure)
      - per-city wind speeds (kept separate; wind generation is regional)
    """
    out = pd.DataFrame(index=weather.index)
    cities = list(CITY_WEIGHTS.keys())

    # Variables to aggregate as weighted national means
    weighted = {
        "temp": "national_temp",      # for demand
        "rhum": "national_humidity",
        "pres": "national_pressure",
        "tsun": "national_sunshine",  # solar proxy if no radiation column
        "cldc": "national_cloud",
    }

    for var, out_name in weighted.items():
        cols = [f"{c}_{var}" for c in cities if f"{c}_{var}" in weather.columns]
        if not cols:
            continue
        # Build weight vector aligned with cols
        weights = np.array([CITY_WEIGHTS[c.split("_")[0]] for c in cols])
        weights /= weights.sum()
        # Weighted mean ignoring NaN per-row
        vals = weather[cols].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            mask = ~np.isnan(vals)
            wmat = np.broadcast_to(weights, vals.shape) * mask
            num = np.where(mask, vals * wmat, 0).sum(axis=1)
            den = wmat.sum(axis=1)
            out[out_name] = np.where(den > 0, num / den, np.nan)

    # Per-city wind speeds (renewables are spatial, keep separate)
    for city in cities:
        wcol = f"{city}_wspd"
        if wcol in weather.columns:
            out[f"wind_{city}"] = weather[wcol]

    # National max wind (proxy for severe-weather signal)
    wind_cols = [c for c in out.columns if c.startswith("wind_")]
    if wind_cols:
        out["wind_max"] = out[wind_cols].max(axis=1)
        out["wind_mean"] = out[wind_cols].mean(axis=1)

    return out


# ---------- Main assembly ----------
def main():
    print("Loading raw datasets...")
    prices = load("prices")                          # Series-shape (1 col)
    load_actual = load("load_actual")                # 1 col
    load_forecast = load("load_forecast")            # 1 col
    gen_raw = load("generation_actual")              # 25 cols
    ws_forecast = load("wind_solar_forecast")        # 3 cols
    weather = load("weather")                        # 77 cols

    for name, df in [("prices", prices), ("load_actual", load_actual),
                     ("load_forecast", load_forecast), ("gen_raw", gen_raw),
                     ("ws_forecast", ws_forecast), ("weather", weather)]:
        print(f"  {name:14s} shape={df.shape}, range={df.index.min()} -> {df.index.max()}")

    print("\nCleaning generation...")
    gen = clean_generation(gen_raw)
    print(f"  -> {gen.shape[1]} clean generation columns")

    print("Aggregating weather...")
    wx = aggregate_weather(weather)
    print(f"  -> {wx.shape[1]} aggregated weather columns")

    print("\nJoining all sources on timestamp...")
    df = (
        prices
        .rename(columns={"prices": "price"})
        .join(load_actual.rename(columns={"Actual Load": "load_actual"}), how="left")
        .join(load_forecast.rename(columns={"Forecasted Load": "load_forecast"}), how="left")
        .join(ws_forecast.rename(columns={
            "Solar":         "fcst_solar",
            "Wind Onshore":  "fcst_wind_on",
            "Wind Offshore": "fcst_wind_off",
        }), how="left")
        .join(gen, how="left")
        .join(wx, how="left")
    )

    print(f"  Joined shape: {df.shape}")

    # ---------- Quality checks ----------
    print("\nQuality checks:")
    n_total = len(df)
    print(f"  Total rows: {n_total:,}")
    print(f"  Duplicated indices: {df.index.duplicated().sum()}")
    print(f"  Date range: {df.index.min()} -> {df.index.max()}")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="Europe/Berlin")
    missing = expected.difference(df.index)
    print(f"  Missing hourly timestamps in range: {len(missing)}")

    print("\n  Top 10 columns by null count:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    # Sanity ranges
    print("\n  Sanity stats on key columns:")
    for col in ["price", "load_actual", "national_temp", "fcst_wind_on", "gen_renewable_total"]:
        if col in df.columns:
            s = df[col]
            print(f"    {col:24s}  min={s.min():>10.2f}  mean={s.mean():>10.2f}  max={s.max():>10.2f}  nulls={s.isnull().sum()}")

    # ---------- Light cleanup ----------
    print("\nLight cleanup:")
    # Forward-fill very short gaps (up to 3 hours)
    before = df.isnull().sum().sum()
    df = df.ffill(limit=3)
    after = df.isnull().sum().sum()
    print(f"  Forward-filled {before - after:,} null cells (gaps up to 3 hours)")

    # Drop rows where the target (price) is null - can't train on those
    n_before = len(df)
    df = df.dropna(subset=["price"])
    print(f"  Dropped {n_before - len(df)} rows where price was null")

    # ---------- Save ----------
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(OUT_PATH, engine="fastparquet")
    print(f"\n[DONE] Saved {df.shape[0]:,} rows x {df.shape[1]} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()