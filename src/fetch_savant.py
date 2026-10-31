"""Fetch regular-season pitches from Baseball Savant, one day per request, into a season parquet.

Usage:
  python src/fetch_savant.py <season> <out.parquet> [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--base existing.parquet]
  python src/fetch_savant.py --check <season> <kaggle_season.parquet> <start> <end>
      (fetch those days and require the same rows and values as the Kaggle file)
Only the columns the pipeline reads are kept (pfc.COLS).
"""
import io
import sys
import time
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

import pfc

URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=pitcher"
       "&hfGT=R%7C&hfSea={season}%7C&game_date_gt={d}&game_date_lt={d}")
UA = "mlb-pitch-forecast (github.com/yasumorishima/mlb-pitch-forecast)"


def fetch_day(season, d, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(URL.format(season=season, d=d), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            if not body.strip():
                return pd.DataFrame(columns=pfc.COLS)
            df = pd.read_csv(io.BytesIO(body), low_memory=False)
            if "game_pk" not in df.columns:
                raise ValueError("unexpected response")
            assert len(df) < 25000, f"{d}: {len(df)} rows hits the export cap"
            return df[pfc.COLS]
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                raise
            print(f"{d}: retry after {type(e).__name__}", flush=True)
            time.sleep(10 * (t + 1))


def fetch_range(season, start, end):
    parts, d = [], start
    while d <= end:
        df = fetch_day(season, d.isoformat())
        print(d, len(df), flush=True)
        if len(df):
            parts.append(df)
        d += timedelta(days=1)
        time.sleep(1.0)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=pfc.COLS)


def normalise(df):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def check(season, kaggle_path, start, end):
    got = normalise(fetch_range(season, start, end))
    ref = pd.read_parquet(kaggle_path, columns=pfc.COLS)
    ref = ref[ref["game_type"] == "R"]
    ref = ref[(pd.to_datetime(ref["game_date"]).dt.date >= start) & (pd.to_datetime(ref["game_date"]).dt.date <= end)]
    ref = normalise(ref)
    assert len(got) == len(ref), f"rows: savant {len(got)} vs kaggle {len(ref)}"
    bad = []
    for c in pfc.COLS:
        a, b = got[c], ref[c]
        if pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(pd.to_numeric(a, errors="coerce").astype("float64").to_numpy(),
                            pd.to_numeric(b, errors="coerce").astype("float64").to_numpy(),
                            equal_nan=True, rtol=0, atol=1e-6)      # ids must match exactly
        else:
            ok = a.astype("string").fillna("<NA>").to_numpy() == b.astype("string").fillna("<NA>").to_numpy()
        if not ok.all():
            bad.append((c, int((~ok).sum())))
    print("rows", len(got), "columns differing:", bad)
    # Savant revises some pitch-type labels after the fact (52 of 26,303 pitches, 0.2%, between the
    # Kaggle build of 2026-10-09 and 2026-10-31). Every other column must be identical.
    revised = dict(bad).pop("pitch_type", 0)
    other = [b for b in bad if b[0] != "pitch_type"]
    if revised:
        m = got["pitch_type"].astype("string").fillna("<NA>").to_numpy() != ref["pitch_type"].astype("string").fillna("<NA>").to_numpy()
        pairs = pd.Series(list(zip(ref["pitch_type"][m].astype("string").fillna("<NA>"), got["pitch_type"][m].astype("string").fillna("<NA>"))))
        print("pitch_type revisions (kaggle -> savant now):", pairs.value_counts().to_dict())
    assert not other, other
    assert revised <= 0.01 * len(got), f"{revised} pitch-type labels differ (more than 1%)"


def main(a):
    if a[0] == "--check":
        check(int(a[1]), a[2], date.fromisoformat(a[3]), date.fromisoformat(a[4]))
        return
    season, out = int(a[0]), a[1]
    opt = dict(zip(a[2::2], a[3::2]))
    base = pd.read_parquet(opt["--base"]) if "--base" in opt else None
    # re-fetch the last 3 days already held: Savant fills in late games and corrections
    start = date.fromisoformat(opt["--start"]) if "--start" in opt else (
        pd.to_datetime(base["game_date"]).max().date() - timedelta(days=2) if base is not None and len(base) else date(season, 3, 1))
    if base is not None and len(base):
        base = base[pd.to_datetime(base["game_date"]).dt.date < start]
    end = date.fromisoformat(opt["--end"]) if "--end" in opt else date.today() - timedelta(days=1)
    new = fetch_range(season, start, end)
    df = pd.concat([base, new], ignore_index=True) if base is not None else new
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="last")
    df.to_parquet(out, index=False)
    print("wrote", out, len(df), "rows through", pd.to_datetime(df["game_date"]).max().date() if len(df) else None)


if __name__ == "__main__":
    main(sys.argv[1:])
