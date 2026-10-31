"""Development run: history 2024, train 2025 through June, validate 2025 July onward.

Usage: python src/dev.py <statcast_2024.parquet> <statcast_2025.parquet> <out.json> [--quick]
--quick keeps the first 40 days of each season (checks the code; its numbers mean nothing).
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import pfc

H_GRID = [5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, None]
A_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
LGB_GRID = [{"num_leaves": 63}, {"num_leaves": 255}]
LGB_BASE = {"objective": "multiclass", "num_class": pfc.K, "learning_rate": 0.05,
            "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 4,
            "seed": 20261031}
CAT = ["pa_cls", "g_prev_cls"]
SHUFFLE_DELTA = 0.02
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load(paths, quick):
    raw = pfc.load(paths)
    if quick:
        start = raw.groupby("season")["game_date"].transform("min")
        raw = raw[(raw["game_date"] - start).dt.days < 40].reset_index(drop=True)
    return raw


def split(d, quick):
    s25 = ((d["season"] == 2025) & d["scored"]).to_numpy()
    date = d["game_date"]
    if quick:
        cut = date[d["season"] == 2025].min() + pd.Timedelta(days=20)
    else:
        cut = pd.Timestamp("2025-07-01")
    return np.where(s25 & (date < cut).to_numpy())[0], np.where(s25 & (date >= cut).to_numpy())[0]


def role_prior(d, rows):
    y, st = d["cls"].to_numpy()[rows], d["starter"].to_numpy()[rows]
    return [(np.bincount(y[st == r], minlength=pfc.K) / max(1, (st == r).sum())).tolist() for r in (0, 1)]


def tune_floors(d, tr, va):
    """Grid the counts once per (cell, h); F1 -> F2 -> F3 each picks (h, a) for its own validation loss."""
    counts = {(name, h): pfc.asof_counts(d, "pitcher", cols, h)
              for name, cols in pfc.CELLS.items() for h in H_GRID}
    log("counts grid done")
    y = d["cls"].to_numpy()[va]
    cfg = {"prior": role_prior(d, tr)}
    base = np.asarray(cfg["prior"])[d["starter"].to_numpy()]
    for name in ("F1", "F2", "F3"):
        best = None
        for h in H_GRID:
            c, t = counts[(name, h)]
            for a in A_GRID:
                ll = pfc.logloss_rows(pfc.shrink(c[va], t[va], base[va], a), y).mean()
                if best is None or ll < best[0]:
                    best = (float(ll), h, a)
        ll, h, a = best
        c, t = counts[(name, h)]
        base = pfc.shrink(c, t, base, a)
        cfg[name] = {"h": h, "a": a, "val_logloss": ll}
        log(name, cfg[name])
    return cfg, counts


def fit(X, y, tr, params, va=None, rounds=None):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=CAT, free_raw_data=False)
    if rounds is not None:
        return lgb.train(params, dtr, rounds)
    dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr, categorical_feature=CAT)
    return lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def run(paths, out, quick=False):
    raw = load(paths, quick)
    bad = pfc.sequence_gate(raw)
    log("loaded", len(raw), "rows; illegal transitions", len(bad))
    pos = pfc.position_players(raw)
    d = pfc.prepare(raw, pos)
    tr, va = split(d, quick)
    cfg, counts = tune_floors(d, tr, va)
    X, fl = pfc.make_X(d, cfg, counts=counts)
    del counts
    y = d["cls"].to_numpy()
    yv = y[va]
    res = {"quick": quick, "rows": {"all": len(d), "train": len(tr), "val": len(va)},
           "illegal_transitions": bad.astype(str).to_dict("records"),
           "position_players": sorted(f"{s}:{p}" for s, p in pos),
           "floors": {k: cfg[k] for k in ("F1", "F2", "F3")}, "prior": cfg["prior"]}
    f0 = np.bincount(y[tr], minlength=pfc.K) / len(tr)
    res["floors"]["F0"] = {"val_logloss": float(pfc.logloss_rows(np.tile(f0, (len(va), 1)), yv).mean())}
    best = None
    for g in LGB_GRID:
        m = fit(X, y, tr, {**LGB_BASE, **g}, va=va)
        ll = float(pfc.logloss_rows(m.predict(X.iloc[va], num_iteration=m.best_iteration), yv).mean())
        log("lgb", g, "iter", m.best_iteration, "val", round(ll, 5))
        if best is None or ll < best[0]:
            best = (ll, g, m.best_iteration, m)
    ll, g, it, m = best
    pm = m.predict(X.iloc[va], num_iteration=it)
    res["model"] = {"params": g, "best_iteration": int(it), "val_logloss": ll,
                    "val_accuracy": float((pm.argmax(1) == yv).mean()),
                    "minus_F2": ll - cfg["F2"]["val_logloss"], "minus_F3": ll - cfg["F3"]["val_logloss"],
                    "top_gain": {k: float(v) for k, v in pd.Series(m.feature_importance("gain"), index=X.columns)
                                 .sort_values(ascending=False).head(15).items()}}
    log("model", res["model"]["params"], "val", round(ll, 5), "minus F3", round(res["model"]["minus_F3"], 5))

    gates = {}
    gates["independent_counts"] = independent_counts_gate(d, cfg, va)
    gates["independent_counts_detects_sameday_leak"] = invert(independent_counts_gate(d, cfg, va, broken="no_sameday"))
    gates["cut_frame"] = cut_frame_gate(raw, d, X, cfg, pos, va, n_uniform=150, n_double=100)
    gates["cut_frame_detects_current_leak"] = invert(cut_frame_gate(raw, d, None, cfg, pos, va, 40, 0, broken="include_current"))
    gates["cut_frame_detects_sameday_leak"] = invert(cut_frame_gate(raw, d, None, cfg, pos, va, 0, 40, broken="no_sameday"))
    params = {**LGB_BASE, **g}
    Xl = X.copy()
    Xl["leak_speed"] = d["release_speed"].astype(np.float32).to_numpy()
    lll = float(pfc.logloss_rows(fit(Xl, y, tr, params, rounds=it).predict(Xl.iloc[va]), yv).mean())
    gates["planted_leak"] = {"val_logloss": lll, "drop": ll - lll, "pass": bool(ll - lll > 0.2)}
    log("planted leak drop", round(ll - lll, 4))
    gates["shuffle"] = shuffle_gate(paths, quick, pos, params)
    res["gates"] = gates
    res["pass"] = all(v["pass"] for v in gates.values())
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    log("done; gates pass =", res["pass"], {k: v["pass"] for k, v in gates.items()})
    if not res["pass"]:
        sys.exit(1)


def invert(r):
    """A planted defect must be caught: passes only if rows were checked and mismatches were found."""
    found = r.get("mismatches", 0) > 0 or r.get("worst_rel_diff", 0.0) > 1e-5
    return {**r, "pass": bool(r["checked"] > 0 and found)}


def doubleheader_rows(d, rows, ent):
    """Rows whose `ent` already played another game earlier the same date."""
    first_game = d.groupby([ent, "game_date"])["game_pk"].transform("first").to_numpy()
    return rows[(d["game_pk"].to_numpy()[rows] != first_game[rows])]


def independent_counts_gate(d, cfg, va, n=1500, broken=None):
    """Second implementation of the pitcher F1-F3 and batter counts (plain weighted sums over earlier rows).
    Samples include doubleheader second-game rows for the pitcher and for the batter."""
    rng = np.random.default_rng(1)
    dh_p, dh_b = doubleheader_rows(d, va, "pitcher"), doubleheader_rows(d, va, "batter")

    def pick(a, m):
        return rng.choice(a, min(m, len(a)), replace=False) if len(a) else np.array([], int)

    rows_u = pick(va, n)
    pit = d["cls"].to_numpy() >= 0
    date, game, cls = d["game_date"].to_numpy(), d["game_pk"].to_numpy(), d["cls"].to_numpy()
    checks = [("pitcher", cols, cfg[name]["h"], np.r_[rows_u, pick(dh_p, 200)]) for name, cols in pfc.CELLS.items()]
    checks.append(("batter", [], pfc.BATTER_H, np.r_[rows_u, pick(dh_b, 200)]))
    worst, checked = 0.0, 0
    for ent, cols, h, rows in checks:
        lam = 0.0 if h is None else np.log(2.0) / h
        fast, _ = pfc.asof_counts(d, ent, cols, h, broken)
        by_ent = {e: np.asarray(ix) for e, ix in pd.Series(np.arange(len(d))[pit]).groupby(d[ent].to_numpy()[pit])}
        colv = [d[c].to_numpy() for c in cols]
        ev = d[ent].to_numpy()
        for r in rows.astype(int):
            own = by_ent.get(ev[r], np.array([], int))
            j = own[own < r]
            keep = (date[j] < date[r]) | (game[j] == game[r])
            for cv in colv:
                keep &= cv[j] == cv[r]
            w = np.exp(-lam * (len(j) - np.searchsorted(own, j[keep])))
            v = np.bincount(cls[j[keep]], weights=w, minlength=pfc.K)
            worst = max(worst, float(np.abs(v - fast[r]).max() / max(1.0, v.sum())))
            checked += 1
    log("independent counts", broken, "checked", checked, "worst relative diff", worst)
    return {"broken": broken, "checked": checked, "pitcher_doubleheader_rows": int(min(200, len(dh_p))),
            "batter_doubleheader_rows": int(min(200, len(dh_b))), "worst_rel_diff": worst, "pass": worst < 1e-5}


def cut_frame_gate(raw, d, X, cfg, pos, va, n_uniform, n_double, broken=None):
    """Rebuild row k's whole feature vector from a frame cut at k, with k's own pitch information and
    every other game of the same date scrambled; it must equal the full-data row exactly.
    With `broken`, both the full-data row and the cut frame use the planted defect."""
    rng = np.random.default_rng(4 if broken is None else 5)
    dh = np.r_[doubleheader_rows(d, va, "pitcher"), doubleheader_rows(d, va, "batter")]
    rows = np.r_[rng.choice(va, n_uniform, replace=False) if n_uniform else np.array([], int),
                 rng.choice(dh, min(n_double, len(dh)), replace=False) if len(dh) else np.array([], int)].astype(int)
    pa_key = raw["game_pk"].to_numpy() * 1000 + raw["at_bat_number"].to_numpy()
    pit_pa = pd.Series(raw["pitcher"].to_numpy()).groupby(pa_key).agg(set)
    bat_pa = pd.Series(raw["batter"].to_numpy()).groupby(pa_key).agg(set)
    types = np.array(["FF", "SL", "CH", "CU", "SI"])
    mism, checked = 0, 0
    if broken is not None:
        X, _ = pfc.make_X(d, cfg, broken=broken)
    for k in rows:
        p, b, g = raw.at[k, "pitcher"], raw.at[k, "batter"], raw.at[k, "game_pk"]
        dt = raw["game_date"].to_numpy()[k]
        pas = set(pit_pa.index[pit_pa.map(lambda s: p in s)]) | set(bat_pa.index[bat_pa.map(lambda s: b in s)])
        sel = (np.isin(pa_key, list(pas)) | (raw["game_pk"].to_numpy() == g)) & (np.arange(len(raw)) <= k)
        idx = np.where(sel)[0]
        c = raw.iloc[idx].copy()
        counted = (c["pitch_type"].notna() & ~c["pitch_type"].isin(pfc.NOT_SCORED_TYPES)
                   & (c["description"] != "pitchout")).to_numpy()
        scr = (c["game_date"].to_numpy() == dt) & (c["game_pk"].to_numpy() != g) & counted
        scr |= (idx == k)
        n_s = int(scr.sum())
        c.loc[scr, "pitch_type"] = rng.choice(types, n_s)
        c.loc[scr, "release_speed"] = rng.uniform(70, 100, n_s)
        c.loc[scr, "zone"] = rng.integers(1, 15, n_s).astype(float)
        c.loc[scr, "description"] = rng.choice(["ball", "called_strike", "swinging_strike", "foul"], n_s)
        c = c.reset_index(drop=True)
        cx, _ = pfc.make_X(pfc.prepare(c, pos), cfg, broken=broken)
        a = X.iloc[k].to_numpy(np.float64)
        bvec = cx.iloc[len(c) - 1].to_numpy(np.float64)
        checked += 1
        if not np.array_equal(a, bvec, equal_nan=True):
            mism += 1
            if broken is None and mism <= 5:
                diff = [col for col, u, v in zip(X.columns, a, bvec) if not (u == v or (np.isnan(u) and np.isnan(v)))]
                log("cut-frame mismatch at row", int(k), diff[:10])
    log("cut-frame", broken, "checked", checked, "mismatches", mism)
    return {"broken": broken, "checked": checked, "mismatches": mism, "pass": mism == 0 and checked > 0}


def shuffle_gate(paths, quick, pos, params):
    """Shuffle pitch types within pitcher-season; refit M early-stopped on the last 20% of training dates.
    One-sided: fail if M beats F1 by more than SHUFFLE_DELTA (structure found in noise)."""
    raw = load(paths, quick)
    rng = np.random.default_rng(3)
    m = raw["pitch_type"].notna() & ~raw["pitch_type"].isin(pfc.NOT_SCORED_TYPES)
    for _, ix in raw[m].groupby(["season", "pitcher"]).groups.items():
        v = raw.loc[ix, "pitch_type"].to_numpy().copy()
        rng.shuffle(v)
        raw.loc[ix, "pitch_type"] = v
    d = pfc.prepare(raw, pos)
    tr, va = split(d, quick)
    cfg, counts = tune_floors(d, tr, va)
    X, _ = pfc.make_X(d, cfg, counts=counts)
    del counts
    y = d["cls"].to_numpy()
    dates = d["game_date"].to_numpy()[tr]
    cut = np.quantile(dates.astype("int64"), 0.8)
    inner_tr, inner_va = tr[dates.astype("int64") < cut], tr[dates.astype("int64") >= cut]
    mm = fit(X, y, inner_tr, params, va=inner_va)
    llm = float(pfc.logloss_rows(mm.predict(X.iloc[va], num_iteration=mm.best_iteration), y[va]).mean())
    gap = llm - cfg["F1"]["val_logloss"]
    log("shuffle M - F1", round(gap, 5))
    return {"M_minus_F1": gap, "delta": SHUFFLE_DELTA, "pass": bool(gap >= -SHUFFLE_DELTA)}


if __name__ == "__main__":
    a = sys.argv[1:]
    quick = "--quick" in a
    a = [x for x in a if x != "--quick"]
    run([(2024, a[0]), (2025, a[1])], a[2], quick)
