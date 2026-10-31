"""Build the serving artifacts (compact model + per-player state) with the frozen settings.

Usage: python src/build_serving.py <out_dir> <serve_season> <season>=<parquet> ... [--current <dir>]
  --current: directory holding the served model.json.gz and meta.json (absent on the first build).

* Candidate: trained on the latest season's scored pitches before the hold-out week (its last
  7 days), previous season as history, frozen floors and rounds. Skipped when the season is too
  young (fewer than MIN_TRAIN training pitches or MIN_HOLD hold-out pitches).
* Publish gate: on the hold-out week, each model scored on features built with its own prior;
  the 95% upper bound (bootstrap over pitchers) of candidate minus current log loss must be <= 0.
  First build: "current" is the frozen test model (trained on the season before).
* Parity gates (a failure uploads nothing): the compact tree evaluator equals LightGBM; features
  rebuilt from a request (earlier dates' state + this game's pitches) match the training features.
* The state is always rebuilt for `serve_season` (previous season + season to date).
"""
import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serve", "api"))
import _serve as sv  # noqa: E402

import dev  # noqa: E402
import pfc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_TRAIN, MIN_HOLD = 60000, 10000
F32 = np.float32


def frozen():
    fz = json.load(open(os.path.join(ROOT, "FREEZE.json")))
    cfg = {k: {x: fz["floors"][k][x] for x in ("h", "a")} for k in ("F1", "F2", "F3")}
    return cfg, {**dev.LGB_BASE, **fz["model"]["params"]}, int(fz["model"]["best_iteration"])


def frame(paths, seasons):
    raw = pfc.load([(s, paths[s]) for s in seasons if s in paths])
    pfc.sequence_gate(raw)
    return raw, pfc.prepare(raw, pfc.position_players(raw))


def train_rows(d, season, before=None):
    m = ((d["season"] == season) & d["scored"]).to_numpy()
    if before is not None:
        m &= (d["game_date"] < before).to_numpy()
    return np.where(m)[0]


def fit_on(d, rows, cfg, params, rounds):
    c = dict(cfg, prior=dev.role_prior(d, rows))
    X, _ = pfc.make_X(d, c)
    return dev.fit(X, d["cls"].to_numpy(), rows, params, rounds=rounds), c, X


def boot_upper(diff, pitcher, n=2000, q=0.95, seed=11):
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(pitcher, return_inverse=True)
    s, c = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
    out = np.empty(n)
    for b in range(n):
        w = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))
        out[b] = (w * s).sum() / (w * c).sum()
    return float(np.quantile(out, q))


def state_from(d, cfg, until=None):
    """Committed counts per pitcher (F1-F3) and batter from rows before date `until` (all if None)."""
    m = d["cls"].to_numpy() >= 0
    if until is not None:
        m &= (d["game_date"] < until).to_numpy()
    pit, bat, cls = d["pitcher"].to_numpy(), d["batter"].to_numpy(), d["cls"].to_numpy()
    sr, b, s, pc = (d[c].to_numpy() for c in ("stand_r", "balls", "strikes", "pa_cls"))
    lams = {n: sv.lam_of(cfg[n]["h"]) for n in ("F1", "F2", "F3")}
    lam_b = sv.lam_of(pfc.BATTER_H)
    P, B = {}, {}
    for r in np.where(m)[0]:
        p = P.setdefault(int(pit[r]), {n: sv.Counts() for n in ("F1", "F2", "F3")})
        keys = sv.cell_keys(int(sr[r]), int(b[r]), int(s[r]), int(pc[r]))
        i = p["F1"].n
        for n in ("F1", "F2", "F3"):
            p[n].add(keys[n], int(cls[r]), i, lams[n])
            p[n].n = i + 1
        bb = B.setdefault(int(bat[r]), sv.Counts())
        bb.add((), int(cls[r]), bb.n, lam_b)
        bb.n += 1
    return {"pitchers": P, "batters": B}


def _nn(v):
    return None if pd.isna(v) else float(v)


def request_from(raw, d, k, explicit_pa=True):
    """The request a dugout would send for row k, built from the data before k."""
    g, p, bt, ab, dt = (raw.at[k, c] for c in ("game_pk", "pitcher", "batter", "at_bat_number", "game_date"))
    gp, gpit, gbat = raw["game_pk"].to_numpy(), raw["pitcher"].to_numpy(), raw["batter"].to_numpy()
    idx = np.arange(len(raw))
    counted = d["cls"].to_numpy() >= 0
    pcls = d["pa_cls"].to_numpy()

    def entry(j):
        t = raw.at[j, "pitch_type"]
        e = {"type": None if pd.isna(t) else t, "speed": _nn(raw.at[j, "release_speed"]),
             "zone": _nn(raw.at[j, "zone"]), "description": raw.at[j, "description"],
             "balls": int(raw.at[j, "balls"]), "strikes": int(raw.at[j, "strikes"]),
             "stand": raw.at[j, "stand"], "at_bat": int(raw.at[j, "at_bat_number"])}
        if explicit_pa:
            e["pa_prev"] = int(pcls[j])
        return e

    req = {"pitcher": int(p), "batter": int(bt), "stand": raw.at[k, "stand"], "p_throws": raw.at[k, "p_throws"],
           "starter": bool(d.at[k, "starter"]), "balls": int(raw.at[k, "balls"]), "strikes": int(raw.at[k, "strikes"]),
           "outs": int(raw.at[k, "outs_when_up"]), "on_1b": bool(pd.notna(raw.at[k, "on_1b"])),
           "on_2b": bool(pd.notna(raw.at[k, "on_2b"])), "on_3b": bool(pd.notna(raw.at[k, "on_3b"])),
           "inning": int(raw.at[k, "inning"]), "top": raw.at[k, "inning_topbot"] == "Top",
           "score_diff": int(raw.at[k, "fld_score"] - raw.at[k, "bat_score"]),
           "pitch_number": int(raw.at[k, "pitch_number"]), "tto": int(d.at[k, "tto"]), "at_bat": int(ab),
           "game_pitches": [entry(j) for j in idx[(gp == g) & (gpit == p) & (idx < k)]],
           "batter_game_pitches": [{"type": None if pd.isna(raw.at[j, "pitch_type"]) else raw.at[j, "pitch_type"],
                                    "description": raw.at[j, "description"]}
                                   for j in idx[(gp == g) & (gbat == bt) & (idx < k)]]}
    if explicit_pa:
        pa_rows = idx[(gp == g) & (raw["at_bat_number"].to_numpy() == ab) & (idx < k) & counted]
        req["pa_prev"] = entry(pa_rows[-1]) if len(pa_rows) else None
    same_day = (raw["game_date"].to_numpy() == np.datetime64(dt)) & (gp != g) & (idx < k) & counted
    req["pitcher_pitches_other_games_today"] = int((same_day & (gpit == p)).sum())
    req["batter_pitches_other_games_today"] = int((same_day & (gbat == bt)).sum())
    return req


def single_pitcher_pas(raw, g, p, k):
    """True if every plate appearance this pitcher faced today (up to k) had only his pitches."""
    idx = np.arange(len(raw))
    gm = (raw["game_pk"].to_numpy() == g) & (idx <= k)
    abs_p = set(raw.loc[gm & (raw["pitcher"].to_numpy() == p), "at_bat_number"])
    other = raw.loc[gm & raw["at_bat_number"].isin(abs_p).to_numpy() & (raw["pitcher"].to_numpy() != p)]
    return len(other) == 0


def parity_gate(raw, d, X, model, cfg, booster, dates, dh_date, per_date=120, n_dh=100):
    rng = np.random.default_rng(13)
    names = list(X.columns)
    first = d.groupby(["pitcher", "game_date"])["game_pk"].transform("first").to_numpy()
    firstb = d.groupby(["batter", "game_date"])["game_pk"].transform("first").to_numpy()
    gp = d["game_pk"].to_numpy()
    feat_bad, prob_bad, worst_rel, worst_p, checked, fallback, dh = 0, 0, 0.0, 0.0, 0, 0, 0
    for dt in dates:
        st = state_from(d, cfg, until=pd.Timestamp(dt))
        ms = {"cfg": cfg, "batter_h": pfc.BATTER_H, **st}
        today = np.where(((d["game_date"] == pd.Timestamp(dt)) & d["scored"]).to_numpy())[0]
        pick = list(rng.choice(today, min(per_date, len(today)), replace=False))
        if dt == dh_date:
            sec = today[(gp[today] != first[today]) | (gp[today] != firstb[today])]
            pick += list(rng.choice(sec, min(n_dh, len(sec)), replace=False)) if len(sec) else []
        for k in pick:
            k = int(k)
            is_dh = gp[k] != first[k] or gp[k] != firstb[k]
            use_fallback = bool(rng.integers(2)) and single_pitcher_pas(raw, raw.at[k, "game_pk"], raw.at[k, "pitcher"], k)
            x = sv.features(request_from(raw, d, k, explicit_pa=not use_fallback), ms)
            a = np.array([x[n] for n in names], dtype=F32)
            b = X.iloc[k].to_numpy(F32)
            fin = ~(np.isnan(a) & np.isnan(b))
            rel = float(np.max(np.abs(a[fin] - b[fin]) / np.maximum(1.0, np.abs(b[fin])))) if fin.any() else 0.0
            worst_rel = max(worst_rel, rel)
            if rel > 1e-6 or (np.isnan(a) != np.isnan(b)).any():
                feat_bad += 1
                if feat_bad <= 5:
                    bad = [n for n, u, v in zip(names, a, b) if not (abs(float(u) - float(v)) <= 1e-6 * max(1.0, abs(float(v))) or (np.isnan(u) and np.isnan(v)))]
                    dev.log("parity mismatch", k, bad[:8])
            pj = booster.predict([float(v) for v in a])           # serving path end to end
            pl = model.predict(X.iloc[[k]])[0]                    # training path
            dp = float(np.abs(pj - pl).max())
            worst_p = max(worst_p, dp)
            if dp > 1e-6:
                prob_bad += 1
                dev.log("probability differs at row", k, "by", dp)
            checked += 1
            fallback += use_fallback
            dh += is_dh
    res = {"checked": checked, "fallback_requests": fallback, "doubleheader_rows": dh,
           "feature_rows_off": feat_bad, "worst_feature_rel_diff": worst_rel,
           "prob_rows_off_1e-6": int(prob_bad), "worst_prob_diff": worst_p}
    res["pass"] = bool(checked > 0 and fallback > 0 and dh > 0 and feat_bad == 0
                       and prob_bad <= 0.01 * checked and worst_p < 1e-3)
    dev.log("parity", res)
    return res


def booster_gate(model, X, rows, booster):
    pl = model.predict(X.iloc[rows])
    worst = max(float(np.abs(booster.predict([float(v) for v in X.iloc[int(k)].to_numpy(F32)]) - pl[i]).max())
                for i, k in enumerate(rows))
    return {"checked": int(len(rows)), "worst_prob_diff": worst, "pass": worst < 1e-9}


def dump_state(st):
    def cells(c):
        return {"n": c.n, "cells": {"|".join(str(v) for v in key): [*map(float, vec), int(ref)] for key, (vec, ref) in c.cells.items()}}
    return {"pitchers": {str(p): {n: cells(c) for n, c in v.items()} for p, v in st["pitchers"].items()},
            "batters": {str(b): cells(c) for b, c in st["batters"].items()}}


def write_gz(path, obj):
    data = gzip.compress(json.dumps(obj, separators=(",", ":")).encode(), mtime=0)
    open(path, "wb").write(data)
    return hashlib.sha256(data).hexdigest()


def main(out, serve_season, paths, current=None):
    os.makedirs(out, exist_ok=True)
    cfg, params, rounds = frozen()
    latest = max(s for s in paths if s < serve_season or s == serve_season)
    raw, d = frame(paths, [latest - 1, latest])
    last = d.loc[d["season"] == latest, "game_date"].max()
    h0 = last - pd.Timedelta(days=6)
    tr = train_rows(d, latest, before=h0)
    hold = np.where(((d["season"] == latest) & d["scored"] & (d["game_date"] >= h0)).to_numpy())[0]
    y = d["cls"].to_numpy()[hold]
    gates, model_dump, model_info, new_model = {}, None, None, False
    cur = None
    if current:
        cur_meta = json.load(open(os.path.join(current, "meta.json")))
        cur_model = json.loads(gzip.decompress(open(os.path.join(current, "model.json.gz"), "rb").read()))
        cur = {"booster": sv.Booster(cur_model), "cfg": {**cur_meta["cfg"]}, "features": cur_meta["features"],
               "model": cur_model, "info": {k: cur_meta[k] for k in ("model_trained_on", "cfg", "features")}}
    if len(tr) >= MIN_TRAIN and len(hold) >= MIN_HOLD:
        cand, ccfg, X = fit_on(d, tr, cfg, params, rounds)
        dev.log("candidate trained on", len(tr), "pitches of", latest, "before", h0.date())
        lc = pfc.logloss_rows(cand.predict(X.iloc[hold]), y)
        if cur:
            Xc, _ = pfc.make_X(d, cur["cfg"])
            Xc = Xc[cur["features"]]
            pc = np.array([cur["booster"].predict([float(v) for v in Xc.iloc[int(k)].to_numpy(F32)]) for k in hold])
            label = "served model"
        else:
            rawp, dp = frame(paths, [latest - 2, latest - 1])
            trp = train_rows(dp, latest - 1)
            curm, pcfg, _ = fit_on(dp, trp, cfg, params, rounds)
            del rawp, dp
            Xc, _ = pfc.make_X(d, pcfg)
            pc = curm.predict(Xc.iloc[hold])
            label = f"frozen test model (trained on {latest - 1})"
        lu = pfc.logloss_rows(pc, y)
        up = boot_upper(lc - lu, d["pitcher"].to_numpy()[hold])
        gates["publish"] = {"holdout": [str(h0.date()), str(last.date())], "rows": int(len(hold)), "current": label,
                            "candidate_logloss": float(lc.mean()), "current_logloss": float(lu.mean()),
                            "upper95_candidate_minus_current": up, "pass": bool(up <= 0)}
        dev.log("publish gate", gates["publish"])
        cdump = sv.compact(cand.dump_model())
        booster = sv.Booster(cdump)
        rng = np.random.default_rng(12)
        gates["booster"] = booster_gate(cand, X, rng.choice(hold, min(2000, len(hold)), replace=False), booster)
        days = pd.Series(d.loc[(d["season"] == latest) & d["scored"], "game_date"].unique())
        dh_counts = d[(d["season"] == latest)].groupby("game_date")["game_pk"].nunique() - \
            d[(d["season"] == latest)].groupby("game_date")["home_team"].nunique()
        dh_date = str(dh_counts.idxmax().date())
        dates = [str(pd.Timestamp(t).date()) for t in days.sample(2, random_state=5)] + [dh_date]
        gates["parity"] = parity_gate(raw, d, X, cand, ccfg, booster, dates, dh_date)
        if not (gates["booster"]["pass"] and gates["parity"]["pass"]):
            json.dump({"gates": gates}, open(os.path.join(out, "gates.json"), "w"), indent=1, default=str)
            dev.log("parity failed; nothing is written for upload")
            sys.exit(1)
        new_model = gates["publish"]["pass"] or not cur
        if not gates["publish"]["pass"] and not cur:
            dev.log("first build lost to the frozen test model; nothing is published")
            json.dump({"gates": gates}, open(os.path.join(out, "gates.json"), "w"), indent=1, default=str)
            sys.exit(1)
        if new_model:
            model_dump = cdump
            model_info = {"model_trained_on": f"{latest} before {h0.date()}", "cfg": ccfg, "features": list(X.columns)}
        del X
    else:
        gates["skipped"] = {"reason": f"season too young: {len(tr)} training / {len(hold)} hold-out pitches", "pass": True}
        if not cur:
            dev.log("no served model and no candidate; nothing to publish")
            sys.exit(1)
    if not new_model:
        model_dump, model_info = cur["model"], cur["info"]

    seasons = [s for s in (serve_season - 1, serve_season) if s in paths]
    raw_s, d_s = frame(paths, seasons)
    st = state_from(d_s, {n: model_info["cfg"][n] for n in ("F1", "F2", "F3")})
    h_model = write_gz(os.path.join(out, "model.json.gz"), model_dump)
    h_state = write_gz(os.path.join(out, "state.json.gz"), dump_state(st))
    meta = {"serve_season": serve_season, "state_seasons": seasons, "data_through": str(d_s["game_date"].max().date()),
            **model_info, "batter_h": pfc.BATTER_H, "gates": gates, "new_model": bool(new_model),
            "sha256": {"model.json.gz": h_model, "state.json.gz": h_state}}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1, default=str)
    dev.log("written; new model =", new_model, "pitchers", len(st["pitchers"]), "batters", len(st["batters"]))


if __name__ == "__main__":
    a = sys.argv[1:]
    cur = None
    if "--current" in a:
        i = a.index("--current")
        cur = a[i + 1]
        a = a[:i] + a[i + 2:]
    paths = {int(x.split("=")[0]): x.split("=")[1] for x in a[2:]}
    paths = {s: p for s, p in paths.items() if os.path.exists(p)}
    main(a[0], int(a[1]), paths, cur)
