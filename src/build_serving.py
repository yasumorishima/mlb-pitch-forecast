"""Build the serving artifacts (model JSON + per-player state) with the frozen settings.

Usage: python src/build_serving.py <out_dir> <serve_season> <season>=<parquet> ... [--current <dir with model.json, meta.json>]

* Candidate model: trained on the latest season's scored pitches before the hold-out week
  (the last 7 days of data), with the previous season as history, frozen floors and rounds.
* Current model: --current (the model now served) or, on the first build, the frozen test model
  (trained on the season before, with its own previous season as history).
* Publish gate: on the hold-out week, the 95% upper bound (bootstrap over pitchers) of
  candidate minus current log loss must be <= 0.
* Parity gates: the JSON tree evaluator equals LightGBM; features rebuilt from a request (state of
  earlier dates + this game's pitches) equal the training features.
* State for serving `serve_season`: earlier seasons are history only if they are the previous season.
"""
import gzip
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serve", "api"))
import _serve as sv  # noqa: E402

import dev  # noqa: E402
import pfc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def frozen():
    fz = json.load(open(os.path.join(ROOT, "FREEZE.json")))
    cfg = {k: {x: fz["floors"][k][x] for x in ("h", "a")} for k in ("F1", "F2", "F3")}
    return cfg, {**dev.LGB_BASE, **fz["model"]["params"]}, int(fz["model"]["best_iteration"])


def frame(paths, seasons):
    raw = pfc.load([(s, paths[s]) for s in seasons])
    pfc.sequence_gate(raw)
    pos = pfc.position_players(raw)
    return raw, pfc.prepare(raw, pos)


def train(paths, season, cfg, params, rounds, before=None):
    """Model for `season` rows (before date `before`), previous season as history."""
    raw, d = frame(paths, [season - 1, season])
    m = ((d["season"] == season) & d["scored"]).to_numpy()
    if before is not None:
        m &= (d["game_date"] < before).to_numpy()
    tr = np.where(m)[0]
    c = dict(cfg, prior=dev.role_prior(d, tr))
    X, _ = pfc.make_X(d, c)
    model = dev.fit(X, d["cls"].to_numpy(), tr, params, rounds=rounds)
    return model, c, raw, d, X


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
    rows = np.where(m)[0]
    pit, bat, cls = d["pitcher"].to_numpy(), d["batter"].to_numpy(), d["cls"].to_numpy()
    sr, b, s, pc = (d[c].to_numpy() for c in ("stand_r", "balls", "strikes", "pa_cls"))
    lams = {n: sv.lam_of(cfg[n]["h"]) for n in ("F1", "F2", "F3")}
    lam_b = sv.lam_of(pfc.BATTER_H)
    P, B = {}, {}
    for r in rows:
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


def request_from(raw, d, k):
    """The request a dugout would send for row k, built from the data before k."""
    g, p, bt, ab, dt = (raw.at[k, c] for c in ("game_pk", "pitcher", "batter", "at_bat_number", "game_date"))
    gp, gpit, gbat = raw["game_pk"].to_numpy(), raw["pitcher"].to_numpy(), raw["batter"].to_numpy()
    idx = np.arange(len(raw))
    pcls = d["pa_cls"].to_numpy()

    def entry(j):
        t = raw.at[j, "pitch_type"]
        return {"type": None if pd.isna(t) else t, "speed": _nn(raw.at[j, "release_speed"]),
                "zone": _nn(raw.at[j, "zone"]), "description": raw.at[j, "description"],
                "balls": int(raw.at[j, "balls"]), "strikes": int(raw.at[j, "strikes"]),
                "stand": raw.at[j, "stand"], "at_bat": int(raw.at[j, "at_bat_number"]),
                "pa_prev": int(pcls[j])}

    game = [entry(j) for j in idx[(gp == g) & (gpit == p) & (idx < k)]]
    pa_rows = idx[(gp == g) & (raw["at_bat_number"].to_numpy() == ab) & (idx < k) & (d["cls"].to_numpy() >= 0)]
    pa_prev = entry(pa_rows[-1]) if len(pa_rows) else None
    bgame = [{"type": None if pd.isna(raw.at[j, "pitch_type"]) else raw.at[j, "pitch_type"],
              "description": raw.at[j, "description"]} for j in idx[(gp == g) & (gbat == bt) & (idx < k)]]
    same_day = (raw["game_date"].to_numpy() == np.datetime64(dt)) & (gp != g) & (idx < k) & (d["cls"].to_numpy() >= 0)
    return {"pitcher": int(p), "batter": int(bt), "stand": raw.at[k, "stand"], "p_throws": raw.at[k, "p_throws"],
            "starter": bool(d.at[k, "starter"]), "balls": int(raw.at[k, "balls"]), "strikes": int(raw.at[k, "strikes"]),
            "outs": int(raw.at[k, "outs_when_up"]), "on_1b": bool(pd.notna(raw.at[k, "on_1b"])),
            "on_2b": bool(pd.notna(raw.at[k, "on_2b"])), "on_3b": bool(pd.notna(raw.at[k, "on_3b"])),
            "inning": int(raw.at[k, "inning"]), "top": raw.at[k, "inning_topbot"] == "Top",
            "score_diff": int(raw.at[k, "fld_score"] - raw.at[k, "bat_score"]),
            "pitch_number": int(raw.at[k, "pitch_number"]), "tto": int(d.at[k, "tto"]), "at_bat": int(ab),
            "game_pitches": game, "pa_prev": pa_prev, "batter_game_pitches": bgame,
            "pitcher_pitches_other_games_today": int((same_day & (gpit == p)).sum()),
            "batter_pitches_other_games_today": int((same_day & (gbat == bt)).sum())}


def _nn(v):
    return None if pd.isna(v) else float(v)


def parity_gate(raw, d, X, model, cfg, booster, dates, per_date=150):
    rng = np.random.default_rng(13)
    feat_mismatch, worst_p, checked = 0, 0.0, 0
    names = list(X.columns)
    for dt in dates:
        st = state_from(d, cfg, until=pd.Timestamp(dt))
        ms = {"cfg": cfg, "batter_h": pfc.BATTER_H, **st}
        today = np.where(((d["game_date"] == pd.Timestamp(dt)) & d["scored"]).to_numpy())[0]
        for k in rng.choice(today, min(per_date, len(today)), replace=False):
            x = sv.features(request_from(raw, d, int(k)), ms)
            a = np.array([x[n] for n in names], dtype=np.float32)
            b = X.iloc[int(k)].to_numpy(np.float32)
            checked += 1
            if not np.array_equal(a, b, equal_nan=True):
                feat_mismatch += 1
                if feat_mismatch <= 5:
                    bad = [n for n, u, v in zip(names, a, b) if not (u == v or (np.isnan(u) and np.isnan(v)))]
                    dev.log("parity mismatch", int(k), bad[:8], [float(x[n]) for n in bad[:3]], [float(X.iloc[int(k)][n]) for n in bad[:3]])
                continue                      # a float32 rounding flip can move a split; counted above
            pj = booster.predict([float(v) for v in a])
            pl = model.predict(X.iloc[[int(k)]])[0]
            worst_p = max(worst_p, float(np.abs(pj - pl).max()))
    share = 1 - feat_mismatch / max(checked, 1)
    dev.log("parity checked", checked, "feature rows equal", round(share, 4), "worst prob diff", worst_p)
    return {"checked": checked, "feature_rows_equal_share": share, "worst_prob_diff": worst_p,
            "pass": checked > 0 and share >= 0.999 and worst_p < 1e-6}


def booster_gate(model, X, rows, booster):
    pl = model.predict(X.iloc[rows])
    worst = 0.0
    for i, k in enumerate(rows):
        pj = booster.predict([float(v) for v in X.iloc[int(k)].to_numpy(np.float32)])
        worst = max(worst, float(np.abs(pj - pl[i]).max()))
    return {"checked": int(len(rows)), "worst_prob_diff": worst, "pass": worst < 1e-9}


def dump_state(st):
    def cells(c):
        return {"n": c.n, "cells": {"|".join(str(v) for v in key): [*map(float, vec), int(ref)] for key, (vec, ref) in c.cells.items()}}
    return {"pitchers": {str(p): {n: cells(c) for n, c in v.items()} for p, v in st["pitchers"].items()},
            "batters": {str(b): cells(c) for b, c in st["batters"].items()}}


def main(out, serve_season, paths, current=None):
    os.makedirs(out, exist_ok=True)
    cfg, params, rounds = frozen()
    latest = max(paths)
    # hold-out week = last 7 days of data
    last = pfc.load([(latest, paths[latest])])["game_date"].max()
    h0 = last - pd.Timedelta(days=6)
    cand, ccfg, raw, d, X = train(paths, latest, cfg, params, rounds, before=h0)
    dev.log("candidate trained on", latest, "before", h0.date())
    hold = np.where(((d["season"] == latest) & d["scored"] & (d["game_date"] >= h0)).to_numpy())[0]
    y = d["cls"].to_numpy()[hold]
    lc = pfc.logloss_rows(cand.predict(X.iloc[hold]), y)
    if current:
        cur_dump = json.load(open(os.path.join(current, "model.json")))
        cur_meta = json.load(open(os.path.join(current, "meta.json")))
        cur = sv.Booster(cur_dump)
        pc = np.array([cur.predict([float(v) for v in X.iloc[int(k)].to_numpy(np.float32)]) for k in hold])
        cur_label = "served model"
    else:
        curm, _, _, _, _ = train(paths, latest - 1, cfg, params, rounds)
        pc = curm.predict(X.iloc[hold])
        cur_label = f"frozen test model (trained on {latest - 1})"
    lu = pfc.logloss_rows(pc, y)
    up = boot_upper(lc - lu, d["pitcher"].to_numpy()[hold])
    publish = {"holdout": [str(h0.date()), str(last.date())], "rows": int(len(hold)), "current": cur_label,
               "candidate_logloss": float(lc.mean()), "current_logloss": float(lu.mean()),
               "upper95_candidate_minus_current": up, "pass": up <= 0}
    dev.log("publish gate", publish)

    dump = cand.dump_model()
    booster = sv.Booster(dump)
    rng = np.random.default_rng(12)
    gates = {"publish": publish,
             "booster": booster_gate(cand, X, rng.choice(hold, min(2000, len(hold)), replace=False), booster)}
    dates = sorted(pd.Series(d.loc[(d["season"] == latest) & d["scored"], "game_date"].unique()).sample(3, random_state=5))
    gates["parity"] = parity_gate(raw, d, X, cand, ccfg, booster, [str(pd.Timestamp(t).date()) for t in dates])
    parity_ok = gates["booster"]["pass"] and gates["parity"]["pass"]
    new_model = parity_ok and publish["pass"]

    # state for serving `serve_season`: previous season + season to date (independent of the model)
    seasons = [s for s in (serve_season - 1, serve_season) if s in paths]
    raw_s, d_s = frame(paths, seasons)
    st = state_from(d_s, cfg)
    if new_model or not current:
        model_dump, model_info = dump, {"model_trained_on": f"{latest} before {h0.date()}",
                                        "cfg": {**cfg, "prior": ccfg["prior"]}, "features": list(X.columns)}
    else:   # keep serving the current model, refresh only the state
        model_dump, model_info = cur_dump, {k: cur_meta[k] for k in ("model_trained_on", "cfg", "features")}
    meta = {"serve_season": serve_season, "state_seasons": seasons, "data_through": str(d_s["game_date"].max().date()),
            **model_info, "batter_h": pfc.BATTER_H, "gates": gates, "new_model": new_model}
    json.dump(model_dump, open(os.path.join(out, "model.json"), "w"))
    with gzip.open(os.path.join(out, "state.json.gz"), "wt") as f:
        json.dump(dump_state(st), f)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1, default=str)
    dev.log("written; new model =", new_model, "pitchers", len(st["pitchers"]), "batters", len(st["batters"]))
    if not parity_ok or (not current and not publish["pass"]):
        sys.exit(1)     # nothing is uploaded when the serving path disagrees with training


if __name__ == "__main__":
    a = sys.argv[1:]
    cur = None
    if "--current" in a:
        i = a.index("--current")
        cur = a[i + 1]
        a = a[:i] + a[i + 2:]
    paths = {int(x.split("=")[0]): x.split("=")[1] for x in a[2:]}
    main(a[0], int(a[1]), paths, cur)
