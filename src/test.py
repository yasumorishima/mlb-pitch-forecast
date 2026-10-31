"""Test run (read once): train on all of 2025 (history 2024), score 2026 (history 2025).

Usage: python src/test.py <2024.parquet> <2025.parquet> <2026.parquet> <FREEZE.json> <out.json>
Refuses to run unless the code is the frozen code and the 2026 file is the frozen file.
"""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import dev
import pfc

BOOT = 2000
BAR = -0.010
REGULAR_SEASON_END_2026 = "2026-09-27"   # statsapi.mlb.com/api/v1/seasons/2026?sportId=1


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_freeze(fz, p26):
    for path in ("src", "requirements.txt", ".github/workflows"):
        r = subprocess.run(["git", "diff", "--quiet", fz["code_sha"], "HEAD", "--", path], cwd="..")
        assert r.returncode == 0, f"{path} changed since the frozen code {fz['code_sha']}"
    got = md5(p26)
    assert got == fz["md5_2026"], f"2026 file md5 {got} != frozen {fz['md5_2026']}"
    return got


def boot_pitchers(diff, pitcher, rng, n=BOOT):
    """Bootstrap over pitchers of the pitch-weighted mean difference."""
    codes, inv = np.unique(pitcher, return_inverse=True)
    s = np.bincount(inv, weights=diff)
    c = np.bincount(inv).astype(float)
    out = np.empty(n)
    for b in range(n):
        w = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))
        out[b] = (w * s).sum() / (w * c).sum()
    return out


def boot_two_way(diff, pitcher, game, rng, n=BOOT):
    pc, pi = np.unique(pitcher, return_inverse=True)
    gc, gi = np.unique(game, return_inverse=True)
    out = np.empty(n)
    for b in range(n):
        wp = np.bincount(rng.integers(0, len(pc), len(pc)), minlength=len(pc))[pi]
        wg = np.bincount(rng.integers(0, len(gc), len(gc)), minlength=len(gc))[gi]
        w = wp * wg
        out[b] = (w * diff).sum() / w.sum()
    return out


def auc(score, label):
    r = pd.Series(score).rank().to_numpy()
    n1 = label.sum()
    n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def independent_f2(d, cfg, fl, rows):
    """Second implementation of the F1 -> F2 chain on sampled rows (plain weighted sums)."""
    pit = d["cls"].to_numpy() >= 0
    by_p = {e: np.asarray(ix) for e, ix in pd.Series(np.arange(len(d))[pit]).groupby(d["pitcher"].to_numpy()[pit])}
    date, game, cls = d["game_date"].to_numpy(), d["game_pk"].to_numpy(), d["cls"].to_numpy()
    prior = np.asarray(cfg["prior"])
    worst = 0.0
    for r in rows:
        own = by_p.get(d.at[r, "pitcher"], np.array([], int))
        j = own[own < r]
        j = j[(date[j] < date[r]) | (game[j] == game[r])]
        pos = np.searchsorted(own, j)
        i_r = int((own < r).sum())
        p = prior[d.at[r, "starter"]]
        for name in ("F1", "F2"):
            h, a = cfg[name]["h"], cfg[name]["a"]
            lam = 0.0 if h is None else np.log(2.0) / h
            keep = np.ones(len(j), bool)
            for c in pfc.CELLS[name]:
                keep &= d[c].to_numpy()[j] == d.at[r, c]
            v = np.bincount(cls[j[keep]], weights=np.exp(-lam * (i_r - pos[keep])), minlength=pfc.K)
            p = (v + a * p) / (v.sum() + a)
        worst = max(worst, float(np.abs(p - fl["F2"][0][r]).max()))
    return worst


def run(p24, p25, p26, freeze_path, out):
    fz = json.load(open(freeze_path))
    got_md5 = check_freeze(fz, p26)
    cfg = {"prior": None, **{k: fz["floors"][k] for k in ("F1", "F2", "F3")}}
    params = {**dev.LGB_BASE, **fz["model"]["params"]}
    rounds = int(fz["model"]["best_iteration"])

    # training: all of 2025 with 2024 as history
    rawA = pfc.load([(2024, p24), (2025, p25)])
    pfc.sequence_gate(rawA)
    posA = pfc.position_players(rawA)
    dA = pfc.prepare(rawA, posA)
    trA = np.where(((dA["season"] == 2025) & dA["scored"]).to_numpy())[0]
    cfg["prior"] = dev.role_prior(dA, trA)
    XA, _ = pfc.make_X(dA, cfg)
    yA = dA["cls"].to_numpy()
    model = dev.fit(XA, yA, trA, params, rounds=rounds)
    dev.log("trained on", len(trA), "rows of 2025")
    f0 = np.bincount(yA[trA], minlength=pfc.K).astype(float)
    f0 /= f0.sum()
    del XA, dA, rawA

    # scoring: 2026 with 2025 as history
    rawB = pfc.load([(2025, p25), (2026, p26)])
    last = rawB.loc[rawB["season"] == 2026, "game_date"].max()
    assert str(last.date()) >= REGULAR_SEASON_END_2026, f"2026 ends at {last.date()}"
    bad = pfc.sequence_gate(rawB)
    posB = pfc.position_players(rawB)
    dB = pfc.prepare(rawB, posB)
    te = np.where(((dB["season"] == 2026) & dB["scored"]).to_numpy())[0]
    XB, fl = pfc.make_X(dB, cfg)
    y = dB["cls"].to_numpy()[te]
    pm = model.predict(XB.iloc[te])
    loss = {"M": pfc.logloss_rows(pm, y)}
    for name in ("F1", "F2", "F3"):
        loss[name] = pfc.logloss_rows(fl[name][0][te], y)
    loss["F0"] = pfc.logloss_rows(np.tile(f0, (len(te), 1)), y)

    rng = np.random.default_rng(20261101)
    pitcher = dB["pitcher"].to_numpy()[te]
    game = dB["game_pk"].to_numpy()[te]
    res = {"md5_2026": got_md5, "rows_2026": int(len(te)), "last_date_2026": str(last.date()),
           "illegal_transitions": bad.astype(str).to_dict("records"),
           "position_players": sorted(f"{s}:{p}" for s, p in posB),
           "mean_logloss": {k: float(v.mean()) for k, v in loss.items()}, "hypotheses": {}}
    pvals = {}
    for h, floor in (("H1", "F2"), ("H2", "F3")):
        diff = loss["M"] - loss[floor]
        bs = boot_pitchers(diff, pitcher, rng)
        tw = boot_two_way(diff, pitcher, game, rng)
        pvals[h] = float((bs >= BAR).mean())
        res["hypotheses"][h] = {"floor": floor, "mean_diff": float(diff.mean()),
                                "upper_95_one_sided": float(np.quantile(bs, 0.95)),
                                "p_vs_bar": pvals[h],
                                "two_way_upper_95": float(np.quantile(tw, 0.95))}
    order = sorted(pvals, key=pvals.get)
    alive = True
    for rank, h in enumerate(order):
        alpha = 0.05 / (len(order) - rank)
        ok = alive and pvals[h] <= alpha
        res["hypotheses"][h]["holm_alpha"] = alpha
        res["hypotheses"][h]["verdict"] = "SUPPORTED" if ok else "NOT SUPPORTED"
        alive = ok

    # reported, no verdict
    rep = {"accuracy": float((pm.argmax(1) == y).mean()),
           "top2_accuracy": float((np.argsort(-pm, 1)[:, :2] == y[:, None]).any(1).mean()),
           "brier": float(((pm - np.eye(pfc.K)[y]) ** 2).sum(1).mean())}
    rep["calibration"] = {c: {"mean_pred": float(pm[:, k].mean()), "observed": float((y == k).mean())}
                          for k, c in enumerate(pfc.CLASSES)}
    fb = np.isin(y, [0, 1, 2])
    rep["fastball_auc"] = {"M": auc(pm[:, :3].sum(1), fb), "F3": auc(fl["F3"][0][te][:, :3].sum(1), fb)}
    diff3 = loss["M"] - loss["F3"]
    st = dB.iloc[te]
    rep["gain_vs_F3_by_count"] = {f"{b}-{s}": float(diff3[(st["balls"] == b).to_numpy() & (st["strikes"] == s).to_numpy()].mean())
                                  for b in range(4) for s in range(3)}
    rep["gain_vs_F3_by_tto"] = {int(t): float(diff3[(st["tto"].clip(upper=4) == t).to_numpy()].mean()) for t in range(1, 5)}
    cold = fl["F1"][1][te] < 300
    rep["cold_start"] = {"pitches": int(cold.sum()), "gain_vs_F3": float(diff3[cold].mean())}
    per = pd.DataFrame({"p": pitcher, "d": diff3}).groupby("p")["d"].mean()
    rep["per_pitcher"] = {"pitchers": int(len(per)), "pitcher_weighted_mean_gain_vs_F3": float(per.mean()),
                          "share_where_M_beats_F3": float((per < 0).mean())}
    res["reported"] = rep

    # gates
    sample = np.random.default_rng(7).choice(te, min(5000, len(te)), replace=False)
    worst = independent_f2(dB, cfg, fl, sample)
    res["gates"] = {"independent_F2": {"n": int(len(sample)), "worst_abs_diff": worst, "pass": worst < 1e-6}}
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    dev.log("H1", res["hypotheses"]["H1"]["verdict"], "H2", res["hypotheses"]["H2"]["verdict"])
    if not res["gates"]["independent_F2"]["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    run(*sys.argv[1:6])
