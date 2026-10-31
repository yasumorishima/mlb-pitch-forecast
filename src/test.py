"""Test run (read once): train on all of 2025 (history 2024), score 2026 (history 2025).

Usage: python src/test.py <2024.parquet> <2025.parquet> <2026.parquet> <out.json>
Reads FREEZE.json at the repository root and refuses to run unless
  * the frozen code commit exists, and src/, requirements.txt, .github/workflows/ and PLAN.md
    are unchanged from it to HEAD;
  * the modelling code (pfc.py, dev.py, requirements.txt) is unchanged from the development run;
  * the floors and model settings equal the non-quick development result on the `results` branch;
  * the three parquet files have the frozen md5s.
Every gate runs before any pitch of 2026 is scored; a failed gate exits without scoring.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import dev
import pfc

BOOT = 2000
BAR = -0.010
F2_TOL = 1e-6
REGULAR_SEASON_END_2026 = "2026-09-27"   # statsapi.mlb.com/api/v1/seasons/2026?sportId=1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_freeze(fz, paths):
    commits = git("log", "--format=%H", "--", "FREEZE.json").stdout.split()
    assert len(commits) == 1, f"FREEZE.json was written by {len(commits)} commits"
    f = commits[0]
    assert git("rev-parse", f + "^").stdout.strip() == fz["code_sha"], "the freeze commit is not on top of the frozen code"
    changed = git("diff", "--name-only", fz["code_sha"], f).stdout.split()
    assert changed == ["FREEZE.json"], f"the freeze commit changed {changed}"
    for sha in (fz["code_sha"], fz["dev_sha"]):
        assert git("cat-file", "-e", sha + "^{commit}").returncode == 0, f"commit {sha} not found"
    r = git("diff", "--quiet", fz["code_sha"], "HEAD", "--", "src", "requirements.txt", ".github/workflows", "PLAN.md")
    assert r.returncode == 0, f"code changed since the frozen commit {fz['code_sha']}"
    r = git("diff", "--quiet", fz["dev_sha"], fz["code_sha"], "--", "src/pfc.py", "src/dev.py", "requirements.txt")
    assert r.returncode == 0, "modelling code differs from the development run"
    name = fz["dev_result"]
    assert fz["dev_sha"] in name and "_quick" not in name, f"{name} is not a full development result"
    devres = json.loads(git("show", f"origin/results:results/{name}").stdout)
    assert not devres["quick"] and devres["pass"], "development result is quick or failed its gates"
    for k in ("F1", "F2", "F3"):
        assert {x: devres["floors"][k][x] for x in ("h", "a")} == {x: fz["floors"][k][x] for x in ("h", "a")}, k
    assert devres["model"]["params"] == fz["model"]["params"]
    assert int(devres["model"]["best_iteration"]) == int(fz["model"]["best_iteration"])
    got = {y: md5(p) for y, p in paths.items()}
    for y in paths:
        assert got[y] == fz["md5"][y], f"{y} md5 {got[y]} != frozen {fz['md5'][y]}"
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
        before = own[own < r]
        i_r = len(before)
        j = before[(date[before] < date[r]) | (game[before] == game[r])]
        pos = np.searchsorted(own, j)
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


def run(p24, p25, p26, out):
    fz = json.load(open(os.path.join(ROOT, "FREEZE.json")))
    md5s = check_freeze(fz, {"2024": p24, "2025": p25, "2026": p26})
    cfg = {"prior": None, **{k: {x: fz["floors"][k][x] for x in ("h", "a")} for k in ("F1", "F2", "F3")}}
    params = {**dev.LGB_BASE, **fz["model"]["params"]}
    rounds = int(fz["model"]["best_iteration"])

    # ---- training: all of 2025 with 2024 as history (rounds as chosen in development, unchanged)
    rawA = pfc.load([(2024, p24), (2025, p25)])
    pfc.sequence_gate(rawA)
    dA = pfc.prepare(rawA, pfc.position_players(rawA))
    trA = np.where(((dA["season"] == 2025) & dA["scored"]).to_numpy())[0]
    cfg["prior"] = dev.role_prior(dA, trA)
    XA, _ = pfc.make_X(dA, cfg)
    yA = dA["cls"].to_numpy()
    f0 = np.bincount(yA[trA], minlength=pfc.K).astype(float)
    f0 /= f0.sum()
    model = dev.fit(XA, yA, trA, params, rounds=rounds)
    dev.log("trained on", len(trA), "rows of 2025 with", rounds, "rounds")
    del XA, dA, rawA

    # ---- scoring frame: 2026 with 2025 as history
    rawB = pfc.load([(2025, p25), (2026, p26)])
    last = rawB.loc[rawB["season"] == 2026, "game_date"].max()
    assert str(last.date()) >= REGULAR_SEASON_END_2026, f"2026 ends at {last.date()}"
    bad = pfc.sequence_gate(rawB)
    posB = pfc.position_players(rawB)
    dB = pfc.prepare(rawB, posB)
    del rawB
    te = np.where(((dB["season"] == 2026) & dB["scored"]).to_numpy())[0]
    XB, fl = pfc.make_X(dB, cfg)
    st = dB[["balls", "strikes", "tto"]].iloc[te].reset_index(drop=True)
    prior_n = pfc.asof_counts(dB, "pitcher", [], None)[1][te]

    # ---- gates, before anything is scored
    rng7 = np.random.default_rng(7)
    dh = dev.doubleheader_rows(dB, te, "pitcher")
    sample = np.r_[rng7.choice(te, min(5000, len(te)), replace=False),
                   rng7.choice(dh, min(200, len(dh)), replace=False) if len(dh) else np.array([], int)].astype(int)
    worst = independent_f2(dB, cfg, fl, sample)
    worst_broken = independent_f2(dB, cfg, pfc.floors(dB, cfg, broken="no_sameday"), sample) if len(dh) else None
    gates = {"coverage": {"last_date_2026": str(last.date()), "pass": True},
             "independent_F2": {"n": int(len(sample)), "doubleheader_rows": int(min(200, len(dh))),
                                "worst_abs_diff": worst, "pass": worst < F2_TOL},
             "independent_F2_detects_sameday_leak": {"worst_abs_diff": worst_broken,
                                                     "pass": worst_broken is not None and worst_broken > F2_TOL}}
    dev.log("gates", {k: v["pass"] for k, v in gates.items()})
    dropped = sorted({f"{s}:{p}" for s, p in zip(dB["season"].to_numpy()[dB["cls"].to_numpy() >= 0],
                                                  dB["pitcher"].to_numpy()[dB["cls"].to_numpy() >= 0]) if (s, p) in posB})
    res = {"md5": md5s, "rows_2026": int(len(te)), "illegal_transitions": bad.astype(str).to_dict("records"),
           "position_players_dropped": dropped, "gates": gates, "scoring_started": False}

    def dump():
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=str)

    if not all(v["pass"] for v in gates.values()):
        dump()
        sys.exit(1)

    # ---- scoring (recorded as started before the first prediction, so a crash still counts as the one run)
    res["scoring_started"] = True
    dump()
    y = dB["cls"].to_numpy()[te]
    pm = model.predict(XB.iloc[te])
    loss = {"M": pfc.logloss_rows(pm, y)}
    for name in ("F1", "F2", "F3"):
        loss[name] = pfc.logloss_rows(fl[name][0][te], y)
    loss["F0"] = pfc.logloss_rows(np.tile(f0, (len(te), 1)), y)
    rng = np.random.default_rng(20261101)
    pitcher = dB["pitcher"].to_numpy()[te]
    game = dB["game_pk"].to_numpy()[te]
    res["mean_logloss"] = {k: float(v.mean()) for k, v in loss.items()}
    res["hypotheses"] = {}
    pvals, boots = {}, {}
    for h, floor in (("H1", "F2"), ("H2", "F3")):
        diff = loss["M"] - loss[floor]
        boots[h] = boot_pitchers(diff, pitcher, rng)
        tw = boot_two_way(diff, pitcher, game, rng)
        pvals[h] = float((boots[h] >= BAR).mean())
        res["hypotheses"][h] = {"floor": floor, "mean_diff": float(diff.mean()), "p_vs_bar": pvals[h],
                                "two_way_upper_95": float(np.quantile(tw, 0.95))}
    alive = True
    for rank, h in enumerate(sorted(pvals, key=pvals.get)):
        alpha = 0.05 / (len(pvals) - rank)
        ok = alive and pvals[h] <= alpha
        res["hypotheses"][h].update({"holm_alpha": alpha,
                                     "upper_at_holm_level": float(np.quantile(boots[h], 1 - alpha)),
                                     "verdict": "SUPPORTED" if ok else "NOT SUPPORTED"})
        alive = ok
    dump()
    try:
        res["reported"] = reported(pm, y, fl, te, loss, st, prior_n, pitcher)
    except Exception as e:  # the verdicts above are already recorded
        res["reported"] = {"error": repr(e)}
    dump()
    dev.log("H1", res["hypotheses"]["H1"]["verdict"], "H2", res["hypotheses"]["H2"]["verdict"])


def reported(pm, y, fl, te, loss, st, prior_n, pitcher):
    rep = {"accuracy": float((pm.argmax(1) == y).mean()),
           "top2_accuracy": float((np.argsort(-pm, 1)[:, :2] == y[:, None]).any(1).mean()),
           "brier": float(((pm - np.eye(pfc.K)[y]) ** 2).sum(1).mean())}
    rep["calibration"] = {c: {"mean_pred": float(pm[:, k].mean()), "observed": float((y == k).mean())}
                          for k, c in enumerate(pfc.CLASSES)}
    fb = np.isin(y, [0, 1, 2])
    rep["fastball_auc"] = {"M": auc(pm[:, :3].sum(1), fb), "F3": auc(fl["F3"][0][te][:, :3].sum(1), fb)}
    diff3 = loss["M"] - loss["F3"]
    rep["gain_vs_F3_by_count"] = {f"{b}-{s}": float(diff3[(st["balls"] == b).to_numpy() & (st["strikes"] == s).to_numpy()].mean())
                                  for b in range(4) for s in range(3)}
    rep["gain_vs_F3_by_tto"] = {int(t): float(diff3[(st["tto"].clip(upper=4) == t).to_numpy()].mean()) for t in range(1, 5)}
    cold = prior_n < 300
    rep["cold_start"] = {"pitches": int(cold.sum()), "gain_vs_F3": float(diff3[cold].mean()) if cold.any() else None}
    per = pd.DataFrame({"p": pitcher, "d": diff3}).groupby("p")["d"].mean()
    rep["per_pitcher"] = {"pitchers": int(len(per)), "pitcher_weighted_mean_gain_vs_F3": float(per.mean()),
                          "share_where_M_beats_F3": float((per < 0).mean())}
    return rep


if __name__ == "__main__":
    run(*sys.argv[1:5])
