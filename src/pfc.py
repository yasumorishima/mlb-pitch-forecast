"""Next-pitch forecast: loading, as-of floors and features.

Everything for pitch k is built from rows strictly before k in time order
(earlier dates, and earlier pitches of the same game). Other games of the
same date are left out for every method (doubleheaders).
"""
import numpy as np
import pandas as pd

CLASSES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS", "OTHER"]
K = len(CLASSES)
CODE = {"FF": 0, "SI": 1, "FC": 2, "SL": 3, "ST": 4, "CU": 5, "KC": 5, "CS": 5,
        "SV": 5, "CH": 6, "FS": 7, "FO": 7}
NOT_SCORED_TYPES = {"PO", "IN", "UN"}
FIRST = K          # "no earlier pitch in this plate appearance"
OHTANI = 660271

COLS = ["game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
        "pitcher", "batter", "pitch_type", "release_speed", "zone",
        "description", "balls", "strikes", "outs_when_up", "on_1b", "on_2b",
        "on_3b", "inning", "inning_topbot", "stand", "p_throws", "bat_score",
        "fld_score", "home_team", "away_team",
        "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7",
        "fielder_8", "fielder_9"]
PITCH_INFO = ["pitch_type", "release_speed", "zone", "description"]   # what a pitch reveals about itself

BALL = {"ball", "blocked_ball", "pitchout", "intent_ball", "automatic_ball"}
STRIKE = {"called_strike", "swinging_strike", "swinging_strike_blocked",
          "missed_bunt", "foul_tip", "automatic_strike", "bunt_foul_tip",
          "foul_bunt"}
FOUL = {"foul", "foul_pitchout"}
DESC_GROUP = {"ball": 0, "blocked_ball": 0, "automatic_ball": 0, "pitchout": 0,
              "intent_ball": 0, "called_strike": 1, "automatic_strike": 1,
              "swinging_strike": 2, "swinging_strike_blocked": 2,
              "missed_bunt": 2, "foul": 3, "foul_tip": 3, "foul_bunt": 3,
              "bunt_foul_tip": 3, "foul_pitchout": 3}

STATE = ["balls", "strikes", "outs_when_up", "on_1b_occ", "on_2b_occ", "on_3b_occ",
         "inning", "fld_home", "score_diff", "stand_r", "throw_r", "same_hand",
         "starter", "pitch_number", "game_pitch_n", "tto", "pa_cls", "pa_speed",
         "pa_zone", "pa_desc", "g_prev_cls", "g_prev_speed"]
CELLS = {"F1": [], "F2": ["stand_r", "balls", "strikes"], "F3": ["balls", "strikes", "pa_cls"]}
BATTER_H = 1000
SORT = ["game_date", "game_pk", "at_bat_number", "pitch_number"]


def load(paths):
    """Read seasons through the column whitelist, regular season only, in time order."""
    parts = []
    for season, path in paths:
        d = pd.read_parquet(path, columns=COLS)
        d = d[d["game_type"] == "R"].copy()
        d["season"] = season
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"])
    n_dates = d.groupby("game_pk")["game_date"].nunique()
    assert (n_dates == 1).all(), f"games spanning dates: {list(n_dates[n_dates > 1].index[:20])}"
    return d.sort_values(SORT, kind="mergesort").reset_index(drop=True)


def sequence_gate(d, max_bad=5):
    """Unique keys, pitch_number 1..n with no gaps, legal count transitions."""
    key = ["game_pk", "at_bat_number", "pitch_number"]
    assert not d.duplicated(key).any(), "duplicate pitch keys"
    g = d.groupby(["game_pk", "at_bat_number"])["pitch_number"]
    assert (g.transform("min") == 1).all(), "plate appearance not starting at pitch 1"
    assert (g.transform("max") == g.transform("count")).all(), "gap in pitch_number"
    p = d.groupby(["game_pk", "at_bat_number"])[["description", "balls", "strikes"]].shift(1)
    m = p["description"].notna()
    db = (d["balls"] - p["balls"])[m]
    ds = (d["strikes"] - p["strikes"])[m]
    pdesc = p["description"][m]
    ps = p["strikes"][m]
    exp_b = pdesc.isin(BALL).astype(int)
    exp_s = (pdesc.isin(STRIKE) | (pdesc.isin(FOUL) & (ps < 2))).astype(int)
    known = pdesc.isin(BALL | STRIKE | FOUL)
    bad = (~known) | (db != exp_b) | (ds != exp_s)
    rows = d.loc[bad[bad].index, key + ["season", "balls", "strikes"]]
    for season, n in rows["season"].value_counts().items():
        assert n <= max_bad, f"{n} illegal count transitions in {season}"
    return rows


def position_players(d):
    """Pitcher-seasons whose id is a fielder (1B..RF) in >= 5 games that season or the previous one."""
    f = d.drop_duplicates(["game_pk", "inning_topbot"])[["season", "game_pk"] + [f"fielder_{i}" for i in range(3, 10)]]
    f = f.melt(id_vars=["season", "game_pk"], value_name="pid").dropna()
    games = f.drop_duplicates(["season", "game_pk", "pid"]).groupby(["season", "pid"]).size()
    out = set()
    for (s, p), n in games.items():
        if n >= 5:
            out.add((int(s), int(p)))
            out.add((int(s) + 1, int(p)))
    assert all(p != OHTANI for _, p in out), "Ohtani flagged as a position player"
    return out


def prepare(d, pos=None):
    """Row-level labels, flags and pre-release state. `pos` = position-player set (computed if None)."""
    d = d.copy()
    pt = d["pitch_type"]
    d["is_pitch"] = pt.notna()
    d["cls"] = pt.map(CODE).fillna(K - 1).astype(np.int16)
    # not counted anywhere as a pitch type: automatic balls/strikes, pitchouts, intentional balls, unknown
    d.loc[pt.isna() | pt.isin(NOT_SCORED_TYPES) | (d["description"] == "pitchout"), "cls"] = -1
    for c in ("balls", "strikes", "stand", "pitcher", "batter", "game_pk"):
        assert d[c].notna().all(), f"missing {c}"
    if pos is None:
        pos = position_players(d)
    is_pos = np.array([(s, p) in pos for s, p in zip(d["season"], d["pitcher"])], dtype=bool)
    d["scored"] = (d["cls"] >= 0) & ~is_pos
    d["fld_home"] = (d["inning_topbot"] == "Top").astype(np.int8)
    d["score_diff"] = (d["fld_score"] - d["bat_score"]).astype(np.int16)
    for b in ("on_1b", "on_2b", "on_3b"):
        d[b + "_occ"] = d[b].notna().astype(np.int8)
    d["stand_r"] = (d["stand"] == "R").astype(np.int8)
    d["throw_r"] = (d["p_throws"] == "R").astype(np.int8)
    d["same_hand"] = (d["stand"] == d["p_throws"]).astype(np.int8)
    first = d.groupby(["game_pk", "fld_home"])["pitcher"].transform("first")
    d["starter"] = (d["pitcher"] == first).astype(np.int8)
    d["game_pitch_n"] = d.groupby(["game_pk", "pitcher"])["is_pitch"].cumsum() - d["is_pitch"].astype(int)
    pa = d.drop_duplicates(["game_pk", "at_bat_number"])[["game_pk", "at_bat_number", "pitcher", "batter"]].copy()
    pa["tto"] = pa.groupby(["game_pk", "pitcher", "batter"]).cumcount() + 1
    d = d.merge(pa[["game_pk", "at_bat_number", "tto"]], on=["game_pk", "at_bat_number"], how="left")
    prev = _prev_pitch(d, ["game_pk", "at_bat_number"])
    for c in prev:
        d["pa_" + c] = prev[c].to_numpy()
    prevg = _prev_pitch(d, ["game_pk", "pitcher"])
    d["g_prev_cls"] = prevg["cls"].to_numpy()
    d["g_prev_speed"] = prevg["speed"].to_numpy()
    return d


def _prev_pitch(d, keys):
    """Type, speed, zone and outcome group of the previous counted pitch within `keys` (one row, all fields)."""
    n = len(d)
    own = np.where(d["cls"].to_numpy() >= 0, np.arange(n, dtype=float), np.nan)
    by = [d[k].to_numpy() for k in keys]
    last = pd.Series(own).groupby(by).shift(1).groupby(by).ffill().to_numpy()
    ok = ~np.isnan(last)
    j = last[ok].astype(int)
    out = pd.DataFrame(index=d.index)
    cls = np.full(n, FIRST, np.int16)
    cls[ok] = d["cls"].to_numpy()[j]
    out["cls"] = cls
    for name, col in (("speed", d["release_speed"]), ("zone", d["zone"]),
                      ("desc", d["description"].map(DESC_GROUP))):
        v = np.full(n, np.nan)
        v[ok] = col.astype(float).to_numpy()[j]
        out[name] = v
    return out


def asof_counts(d, entity, cell_cols, h, broken=None):
    """Decayed class counts of `entity`'s earlier pitches in the same cell, as of each row.

    Decay is by the entity's own counted-pitch index i (all its earlier counted pitches,
    including those of another game on the same date): weight 0.5 ** (age / h); h=None = no decay.
    Earlier dates are always visible; on the same date only the row's own game is visible.
    Returns (counts[n, K] float32, total[n] float32), both measured before the row.
    `broken` plants a defect for the gate self-test: "include_current" or "no_sameday".
    """
    n = len(d)
    out = np.zeros((n, K), np.float32)
    tot = np.zeros(n, np.float32)
    ent = d[entity].to_numpy()
    date = d["game_date"].to_numpy()
    game = d["game_pk"].to_numpy()
    if broken == "no_sameday":
        game = date
    cls = d["cls"].to_numpy()
    cells = list(zip(*[d[c].to_numpy().tolist() for c in cell_cols])) if cell_cols else [()] * n
    lam = 0.0 if h is None else np.log(2.0) / h
    st = {}
    for r in range(n):
        e = ent[r]
        s = st.get(e)
        if s is None:
            s = st[e] = [0, {}, date[r], {}]
        if date[r] != s[2]:
            for gd in s[3].values():
                for c, (v, ref) in gd.items():
                    _add(s[1], c, v, ref, lam)
            s[2] = date[r]
            s[3] = {}
        gd = s[3].setdefault(game[r], {})
        i = s[0]
        c = cells[r]
        if broken == "include_current" and cls[r] >= 0:
            one = np.zeros(K)
            one[cls[r]] = 1.0
            _add(gd, c, one, i, lam)
            s[0] = i = i + 1
        acc = np.zeros(K)
        for store in (s[1], gd):
            hit = store.get(c)
            if hit is not None:
                v, ref = hit
                acc += v * np.exp(-lam * (i - ref))
        out[r] = acc
        tot[r] = acc.sum()
        if cls[r] >= 0 and broken != "include_current":
            one = np.zeros(K)
            one[cls[r]] = 1.0
            _add(gd, c, one, i, lam)
            s[0] = i + 1
    return out, tot


def _add(store, c, v, ref, lam):
    hit = store.get(c)
    if hit is None:
        store[c] = (v.copy(), ref)
        return
    v0, ref0 = hit
    if ref >= ref0:
        store[c] = (v0 * np.exp(-lam * (ref - ref0)) + v, ref)
    else:
        store[c] = (v0 + v * np.exp(-lam * (ref0 - ref)), ref0)


def shrink(counts, total, prior, a):
    """(counts + a * prior) / (total + a), rows of `prior` sum to 1."""
    return (counts + a * prior) / (total[:, None] + a)


def floors(d, cfg, broken=None, counts=None):
    """F1 -> F2 -> F3 probabilities with the chosen (h, a); returns dict name -> (p, n)."""
    base = np.asarray(cfg["prior"])[d["starter"].to_numpy()]
    out = {}
    for name in ("F1", "F2", "F3"):
        h, a = cfg[name]["h"], cfg[name]["a"]
        c, t = counts[(name, h)] if counts is not None else asof_counts(d, "pitcher", CELLS[name], h, broken)
        base = shrink(c, t, base, a)
        out[name] = (base, t)
    return out


def make_X(d, cfg, broken=None, counts=None):
    """Model inputs for every row of a prepared frame."""
    fl = floors(d, cfg, broken, counts)
    X = d[STATE].astype(np.float32).copy()
    for name in ("F1", "F2", "F3"):
        for k, c in enumerate(CLASSES):
            X[f"{name}_{c}"] = fl[name][0][:, k].astype(np.float32)
    X["n_f2"] = fl["F2"][1].astype(np.float32)
    bc, bt = asof_counts(d, "batter", [], BATTER_H, broken)
    bp = (bc + 1.0) / (bt[:, None] + K)
    one = np.zeros((len(d), K), np.float32)
    m = d["cls"].to_numpy() >= 0
    one[np.where(m)[0], d["cls"].to_numpy()[m]] = 1.0
    if broken == "include_current":
        cum = pd.DataFrame(one).groupby([d["game_pk"].to_numpy(), d["pitcher"].to_numpy()]).cumsum().to_numpy()
    else:
        cum = pd.DataFrame(one).groupby([d["game_pk"].to_numpy(), d["pitcher"].to_numpy()]).cumsum().to_numpy() - one
    gt = cum.sum(1)
    for k, c in enumerate(CLASSES):
        X[f"bat_{c}"] = bp[:, k].astype(np.float32)
        X[f"game_{c}"] = (cum[:, k] / np.maximum(gt, 1)).astype(np.float32)
    X["game_n"] = gt.astype(np.float32)
    return X, fl


def clip(p, eps=1e-6):
    p = np.clip(p, eps, None)
    return p / p.sum(1, keepdims=True)


def logloss_rows(p, y):
    return -np.log(clip(p)[np.arange(len(y)), y])
