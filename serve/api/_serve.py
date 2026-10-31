"""Serving logic shared by the API and by the build gates. numpy only.

A request describes one pitch about to be thrown. The server holds, per pitcher and
per batter, the decayed class counts of every earlier date (the "committed" state);
the request carries what happened earlier in *this* game. Features are rebuilt exactly
as in training (src/pfc.py), so the gates can require equality with the training rows.
"""
import math

import numpy as np

CLASSES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS", "OTHER"]
K = len(CLASSES)
CODE = {"FF": 0, "SI": 1, "FC": 2, "SL": 3, "ST": 4, "CU": 5, "KC": 5, "CS": 5,
        "SV": 5, "CH": 6, "FS": 7, "FO": 7}
NOT_COUNTED = {"PO", "IN", "UN"}
FIRST = K
DESC_GROUP = {"ball": 0, "blocked_ball": 0, "automatic_ball": 0, "pitchout": 0,
              "intent_ball": 0, "called_strike": 1, "automatic_strike": 1,
              "swinging_strike": 2, "swinging_strike_blocked": 2,
              "missed_bunt": 2, "foul": 3, "foul_tip": 3, "foul_bunt": 3,
              "bunt_foul_tip": 3, "foul_pitchout": 3}
STATE = ["balls", "strikes", "outs_when_up", "on_1b_occ", "on_2b_occ", "on_3b_occ",
         "inning", "fld_home", "score_diff", "stand_r", "throw_r", "same_hand",
         "starter", "pitch_number", "game_pitch_n", "tto", "pa_cls", "pa_speed",
         "pa_zone", "pa_desc", "g_prev_cls", "g_prev_speed"]


def type_class(pitch_type, description=None):
    """Class index of a thrown pitch, or -1 when it is not counted as a pitch type."""
    if pitch_type is None or pitch_type in NOT_COUNTED or description == "pitchout":
        return -1
    return CODE.get(pitch_type, K - 1)


def cell_keys(stand_r, balls, strikes, pa_cls):
    return {"F1": (), "F2": (stand_r, balls, strikes), "F3": (balls, strikes, pa_cls)}


class Counts:
    """Decayed counts of one entity: cells -> (vector, ref index), plus its pitch clock."""

    def __init__(self, cells=None, n=0):
        self.cells = cells or {}
        self.n = n

    def read(self, cell, i, lam):
        hit = self.cells.get(cell)
        if hit is None:
            return np.zeros(K)
        v, ref = hit
        return v * math.exp(-lam * (i - ref))

    def add(self, cell, k, i, lam):
        one = np.zeros(K)
        one[k] = 1.0
        hit = self.cells.get(cell)
        if hit is None:
            self.cells[cell] = (one, i)
            return
        v0, ref0 = hit
        if i >= ref0:
            self.cells[cell] = (v0 * math.exp(-lam * (i - ref0)) + one, i)
        else:
            self.cells[cell] = (v0 + one * math.exp(-lam * (ref0 - i)), ref0)


def lam_of(h):
    return 0.0 if h is None else math.log(2.0) / h


def features(req, model_state):
    """Feature vector (dict) for one request; same definitions as training."""
    cfg = model_state["cfg"]
    prior = np.asarray(cfg["prior"])[1 if req["starter"] else 0]
    stand_r = 1 if req["stand"] == "R" else 0
    throw_r = 1 if req["p_throws"] == "R" else 0
    game = req.get("game_pitches", [])          # this pitcher's earlier pitches today, in order
    cur_ab = req["at_bat"]
    # previous counted pitch in the plate appearance and in the game
    if "pa_prev" in req:                        # explicit: previous counted pitch of this PA by any pitcher
        pa_prev = req["pa_prev"]
    else:
        pa_prev = next((p for p in reversed(game) if p["at_bat"] == cur_ab and type_class(p["type"], p.get("description")) >= 0), None)
    g_prev = next((p for p in reversed(game) if type_class(p["type"], p.get("description")) >= 0), None)
    x = {"balls": req["balls"], "strikes": req["strikes"], "outs_when_up": req["outs"],
         "on_1b_occ": int(bool(req["on_1b"])), "on_2b_occ": int(bool(req["on_2b"])),
         "on_3b_occ": int(bool(req["on_3b"])), "inning": req["inning"],
         "fld_home": int(bool(req["top"])), "score_diff": req["score_diff"],
         "stand_r": stand_r, "throw_r": throw_r, "same_hand": int(req["stand"] == req["p_throws"]),
         "starter": int(bool(req["starter"])), "pitch_number": req["pitch_number"],
         "game_pitch_n": sum(1 for p in game if p["type"] is not None), "tto": req["tto"],
         "pa_cls": FIRST if pa_prev is None else type_class(pa_prev["type"], pa_prev.get("description")),
         "pa_speed": np.nan if pa_prev is None else _f(pa_prev.get("speed")),
         "pa_zone": np.nan if pa_prev is None else _f(pa_prev.get("zone")),
         "pa_desc": np.nan if pa_prev is None else _f(DESC_GROUP.get(pa_prev.get("description"))),
         "g_prev_cls": FIRST if g_prev is None else type_class(g_prev["type"], g_prev.get("description")),
         "g_prev_speed": np.nan if g_prev is None else _f(g_prev.get("speed"))}
    # floors: committed state of earlier dates + this game's earlier pitches
    pst = model_state["pitchers"].get(req["pitcher"])
    out = {}
    base = prior
    for name in ("F1", "F2", "F3"):
        h, a = cfg[name]["h"], cfg[name]["a"]
        lam = lam_of(h)
        c = Counts(dict(pst[name].cells) if pst else {}, pst[name].n if pst else 0)
        i = c.n + int(req.get("pitcher_pitches_other_games_today", 0))
        prev_in_pa = {}
        for p in game:
            k = type_class(p["type"], p.get("description"))
            sr = 1 if p["stand"] == "R" else 0
            pc = _cls_of(p["pa_prev"]) if "pa_prev" in p else prev_in_pa.get(p["at_bat"], FIRST)
            if k >= 0:
                c.add(cell_keys(sr, p["balls"], p["strikes"], pc)[name], k, i, lam)
                prev_in_pa[p["at_bat"]] = k
                i += 1
        cell = cell_keys(stand_r, req["balls"], req["strikes"], x["pa_cls"])[name]
        v = c.read(cell, i, lam)
        base = (v + a * base) / (v.sum() + a)
        out[name] = base
        if name == "F2":
            x["n_f2"] = float(v.sum())
    for name in ("F1", "F2", "F3"):
        for k, cn in enumerate(CLASSES):
            x[f"{name}_{cn}"] = out[name][k]
    # batter: committed state + pitches he has seen earlier today (any pitcher), in order
    bst = model_state["batters"].get(req["batter"])
    bc = Counts(dict(bst.cells) if bst else {}, bst.n if bst else 0)
    lam_b = lam_of(model_state["batter_h"])
    i = bc.n + int(req.get("batter_pitches_other_games_today", 0))
    for t in req.get("batter_game_pitches", []):
        k = type_class(t.get("type"), t.get("description"))
        if k >= 0:
            bc.add((), k, i, lam_b)
            i += 1
    bv = bc.read((), i, lam_b)
    bp = (bv + 1.0) / (bv.sum() + K)
    gm = np.zeros(K)
    for p in game:
        k = type_class(p["type"], p.get("description"))
        if k >= 0:
            gm[k] += 1
    gt = gm.sum()
    for k, cn in enumerate(CLASSES):
        x[f"bat_{cn}"] = bp[k]
        x[f"game_{cn}"] = gm[k] / max(gt, 1.0)
    x["game_n"] = gt
    return x


def _f(v):
    return np.nan if v is None else float(v)


def _cls_of(v):
    """Previous-pitch class given as an index (0..9) or a pitch-type code; None/"FIRST" = first pitch."""
    if v is None or v == "FIRST":
        return FIRST
    if isinstance(v, (int, np.integer)):
        return int(v)
    k = type_class(v)
    return FIRST if k < 0 else k


# ---- LightGBM model evaluated from its JSON dump (no lightgbm at serving time) ----------

class Booster:
    def __init__(self, dump):
        self.names = dump["feature_names"]
        self.k = dump["num_class"]
        self.trees = [t["tree_structure"] for t in dump["tree_info"]]

    def raw(self, row):
        """row: list of float in feature order (NaN = missing)."""
        s = np.zeros(self.k)
        for t, node in enumerate(self.trees):
            while "leaf_value" not in node:
                node = node["left_child"] if _go_left(node, row[node["split_feature"]]) else node["right_child"]
            s[t % self.k] += node["leaf_value"]
        return s

    def predict(self, row):
        z = self.raw(row)
        z = np.exp(z - z.max())
        return z / z.sum()


def _go_left(node, v):
    missing = v is None or (isinstance(v, float) and math.isnan(v))
    if node["decision_type"] == "==":
        if missing or v < 0:
            return False
        cats = node.get("_cats")
        if cats is None:
            cats = node["_cats"] = {int(c) for c in str(node["threshold"]).split("||")}
        return int(v) in cats
    mt = node.get("missing_type", "None")
    if mt == "NaN" and missing:
        return node["default_left"]
    if mt == "Zero" and (missing or v == 0.0):
        return node["default_left"]
    if missing:
        v = 0.0
    return v <= node["threshold"]
