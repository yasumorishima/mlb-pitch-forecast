"""POST /api/predict -> next-pitch-type probabilities.  GET /api/predict -> model info and an example.

Artifacts (model.json.gz, state.json.gz, meta.json) come from the GitHub release `model-latest`
of yasumorishima/mlb-pitch-forecast. meta.json carries the sha256 of the other two, so a model and
a state from different builds are never served together. Reloaded after CACHE_SECONDS.
"""
import gzip
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _serve as sv  # noqa: E402

BASE = os.environ.get("ARTIFACT_BASE",
                      "https://github.com/yasumorishima/mlb-pitch-forecast/releases/download/model-latest/")
CACHE_SECONDS = 6 * 3600
MAX_BODY = 200_000
_cache = {}

EXAMPLE = {
    "pitcher": 669373, "batter": 592450, "stand": "R", "p_throws": "L", "starter": True,
    "balls": 1, "strikes": 2, "outs": 1, "on_1b": False, "on_2b": True, "on_3b": False,
    "inning": 6, "top": True, "score_diff": 1, "pitch_number": 4, "tto": 3, "at_bat": 42,
    "game_pitches": [
        {"type": "FF", "speed": 95.1, "zone": 5, "description": "called_strike",
         "balls": 0, "strikes": 0, "stand": "R", "at_bat": 42},
        {"type": "SL", "speed": 86.0, "zone": 14, "description": "ball",
         "balls": 0, "strikes": 1, "stand": "R", "at_bat": 42},
        {"type": "FF", "speed": 95.8, "zone": 2, "description": "foul",
         "balls": 1, "strikes": 1, "stand": "R", "at_bat": 42}],
    "batter_game_pitches": [{"type": "FF"}, {"type": "SL"}, {"type": "FF"}]}

INT_FIELDS = ("pitcher", "batter", "balls", "strikes", "outs", "inning", "score_diff", "pitch_number", "tto", "at_bat")
BOOL_FIELDS = ("starter", "on_1b", "on_2b", "on_3b", "top")


def _get(name):
    with urllib.request.urlopen(BASE + name, timeout=20) as r:
        return r.read()


def _load():
    for attempt in range(3):
        meta = json.loads(_get("meta.json"))
        mz, sz = _get("model.json.gz"), _get("state.json.gz")
        if (hashlib.sha256(mz).hexdigest() == meta["sha256"]["model.json.gz"]
                and hashlib.sha256(sz).hexdigest() == meta["sha256"]["state.json.gz"]):
            return meta, mz, sz
        time.sleep(2 * (attempt + 1))        # an upload is in progress; try again
    raise RuntimeError("artifacts do not match their meta.json")


def artifacts():
    if "booster" not in _cache or time.time() - _cache["loaded"] > CACHE_SECONDS:
        meta, mz, sz = _load()
        booster = sv.Booster(json.loads(gzip.decompress(mz)))
        raw = json.loads(gzip.decompress(sz))

        def counts(c):
            cells = {}
            for key, val in c["cells"].items():
                parts = tuple(int(p) for p in key.split("|")) if key else ()
                cells[parts] = (sv.np.array(val[:sv.K]), int(val[sv.K]))
            return sv.Counts(cells, c["n"])

        state = {"cfg": meta["cfg"], "batter_h": meta["batter_h"],
                 "pitchers": {int(p): {n: counts(c) for n, c in v.items()} for p, v in raw["pitchers"].items()},
                 "batters": {int(b): counts(c) for b, c in raw["batters"].items()}}
        _cache.clear()
        _cache.update(meta=meta, booster=booster, state=state, loaded=time.time())
    return _cache


def clean(req):
    """Coerce types; raise ValueError on anything malformed."""
    missing = [k for k in INT_FIELDS + BOOL_FIELDS + ("stand", "p_throws") if k not in req]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    out = dict(req)
    for k in INT_FIELDS:
        out[k] = int(req[k])
    for k in BOOL_FIELDS:
        if not isinstance(req[k], bool):
            raise ValueError(f"{k} must be true or false")
    for k in ("stand", "p_throws"):
        if req[k] not in ("L", "R"):
            raise ValueError(f"{k} must be L or R")
    if not (0 <= out["balls"] <= 3 and 0 <= out["strikes"] <= 2):
        raise ValueError("balls must be 0-3 and strikes 0-2")
    for k in ("game_pitches", "batter_game_pitches"):
        v = req.get(k, [])
        if not isinstance(v, list) or len(v) > 400:
            raise ValueError(f"{k} must be a list of at most 400 pitches")
    for p in req.get("game_pitches", []):
        for k in ("balls", "strikes", "at_bat"):
            p[k] = int(p[k])
        if p.get("stand") not in ("L", "R"):
            raise ValueError("each game pitch needs stand L or R")
    for k in ("pitcher_pitches_other_games_today", "batter_pitches_other_games_today"):
        if k in req:
            out[k] = int(req[k])
    return out


def predict(req):
    a = artifacts()
    x = sv.features(req, a["state"])
    row = [float(sv.np.float32(x[n])) for n in a["meta"]["features"]]
    p = a["booster"].predict(row)
    probs = {c: round(float(v), 6) for c, v in zip(sv.CLASSES, p)}
    known = {"pitcher": req["pitcher"] in a["state"]["pitchers"], "batter": req["batter"] in a["state"]["batters"]}
    return {"probabilities": probs, "most_likely": max(probs, key=probs.get), "known": known,
            "data_through": a["meta"]["data_through"], "model": a["meta"]["model_trained_on"]}


class handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            m = artifacts()["meta"]
            self._send(200, {"classes": sv.CLASSES, "data_through": m["data_through"],
                             "model": m["model_trained_on"], "serve_season": m["serve_season"],
                             "repo": "https://github.com/yasumorishima/mlb-pitch-forecast",
                             "example_request": EXAMPLE, "example_response": predict(clean(EXAMPLE))})
        except Exception as e:  # noqa: BLE001
            self._send(503, {"error": "artifacts unavailable", "detail": type(e).__name__})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send(400, {"error": "bad Content-Length"})
        if n <= 0 or n > MAX_BODY:
            return self._send(413 if n > MAX_BODY else 400, {"error": "send a JSON body (max 200 KB)"})
        try:
            req = clean(json.loads(self.rfile.read(n)))
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            return self._send(400, {"error": "bad request", "detail": f"{e}"[:300]})
        try:
            out = predict(req)
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": "bad request", "detail": f"{type(e).__name__}: {e}"[:300]})
        except Exception as e:  # noqa: BLE001
            return self._send(503, {"error": "artifacts unavailable", "detail": type(e).__name__})
        if any(math.isnan(v) for v in out["probabilities"].values()):
            return self._send(500, {"error": "prediction failed"})
        self._send(200, out)
