"""Write FREEZE.json from a full development result. Reads no pitch of 2026 (md5 only).

Usage: python src/freeze.py <dev_result.json> <dev_result_name> <code_sha> <2024> <2025> <2026> <kaggle_version>
"""
import json
import os
import sys

from test import md5, ROOT


def main(path, name, code_sha, p24, p25, p26, version):
    r = json.load(open(path))
    assert not r["quick"] and r["pass"], "not a full, passing development result"
    dev_sha = name.rsplit("_", 1)[-1].removesuffix(".json")
    fz = {"code_sha": code_sha, "dev_sha": dev_sha, "dev_result": name,
          "floors": r["floors"], "model": {"params": r["model"]["params"],
                                           "best_iteration": r["model"]["best_iteration"]},
          "validation": {"model_val_logloss": r["model"]["val_logloss"],
                         "minus_F2": r["model"]["minus_F2"], "minus_F3": r["model"]["minus_F3"]},
          "md5": {"2024": md5(p24), "2025": md5(p25), "2026": md5(p26)},
          "kaggle_dataset": "yasunorim/mlb-statcast-bat-tracking-2024-2025", "kaggle_version": version}
    with open(os.path.join(ROOT, "FREEZE.json"), "w") as f:
        json.dump(fz, f, indent=1)
    print(json.dumps(fz, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:8])
