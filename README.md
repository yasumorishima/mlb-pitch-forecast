# Next-pitch forecast (MLB)

Before a pitch is thrown, how well can the next pitch type be predicted, and does a model add anything over the tables a scout already carries (a pitcher's usage by count and batter side, and by count and previous pitch)?

The plan is pre-registered in [PLAN.md](PLAN.md). Development uses 2024 (history only) and 2025; the 2026 season is scored once, after the plan and code are frozen in `FREEZE.md`. After the test, the model is served as a free JSON API and retrained in season on GitHub Actions.

**Result (2026, scored once after the freeze): both hypotheses SUPPORTED.** The model beats the scout tables by about 0.063 to 0.065 nats per pitch (log loss 1.231 vs 1.294 and 1.296), for 97.5% of 832 pitchers; it still names the right pitch type only 46% of the time. Details in [RESULTS.md](RESULTS.md).



Data: every regular-season pitch from Baseball Savant, via the Kaggle dataset [MLB Statcast + Bat Tracking (2024-2026)](https://www.kaggle.com/datasets/yasunorim/mlb-statcast-bat-tracking-2024-2025). Not affiliated with MLB.

## API

The frozen model is served as a free JSON API from `serve/` (Vercel, Python function, numpy only).

- `GET /api/predict`: model info, the date the data runs through, and an example request with its answer.
- `POST /api/predict`: one pitch about to be thrown. Required fields: `pitcher`, `batter` (MLBAM ids), `stand`, `p_throws` (`"L"`/`"R"`), `starter`, `balls`, `strikes`, `outs`, `on_1b`, `on_2b`, `on_3b`, `inning`, `top`, `score_diff` (fielding side minus batting side), `pitch_number` (in the plate appearance), `tto` (times this batter has faced the pitcher today, this one included), `at_bat`. Optional: `game_pitches` (this pitcher's earlier pitches today, in order: `type`, `speed`, `zone`, `description`, `balls`, `strikes`, `stand`, `at_bat`), `batter_game_pitches` (pitches the batter has seen today), `pa_prev` (the previous pitch of this plate appearance when another pitcher threw it).
- The answer: probabilities for the nine classes, the most likely class, and whether the pitcher and batter are known to the model.

The features are rebuilt exactly as in training; the build gate checks this on 460 real pitches (no feature differs, largest probability difference 2e-16).

## In-season updates

`build-model.yml` runs daily from March to October. It fetches the season from Baseball Savant day by day (re-fetching the last 3 days), refreshes every pitcher's and batter's state, and retrains with the frozen settings on all but the last 7 days. The new model replaces the served one only if, on those 7 days, the 95% upper bound (bootstrap over pitchers) of new minus current log loss is at or below 0. The first build (data through 2026-09-27) passed: 1.2040 against 1.2105 for the frozen test model. Savant revises a few pitch-type labels after the fact (52 of 26,303 pitches in the last week of 2026 between 9 and 31 October); the fetch check allows up to 1% of such revisions and requires every other column to be identical.
