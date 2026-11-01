# CLAUDE.md — mlb-pitch-forecast

Working rules for Claude Code (cloud or local). This file is the source of truth for this project; the owner's local notes are not visible to cloud sessions.

## Session protocol (cloud)
This project is worked on in Claude Code cloud. The owner starts a session with just 「続き」/「つづきから」.
1. Start the session in the environment 「クラウド許可1」 (full network). Read this file first, then continue from the first unfinished item in **Next tasks**.
2. Before ending, record what you did and what comes next **in this file** (update "Next tasks" and add a dated line to "Progress log"). This file is the only memory that survives between cloud sessions.
3. Cloud can push only its session branch: open a PR to `main` and merge it yourself (own repo). Report the merged commit. Use a **merge commit (not squash)** if the PR contains a frozen pre-registration commit.
4. **A merge to `main` redeploys the live API** (Vercel is connected to this repo; see "API"). If the PR touches `serve/`, check the live API after the merge, or say it is unverified.
5. Cloud has no `gh` CLI: use the GitHub MCP tools. `workflow_dispatch` returns **403** from cloud, so cloud cannot start `dev` / `freeze` / `test` / `build-model` by hand. Anything that must run on GitHub Actions has to be triggered by a push or PR merge (no such trigger exists yet).
6. Write replies to the owner in Japanese (です/ます). Do not claim anything you did not run in this session. Say where every number was measured (cloud / GitHub Actions run id / Raspberry Pi). The numbers recorded up to 2026-11-01 come from GitHub Actions runs and the owner's Raspberry Pi.
7. The Raspberry Pi and the owner's PC are not reachable from cloud.

## Progress log
- 2026-10-31: PLAN v3; four full development runs (two unsaved because of a `bash -e` bug in the save step, grids widened twice; all in PLAN.md "Development log"). Frozen `9d2497a` (dev run 41198675420), scored 2026 once in run 41201400282: **H1 and H2 SUPPORTED** (RESULTS.md).
- 2026-10-31: serving build `build_serving.py` + `build-model.yml`; first build run 41205101259 adopted a model trained on 2026 before 2026-09-21 (hold-out log loss 1.2040 vs 1.2105 for the frozen test model). Release `model-latest` + `model-20261031`.
- 2026-11-01: API live at https://mlb-pitch-forecast.vercel.app/api/predict (README `dfe6db6`). The working copy on the owner's Raspberry Pi equals `main` (`dfe6db6`, clean) — nothing unpushed.
- 2026-11-01: this CLAUDE.md written for cloud.

## What this is
Before a pitch is thrown, how well can the next pitch type (9 classes) be predicted, and does a model beat a scout's usage tables (F2 = pitcher × batter side × count; F3 = pitcher × count × previous pitch)? Pre-registered in `PLAN.md`, tested once on the 2026 season: M 1.2309 vs F3 1.2942 / F2 1.2955 nats per pitch (712,672 pitches, 832 pitchers), both hypotheses SUPPORTED; accuracy 46%. Then served as a free JSON API with daily in-season retraining. See `README.md` and `RESULTS.md`.

Layout: `src/pfc.py` (features, floors, gates), `src/dev.py` (development run), `src/freeze.py`, `src/test.py` (the one 2026 scoring), `src/build_serving.py` (serving artifacts + gates), `src/fetch_savant.py` (Savant day-by-day fetch), `serve/api/` (Vercel function, numpy only). Results JSON live on the **`results` branch** (`results/dev_*.json`, `results/test_*.json`).

## Non-negotiable discipline
1. **The 2026 test is done and is never re-run or re-scored.** `test.yml` refuses if a `results/test_*` file exists; do not work around it. No new 2026 scoring of any kind is presented as a test.
2. **Never edit frozen records:** `FREEZE.json`, the pre-registered sections of `PLAN.md` (only append to "Development log"), the verdict numbers in `RESULTS.md`, and anything on the `results` branch. The md5s of the three season files are in `FREEZE.json`.
3. **The served model must stay the tested one.** `build_serving.py` reads the frozen floors, parameters and rounds from `FREEZE.json` and imports `src/pfc.py` / `src/dev.py`. Changing features or settings there changes what the API serves — that needs a new pre-registration (in a new file) before any run, not an edit.
4. Gates are never loosened to pass (the build's parity gate, booster gate and publish gate; PLAN.md "Gates"). A failed gate is reported as it is.
5. Report nulls as facts. Code that produces a result gets an independent review before it runs; verify review claims against the code before accepting them.

## Next tasks (in order)
1. **Small fixes:** README says the plan is frozen in `FREEZE.md` — the file is `FREEZE.json`. `build-model.yml` cron comment says "does nothing outside March to September" but the guard runs March–October (`m < 3 || m > 10` → off). Fix both (comment/README only; no logic change).
2. **A cloud-checkable smoke test for `serve/api`:** a pytest that loads `model-latest` (public release assets), calls `predict.predict(predict.EXAMPLE)` (nine probabilities summing to 1, `data_through` present) and checks that the three kinds of bad input the README describes return 400 through `handler`. Add a workflow triggered on push/PR (not `workflow_dispatch`, which cloud cannot start) that runs it, so a PR touching `serve/` is checked before the merge redeploys the API.
3. Nothing else is scheduled for this repo in the off-season. Do not start new modelling without a new pre-registration agreed with the owner.

## How it runs
### Pre-registration pipeline (finished)
`dev.yml` (2024 + 2025 only; `--quick` = first 40 days, code check, no numbers) → `freeze.yml` (writes `FREEZE.json` from a saved dev result, checks the dev run's md5s) → `test.yml` (scores 2026 once; refuses if code/settings/md5s differ from the freeze or a test result exists). All `workflow_dispatch` with a required `memo`; secrets `KAGGLE_USERNAME` / `KAGGLE_KEY`.

### Daily rebuild (`build-model.yml`)
- Schedule `23 11 * * *` UTC; a guard makes scheduled runs do nothing in November–February. From October the served season is next year (`serve_season = year + 1`).
- Data: Kaggle `statcast_2024/2025/2026.parquet`; seasons from 2027 on are fetched from Baseball Savant day by day (`fetch_savant.py`, re-fetching the last 3 days) and cached as Release `data-<year>`.
- Candidate = frozen settings trained on the latest season minus its last 7 days. **Publish gate:** it replaces the served model only if the 95% upper bound (bootstrap over pitchers) of candidate − served log loss on those 7 days is ≤ 0. Parity gates (compact evaluator = LightGBM; features rebuilt from a request = training features) must pass or nothing is uploaded. Per-player state is rebuilt every run.
- Uploads `model.json.gz`, `state.json.gz`, then `meta.json` (carries both sha256) to Release `model-latest`; an adopted model is also kept as Release `model-YYYYMMDD`.
- Live state now: `model-latest` = model trained on 2026 before 2026-09-21, data through 2026-09-27, serve season 2027. The early-season path ("season too young": state refreshed, model kept) and the Savant fetch for a season ≥ 2027 have not yet run on a live schedule.
- The repo will get no pushes over the off-season; GitHub disables scheduled workflows after 60 days without a push. The owner's account runs a weekly keep-alive (in another repo) that re-enables them — if `build-model` shows as disabled, report it rather than adding a dummy commit.

### API (Vercel)
- https://mlb-pitch-forecast.vercel.app/api/predict — `GET` = model info + example; `POST` = one pitch (fields in README) → nine probabilities. Vercel Hobby (non-commercial), Root Directory `serve`, preset Other, deploys automatically on every push to `main`. Model updates come from the Release, so they need no deploy; the function re-reads the assets after 6 h (`CACHE_SECONDS`), checks their sha256 against `meta.json`, and keeps the last good copy if a reload fails. `ARTIFACT_BASE` overrides the asset URL.
- Creating the Vercel project required the owner's 2-step verification; cloud cannot open the Vercel dashboard. Deploys need nothing from cloud beyond the merge.
- From cloud: `curl -s https://mlb-pitch-forecast.vercel.app/api/predict` — **whether cloud can reach `*.vercel.app` is unverified**; if it cannot, say the live API is unverified. Local check instead: `pip install numpy==2.1.3`, then in `serve/api` run `python -c "import predict, json; print(predict.predict(predict.EXAMPLE))"` (downloads the three assets from `github.com`; not yet run in cloud).

## Data access
- Kaggle `yasunorim/mlb-statcast-bat-tracking-2024-2025` (`statcast_2024/2025/2026.parquet`, 115 columns; the pipeline reads a whitelist `pfc.COLS`). The cloud environment 「クラウド許可1」 has a network secret for `www.kaggle.com` (authenticated `GET https://www.kaggle.com/api/v1/...` works); the `kaggle` CLI goes to `api.kaggle.com`, which may not get the secret — unverified. Full-data `dev.py` / `build_serving.py` runs are for GitHub Actions (secrets `KAGGLE_USERNAME` / `KAGGLE_KEY`); do not paste or create keys.
- Baseball Savant (`baseballsavant.mlb.com`, keyless) for seasons after 2026, via `fetch_savant.py`. Savant revises a few pitch-type labels after the fact (52 of 26,303 in the last week of 2026); the fetch check tolerates up to 1% of such revisions only.
- Release assets (`model-latest`, `model-YYYYMMDD`, later `data-<year>`) are public: `github.com/yasumorishima/mlb-pitch-forecast/releases/download/<tag>/<file>`.
- Do not commit data files (`data/`, `out/` are git-ignored).

## Credits (keep in README)
Pitch data from Baseball Savant via the Kaggle dataset above. Not affiliated with MLB.
