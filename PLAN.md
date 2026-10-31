# Next-pitch forecast — pre-registration (v3, development phase; the test section is frozen by FREEZE.json before the test runs)

v2 (before any run): code audit findings folded in: cut-frame gate with planted defects, one-sided shuffle gate, feature list, pitchouts and intentional balls left out of type counts.

v3 (during the full development run, before any test code ran): test-phase audit folded in: gates run before scoring, freeze tied to the development result and to all three file md5s, F2 tolerance stated in absolute terms, training length, cold-start definition.

## Question
Before a pitch is thrown, a dugout knows the pitcher, the batter, the count, the runners, the score, and every earlier pitch. How well does that predict **which pitch type comes next**, and does a model add anything over the tables a scout already carries: *this pitcher's usage by count and batter side*, and *by count and the previous pitch*?

## Data
Kaggle `yasunorim/mlb-statcast-bat-tracking-2024-2025` (every regular-season pitch 2024-2026, built on GitHub Actions from Baseball Savant). Regular season only (`game_type == "R"`).

Columns are read through an explicit **whitelist** (state, identities, and the type/speed/zone/description of each pitch so that *earlier* pitches can be summarised). Rows are sorted by (`game_date`, `game_pk`, `at_bat_number`, `pitch_number`); a game spanning two dates stops the run. Every feature is computed on the full sequence **before** any row is dropped from scoring.

Rows not scored: `pitch_type` missing (automatic balls and strikes) or in {PO (pitchout), IN (intentional ball), UN}; `description == "pitchout"`; pitches by position players. Pitchouts, intentional balls and unknown types are also left out of every count of earlier pitch types but still count toward a pitcher's pitch count in the game; automatic balls and strikes are not pitches and count toward neither. A pitcher-season is a position player if his id appears in `fielder_3`–`fielder_9` in ≥ 5 games that season or the previous one; the dropped ids are published, and Ohtani must not be among them.

## Target (9 classes)
FF · SI · FC · SL · ST · CU (CU, KC, CS, SV) · CH · FS (FS, FO) · OTHER (KN, EP, SC, FA, anything else). Savant codes as published. Every method's probabilities are clipped at 1e-6 and renormalised.

## Information set
For pitch *k* of game *g*: the pre-release state (count, outs, runners, inning, half, score difference for the pitching side, batter side, pitcher hand, pitch number in the plate appearance, pitcher's pitch count in the game, times through the order); earlier pitches of the same game; earlier dates of this season; and **the previous season only** (so every scored row has the same depth of history). Other games on the same date are excluded for every method (doubleheaders).

## Seasons
- 2024: history only, never trained on or scored.
- **Development:** train 2025 through June (including the March openers) → validate 2025 from July 1 (history = 2024 + season to date). All choices are made here.
- **Test (read once):** train all of 2025 (history 2024) → score 2026 (history 2025).
- Disclosure: earlier injury studies computed per-pitcher pitch-type shares and primary pitch types on 2026. No pitch-type *forecast* was ever made or scored on 2026.

## Floors (same information set as the model, as of pitch *k*)
Counts use every pitch strictly before *k* (earlier dates and earlier pitches of this game), weighted by a recency half-life chosen on validation. The decay clock is the entity's own counted pitches in time order (a same-date other game advances the clock, but its pitch types are never seen).
- **F0** league class frequencies from the training rows.
- **F1** the pitcher's own mix, shrunk toward the league mix of his role (starter / reliever).
- **F2** pitcher × batter side × count (12 counts), shrunk toward F1.
- **F3** pitcher × count × previous pitch type in the plate appearance ("first pitch" as its own level), shrunk toward F2.
Pseudo-counts (grid 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000) and the half-life (grid 25, 50, 100, 250, 500, 1000, 2000, 4000 pitches, or none; widened twice because the 40-day code check and then the first full development run picked edges for F1 and F3 — a floor tuned at a grid edge would flatter the model) are chosen on validation, separately for each floor, to minimise that floor's own log loss (the floors get the same tuning effort as the model).

## Model
**M:** LightGBM multiclass on the state features, F1–F3 computed as of *k* (never from the full season), the pitcher's mix earlier in this game, and the batter's decayed mix of pitches seen (half-life 1,000 pitches, add-one smoothing). Hyper-parameters and the number of boosting rounds are chosen on validation only (early stopping on the July-onward rows while training on rows through June); the frozen configuration is retrained once on all of 2025 with that same number of rounds.

## Hypotheses (test season 2026)
- **H1:** mean log loss per pitch, M − F2 ≤ −0.010 nats.
- **H2:** mean log loss per pitch, M − F3 ≤ −0.010 nats.
Inference: bootstrap over **pitchers** (2,000 draws) of the pitch-weighted mean difference; p = share of draws at or above −0.010. Holm over the two (α = 0.05): a hypothesis is SUPPORTED when its p is at or below its Holm level, which is the same as the upper end of the one-sided interval at that level being below −0.010 (both bounds are reported). Sensitivity, no verdict: two-way pitcher × game bootstrap.

## Published with every run
The pitcher-seasons dropped as position players (only ids that actually threw a counted pitch).

## Reported, no verdict
Accuracy and top-2 accuracy; Brier; per-class calibration; fastball (FF/SI/FC) vs other AUC (for prior work); the gain by count, by times through the order, for pitchers with < 300 prior pitches (undecayed count of his earlier counted pitches, previous season plus season to date); the gain per pitcher (pitcher-weighted mean and the share of pitchers where M beats F3).

## Gates (a failed gate stops the run; nothing is adjusted to pass it)
- **Sequence:** each pitch is unique on (`game_pk`, `at_bat_number`, `pitch_number`); `pitch_number` runs 1..n without gaps in each plate appearance; each pitch's balls/strikes follow from the previous pitch's description, with at most 5 exceptions per season, all printed (measured before this gate was written: 1 of 529,030 transitions in 2024, which is history only).
- **Cut frame:** for 150 random validation pitches plus 100 pitches in a doubleheader's second game (pitcher's or batter's), the whole feature row of *k* is rebuilt from a frame holding only rows up to *k*, with *k*'s own type, speed, zone and result scrambled and every other game of the same date scrambled; it must equal the full-data row exactly.
- **The cut frame can fail:** the same check run with a planted defect (counting the current pitch; letting the same-date game through) must check at least one pitch and report mismatches.
- **Independent counts:** F1–F3 cell counts and the batter's counts for 1,500 random validation pitches plus up to 200 doubleheader second-game pitches (pitcher's, batter's), recomputed as plain weighted sums, match to 1e-5 (relative); the same check on the planted same-date defect must exceed 1e-5.
- **Planted leak:** adding the current pitch's speed must lower validation loss by more than 0.2 nats (the leak checks can see a leak).
- **Shuffle:** with `pitch_type` shuffled within pitcher-season and M refitted (early-stopped on the last 20% of training dates), M must not beat F1 by more than 0.02 nats on validation (one-sided: the in-game mix legitimately carries this season's mix).
- **Independent F2:** for 5,000 random 2026 pitches plus up to 200 doubleheader second-game pitches, the F1 → F2 chain recomputed by a second implementation must match within 1e-6 (absolute, per class; counts are stored as float32, whose relative error measured 5e-8 in the code check); the same check against the planted same-date defect must exceed 1e-6.
- **Order:** every gate runs before any 2026 pitch is scored; a failed gate writes the gate results only and exits.
- **Coverage:** the 2026 file runs to the last day of the regular season.

## Freeze
The `freeze` workflow writes `FREEZE.json` from the full development result on the `results` branch: the code commit, the development commit and result file, the floors and model settings, the validation numbers, the md5 of all three season files and the Kaggle dataset version (2026 is hashed, not read). The `test` job refuses to run unless the code (src, requirements, workflows, PLAN.md) is unchanged since the frozen commit, the modelling code is unchanged since the development run, the settings equal the development result, and all three md5s match; it also refuses if a test result already exists. FREEZE.json must have been written by exactly one commit, sitting directly on the frozen code and changing nothing else. The freeze also checks that the 2024 and 2025 files have the md5s logged by the development run. The test runs straight after the freeze (the Kaggle CLI cannot pin a dataset version, so the md5 is the pin; if the file changes in between, the test refuses and the change is recorded here). A run stopped by a gate is saved as `testgate_*` and does not count as the test; a run that reached scoring is saved as `test_*` (written before the first prediction, so even a crash counts as the one run).

## Delivery and MLOps (after the test, whatever the verdict)
- The frozen model is served as a free JSON API (Vercel Hobby, Python function): state in, nine probabilities out.
- In-season retraining on GitHub Actions: fetch the new week from Savant, retrain, and publish only if the 95% upper bound of (new − current) log loss on the most recent unseen week is ≤ 0.

## Development log
- 2026-10-31, run 41190371915 (commit afb6cbf, full development data): all development gates passed; validation log loss M 1.1846, F3 1.2429, F2 1.2435, F1 1.3287. F1 (h 100, a 0.3) and F3 (h 100) sat on grid edges, so the grids were widened and the development run repeated; that run's result (not this one) is the one frozen. The result file of this run was not saved to the `results` branch (a fetch refspec bug in the save step); its numbers are in the run log.
