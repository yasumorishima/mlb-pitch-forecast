# Results: the 2026 test (run once)

Frozen on 2026-10-31 (`FREEZE.json`, commit 9d2497a; development run 41198675420), scored once in run 41201400282. Raw output: [`results/test_20261031T123817Z_9d2497a….json`](https://github.com/yasumorishima/mlb-pitch-forecast/blob/results/results/test_20261031T123817Z_9d2497a33ef9a662c7479ab87c55970f9c8a20f1.json) on the `results` branch.

## Verdicts

712,672 scored pitches of the 2026 regular season (832 pitchers).

| Method | Mean log loss per pitch |
|---|---|
| **M** (LightGBM on state + floors) | **1.2309** |
| F3 pitcher × count × previous pitch | 1.2942 |
| F2 pitcher × batter side × count | 1.2955 |
| F1 pitcher's own mix | 1.3784 |
| F0 league mix | 1.9152 |

| Hypothesis | Mean difference | Upper bound at its Holm level | Verdict |
|---|---|---|---|
| H1: M − F2 ≤ −0.010 | −0.0647 | −0.0621 (97.5%) | **SUPPORTED** |
| H2: M − F3 ≤ −0.010 | −0.0633 | −0.0613 (95%) | **SUPPORTED** |

Bootstrap over pitchers, 2,000 draws; no draw reached −0.010. The two-way pitcher × game bootstrap gives upper bounds −0.0621 and −0.0607.

## Gates (all passed, run before any pitch was scored)
- Coverage: the 2026 file runs to the last day of the regular season.
- Independent F2 (5,000 random pitches plus 200 doubleheader second-game pitches): largest difference 9.7e-8; the planted same-date defect gives 0.034.

## Reported, no verdict
- Accuracy 46.0%, top-2 accuracy 73.7%, Brier 0.648. Mean predicted share vs observed share differs by at most 0.43 points for any class (FF predicted 30.3%, observed 30.8%).
- Fastball (FF/SI/FC) vs other, AUC: M 0.707, F3 0.671.
- The gain over F3 is largest at 3-0 (−0.33) and 3-1 (−0.11), and between −0.05 and −0.08 at every other count. A likely reason: a pitcher's own 3-0 cell holds few pitches, while the model learns the league-wide pull toward fastballs at 3-0. This is an interpretation, not a tested claim.
- By times through the order: −0.070, −0.053, −0.049, −0.044 (1st, 2nd, 3rd, 4th or later).
- Pitchers with fewer than 300 earlier pitches (62,373 pitches): gain −0.101.
- Per pitcher: M beats F3 for 97.5% of 832 pitchers; pitcher-weighted mean gain −0.080.

## What this does and does not say
The model adds about 0.06 nats per pitch over a scout's usage tables, in a season it never saw. It still picks the right pitch type only 46% of the time: the next pitch is far from predictable, and the value is in better-calibrated probabilities rather than a single call.
