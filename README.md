# Next-pitch forecast (MLB)

Before a pitch is thrown, how well can the next pitch type be predicted, and does a model add anything over the tables a scout already carries (a pitcher's usage by count and batter side, and by count and previous pitch)?

The plan is pre-registered in [PLAN.md](PLAN.md). Development uses 2024 (history only) and 2025; the 2026 season is scored once, after the plan and code are frozen in `FREEZE.md`. After the test, the model is served as a free JSON API and retrained in season on GitHub Actions.

Status: development.

Data: every regular-season pitch from Baseball Savant, via the Kaggle dataset [MLB Statcast + Bat Tracking (2024-2026)](https://www.kaggle.com/datasets/yasunorim/mlb-statcast-bat-tracking-2024-2025). Not affiliated with MLB.
