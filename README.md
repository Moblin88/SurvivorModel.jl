# SurvivorModel

[![Build Status](https://github.com/Moblin88/SurvivorModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Moblin88/SurvivorModel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Moblin88/SurvivorModel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Moblin88/SurvivorModel.jl)

SurvivorModel models each drive as a piecewise-constant race between two
independent outcomes:

- an offensive touchdown (`:td`);
- a defensive event (`:defensive`), covering every non-touchdown drive-ending
  result.

`End of half` drives are censored observations. Hazards depend on elapsed drive
time, not field position. The default time bins are 0-2, 2-4, 4-6, and 6+
minutes, and callers can supply explicit time edges.

## Empirical-Bayes workflow

Fit a season-opening prior from historical drives, then update it with current
season data:

```julia
using SurvivorModel

historical = load_drive_pbp(2021:2023)
current = load_drive_pbp(2024)

prior = fit_empirical_bayes_prior(historical; current_season=2024)
model = fit_hazard_model(current; prior=prior)

td_rate = hazard_rate(model, :td, "KC", 2)
defensive_rate = hazard_rate(model, :defensive, "SF", 2)

# Home-adjusted rates use the fitted global multipliers.
home_td_rate = hazard_rate(model, :td, "KC", 2; home=true)
home_defensive_rate = hazard_rate(model, :defensive, "SF", 2; home=true)
```

`fit_empirical_bayes_prior` estimates separate Gamma-Poisson hyperparameters
for each outcome and elapsed-time bin, plus one global offensive touchdown
multiplier and one global defensive-event multiplier. The multipliers are
positive, may be above or below 1.0, and are fitted from historical data
alongside the league hyperparameters. Inspect them with
`home_multiplier(prior, :td)` and `home_multiplier(prior, :defensive)`.
They remain fixed when current-season data are added. The prior uses the
most recent three seasons to form team-specific priors, measuring recency from
`current_season`. When no half-life is supplied, the model estimates
year-to-year persistence from correlations between shrunk team-season hazard
moments, using at most those three seasons. Inspect the resulting historical
weights with `recency_weights(prior)`. New drives can be incorporated without
refitting the historical prior. The effective historical window is capped at
three seasons even when a larger `max_seasons` value is supplied. The legacy
`half_life_candidates` keyword is still accepted, but automatic recency
estimation no longer searches those candidate values.

```julia
update_hazard_model!(model, newly_available_drives)
posterior = hazard_posterior(model, :td, "KC", 2)
```

## Regular-season forecasts

Use the schedule-backed forecast API to produce a frozen pre-week forecast for
every regular-season game from a requested week through week 18:

```julia
context = fit_regular_season_forecast(2024; as_of_week=10)
forecast = forecast_win_probabilities(context)

forecast[:, [
    :game_id,
    :week,
    :away_team,
    :home_team,
    :away_win_probability,
    :home_win_probability,
]]
```

The model uses regular-season drives from prior seasons for the
empirical-Bayes prior and target-season drives through week 9 for the
week-10 snapshot. Later target-season results are not used, so the same call
works for historical evaluation and for a currently unfolding season. The
probability-only output does not evaluate spread or predictive-variance
metrics. Reuse the same context for other views:

```julia
spreads = forecast_spreads(context)
results = regular_season_results(2024; from_week=10)
```

When generating several weekly snapshots for the same target season, fit the
historical prior once and pass it to each call as `prior=...`. This avoids
repeating the empirical-Bayes hyperparameter and recency fits while retaining
the week-specific current-season update:

```julia
first_context = fit_regular_season_forecast(2024; as_of_week=10)
prior = first_context.model.prior
next_context = fit_regular_season_forecast(
    2024;
    as_of_week=11,
    prior=prior,
)
```

`forecast_regular_season(2024; as_of_week=10)` remains the convenience
function that returns schedule scores/results, home and away probabilities,
`expected_spread`, `predictive_spread_variance`, and `game_completed` in one
table. Pass `include_completed=false` to forecast only games without a
recorded result. `regular_season_results` reads the schedule only and does not
load PBP or fit a model. `load_schedule()` can be used to fetch or normalize
the underlying schedule separately. At the matchup level,
`expected_game_win_probability` and `expected_game_spread_metrics` provide the
same split without constructing the full `ExpectedGameMetrics` result.

The posterior rate returned by `hazard_rate` is the Gamma posterior mean.
Pass `posteam_home=true` to `drive_moments` when the possessing team is at
home; the defensive team is assigned the complementary away/home status.
`fit_score_marks`, `drive_moments`, and `game_spread_distribution` provide the
downstream score and game-level approximations.

To propagate conditional hazard-posterior uncertainty through those nonlinear
game calculations, use the log-hazard theta summary and the
second-order metric approximation:

```julia
marks = fit_score_marks(current)
theta = hazard_theta(model, "KC", "SF")
metrics = expected_game_metrics(model, marks, "KC", "SF")

metrics.expected_spread
metrics.expected_win_probability
metrics.predictive_spread_variance
```

`hazard_theta` orders the log hazards as home offense, away defense, away
offense, and home defense, with one block per elapsed-time bin. The expected
spread and win probability use a second-order delta-method correction based on
the Gamma posterior moments. The predictive spread variance also includes
between-hazard-posterior variation in the conditional spread mean. This
calculation treats the empirical-Bayes hyperparameters, fitted home
multipliers, and `ScoreMarks` as fixed.

## Historical calibration

Evaluate fixed pre-week snapshots against completed schedule results with
`evaluate_calibration`. If no seasons are supplied, it selects the most recent
completed regular seasons in the schedule:

```julia
report = evaluate_calibration(;
    cutoff_weeks=(1, 5, 10, 15),
    recent_seasons=3,
)

report.summary
report.reliability
report.games
report.spread_summary
report.spread_reliability
report.spread_coverage
report.spread_games
```

Each snapshot is fit using only drives before its cutoff week and scores games
from that week through the end of the season. `report.summary` contains Brier
score, log loss, mean predicted probability, observed home-win rate, and their
calibration gap. Ties are represented as `0.5` in all of these calculations;
games without a schedule result are excluded from scoring. Reliability rows use
fixed probability bins and include the number of games, observed rate, and
calibration gap for each bin.

The spread tables evaluate the expected home-team score difference against the
schedule's actual home margin. `report.spread_summary` contains signed error,
mean absolute error, RMSE, and the rates at which the actual margin is above,
below, or equal to the model forecast. `report.spread_reliability` groups
games by predicted margin and compares mean predicted and actual margins.
`report.spread_coverage` checks central 50%, 80%, 90%, and 95% Normal
predictive intervals built from `expected_spread` and
`predictive_spread_variance`. These are model-line diagnostics, not
sportsbook ATS or profitability results.

The deterministic calibration tests run with the normal test suite. To produce
the opt-in report from live NFLData schedules and play-by-play:

```sh
SURVIVORMODEL_RUN_CALIBRATION=true julia --project=. test/calibration_live.jl
```

The live report is diagnostic and does not impose an arbitrary model-quality
threshold on CI.

## Survivor-pool planning

Use a fitted regular-season context to create one forward survivor pick per
week while preventing team reuse:

```julia
context = fit_regular_season_forecast(2025; as_of_week=1)

plan = optimize_survivor_pool(
    context;
    picks_made=Dict{Int,String}(),
    strikes_remaining=1,
    weekly_survival_probability=0.65,
)

plan.current_pick
plan.selections
plan.objective_value
```

The optimizer expands each unplayed forecast game into a home-team and
away-team candidate, excludes teams in `picks_made`, and solves one binary
assignment model with JuMP and HiGHS. It selects exactly one team for every
week in the requested horizon and allows each team to be selected at most
once. `plan.selections` includes each selected team's win probability, reach
discount, and discounted objective contribution; `plan.current_pick` is the
row to use for the current week.

The objective is expected future wins weighted by your personal probability of
still being alive before each week. It does not estimate the probability that
the entire pool survives and does not use sportsbook lines. With fixed weekly
survival probability `q`, no remaining strikes uses `d[k] = q^k`, where `k`
is the number of prior planned weeks. With `s` remaining strikes, the
discount is the probability of having at most `s` losses in those prior weeks:
`d[k] = sum(binomial(k, losses) * (1-q)^losses * q^(k-losses))` for
`losses = 0:min(s, k)`. For example, with `q = 0.65` and one unused strike,
the first discounts are `1.0, 1.0, 0.8775, 0.71825`.

After each week, refresh the forecast context with the new `as_of_week`,
record the team picked in `picks_made`, update `strikes_remaining`, and call
`optimize_survivor_pool` again. The optimizer itself performs one forecast
pass and one solve; it does not iteratively recompute discounts from the
selected teams' probabilities. For deterministic evaluation or custom
forecasts, call `build_survivor_candidates` and pass its result to
`optimize_survivor_pool(candidates, state)`.

To replay the strategy against completed historical seasons, run the opt-in
survivor harness:

```sh
SURVIVORMODEL_RUN_SURVIVOR=true julia --project=. test/survivor_live.jl
```

By default it evaluates the three most recent completed regular seasons with
one and two initial strikes and `weekly_survival_probability = 0.65`. It
refits the forecast context before each week using only drives before that
week, reuses the same weekly forecast for both strike scenarios, applies
historical wins and losses (ties count as losses), and prints a summary with
the last week survived, elimination week/reason, wins, losses, and picks.
Override the inputs with environment variables such as:

```sh
SURVIVORMODEL_RUN_SURVIVOR=true \
SURVIVORMODEL_SURVIVOR_SEASONS=2021,2022,2023,2024 \
SURVIVORMODEL_SURVIVOR_WEEKLY_SURVIVAL=0.65 \
julia --project=. test/survivor_live.jl
```

Use `SURVIVORMODEL_SURVIVOR_RECENT_SEASONS` when selecting the latest completed
seasons, and `SURVIVORMODEL_SURVIVOR_MAX_SEASONS` to change the historical
training window. The harness uses automatic moment-based recency calibration
by default; set `SURVIVORMODEL_SURVIVOR_RECENCY_HALF_LIFE` to a positive,
finite value to override it for a fixed-half-life replay. The harness is
opt-in because it downloads live schedules and play-by-play data and reruns
one pre-week forecast per regular-season week (17 weeks for seasons through
2020 and 18 weeks from 2021 onward).
