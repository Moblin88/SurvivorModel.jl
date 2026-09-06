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
previous three seasons to form team-specific priors, measuring recency from
`current_season`. The exponential decay is calibrated with chronological
predictive validation when enough historical seasons are available. Inspect
the resulting historical weights with `recency_weights(prior)`. New drives
can be incorporated without refitting the historical prior:

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
