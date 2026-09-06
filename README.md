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

The posterior rate returned by `hazard_rate` is the Gamma posterior mean.
Pass `posteam_home=true` to `drive_moments` when the possessing team is at
home; the defensive team is assigned the complementary away/home status.
`fit_score_marks`, `drive_moments`, and `game_spread_distribution` provide the
downstream score and game-level approximations.
