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

`fit_empirical_bayes_prior` estimates separate stationary Gamma parameters for
each outcome and elapsed-time bin. It also estimates one global home
multiplier and one season-to-season persistence probability for each outcome.
The persistence probability is shared across the bins of that outcome's hazard
curve.

Historical hyperparameters are fit with the event-process marginal likelihood,
not with exposure-normalized factorial moments. For each observed risk
interval, the likelihood retains the competing-risk contribution
`λᴛ^Nᴛ λᴅ^Nᴅ exp[-(λᴛ + λᴅ)E]`. Home exposure is scaled by the fitted
outcome-specific multiplier, and home events contribute the corresponding
multiplier factor. The latent team/bin rate is integrated through the
season-to-season reset transition, so a historical fit uses the full finite
Gamma-mixture model rather than treating seasons as independent Gamma draws.

For a selected history of `N` seasons, the likelihood sums over every
contiguous reset/persistence partition of those seasons. The implementation
evaluates that sum with an exact forward dynamic program rather than
enumerating the `2^(N-1)` paths. The three-season case is equivalent to the
four familiar paths: all seasons redraw, either adjacent pair persists, or all
three seasons persist. Each segment is evaluated from aggregated counts and
effective exposures, so the historical fit does not revisit individual drive
rows during optimization.

The historical fit uses an EM/ECME decomposition rather than one simultaneous
high-dimensional search. The E-step uses a forward/backward segment filter to
compute posterior group probabilities, expected persistent links, and
posterior moments of the shared Gamma rates. The conditional Gamma updates
then solve each time bin independently given the shared `rho` and home
multiplier, while the shared parameters are updated in a two-dimensional
conditional likelihood step. The exact filter uses quadratic work and linear
state per team/bin cell in the number of supplied seasons, so `max_seasons`
can request longer histories without a package-imposed three-season cap.
Public fitting and forecasting APIs use a five-season historical window by
default; pass `max_seasons` explicitly to choose a different trailing window.

Production fitting uses `EMLBFGSFit()` by default. The other solver
implementations remain available only to the developer benchmark and recovery
tools; they are not part of the weekly application API. Use the default unless
you are explicitly comparing solver behavior on synthetic or historical data.

The Gamma marginal derivative path caches the special-function values at each
bin's base shape and uses the exact integer-count recurrences for
`loggamma`, digamma, and trigamma for small aggregated counts. Zero-count
groups therefore avoid a second special-function evaluation, while larger
counts fall back to direct `SpecialFunctions` calls. No approximate special
function backend is used by default, so the optimizer retains the exact
likelihood and curvature semantics.

The likelihood is evaluated separately for touchdowns and defensive events
because it factorizes conditional on the observed risk intervals. This still
accounts for competing-process exposure: short defensive risk windows and
longer offensive risk windows enter the joint likelihood through their
observed integrated hazards. The fit stops with an error if the historical
data contain no usable risk intervals or if the EM/conditional optimization
fails; it does not silently substitute the weak default prior. Inspect
`likelihood_fit_diagnostics(prior, :td)` or
`likelihood_fit_diagnostics(prior, :defensive)` for the maximized likelihood,
fit status, iteration counts, conditional objective evaluations, and boundary
flags.

The team-specific season-opening prior is a finite Gamma mixture produced by a
probabilistic reset filter. Each component represents a possible last reset
season; with `N` historical seasons, the target-season prior contains at most
`N + 1` components. The exact mixture is available from
`hazard_posterior`, while `hazard_rate` returns its posterior mean. Inspect the
home multipliers with `home_multiplier(prior, :td)` and
`home_multiplier(prior, :defensive)`, and inspect persistence with
`hazard_persistence(prior, :td)` or `hazard_persistence(prior, :defensive)`.
The league parameters and home multipliers remain fixed when current-season
drives are added.

```julia
update_hazard_model!(model, newly_available_drives)
posterior = hazard_posterior(model, :td, "KC", 2)
```

`posterior.weights`, `posterior.components`, and `posterior.source_seasons`
describe the exact finite Gamma mixture. New drives update every component
with the same Gamma-Poisson conjugate rule and reweight the components by their
predictive likelihood.

Renewal simulation, synthetic recovery, and solver-comparison workflows are
developer/research tools. They remain available through the dedicated
benchmark tooling but are intentionally omitted from the weekly application
workflow.

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
metrics and is the canonical weekly forecast surface:

```julia
results = regular_season_results(2024; from_week=10)
```

When generating several weekly snapshots for the same target season, fit the
historical prior once and pass it to each call as `prior=...`. This avoids
repeating the empirical-Bayes moment fit while retaining
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

Pass `include_completed=false` to forecast only games without a recorded
result. `regular_season_results` reads the schedule only and does not load PBP
or fit a model. `load_schedule()` can be used to fetch or normalize the
underlying schedule separately. At the matchup level,
`expected_game_win_probability` provides a direct probability calculation.

The posterior rate returned by `hazard_rate` is the finite-mixture posterior
mean. Spread forecasting, score-mark calculations, renewal simulation, and
predictive-interval diagnostics remain available as qualified research
interfaces but are not required for the weekly survivor run.

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
```

Each snapshot is fit using only drives before its cutoff week and scores games
from that week through the end of the season. `report.summary` contains Brier
score, log loss, mean predicted probability, observed home-win rate, and their
calibration gap. Ties are represented as `0.5` in all of these calculations;
games without a schedule result are excluded from scoring. Reliability rows use
fixed probability bins and include the number of games, observed rate, and
calibration gap for each bin.

The standard report is intentionally limited to the probability metrics used
by survivor selection. Spread error, spread reliability, and predictive
interval diagnostics remain available through the qualified research
interface:

```julia
research = SurvivorModel.evaluate_calibration_research(;
    cutoff_weeks=(1, 5, 10, 15),
    recent_seasons=3,
)

research.spread_summary
research.spread_reliability
research.spread_coverage
```

The deterministic calibration tests run with the normal test suite. To produce
the opt-in report from live NFLData schedules and play-by-play:

```sh
SURVIVORMODEL_RUN_CALIBRATION=true julia --project=. test/calibration_live.jl
```

The live report is diagnostic and does not impose an arbitrary model-quality
threshold on CI.

The default package test suite uses deterministic fixtures. To run the live
drive-data and fitted-model sanity checks against NFLData as well:

```sh
SURVIVORMODEL_RUN_LIVE_SANITY=true julia --project=. -e 'using Pkg; Pkg.test()'
```

## Survivor-pool planning

Use a fitted regular-season context to create one forward survivor pick per
week while preventing team reuse:

```julia
context = fit_regular_season_forecast(2025; as_of_week=1)

plan = optimize_survivor_pool(
    context;
    picks_made=Dict{Int,String}(),
    strikes_remaining=1,
    selection_config=SurvivorSelectionConfig(
        weekly_survival_probability=0.65,
        through_week=18,
    ),
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
discount, selected-team market spread, and discounted objective contribution;
`plan.current_pick` is the row to use for the current week. The effective
`plan.selection_config` records the objective, reach-discount policy, market
guard, missing-line policy, and horizon used by the MILP.

### Survivor command-line app

Julia 1.12 can run the package directly through its `@main` entry point:

```sh
julia --project=. -m SurvivorModel --season 2026 <<'EOF'
KC
SF
EOF
```

The package also declares a named `survivor` app in `Project.toml`. Install the
local checkout into Julia's app environment with:

```sh
julia -e 'using Pkg; Pkg.Apps.develop(path="/path/to/SurvivorModel")'
```

Then run it from any directory:

```sh
survivor --season 2026 < picks.txt
```

The app reads one team abbreviation per nonblank line, starting with week 1.
It infers the next week from the number of picks, loads the season schedule to
count completed losses (ties count as losses), and defaults to two initial
strikes. Use `--strikes N` to choose a different initial loss allowance. The
effective week is one plus the number of supplied picks; games at or after
that week are treated as future games even when the schedule already contains
their results, which allows replaying an earlier week of a completed season.
The default output is only the selected team's abbreviation and a newline.
For an opt-in phase breakdown, set `SURVIVORMODEL_TIMINGS=true`; timing
diagnostics are written to stderr so stdout remains suitable for a picks file:

```sh
SURVIVORMODEL_TIMINGS=true survivor --season 2026 < picks.txt
```

Use `--refresh-data` (or set `SURVIVORMODEL_REFRESH_DATA=true`) for an explicit
data refresh. This clears NFLData's raw-data cache and rebuilds the package's
summarized historical-drive cache before running; normal invocations reuse
summarized historical seasons while still loading the current season through
the normal NFLData path:

```sh
survivor --refresh-data --season 2026 < picks.txt
```

Historical empirical-Bayes priors are stored in the package's Scratch.jl
space and keyed by season, historical-window length, time-bin configuration,
fit method, and a fingerprint of the historical drive data used for the fit.
If that data changes, the cached prior is recomputed. current-season drives and the survivor optimization are refreshed on each
invocation. An opening-week forecast can run before NFLData publishes
target-season PBP and uses historical drives only. Once prior picks imply week
2 or later, target-season PBP must be available so the current-season update is
not omitted. Cache clearing is a maintenance operation; normal weekly runs
should use `--refresh-data` instead.

The default objective is expected future wins weighted by the probability of
still being alive before each week. It does not estimate the exact probability
that the entire pool survives. The MILP always selects one team per week and
uses each team at most once. `SurvivorSelectionConfig` can change the
objective, weekly survival probability, reach-discount policy, market guard,
missing-line policy, and planning horizon. The default market policy protects
the current and following week by requiring a selected team to be favored by at
least `2.0` points; missing lines remain eligible. Positive `market_spread`
values mean the selected team is favored.

With fixed weekly survival probability `q`, no remaining strikes uses
`d[k] = q^k`, where `k` is the number of prior planned weeks. With `s`
remaining strikes, the discount is the probability of having at most `s`
losses in those prior weeks:
`d[k] = sum(binomial(k, losses) * (1-q)^losses * q^(k-losses))` for
`losses = 0:min(s, k)`.

### Developer benchmarks

Solver comparisons, synthetic recovery, and real-data fitting benchmarks are
developer workflows rather than weekly survivor features. Run them directly
with `tools/fit_benchmark.jl`; see that tool's `--help` output for supported
scenarios and options.

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
the last week survived, elimination week/reason, wins, losses, and picks. It
also compares the MILP strategy with a greedy baseline that selects the
largest available selected-team published spread each week while avoiding
previously picked teams. The greedy baseline does not plan around future team
reuse or use model win probabilities; games without a published spread are
not eligible for its pick.
Override the inputs with environment variables such as:

```sh
SURVIVORMODEL_RUN_SURVIVOR=true \
SURVIVORMODEL_SURVIVOR_SEASONS=2021,2022,2023,2024 \
SURVIVORMODEL_SURVIVOR_WEEKLY_SURVIVAL=0.65 \
julia --project=. test/survivor_live.jl
```

Use `SURVIVORMODEL_SURVIVOR_RECENT_SEASONS` when selecting the latest completed
seasons, and `SURVIVORMODEL_SURVIVOR_MAX_SEASONS` to change the historical
training window. The harness is opt-in because it downloads live
schedules and play-by-play data and reruns one pre-week forecast per
regular-season week (17 weeks for seasons through 2020 and 18 weeks from
2021 onward).
