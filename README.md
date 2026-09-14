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

The fitting method is selected with a typed value when comparing optimization
strategies:

```julia
prior = fit_empirical_bayes_prior(
    historical;
    current_season=2024,
    method=DirectLBFGSFit(),
)
```

The default `method=EMLBFGSFit()` uses the EM decomposition with analytic
likelihood-gradient polishing. `MomentFit()` stops after the pooled moment fit and
returns those estimates without optimizing the exact likelihood; its
diagnostics therefore use `status=:moment`, and its likelihood is only the
exact likelihood evaluated at the moment estimates. The moment fit estimates
the bin means and Gamma variances, the shared home multiplier, and persistence
by repeatedly recomputing covariance-dependent weights. In particular, the
team-cluster weights for mean rates and the pair-product weights depend on the
current persistence estimate, so the procedure updates its weights and
parameters until the damped fixed-point iteration converges. Use
`MomentFit()` for a fast standalone estimator or `MomentLBFGSFit()` to use the
same estimates as an initial point for exact likelihood optimization.

`EMECMEFit()` retains the bounded derivative-free conditional baseline, while
`EMLBFGSFit()` keeps the EM decomposition but uses the analytic likelihood
gradient for the conditional steps. `DirectLBFGSFit()` and `DirectBFGSFit()`
optimize all bin and shared parameters jointly with the exact gradient.
`BlockNewtonFit()` starts from the iterated moment estimate, alternates damped 2×2
Newton steps for the shared `(logit(rho), log(h))` block and each per-bin
`(log(mean), log(shape))` block, then applies an exact Schur-complement
correction. Its Cholesky solves avoid explicit inverses, and the Schur
complement is assembled with small low-rank Cholesky downdates.
`SchurNewtonFit()` starts from the same moment estimate and uses the analytic observed-likelihood
Hessian, a block Schur-complement Newton step, bound-aware backtracking, and an
explicit L-BFGS fallback. Both Newton variants are available for benchmarking
while `EMLBFGSFit()` remains the default. All methods use the same event-process
likelihood, bounds, reset mixture, and explicit failure behavior. The
standalone comparison harness is
`benchmark/reset_solver_comparison.jl`.

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

## Abstract schedule simulation

`simulate_renewal_schedule` generates drive-level data from the fitted
two-outcome renewal process over a hypothetical schedule:

```julia
using Random

model = fit_hazard_model(current; prior=prior)
simulation = simulate_renewal_schedule(
    model,
    hypothetical_schedule;
    rng=MersenneTwister(42),
)

simulation.games.drive_count
simulation.drives
simulation.latent_rates
```

The schedule uses the same columns as `load_schedule`: `game_id`, `season`,
`game_type`, `week`, `away_team`, `home_team`, and `result`. The result may be
`missing` for an unplayed hypothetical schedule. Games are simulated in
season/week order, with a random 50/50 opening possession and alternating
possessions thereafter. Each drive samples a piecewise-exponential waiting
time from the touchdown and defensive hazards until the 3,600-second game
horizon is exhausted. Durations are returned as integer `Second` values, and
the final censored interval is labeled `End of half`.

The first simulated season samples one latent Gamma-mixture rate for every
team, outcome, and time bin from the model's posterior and reuses those rates
across that season's games. For each later season, every rate independently
persists or redraws according to the fitted outcome-specific persistence and
Gamma hyperparameters. This preserves schedule-level dependence from shared
rate uncertainty. The returned `drives` table can be passed to
`build_exposure_data` or `fit_hazard_model`; `games` reports endogenous drive
counts, and `latent_rates` exposes the realized baseline rates.

This simulator intentionally excludes score rewards, score marks, win
probabilities, and detailed football possession rules. It is intended for
renewal-process experiments and prior-fitting simulations.

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

`forecast_regular_season(2024; as_of_week=10)` remains the convenience
function that returns schedule scores/results, home and away probabilities,
`expected_spread`, `predictive_spread_variance`, and `game_completed` in one
table. Pass `include_completed=false` to forecast only games without a
recorded result. `regular_season_results` reads the schedule only and does not
load PBP or fit a model. `load_schedule()` can be used to fetch or normalize
the underlying schedule separately. At the matchup level,
`expected_game_win_probability` and `expected_game_spread_metrics` provide the
same split without constructing the full `ExpectedGameMetrics` result.

The posterior rate returned by `hazard_rate` is the finite-mixture posterior
mean.
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
the exact finite-mixture log-rate moments. The predictive spread variance also includes
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
discount, selected-team market spread, and discounted objective contribution;
`plan.current_pick` is the row to use for the current week.

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

Set `SURVIVORMODEL_REFRESH_DATA=true` for an explicit data refresh. This
clears NFLData's raw-data cache and rebuilds the package's summarized
historical-drive cache before running; normal invocations reuse summarized
historical seasons while still loading the current season through the normal
NFLData path.

If Julia startup and package compilation dominate the weekly run, an optional
PackageCompiler sysimage can be built outside the project environment:

```sh
julia -e 'using Pkg; Pkg.add("PackageCompiler")'
julia tools/build_sysimage.jl
julia --project=. --sysimage="$HOME/.cache/SurvivorModel/survivor.dylib" \
  -m SurvivorModel --season 2026 < picks.txt
```

Set `SURVIVORMODEL_SYSIMAGE_PATH` to choose a different output path. The
ordinary `survivor` command remains the portable fallback.

Historical empirical-Bayes priors are stored in the package's Scratch.jl
space and keyed by season, historical-window length, time-bin configuration,
and fit method. Current-season drives and the survivor optimization are
refreshed on each invocation. An opening-week forecast can run before
NFLData publishes target-season PBP and uses historical drives only. Once
prior picks imply week 2 or later, target-season PBP must be available so the
current-season update is not omitted. To remove the cached historical fits:

```julia
using SurvivorModel
clear_historical_prior_cache!()
```

The same cache can be cleared from the command line. This exits after clearing
when used by itself, or clears the cache before the requested forecast or
benchmark:

```sh
survivor --clear-cache
survivor --clear-cache --season 2026 < picks.txt
```

The objective is expected future wins weighted by your personal probability of
still being alive before each week. It does not estimate the probability that
the entire pool survives and does not use sportsbook lines in its objective.
As a near-term safety guard, the optimizer disallows candidates in the current
week and the following week unless the selected team is favored by at least
`2.0` points according to `spread_line`. Positive `market_spread` values mean
the selected team is favored. Missing lines remain eligible, and weeks after
the two-week protected window are not filtered by this guard. With fixed
weekly survival probability `q`, no remaining strikes uses `d[k] = q^k`, where
`k` is the number of prior planned weeks. With `s` remaining strikes, the
discount is the probability of having at most `s` losses in those prior weeks:
`d[k] = sum(binomial(k, losses) * (1-q)^losses * q^(k-losses))` for
`losses = 0:min(s, k)`. For example, with `q = 0.65` and one unused strike,
the first discounts are `1.0, 1.0, 0.8775, 0.71825`.

### Fitting benchmarks

Run the fitting-method benchmark from the same app. The default uses a
deterministic synthetic drive data set, warms each method once, times the
requested repeats, and prints a comparison table with median runtime,
convergence status, total log likelihood, the gap from the best likelihood,
iteration counts, function evaluations, and boundary-parameter counts:

```sh
survivor --benchmark
survivor --benchmark --repeats 5
```

The performance benchmark can be replaced with a correctly specified
parameter-recovery simulation:

```sh
survivor --benchmark --scenario recovery --repeats 3
survivor --benchmark --scenario recovery \
  --recovery-seasons 5 --max-seasons 5 --repeats 3
```

The recovery scenario simulates a configurable number of seasons (three by
default) for 32 teams from the same piecewise competing-risk model used by
fitting. It creates a
17-game-per-team hypothetical schedule, then uses the abstract renewal
simulator to generate drive durations and outcomes endogenously. One Gamma
rate per outcome and time bin is reused across each season's scheduled games;
the rate is carried across seasons with the configured outcome-specific
persistence probability or redrawn, and separate offensive and defensive home
multipliers are applied. The output reports the generating truth,
shared-parameter errors, and per-bin mean and variance errors for every fit
method. The default event-rate means are calibrated to roughly 160-second
drives and a 20% touchdown share; the simulator intentionally does not
generate score rewards. Use `--recovery-seasons N` to change the simulated
history, and pass the same `--max-seasons N` when the fit should use the full
history. The generated truth and schedule are also available through
`synthetic_fit_recovery_drives(seasons=...)`.

Use real historical data by supplying a reference season. The benchmark loads
the preceding `--max-seasons` seasons through `load_drive_pbp` (the default is five):

```sh
survivor --benchmark --data real --season 2024
```

The output also includes the fitted hazard parameters for each method,
outcome, and time bin: the outcome-specific home hazard multiplier,
season-to-season persistence, and the Gamma hazard mean and variance.
The original synthetic performance scenario includes persistent latent team
rates for both outcomes and prints its target home multiplier and persistence.
Those are latent generation targets, so finite-sample estimates can differ;
use the likelihood gap and convergence status alongside the parameter table.
Real-data benchmarks do not have known parameter targets.

Higher total log likelihood is better when comparing methods on the same data;
`gap` is zero for the best successful fit. Synthetic and real benchmarks use
the same event-process likelihood and typed fitting methods as the package API.

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
