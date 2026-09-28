# SurvivorModel

[![Build Status](https://github.com/Moblin88/SurvivorModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Moblin88/SurvivorModel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Moblin88/SurvivorModel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Moblin88/SurvivorModel.jl)

SurvivorModel models each drive as a race between two independent Weibull
event processes:

- an offensive touchdown (`:td`);
- a defensive event (`:defensive`), covering every non-touchdown drive-ending
  result.

`End of half` drives are right-censored observations. Hazards depend on
elapsed drive time, not field position. Each outcome has one Weibull shape
shared across teams, while each team has a separate cumulative-hazard rate:
`H(t) = rho * t^k` and `h(t) = rho * k * t^(k - 1)`. Drive durations and
elapsed-time arguments are measured in minutes. Since source drive times have
one-second precision, a zero-second observed event duration is treated as a
one-second duration.

## Empirical-Bayes workflow

Fit a season-opening prior from historical drives, then update it with current
season data:

```julia
using SurvivorModel

historical = load_drive_pbp(2021:2023)
current = load_drive_pbp(2024)

prior = fit_empirical_bayes_prior(historical; current_season=2024)
model = fit_hazard_model(current; prior=prior)

td_rate = hazard_rate(model, :td, "KC", 2.0)
defensive_rate = hazard_rate(model, :defensive, "SF", 2.0)

# Home-adjusted instantaneous hazards use the fitted global multipliers.
home_td_rate = hazard_rate(model, :td, "KC", 2.0; home=true)
home_defensive_rate = hazard_rate(model, :defensive, "SF", 2.0; home=true)
```

`fit_empirical_bayes_prior` fits separate Weibull shapes and Gamma
hyperparameters for touchdown and defensive-event rates. The Gamma prior is
on the cumulative-hazard coefficient `rho = scale^(-k)`, which is conjugate
under transformed exposure. One global home multiplier and one season-reset
probability are fit for each outcome.

For each cause, every drive contributes transformed risk exposure
`T^k`; home exposure is multiplied by the fitted home factor. An observed
event increments only its cause's event count, while a censored drive adds
exposure to both causes. Conditional on a Gamma component with shape `alpha`
and rate `beta`, the posterior is `Gamma(alpha + N, beta + E)`. The
historical likelihood also includes the Weibull event-time terms
`N * log(k) + (k - 1) * sum(log(T_event))` and the home-event multiplier.
Team rates are integrated out, and the seasonal reset filter carries the
resulting finite Gamma mixtures rather than treating seasons as independent
draws.

Each mixture component represents a possible last reset season. With `N`
historical seasons, a target-season prior contains at most `N + 1` components.
The mixture is available from `hazard_posterior`; `hazard_rate` returns the
posterior-mean instantaneous hazard at a requested elapsed time. Inspect
fitted shapes with `weibull_shape(prior, :td)` or
`weibull_shape(prior, :defensive)`, home factors with
`home_multiplier(prior, :td)` or `home_multiplier(prior, :defensive)`, and
reset probabilities with `hazard_persistence`. Shapes, league hyperparameters,
home multipliers, and reset probabilities remain fixed when current-season
drives are added.

The historical fit uses the most recent five seasons by default; pass
`max_seasons` to choose a different trailing window. Failed fits and histories
without observed events for either cause raise errors rather than silently
substituting a default prior. Inspect
`likelihood_fit_diagnostics(prior, :td)` or
`likelihood_fit_diagnostics(prior, :defensive)` for the maximized likelihood,
convergence, iteration and evaluation counts, and boundary parameters.

Competing-Weibull probabilities and drive-time moments are evaluated with
one-dimensional numerical integration via QuadGK. Score-spread moments are
estimated separately for each drive-ending cause, assuming score change and
drive time are independent conditional on cause. The renewal game forecast
combines those outcome/time moments; rate gradients and Hessians are computed
by integrating their derivative integrands alongside the moments.

```julia
update_hazard_model!(model, newly_available_drives)
posterior = hazard_posterior(model, :td, "KC")
```

`posterior.weights`, `posterior.components`, and `posterior.source_seasons`
describe the exact finite Gamma mixture. New drives update every component
with the same transformed-exposure conjugate rule and reweight the components
by their predictive likelihood.

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
forecast output contains schedule/results fields and home/away win
probabilities:

```julia
results = regular_season_results(2024; from_week=10)
```

When generating several weekly snapshots for the same target season, fit the
historical prior once and pass it to each call as `prior=...`. This avoids
repeating the empirical-Bayes likelihood fit while retaining
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

## Survivor-pool planning

Use a fitted regular-season context to create one forward survivor pick per
week while preventing team reuse:

```julia
context = fit_regular_season_forecast(2025; as_of_week=1)

plan = optimize_survivor_pool(
    context;
    picks_made=Dict{Int,String}(),
    strikes_remaining=2,
    selection_config=SurvivorSelectionConfig(
        through_week=18,
        timeout_seconds=600.0,
    ),
)

plan.current_pick
plan.selections
plan.objective_value
```

The default `:exact_milp` optimizer expands each unplayed forecast game into
model-favorite candidates with win probability at least `0.5`, excludes teams
in `picks_made`, and solves one binary assignment model with JuMP and HiGHS.
It selects exactly one team for every week in the requested horizon and allows
each team to be selected at most once. The covariance-aware finite-state
formulation maximizes expected completed weeks before elimination.

`plan.selections` includes each selected team's win probability, selected-team
market spread, survival and elimination probabilities, parameter-variance
adjustment, and objective contribution. `plan.current_pick` is the row to use
for the current week. `plan.selection_config` records the eligibility
policies, horizon, Hessian-prefix length, and timeout.

The optimizer tracks the probability of being alive after each week at every
loss count below the elimination threshold. It uses the posterior-mean win
probability for every candidate and propagates scalar
gradient and Hessian contractions needed for a configurable initial
second-order prefix of the expected-weeks objective. With
`hessian_weeks=K`, the objective is

`sum(P[w] for every planned week) + 1/2 * sum(H[w] for the first K weeks)`.

The default is `hessian_weeks=3`; `0` uses only posterior-mean probability
terms, and values beyond the available horizon are clamped to that horizon.
The linear tail is not a different probability model: it continues the same
posterior-mean recurrence and omits only the later Hessian corrections. When
the prefix covers the full horizon, this is the usual
`F(mu) + 1/2 * trace(H_F(mu) * Sigma)` approximation.

If `p[w,l]` is the probability of reaching the start of week `w` with `l`
losses, selecting candidate `t` in week `w` gives the successor
`v[w,t] * p[w,l] + (1 - v[w,t]) * p[w,l - 1]`. The initial state is
`p[0,0] = 1`, with other initial loss states and negative loss indices equal to
zero. Candidate-specific successors are affine expressions, not separate
team-specific state variables. For each candidate, the MILP creates a dummy
`d[t] = selected[t] * candidate_successor[t]` and links the state successor to
the sum of the candidate dummies. Each dummy uses the finite candidate
recurrence bounds in the standard four-constraint bounded binary-product
linearization, so it is zero when the candidate is not selected and equals its
candidate recurrence when selected. This gives the LP relaxation the
candidate-wise convex-hull formulation instead of relaxing a shared successor
with other-candidate intervals.

Posterior coordinates are shared by `(hazard kind, team)` across the whole
horizon. The fitted posterior covariance is currently diagonal. The MILP
precomputes candidate gradient Gram constants
`K[t,k] = gradient(v[t])' * Sigma * gradient(v[k])` and tracks
`gradient(p[w,l])' * Sigma * gradient(v[k])` for each selectable candidate
reference `k`. A second scalar state tracks
`trace(Sigma * Hessian(p[w,l]))`; its recurrence includes the candidate
Hessian contraction and the gradient cross term. Signed interval recurrences
provide finite bounds for the probability, gradient, and Hessian dummy
products. Gradient and Hessian states are created only for the retained
Hessian prefix, while probability states continue through the full horizon.
The number of these scalar states depends on selectable candidates, not on the
number of posterior parameter coordinates. This remains a second-order
approximation to posterior uncertainty, not exact posterior integration.

The exact MILP is warm-started with a deterministic feasible greedy plan. For
each week it selects the highest posterior-mean `base_probability` among
eligible teams not already used, with stable candidate-order tie breaking.

```julia
plan = optimize_survivor_pool(
    context;
    strikes_remaining=2,
    selection_config=SurvivorSelectionConfig(
        through_week=18,
    ),
)
```

`strikes_remaining=s` means the `s`-th future loss eliminates the pool, while
zero means the next loss eliminates it.

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

Pass `--timeout SECONDS` to limit the default HiGHS MILP solve. If HiGHS
reaches the limit after finding a feasible incumbent, the app returns the best
incumbent found so far; if no feasible incumbent exists, the optimization
reports an error. Omit the option for an unlimited solve:

```sh
survivor --season 2026 --timeout 600 < picks.txt
```

The app reads one team abbreviation per nonblank line, starting with week 1.
It infers the next week from the number of picks, loads the season schedule to
count completed losses (ties count as losses), and defaults to two initial
strikes. Use `--strikes N` to choose a different initial strike count. The
effective week is one plus the number of supplied picks; games at or after
that week are treated as future games even when the schedule already contains
their results, which allows replaying an earlier week of a completed season.
The default output is only the selected team's abbreviation and a newline.
For an opt-in phase breakdown, pass `--timings`; timing diagnostics are written
to stderr so stdout remains suitable for a picks file:

```sh
survivor --season 2026 --timings < picks.txt
```

With `--timings`, the stderr diagnostics also include one record for each MILP
solve. Each record reports the phase, termination/primal status, whether an
incumbent is available, incumbent objective, objective bound, relative gap, and
branch-and-bound node count.

Use `--refresh-data` (or set `SURVIVORMODEL_REFRESH_DATA=true`) for an explicit
data refresh. This clears NFLData's raw-data cache and rebuilds the package's
summarized historical-drive cache before running; normal invocations reuse
summarized historical seasons while still loading the current season through
the normal NFLData path:

```sh
survivor --refresh-data --season 2026 < picks.txt
```

Historical empirical-Bayes priors are stored in the package's Scratch.jl
space and keyed by model-cache schema, season, historical-window length, and
a fingerprint of the historical drive data used for the fit.
If that data changes, the cached prior is recomputed. current-season drives and the survivor optimization are refreshed on each
invocation. An opening-week forecast can run before NFLData publishes
target-season PBP and uses historical drives only. Once prior picks imply week
2 or later, target-season PBP must be available so the current-season update is
not omitted. If a cached schedule still marks a supplied previous pick as
uncompleted, the CLI automatically clears NFLData's cache and retries the
schedule validation once. `--refresh-data` remains available when an explicit
full data refresh is desired.

The weekly planner uses the covariance-aware expected-weeks MILP described
above. `SurvivorSelectionConfig` controls the market guard, missing-line
policy, planning horizon, `hessian_weeks`, and `timeout_seconds`.
`hessian_weeks` defaults to three, allows zero for posterior-mean probability
terms only, and is clamped to the available horizon. `timeout_seconds` limits
the HiGHS solve and defaults to unlimited. The default market policy protects
the current and following week by requiring a selected team to be favored by
at least `2.0` points; missing lines remain eligible. Positive
`market_spread` values mean the selected team is favored.

### Docker image

The Docker image installs the `survivor` app as its entrypoint. Build and run
it like the CLI, passing options after the image name and picks on standard
input:

```sh
docker build -t survivormodel .
docker run --rm -i survivormodel --season 2026 < picks.txt
```

GitHub Actions builds the image for pull requests without publishing, then
publishes multi-platform `linux/amd64` and `linux/arm64` images to Docker Hub
on pushes to `main` and `v*` tags. Docker selects the image matching the host
architecture. Before the first publish, create the public Docker Hub repository
`moblin88/survivormodel.jl` and configure these repository Actions settings:

- `DOCKERHUB_USERNAME` as a repository Actions variable, set to your Docker
  Hub username.
- `DOCKERHUB_TOKEN` as a repository Actions secret, using a Docker Hub access
  token with read/write permission.

Pull and run the published image with:

```sh
docker pull moblin88/survivormodel.jl:latest
docker run --rm -i \
  moblin88/survivormodel.jl:latest \
  --season 2026 < picks.txt
```

`latest` follows `main`. A GitHub tag such as `v1.2.3` is published under the
same Docker Hub tag; the workflow does not create per-commit SHA tags or
floating major/minor aliases.

After each week, refresh the forecast context with the new `as_of_week`,
record the team picked in `picks_made`, update `strikes_remaining`, and call
`optimize_survivor_pool` again. The optimizer itself performs one forecast
pass and one solve. `build_survivor_candidates(context)` is also available to
inspect eligible teams and their forecast probabilities.
