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
At Debug level, each completed prior fit also logs a compact summary and one
parameter record per cause. The `source` field distinguishes new fits from
cached priors; cache records report the stored optimizer diagnostics, not a
new optimization. Records include the Weibull shape, Gamma shape/rate and
mean cumulative-hazard coefficient, home multiplier, persistence and reset
probabilities, and fit diagnostics. Durations are measured in minutes, so
mean cumulative-hazard coefficients have cause-specific units of
minutes^(-Weibull shape) and should not be compared directly across causes.
Iteration and evaluation counts describe the selected converged initialization.

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
        banned_first_pick_teams=["KC", "SF"],
        timeout_seconds=600.0,
    ),
)

plan.current_pick
plan.selections
plan.objective_value
```

The `:exact_milp` formulation expands each unplayed forecast game into
model-favorite candidates with win probability at least `0.5`, excludes teams
in `picks_made`, and represents the schedule with binary assignment variables.
It selects exactly one team for every week in the requested horizon and allows
each team to be selected at most once. The covariance-aware finite-state
formulation maximizes expected completed weeks before elimination.

By default, the formulation is solved using an external tree of full-model LP
relaxations. Set `branch_and_bound=false` in `SurvivorSelectionConfig` or pass
`--no-branch-and-bound` to use the extensive-form MILP instead.
The tree uses one-thread HiPO root and child LP solves, with independent Julia
worker tasks processing nodes in parallel. Every worker owns a separate copy
of the full LP; model memory therefore grows with the effective worker count.
By default, workers use Julia's available `:default` thread pool. Set
`branch_and_bound_workers` in `SurvivorSelectionConfig` or pass
`--branch-and-bound-workers N` to request a maximum; the count is capped to the
available pool and the number of initial regions. The app leaves Julia's
thread startup settings unchanged, so `JULIA_NUM_THREADS` is respected:

```sh
JULIA_NUM_THREADS=auto survivor --season 2026
JULIA_NUM_THREADS=4 survivor --season 2026 \
  --branch-and-bound-workers 3
```

HiGHS uses a process-global scheduler. SurvivorModel coordinates its own
optimizer calls around tree runs, resetting that scheduler before and after
the tree so one-thread HiPO solves also work after earlier default-thread
solves. Do not run unrelated direct HiGHS solves concurrently with a tree run.

Without an environment or runtime thread setting, Julia 1.13 may provide only
one default-pool thread, so the tree remains single-worker. The coordinator
retains bounds for queued and in-flight nodes and owns incumbent selection,
branching, and progress output. Concurrent solve completion can change search
order and the returned feasible witness, but first-pick certification,
eligibility, and objective semantics are unchanged. The tree does not promise
that the remaining witness schedule is globally optimal. If only one
current-week candidate is eligible, that pick is forced and returned with a
feasible greedy schedule. Both modes use HiGHS and the same covariance-aware
formulation.

`plan.selections` includes each selected team's win probability, selected-team
market spread, survival and elimination probabilities, parameter-variance
adjustment, and objective contribution. `plan.current_pick` is the row to use
for the current week. `plan.selection_config` records the eligibility
policies, horizon, Hessian-prefix length, branch-and-bound selection, and
timeout.
`banned_first_pick_teams` excludes those abbreviations only from the current
week's pick; they remain available in later weeks unless selected elsewhere in
the plan.

To save the built MILP in HiGHS' LP format without solving it, use
`write_survivor_pool_lp(path, context; ...)` or the CLI's `--write-model`
option.

The optimizer tracks the probability of being alive after each week at every
loss count below the elimination threshold. It uses the posterior-mean win
probability for every candidate and propagates scalar
gradient and Hessian contractions needed for a configurable initial
second-order prefix of the expected-weeks objective. With
`hessian_weeks=K`, the objective is

`sum(P[w] for every planned week) + 1/2 * sum(H[w] for the first K weeks)`.

The default is `hessian_weeks=18`; `0` uses only posterior-mean probability
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
each week it ranks eligible picks by the next-week total survival probability
plus half the covariance-contracted Hessian state sum, evaluated under the
chosen prefix. The correction applies only through
`H=min(hessian_weeks, remaining_horizon)`; later picks use survival alone.
Stable candidate order breaks ties, and assignment-matching look-ahead
preserves a feasible remaining schedule. This is a local greedy score, not a
future-objective estimate or a globally optimal schedule.

Branch-and-bound completes queued node paths with the same heuristic,
preserving all fixed picks and retaining only a scalar completion score and a
readiness flag per node, not its schedule. Each node is completed at most once.
Completions can improve the incumbent and their full objectives are used as a
node-order tie-breaker after certified region and node upper bounds. They do
not tighten an upper bound or prune a node by themselves.

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

Julia 1.13 can run the package directly through its `@main` entry point:

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

The installed app loads Julia's `startup.jl`, including user startup settings.
After changing the app's launcher configuration, rerun `Pkg.Apps.develop` to
regenerate an existing launcher.

Then run it from any directory:

```sh
survivor --season 2026 < picks.txt
```

Use `--plot-strength WEEK` to open a team-strength scatter plot without
providing picks or running survivor selection:

```sh
survivor --season 2026 --plot-strength 5
survivor --season 2026 --plot-strength=5
```

Week `WEEK` is the start of that week: the posterior includes current-season
regular-season drives only from earlier weeks, and week 1 uses the historical
prior alone. Each target-season team appears, including teams on a bye. The
x-axis shows the neutral-site offensive touchdown rate's percentile in the
fitted league touchdown Gamma prior; the y-axis uses the defensive-event
rate's percentile in its separate league Gamma prior. Both axes run from 0%
to 100%, with higher percentiles indicating stronger rates. Each cause's
reference distribution is shared across teams, not a team-specific mixture
or a ranking of the current teams. Points transform the posterior mean rates
through these prior CDFs.

Bars show the central 50% posterior rate interval, from the 25th to the 75th
percentile of the full team Gamma mixture, with each endpoint mapped to the
same league-prior percentile scale. They are not necessarily 25%-75% on the
plot axes. These intervals describe uncertainty in the team's rate, not a
prediction interval for a future drive or uncertainty in the fitted league
hyperparameters. They can be asymmetric, and a sufficiently skewed mixture's
mean point can lie outside its interval.

Points, both interval bars, label text and outlines, and leader lines share one
high-contrast color for each team. Team labels use bold text on opaque white
backgrounds. Recognized historical abbreviations use their franchise's color;
unknown abbreviations produce a warning and use neutral gray. Collision-aware
placement keeps labels clear of one another and the mean markers, with leader
lines to their points, and updates when the window is resized. Very small
windows may still crowd labels; enlarge the window if a warning appears.

GLMakie loads only for this option. The command opens a native desktop window
and stays open until you close it; a working graphical/OpenGL display is
required. Headless servers and containers cannot use this popup without a
configured display, and there is no file-output fallback. Cache refreshes can
be combined with plotting:

```sh
survivor --refresh-priors --season 2026 --plot-strength 5
```

Plot mode does not accept pick-selection or model export options such as
`--ban`, `--branch-and-bound`, or `--write-model`.

Use `--grid` to print a survivor grid to stdout instead of selecting a pick:

```sh
survivor --season 2026 --grid < picks.txt
survivor --season 2026 --grid --strikes 3 < picks.txt > grid.txt
```

The grid has one row per unused target-season team and columns from the
current week through week 18. Picks on stdin infer the current week and are
validated just as in a selection run; empty input starts at week 1. Each cell
shows a left-aligned opponent and right-aligned one-decimal model win
percentage (`@SF 67.4%` for an away game, `SF 67.4%` for a home game); byes
are blank. Percentages, decimal points, and the fixed-width star slots align
within each week, and horizontal rules separate groups of five team rows
without a trailing rule. Rows are sorted by unrounded current-week win
probability descending, with current-week byes last and alphabetical
tie-breaking. All unused teams are shown, including underdogs, without the
survivor selection market filters.

An aligned `*` marks the top five unused teams independently in each week,
using that week's unrounded adjusted probabilities and alphabetical ties.
Byes are excluded, and fewer than five stars appear when fewer than five
unused teams are scheduled. Stars are weekly probability rankings, not an
optimized survivor plan.

Every displayed probability includes the posterior Hessian adjustment,
including later weeks, and is computed from the same start-of-current-week
posterior. Later games with recorded results remain forecastable for replay;
future drives and outcomes do not enter the posterior. The grid prints every
row and week without truncation, does not run optimization or load GLMakie,
and keeps diagnostics on stderr. `--strikes`, `--refresh-data`, and
`--refresh-priors` are allowed. Do not combine `--grid` with `--plot-strength`,
`--ban`, `--branch-and-bound`, `--no-branch-and-bound`, `--write-model`,
`--timeout`, or `--hessian-weeks`: Hessian adjustments are always applied to
all grid cells.

Pass `--write-model FILE.lp` to save the main MILP with HiGHS and exit without
optimizing or printing a pick. For a preseason 2026 model with no prior picks,
two strikes, an 18-week horizon, and Hessian adjustments for all 18 weeks:

```sh
survivor --season 2026 \
  --write-model survivor_2026_full18_hessian18.lp < /dev/null
```

Pass `--ban TEAM1,TEAM2` to forbid those teams from the current week's pick
without excluding them from later weeks in the plan:

```sh
survivor --season 2026 --ban KC,SF < picks.txt
```

Branch-and-bound is the default. Pass `--no-branch-and-bound` or set
`SurvivorSelectionConfig(branch_and_bound=false)` to use the extensive-form
MILP instead:

```sh
survivor --season 2026 --timeout 60 < picks.txt
```

The tree solves the full continuous relaxation at the root and partitions all
eligible first-week picks, including candidates with zero LP value. Its root
and child relaxations use HiPO with one solver thread each, crossover, and
automatic presolve. Julia worker tasks process independent nodes using
worker-owned copies of the LP; `--branch-and-bound-workers N` limits the worker
pool. Set `JULIA_NUM_THREADS` before launching the app to select Julia's thread
pool; the app does not override that setting. Child LPs fix picks and tighten
probability, gradient, projected-gradient, and Hessian intervals; the existing
four product-hull inequalities are updated in place. Validated solver dual
bounds may prune nodes; if a recoverable solve has no validated bound, the root
uses the formulation-based global interval bound and children retain their
inherited bounds while branching exhaustively. Primal LP values are only
optional branching guidance, never upper bounds.
The tree certifies the current first pick within a numerical tolerance, not the
entire witness schedule. Its returned schedule is feasible and independently
evaluated. A timeout returns the best feasible schedule with an explicit
unproven warning.

The default path does not retry failed HiPO solves or construct
residual-corrected Lagrangian bounds. HiGHS' own crossover and internal
recovery remain enabled; removing the application-level tree backend does not
disable simplex work performed internally by HiGHS. A fallback that retains
an inherited bound may increase the search tree; no speedup is implied.
`--write-model` continues to export the unsolved extensive-form MILP,
regardless of the selected solve mode.

Pass `--timeout SECONDS` to limit optimization. Branch-and-bound shares its
budget across model construction, root and child solves, and certification.
On timeout it returns the exact greedy or improved witness, retains valid
bounds for interrupted or unsolved nodes, and warns if the first pick remains
unproven. Forecast and data loading occur before this optimization budget.
The extensive-form HiGHS solve returns a feasible incumbent when available
and reports an error if no feasible incumbent exists. Omit the option for an
unlimited solve:

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
The selected team is logged at Info level to stderr and printed as its
abbreviation with a newline to stdout, so stdout remains suitable for a picks
file. Phase timings, final prior-fit parameters and optimizer diagnostics are
Debug-level logs; enable them with `JULIA_DEBUG=SurvivorModel`. Prior records
are emitted only after the full prior is built and identify whether the values
come from a fresh fit or cache. Branch-and-bound emits an aligned progress
table to stderr, including root and final rows, node actions, bound sources,
reasons, incumbent and competing bounds, IPM and crossover iterations, and
timing. Important fallback and proof information stays in the table while it
is enabled instead of appearing as interleaved logger messages. Native HiGHS
output remains silent for the external tree.

For example, redirect stdout and stderr separately to save the next pick and
diagnostics to different files:

```sh
JULIA_DEBUG=SurvivorModel survivor --season 2026 \
  < picks.txt > next-pick.txt 2> survivor.log
```


With `--season`, `--refresh-data` (or `SURVIVORMODEL_REFRESH_DATA=true`) clears
NFLData's raw-data cache and rebuilds summarized historical-drive data before
running. Without `--season`, it clears the raw cache and all summarized
historical-drive caches, then exits without downloading or rebuilding data.
It leaves fitted prior caches intact. Normal invocations reuse summarized
historical seasons while still loading the current season through the normal
NFLData path:

```sh
survivor --refresh-data --season 2026 < picks.txt
survivor --refresh-data
```

Historical empirical-Bayes priors are stored in the package's Scratch.jl
space and keyed by model-cache schema, season, historical-window length, and
a fingerprint of the historical drive data used for the fit. If that data
changes, the cached prior is recomputed. Current-season drives and the
survivor optimization are refreshed on each invocation. Use `--refresh-priors`
to remove all cached historical prior fits without clearing any data caches;
by itself it clears the prior cache and exits, or with `--season` it clears
before running and refits as needed:

```sh
survivor --refresh-priors
survivor --refresh-priors --season 2026 < picks.txt
survivor --refresh-data --refresh-priors
survivor --refresh-data --refresh-priors --season 2026 < picks.txt
```

An opening-week forecast can run before NFLData publishes
target-season PBP and uses historical drives only. Once prior picks imply week
2 or later, target-season PBP must be available so the current-season update is
not omitted. If a cached schedule still marks a supplied previous pick as
uncompleted, the CLI automatically clears NFLData's cache and retries the
schedule validation once.

The weekly planner uses the covariance-aware expected-weeks MILP described
above. `SurvivorSelectionConfig` controls the market guard, missing-line
policy, planning horizon, `hessian_weeks`, whether the
`branch_and_bound` tree is selected (default: true), and `timeout_seconds`.
`hessian_weeks` defaults to 18, allows zero for posterior-mean probability
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

The image defaults to `JULIA_NUM_THREADS=auto` for runtime Julia processes.
Override it with `docker run -e JULIA_NUM_THREADS=N` to set a specific thread
count.

The image installs a standalone Julia `startup.jl` that selects AOCL on AMD
x86 CPUs and MKL on Intel x86 CPUs using the container's Linux CPU vendor ID.
ARM and unknown vendors retain default BLAS. Both x86 backends are installed
in a Docker-only environment, separate from the application's dependencies;
startup does not install packages or change the active project. Backend load
errors propagate normally. Julia apps must enable startup files to use this
selection (the installed `survivor` app already does).

Dependency installation and precompilation run before copying `src/`, in a
separate Docker layer. Source-only rebuilds reuse that layer and recompile the
package and installed app; changes to either project file invalidate dependency
setup. Cache reuse also depends on the base image and target architecture.
Both CI build jobs import and export GitHub Actions layer caches, including
intermediate dependency layers.
Build-time precompilation uses Xvfb for GLMakie; the runtime entrypoint remains
unchanged, and interactive plotting still requires a graphical display.

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
