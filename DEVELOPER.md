# Developer workflows

The supported weekly workflow is documented in `README.md`. This document
records implementation details for the covariance-aware survivor MILP and
cache maintenance.

## Covariance-aware survivor MILP

The context-based `optimize_survivor_pool` expands unplayed games into
model-favorite candidates and solves one binary assignment model. It selects
exactly one team per week, prevents team reuse, and applies the near-term
market guard before solving.

The objective tracks the probability of being alive after each week at every
loss count below the elimination threshold. For each candidate, the model
evaluates win probability and derivatives at the joint posterior mean using
the game-level ForwardDiff path. With `hessian_weeks=K`, the objective includes
posterior-mean probability states for the full horizon and the
`1/2 * H` correction for the first `K` transitions. The default is 18
weeks; zero is linear-only, and a larger value is clamped to the horizon. A
full-horizon correction is a second-order approximation
`F(mu) + 1/2 * trace(H_F(mu) * Sigma)`, not exact posterior integration.

If `p[w,l]` is the probability of reaching week `w` with `l` losses, selecting
candidate `t` in week `w` gives the successor
`v[w,t] * p[w,l] + (1 - v[w,t]) * p[w,l - 1]`, with `p[0,0] = 1` and negative
loss indices equal to zero. Candidate-specific successors use dummies
`d[t] = selected[t] * candidate_successor[t]`; four bounded linear product
constraints enforce each dummy, and the shared successor equals their sum.
This retains the candidate-wise convex-hull formulation in the LP relaxation.

Posterior coordinates are shared by `(hazard kind, team)` across the horizon.
The fitted covariance is diagonal. The model precomputes candidate
gradient Gram constants
`K[t,k] = gradient(v[t])' * Sigma * gradient(v[k])`. Gradient states use a
hybrid representation: while any remaining suffix has more active candidate
references than parameter coordinates, the model tracks the full parameter
gradient `D[w,l] = gradient(p[w,l])`. Its recurrence for selected candidate
`t` is
`D[w+1,l] = v[w,t] * D[w,l] + (1 - v[w,t]) * D[w,l-1] +
gradient(v[t]) * (p[w,l] - p[w,l-1])`.
At the first state whose complete remaining suffix fits within the parameter
dimension, the recurrence switches to only the needed contractions
`gradient(p[w,l])' * Sigma * gradient(v[k])`. The transition projects the
parameter state through the covariance and candidate gradients directly into
the candidate-contraction recurrence, so the full parameter gradient is not
reconstructed later. Each loss-count slice therefore tracks at most the
number of posterior coordinates. A scalar state tracks
`trace(Sigma * Hessian(p[w,l]))`; its recurrence includes the candidate
Hessian contraction and gradient cross term. Signed interval recurrences
provide finite bounds for parameter-gradient, candidate-gradient, probability,
and Hessian dummy products.

The interval recurrences include one pass over all candidate histories and a
team-leave-out pass for each distinct candidate team. The leave-out pass
excludes that team from every earlier transition, making its candidate-specific
interval valid when that candidate is selected. Product dummies use this
conditioned interval on the selected branch and the all-history interval on
the unselected branch, where the team may have appeared earlier. Shared
successor-state bounds are tightened to the envelope of these conditioned
candidate intervals. This reuses the existing product-hull inequalities and
adds no auxiliary variables or separate cut family; a conditioned endpoint
can make an existing hull side active.

Redundant aggregate recurrences directly constrain the weekly probability sum
through the full horizon and the adjusted objective sum through the retained
Hessian prefix. The probability aggregate is also constrained to be
nonincreasing. Gradient reference states are suffix-pruned to the Hessian
prefix and omitted when no earlier candidate can have a nonzero Gram
interaction. The hybrid switch checks all later reference widths rather than
assuming they decrease monotonically. This pruning is exact; zero-support
states are fixed at zero.

The MILP is warm-started with a deterministic feasible greedy plan that
selects the highest next-week survival sum plus half the Hessian state sum
under the chosen prefix, using the shared exact probability/parameter-gradient/
Hessian transition. Only the first `H=min(hessian_weeks,horizon)` weeks receive
the correction; subsequent weeks rank survival alone. This is local greediness,
not an optimization of future contributions. Assignment matching preserves a
feasible remaining schedule while
respecting market eligibility, current-week first-pick bans, and team
uniqueness. The selected plan is forward-evaluated again to verify its reported
objective.

## HiPO branch-and-bound

The default is an external branch-and-bound tree that certifies the current
first pick. Set `SurvivorSelectionConfig(branch_and_bound=false)` or pass
`--no-branch-and-bound` to select the extensive-form MILP instead.
Tests that assert a globally optimal complete schedule explicitly select the
extensive-form MILP: the tree certifies only the first pick and may return a
feasible but suboptimal suffix. Tree tests instead compare the selected
first-pick region with exhaustive per-region optima and exactly evaluate the
returned witness.
`_build_survivor_full_model` and the normal extensive-form solve share the
scalar recurrence builder; branch-and-bound does not duplicate probability,
gradient, projected-gradient, or Hessian recurrences. The model contains all
pick variables and the one-pick-per-week/team-once constraints. Probability
states span the full remaining horizon, while derivative states and Hessian
corrections use `H=min(hessian_weeks, remaining_horizon)`. Fixing path picks
does not shift or shorten H. `--write-model` continues to export the unsolved
extensive-form MILP.

The external tree solves its root relaxation on one direct continuous HiGHS
model, partitions every eligible first-week pick (including zero-valued LP
candidates), then dispatches independent child nodes to Julia worker tasks.
Each worker owns a distinct direct HiGHS model copied from the unconditioned
root formulation with its JuMP variable and hull-row references explicitly
remapped. A worker never shares a mutable optimizer or conditioned reference
with another task. Before each child solve, availability and fixed picks
tighten probability, parameter-gradient, projected-gradient, and
signed-curvature intervals. The four existing product-hull rows are rewritten
in place from those intervals; model rows and columns remain fixed, and
siblings recompute from root bounds. The full objective is maximized by
minimizing its negative in this tree, then solver bounds are normalized back
to the original maximization convention.

Root and worker LPs set `solver=hipo`, `threads=1`, `parallel=off`,
`presolve=choose`, and `run_crossover=on`. HiGHS native output remains silent.
The default worker capacity is Julia's `Threads.nthreads(:default)`, limited
by available first-pick regions. `branch_and_bound_workers` and
`--branch-and-bound-workers` request a positive upper limit, which is capped
to the default-pool size. The coordinator is not assigned a reserved solver
thread; node work runs as `Threads.@spawn :default` tasks, and ownership follows
the task rather than a thread ID. The package/app does not set a Julia thread
flag: Julia startup honors `JULIA_NUM_THREADS`, so set it before launch to
select automatic or explicit pool sizing. On Julia 1.12, the default pool can
contain only one thread when the variable is unset.

HiGHS' scheduler is process-global, so prior ordinary solves can otherwise
prevent a later model from changing to `threads=1`. A writer-preferring gate
allows ordinary SurvivorModel solves to share the default scheduler, then gives
the tree exclusive access while it resets the scheduler before the root solve
and again after all workers join. This lets a later ordinary solve reinitialize
HiGHS with its default thread count. Optimizations made directly on unrelated
HiGHS models are outside this gate and must not overlap a tree run.

Only the coordinator changes the ready queue, incumbent, node summaries,
regional upper bounds, observers, and progress output. In-flight nodes remain
live leaves in their ancestor summaries, so taking a node out of the ready
queue never drops its inherited regional certificate. A tree completion is
not considered exhausted while there is queued or in-flight work. On every
normal stop, timeout, or error, the coordinator stops dispatch, closes worker
inputs, drains bounded result delivery, and joins all worker tasks before
returning or rethrowing. It does not destroy or mutate an optimizer while
HiGHS is solving. Workers return node IDs and immutable result data, never
solver references. Coordinator callbacks are serialized.

Conditioned hull totals use outward-rounded `BigFloat` arithmetic. That scoped
precision/rounding section is protected by a shared reentrant lock; the
Float64 bound calculations and solver runs remain parallel.

## App startup and BLAS threads

The Pkg app enables Julia startup-file loading, so installed `survivor` apps
respect the user's `startup.jl`. Regenerate an existing shim with
`Pkg.Apps.develop(path="...")` or `Pkg.Apps.update("survivor")` to apply this
setting. The CLI sets the active BLAS backend to one thread by default when it
starts; `SURVIVORMODEL_BLAS_THREADS=N` overrides that default, while
backend-native thread environment variables are preserved. Importing
`SurvivorModel` as a library does not change BLAS state.

Docker images set `JULIA_NUM_THREADS=auto` unless overridden at runtime, and
load MKL for Intel Linux x86_64 or AOCL for AMD Linux x86_64 before application
packages. ARM and unknown vendors retain OpenBLAS. Vendor packages and CPU
detection are Docker-only; the app itself does not select or install a vendor
backend. Missing or inactive selected Docker backends are startup errors.
Container construction strictly precompiles app dependencies and explicitly
loads GLMakie under Xvfb/software Mesa in a builder stage, then copies the
Julia depot into a clean runtime stage. Precompile or GLMakie load failures
fail the build. The published image omits Xvfb and build-time display packages,
so interactive container plotting still needs a separately configured
display.

The tree accepts a solver upper bound only when a fresh MOI result reports a
feasible dual and the native HiGHS dual status agrees. For nonoptimal results,
the current dual-infeasibility and stationarity diagnostics must also pass
the configured tolerances. Optimal results with feasible dual status are
trusted within HiGHS' tolerances even if crossover leaves some residual fields
unavailable. A finite objective-bound value alone is not sufficient, and a
primal objective is never treated as an upper bound. The accepted bound is
not independently residual-corrected.

If a recoverable root solve has no validated dual bound, every first-pick
region inherits the formulation-based global interval upper bound. A child
uses the tighter of a validated solver bound and its inherited bound; without
a validated solver bound it keeps the inherited bound and branches over all
eligible choices in an unfixed week. A feasible primal assignment is only
branching guidance. A complete assignment can improve the incumbent after
independent exact evaluation, but does not close an unfixed subtree. A fully
path-fixed schedule is evaluated exactly and closes its singleton. Supported
infeasibility certificates may prune infeasible paths; unexpected statuses
and contradictions with known feasible schedules remain errors.

The root and descendants share a deterministic feasible greedy incumbent.
Node completions preserve fixed picks and only provide a lower bound for
ranking/incumbent updates. The coordinator selects nodes with the existing
competing-first-pick priority and interleaves incumbent-region work; solve
results are integrated as workers finish. The current first pick is proven when its
feasible objective is at least the best competing-region upper bound within
the scale-aware `1e-6` tolerance. The returned full schedule is feasible and
its objective is independently evaluated, but future picks are not promised
globally optimal after the first pick is certified. The timeout covers model
construction, root and worker model copies, node conditioning, relaxations,
and certification; on timeout the best feasible schedule is returned with an
explicit unproven warning and all queued and in-flight bounds retained.

The default HiPO tree performs no application-level retry with presolve off
and does not build residual-corrected Lagrangian certificates. HiGHS' internal
recovery and IPX crossover remain enabled. Crossover can add solve cost, and
inherited-bound fallback can increase the number of processed nodes; neither
fewer retries nor crossover alone implies an end-to-end speedup. The table's
`XO` column reports the solver's simplex-iteration statistic as a crossover
diagnostic; there is no application-selected simplex tree backend or basis
transfer between nodes. Removing that application-level backend does not
disable simplex work used internally by HiGHS during solver cleanup or
crossover.

When Debug logging is enabled, B&B diagnostics are rendered as a live,
aligned stderr table rather than separate debug messages. Root and final rows
use the same renderer as node actions. Columns include first-pick region,
depth, branch week, queue size, incumbent lower bound, competing/node upper
bounds, action, bound source, diagnostic reason, IPM and crossover iterations,
and elapsed time. Fallbacks, rejected bounds/guidance, exact closure, pruning,
timeouts, and proof status appear in these rows. Important warnings remain
immediate when the table is disabled; stdout remains reserved for the selected
team abbreviation. Observer events remain available to tests and internal
callers.

## Team-strength plot

The internal `_team_strength_plot_data` helper derives a sorted row for every
team in the target-season regular-season schedule from the forecast context.
The plot uses the neutral-site posterior means of the touchdown and defensive
event cumulative-hazard coefficients, with central 50% posterior intervals.
`_gamma_mixture_quantile` computes the 25th and 75th percentiles of the full
Gamma mixture through Distributions' shape/scale parameterization, ignoring
zero-weight components. These are posterior rate intervals, not
future-observation prediction intervals.

`_team_strength_plot_percentiles` preserves the raw data and transforms each
mean and interval endpoint through the corresponding fitted league Gamma
CDF, multiplied by 100. The shared references are
`HazardPrior.td_hyperparameters` and `.defensive_hyperparameters`, using
`Gamma(shape, inv(rate))`. Do not substitute team-specific mixtures, current
team ranks, or the posterior expectation of the CDF for the CDF of the mean
rate. Both axes have fixed initial 0%-100% limits. Saturated CDFs and
coincident points are valid; do not artificially spread them. Makie range
bars preserve the transformed endpoints even when a skewed mixture's mean
lies outside its central interval.

The per-team palette in `src/team_strength_plot.jl` uses dark team-associated
shades based on published NFL colors, including
[nflverse's team color data](https://github.com/nflverse/nflverse-pbp/blob/master/teams_colors_logos.csv).
Each shade has at least 4.5:1 contrast against white. The same row-ordered
color is used for its point, both range bars, label text and outline, and both
endpoints of each leader line; label backgrounds remain white. Historical
aliases resolve to their current franchise colors: ARZ to ARI, JAC to JAX,
LA and STL to LAR, OAK to LV, SD to LAC, and WFT and WSH to WAS. Unknown
abbreviations warn and use neutral gray.

`src/team_strength_labels.jl` separates deterministic pixel-space layout from
Makie rendering. Native `textlabel!` plots provide bold text, opaque white
backgrounds, padding, and outlines; measure their background `Poly` bounds
rather than the parent plot's zero-sized anchor bounds. Placement avoids
padded label intersections and mean markers, keeps boxes in the viewport,
and connects box edges to the exact projected means with leader lines. The
layout observes viewport, camera projection, and camera resolution changes
through scene-owned callbacks. Crowded tiny viewports retain all labels and
warn rather than silently hiding teams.

CLI `--plot-strength WEEK` fits the same forecast context as a normal run, but
uses the start of `WEEK` as its cutoff and returns before reading picks or
building a survivor optimization. GLMakie is a direct app dependency but is
imported only by this plot path. It opens a native window and waits until it is
closed, so an available desktop/OpenGL display is required. On Linux the
display check fails before GLMakie loads if both `DISPLAY` and
`WAYLAND_DISPLAY` are unset. The rendering smoke test runs under Xvfb with
software Mesa in CI; headless operation is not silently treated as a
successful plot. The display wrapper closes its owned screen and empties the
figure on completion or failure to release scene callbacks. The standalone
two-thread branch-and-bound CI test also runs under Xvfb with software Mesa
because GLMakie initialization can occur during precompilation even when that
test does not render. Headless percentile tests are in
`test/forecast_unit.jl`, layout tests in `test/team_strength_labels_unit.jl`,
and native geometry, dense-team, resize, and lifecycle coverage in
`test/team_strength_plot_rendering.jl`.

## Survivor grid

`--grid` reuses normal stdin pick/strike validation and
`_survivor_cli_build_forecast_context`, then returns before constructing a
selection configuration or solver. `_survivor_grid_data` calls
`forecast_win_probabilities(context; include_completed=true)` once: this path
already applies `p + 0.5 * trace(H * Sigma)` to every game, with the existing
finite-value checks and [0,1] clamping. There is no Hessian-week prefix for
grid probabilities and no hypothetical future posterior update.

Rows contain all unused teams from the full target-season regular-season
schedule, without market eligibility filters, and columns span the current
week through week 18. The numeric current-week probability determines
descending row order before formatting; byes come last and ties are
alphabetical. Duplicate team/week games are errors.
`_survivor_grid_top_five` ranks each column's scheduled unused teams by
unrounded probability and alphabetical ties, returning a row-aligned Boolean
mask after the current-week row sort. It selects exactly the lesser of five
and the number of scheduled unused teams; byes never receive a star. These
independent weekly rankings do not optimize a survivor plan.

`_write_survivor_grid` prints an untruncated plain-text table with separate
left-aligned opponent, right-aligned six-character percentage, and
one-character `*` slots. Percentages retain one decimal place, `@` denotes
away games, and byes are blank. Week headers are centered. The header rule
also separates groups of five team rows, without a trailing rule. Recorded
future results are included for replay but do not replace model probabilities.

Grid mode allows strikes and cache-refresh flags, rejects selection/export
options and `--plot-strength`, and never imports GLMakie. Numerical, sorting,
formatting, and replay coverage is in `test/survivor_grid_unit.jl`; CLI
integration coverage remains in `test/cli_unit.jl`.

## Cache maintenance

Historical prior caches are fingerprinted and refresh automatically when
their inputs change. The helper's default clears the package-owned Scratch.jl
cache; a `cache_directory` override removes only `historical_prior_*.jls`
files in that directory and leaves unrelated or summarized-drive cache files
untouched:

```julia
SurvivorModel.clear_historical_prior_cache!()
SurvivorModel.clear_historical_prior_cache!(; cache_directory="/path/to/cache")
```

Use the CLI's `--refresh-priors` to clear fitted prior caches, either alone or
before a season-based run. Standalone `--refresh-data` clears NFLData's raw
cache and all summarized historical-drive caches, then exits without
rebuilding them; with `--season`, it refreshes data and continues the run.
`--refresh-data` leaves prior fits intact, and `--refresh-priors` leaves data
caches intact. Combine them to clear both cache groups.

`fit_empirical_bayes_prior` emits Debug-level structured diagnostics only after
both cause fits and their reset mixtures are complete.
`_cached_historical_prior` uses the same diagnostic logger for cache hits, with
`source=:cache` and the cache path; misses are logged once by the fitter. Each
acquisition produces a summary and one record per cause, including fitted
parameters and the selected optimizer result. Iteration and function-evaluation
counts belong to the selected converged initialization, not the aggregate cost
of all starts.
