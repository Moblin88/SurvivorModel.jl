# Developer and research workflows

The normal weekly workflow is documented in `README.md`. This document covers
research and maintenance surfaces that are intentionally not part of the
survivor CLI or the default package exports.

## Fitting benchmarks

Run solver comparisons without loading benchmark modes into the weekly app:

```sh
julia --project=. tools/fit_benchmark.jl --help
julia --project=. tools/fit_benchmark.jl --repeats 3
julia --project=. tools/fit_benchmark.jl \
  --scenario recovery \
  --recovery-seasons 5 \
  --max-seasons 5 \
  --repeats 3
julia --project=. tools/fit_benchmark.jl \
  --data real \
  --season 2024
```

The tool retains the non-production solver implementations and reports timing,
likelihood, convergence, fitted parameters, and synthetic recovery quality.
Those fitting types are available by qualified name for tests and experiments,
for example `SurvivorModel.DirectLBFGSFit()`. Production fitting continues to
use `SurvivorModel.EMLBFGSFit()`.

## Renewal-process simulation

`simulate_renewal_schedule` is a model-development simulator. It generates
drive-level outcomes from a fitted two-outcome renewal process over a
hypothetical schedule; it does not simulate scores or survivor decisions.

```julia
using Random
using SurvivorModel

model = fit_hazard_model(current; prior=prior)
simulation = SurvivorModel.simulate_renewal_schedule(
    model,
    hypothetical_schedule;
    rng=MersenneTwister(42),
)

simulation.games
simulation.drives
simulation.latent_rates
```

The returned drives can be passed to `build_exposure_data` or
`fit_hazard_model`. The hypothetical schedule uses the same schedule columns
as `load_schedule`, with `result=missing` for unplayed games.

## Spread and score diagnostics

Spread forecasting is retained for model research but is not required for
survivor selection:

```julia
context = fit_regular_season_forecast(2024; as_of_week=10)
spread_forecast = SurvivorModel.forecast_spreads(context)
full_forecast = SurvivorModel.forecast_regular_season(
    2024;
    as_of_week=10,
)
```

The `:fixed_exact_milp` survivor objective is the exact finite-state MILP for
fixed candidate probabilities. It tracks the probability of remaining alive at
each loss count below the terminal loss threshold and maximizes the sum of
weekly survival probabilities. Binary selection/state-product terms are
linearized with bounds in `[0, 1]`, so this formulation avoids terminal-path
enumeration and nonlinear subproblems.

The default context-based `:exact_milp` objective expands that recursion with a
configurable initial Hessian correction for posterior parameter uncertainty.
For each candidate, the model evaluates the win probability and its
derivatives at the joint posterior mean using the existing game-level
ForwardDiff path. With `hessian_weeks=K`, the objective includes all
posterior-mean probability states and only the `1/2 * H` correction for the
first `K` transitions. The default is three weeks; zero is linear-only, and a
larger value is clamped to the available horizon. When `K` covers the full
horizon, this is the second-order approximation
`F(mu) + 1/2 * trace(H_F(mu) * Sigma)`, not exact posterior integration.

The recursion defines `p[w,l]` as the probability of reaching the start of
week `w` with exactly `l` losses. If candidate `t` is selected in week `w`,
its successor is
`v[w,t] * p[w,l] + (1 - v[w,t]) * p[w,l - 1]`, with `p[0,0] = 1` and negative
loss indices equal to zero. Candidate-specific successors are represented
by candidate-specific continuous dummies
`d[t] = selected[t] * candidate_successor[t]`. Four bounded linear product
constraints enforce each dummy, and the shared successor equals the sum of all
dummies. Therefore each dummy is zero for an unselected candidate and equals
its recurrence for the selected candidate, while the LP relaxation retains the
candidate-wise convex-hull formulation.

To include shared-team uncertainty without parameter-sized MILP state, the
model precomputes the candidate gradient Gram constants
`K[t,k] = gradient(v[t])' * Sigma * gradient(v[k])` and the scalar
`trace(Sigma * Hessian(v[t]))` values. It tracks
`g[w,l,k] = gradient(p[w,l])' * Sigma * gradient(v[k])` for each selectable
candidate reference `k`, followed by
`h[w,l] = trace(Sigma * Hessian(p[w,l]))`. The selected candidate recurrence
for `g` includes `K[t,k] * (p[w,l] - p[w,l - 1])`; the `h` recurrence includes
the candidate Hessian contraction and
`2 * (g[w,l,t] - g[w,l - 1,t])`. Signed lower and upper intervals for all
three state families are propagated recursively and reused as bounds for the
candidate dummy product constraints.

The model also adds redundant aggregate recurrence cuts for
`P[w] = sum(p[w,l])` over the full horizon and
`A[w] = sum(p[w,l] + 0.5 * h[w,l])` only through the retained prefix. For a
selected candidate, these recurrences telescope the loss-state transitions and
directly constrain the objective state in the LP relaxation. The probability
aggregate is additionally constrained to be nonincreasing over time.

Gradient reference states are created only for the retained Hessian prefix and
are omitted when the Gram matrix proves that no candidate in any earlier week
can contribute to that reference. A reference state is added again at the
first week where a nonzero covariance-gradient contraction is possible. No
gradient successor is created after the final retained Hessian transition.
This prefix and support pruning is exact; zero-support states are fixed at zero
rather than approximated.

The candidate Gram matrix is sized by selectable rows rather than by the
posterior parameter vector, and the current fitted covariance remains
diagonal. The covariance MILP is warm-started with a deterministic greedy
feasible plan that selects the highest posterior-mean candidate probability
each week while respecting market eligibility and team uniqueness. It seeds
the selected binaries and the forward-evaluated probability,
gradient-contraction, and retained Hessian-contraction states. The selected
plan is then forward-evaluated again to verify the reported hybrid objective;
it does not solve `:fixed_exact_milp` as a preliminary warm-start problem.

When `prove_first_pick=true`, the selected plan is evaluated again with
Hessian states through the full horizon. A second scalar MILP then keeps all
full-Hessian terms, forbids the selected first candidate, and imposes a
non-strict lower bound equal to that full-Hessian score. This is a
feasibility check rather than a second optimization: only `INFEASIBLE` proves
the first pick under the requested threshold. A feasible alternate, timeout,
or any other unresolved status raises an explicit error, and the proof reuses
the configured `timeout_seconds`.

`SurvivorSelectionConfig(timeout_seconds=...)` passes a HiGHS `time_limit`
attribute to the default optimizer. If the limit is reached with a feasible
incumbent, the MILP returns that best-known plan; a timeout without any
feasible incumbent is reported as an optimization failure.

The CLI's `--timings` mode raises the timing logger to debug level so each
MILP phase records its termination and primal statuses, result count,
incumbent availability and objective, objective bound, relative gap, and
branch-and-bound node count. Library callers retain the normal quiet logging
behavior unless they enable debug logging themselves.

Candidate rows below the model-favorite threshold of `0.5` are excluded before
either survivor objective is solved.

For a matchup-level diagnostic, use
`SurvivorModel.expected_game_metrics`. Score-mark, drive-moment, hazard-theta,
and predictive-spread helpers are likewise qualified research interfaces.
These outputs are model-line diagnostics, not sportsbook ATS or profitability
results.

Probability calibration is the supported weekly quality surface. To run the
additional spread calibration tables, use:

```julia
research = SurvivorModel.evaluate_calibration_research(;
    cutoff_weeks=(1, 5, 10, 15),
    recent_seasons=3,
)

research.spread_summary
research.spread_reliability
research.spread_coverage
```

## Cache maintenance

Historical prior caches are fingerprinted and normally refresh automatically
when their inputs change. Manual cache clearing is reserved for maintenance
and tests:

```julia
SurvivorModel.clear_historical_prior_cache!()
```

For a normal weekly run, use the CLI's explicit `--refresh-data` option instead
of clearing caches manually.
