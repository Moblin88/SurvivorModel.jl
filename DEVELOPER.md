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
`1/2 * H` correction for the first `K` transitions. The default is three
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

Redundant aggregate recurrences directly constrain the weekly probability sum
through the full horizon and the adjusted objective sum through the retained
Hessian prefix. The probability aggregate is also constrained to be
nonincreasing. Gradient reference states are suffix-pruned to the Hessian
prefix and omitted when no earlier candidate can have a nonzero Gram
interaction. The hybrid switch checks all later reference widths rather than
assuming they decrease monotonically. This pruning is exact; zero-support
states are fixed at zero.

The MILP is warm-started with a deterministic feasible greedy plan that
selects the highest posterior-mean candidate probability each week while
respecting market eligibility and team uniqueness. The selected plan is
forward-evaluated again to verify its reported objective.

`SurvivorSelectionConfig(timeout_seconds=...)` passes a HiGHS `time_limit`.
When the limit is reached with a feasible incumbent, the optimizer returns the
best-known plan; a timeout without a feasible incumbent is an optimization
failure. Debug logging records phase timings, termination and primal status,
incumbent availability and objective, bound, relative gap, and branch-and-bound
node count. With the default optimizer, Debug logging also enables HiGHS root
and MIP progress output, routed away from stdout so CLI picks remain clean;
normal library calls remain silent.

## Cache maintenance

Historical prior caches are fingerprinted and refresh automatically when
their inputs change. Manual cache clearing is reserved for maintenance and
tests:

```julia
SurvivorModel.clear_historical_prior_cache!()
```

For a normal weekly run, use the CLI's `--refresh-data` option instead of
clearing caches manually.
