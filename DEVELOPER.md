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
selects the highest posterior-mean candidate probability each week while
respecting market eligibility, current-week first-pick bans, and team
uniqueness. The selected plan is forward-evaluated again to verify its reported
objective.

## Benders backend

`SurvivorSelectionConfig(benders=true)` keeps the weekly team assignment in a
binary HiGHS master built as a JuMP direct model. The master also contains the
exact probability states and their one-hot recurrence gates; its objective is
the base probability contribution plus `theta`, which represents the Hessian
correction. The selected schedule's parameter-gradient, covariance-gradient,
and Hessian recurrences are evaluated directly, without constructing or
optimizing an LP recourse model. Probability and Hessian intervals provide
finite bounds for the two objective components.

If Hessian corrections are modeled and the global bound leaves a gap above the
greedy incumbent, a reverse pass through the recurrence graph generates an
analytic cut at the greedy schedule before the first master solve. Each
one-hot product gate uses the active McCormick side selected by the sign of its
back-propagated objective adjoint. The resulting pick and probability-state
slopes form a dual-feasible supporting cut, tight at the directly evaluated
recourse value. The initial cut is skipped when the global bound already
closes the gap or when there is no Hessian correction to decompose.

Cuts are added to the same direct master between solves. The master is seeded
with the deterministic greedy plan, its probability trajectory, and its
Hessian correction; starts are refreshed from the best feasible plan after
each cut. Caller-supplied optimizers must support direct incremental
constraints and MIP starts. The master is re-solved after each cut; the loop
stops when its upper bound meets the best directly evaluated feasible plan
within tolerance. This is an outer Benders loop and does not depend on solver
callbacks.

`SurvivorSelectionConfig(timeout_seconds=...)` passes a HiGHS `time_limit` for
the extensive-form solve. Benders mode updates the master time limit from the
shared solve budget; deterministic recurrence evaluation and cut generation
also count toward elapsed time. It retains the deterministic greedy plan as a
feasible incumbent, so a timeout returns the best feasible plan seen so far.
The extensive-form optimizer returns its best solver incumbent when available;
a timeout without one is an optimization failure. Debug logging records
extensive-form phase timings, termination and primal status, incumbent
availability and objective, bound, relative gap, and branch-and-bound node
count. For the extensive-form default optimizer, it also enables native HiGHS
root and MIP progress output, routed away from stdout so CLI picks remain
clean.

Benders keeps native HiGHS output silent, even with Debug enabled, and emits
structured diagnostics instead. It logs master model size (variables,
constraint rows, and affine nonzeros); each master solve's status, elapsed
time, base probability objective, Hessian-correction `theta`, bound, relative
gap, node count, simplex iterations, cut count, and the week-ordered team
list; each direct recurrence evaluation's elapsed time, Hessian-correction
objective, and fixed team list; and each analytic cut's intercept and
pick/probability slope ranges. Team lists contain only abbreviations, with no
week labels. Simplex iterations are reported for master solves when the
optimizer exposes the MathOptInterface attribute. All package diagnostics
remain Debug-level; normal library calls stay silent.

`write_survivor_pool_lp` always exports the initial extensive-form HiGHS model,
including when the selection config has `benders=true`.

## Cache maintenance

Historical prior caches are fingerprinted and refresh automatically when
their inputs change. Manual cache clearing is reserved for maintenance and
tests:

```julia
SurvivorModel.clear_historical_prior_cache!()
```

For a normal weekly run, use the CLI's `--refresh-data` option instead of
clearing caches manually.
