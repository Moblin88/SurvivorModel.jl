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

`SurvivorSelectionConfig(branch_and_bound=true)` selects an external tree that
certifies the current first pick. The default remains the extensive-form MILP.
`_build_survivor_full_model` and the normal extensive-form solve share the
scalar recurrence builder; branch-and-bound does not duplicate probability,
gradient, projected-gradient, or Hessian recurrences. The model contains all
pick variables and the one-pick-per-week/team-once constraints. Probability
states span the full remaining horizon, while derivative states and Hessian
corrections use `H=min(hessian_weeks, remaining_horizon)`. Fixing path picks
does not shift or shorten H. `--write-model` continues to export the unsolved
extensive-form MILP.

The external tree uses one direct continuous HiGHS model. It solves the root
relaxation, partitions every eligible first-week pick (including zero-valued
LP candidates), and adds path fixings through variable bounds. Before each
child solve, availability and fixed picks tighten probability,
parameter-gradient, projected-gradient, and signed-curvature intervals. The
four existing product-hull rows are rewritten in place from those intervals;
model rows and columns remain fixed, and siblings recompute from root bounds.
The full objective is maximized by minimizing its negative in this tree, then
solver bounds are normalized back to the original maximization convention.

Root and child relaxations set `solver=hipo`, `threads=0`, `parallel=on`,
`presolve=choose`, and `run_crossover=on`. HiGHS native output remains silent.
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
ranking/incumbent updates. Competing first-pick regions are prioritized, with
incumbent-region work interleaved. The current first pick is proven when its
feasible objective is at least the best competing-region upper bound within
the scale-aware `1e-6` tolerance. The returned full schedule is feasible and
its objective is independently evaluated, but future picks are not promised
globally optimal after the first pick is certified. The timeout covers model
construction, relaxations, and certification; on timeout the best feasible
schedule is returned with an explicit unproven warning and pending bounds
retained.

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

## Cache maintenance

Historical prior caches are fingerprinted and refresh automatically when
their inputs change. Manual cache clearing is reserved for maintenance and
tests:

```julia
SurvivorModel.clear_historical_prior_cache!()
```

For a normal weekly run, use the CLI's `--refresh-data` option instead of
clearing caches manually.
