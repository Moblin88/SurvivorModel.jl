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

`SurvivorSelectionConfig(benders_weeks=K)` enables a meet-in-the-middle
decomposition. `K` is clamped to the remaining horizon. The HiGHS direct
master contains every binary pick, all one-pick-per-week and team-once
constraints, and the survival-probability recurrence through the first `K`
weeks only. All parameter/projected-gradient and Hessian states and gates
remain in fully analytic recourse. The gradient switch and reference pruning
use the original full correction horizon, independently of `K`.

The master objective is survival through `K` plus `theta`. Its upper bound
sums omitted survival and all Hessian interval upper bounds, including when
`K` covers the full horizon. Shorter prefixes retain a
`1e-8 * max(1,abs(bound))` numerical
margin. `hessian_weeks` semantics do not change. Tail recurrences are
evaluated analytically from a selected schedule; no recourse LP is built or
solved in the seed or iteration loop.

The initial master cut is derived from the dual of the full extensive-form LP
relaxation. Rows containing recourse variables provide the dual-feasible
projection onto pick and prefix-state variables; master-only assignment and
prefix constraints remain in the master. This LP is solved to optimality,
unless the shared timeout expires, using HiGHS HiPO with `parallel=on`,
automatic thread selection (`threads=0`), and `run_crossover=on`; these
settings are limited to the initial LP relaxation, not the master MIPs. The
solver's dual status and omitted-column stationarity are checked before
constructing the cut. Signed maximization multipliers are `-JuMP.dual(row)`;
equality shadow prices must not be used because they discard the required
sign. Only picks and prefix probability states are master columns. An additional
product-hull-based interval bound constrains `theta`. Two cached
greedy schedules seed analytic recourse cuts: one greedily selects the
highest-mean candidate each week, while the second forces the second-highest-
mean candidate in week one. The analytic reverse-pass construction uses the existing
projection of tail parameter gradients onto games to limit the number of
gradient gates. No reference/Pareto selection or core construction is used.

At each schedule, forward recurrences supply the exact states. The reverse
pass propagates recourse objective adjoints through the probability,
parameter/projected-gradient, Hessian and product-hull recurrences, initialized
with omitted survival terms and `0.5` on every Hessian objective state.
All derivative recurrences propagate through the full correction horizon;
only probability propagation stops at the master split, producing explicit
prefix-probability coefficients alongside pick coefficients. Initial-state and gate RHS
contributions produce the dual intercept. The cut is
`theta <= intercept + pick_slopes'picks + probability_slopes'prefix_probabilities`.
Its intercept comes from the dual RHS, never from subtracting slopes from
an evaluated objective. The existing scale-dependent Benders objective
tolerance checks dual tightness at the generating schedule; a failed
certificate raises an explicit error. Seeds, main schedules and alternative
schedules all use this same analytic cut construction.
Their prefix objectives contain survival only. Master warm starts cover
picks, `theta`, and prefix probabilities; scalar extensive-model warm starts
remain unchanged. Alternative solve results, including the prefix survival
objective, are captured before restoring
forbidden-pick bounds or starts.

For a gate adjoint `a`, the certificate uses the selected recurrence upper
side if `a>0`, the lower side if `a<0`, and the corresponding unselected
pick bounds; collapsed gates are affine pick terms. Reverse substitution
cancels every omitted state column and leaves retained state slopes. All
chosen rows are tight at the generating binary schedule, so RHS-derived
intercept plus slopes equals exact recourse and certifies dual optimality.
The unit suite checks global-cut stationarity, analytic cut tightness and
validity exhaustively over feasible schedules, and intercept invariance
under small objective perturbations. It
covers `K=0`, full `K`, `H=0`, all relative split/horizon positions, both
gradient representations and switch boundaries, and multiple loss states.

At each iteration, the master is solved to optimality and its selected
schedule is evaluated exactly. A zero recourse correction certifies that
schedule as globally optimal. Otherwise its exact objective is a feasible
lower bound; a second master solve forbids its first pick, and its bound is an
upper bound for all alternatives. Comparing those bounds can certify the
current-week pick without proving that the returned future schedule is
globally optimal. If the bounds intersect, an analytic optimality cut is
added and the loop continues. The mode therefore guarantees the optimal
current-week pick, not necessarily the globally optimal complete schedule. If
only one current-week candidate is eligible, the pick is forced and its
feasible greedy witness is returned without constructing the master.

`SurvivorSelectionConfig(timeout_seconds=...)` provides a shared solve budget
for the full LP relaxation and Benders master MIPs. Model construction, deterministic recurrence
evaluation and cut generation also count toward elapsed time; the remaining
budget is reapplied before each solve. The best
feasible full schedule is retained, so timeout returns it without claiming an
optimality proof.
Unexpected LP statuses or failed numerical certificates raise explicit
errors. All survivor optimizations use HiGHS; callers cannot supply a
custom optimizer. Debug logging reports LP-cut provenance, master sizes,
the initial LP solver and barrier/crossover iterations, and a plain progress
table with one row per master or forbidden-pick solve. The `cuts` column
counts cuts present at solve time: the alternative row includes the
main-schedule cut, and the next master row includes both new cuts when bounds
overlap. The table shows each solve's status, selected and forbidden first
picks, best feasible lower bound, master objective and upper bound, exact
schedule objective, recourse correction, cut count, and solve time. Its header
repeats every 20 rows. The table is written to stderr only when package Debug
logging is enabled. Detailed initialization and final proof/timeout records
remain available. Normal library calls stay silent.

`write_survivor_pool_lp` always exports the initial extensive-form HiGHS model
and exits without solving, regardless of `benders_weeks`.

## Cache maintenance

Historical prior caches are fingerprinted and refresh automatically when
their inputs change. Manual cache clearing is reserved for maintenance and
tests:

```julia
SurvivorModel.clear_historical_prior_cache!()
```

For a normal weekly run, use the CLI's `--refresh-data` option instead of
clearing caches manually.
