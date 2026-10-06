struct SurvivorNativeBasis
    owner::HiGHS.Optimizer
    columns::Vector{HiGHS.HighsInt}
    rows::Vector{HiGHS.HighsInt}
    mapping::Vector{Tuple{JuMP.MOI.VariableIndex,Int}}
    row_mapping::Vector{Tuple{JuMP.MOI.ConstraintIndex,Int}}
end

function _survivor_basis_backend(model)
    optimizer = JuMP.backend(model)
    optimizer isa HiGHS.Optimizer ||
        throw(ArgumentError("survivor basis transfer requires a direct HiGHS model"))
    return optimizer
end

function _survivor_basis_mapping(model, optimizer)
    mapping = [(JuMP.index(v), Int(HiGHS.column(optimizer, JuMP.index(v))))
               for v in JuMP.all_variables(model)]
    sort(last.(mapping)) == collect(0:(length(mapping) - 1)) ||
        error("survivor native column mapping is not a bijection")
    return mapping
end

function _survivor_basis_row_mapping(model, optimizer)
    mapping = Tuple{JuMP.MOI.ConstraintIndex,Int}[
        (JuMP.index(ref), Int(HiGHS.row(optimizer, JuMP.index(ref))))
        for (F, S) in JuMP.list_of_constraint_types(model) if F == JuMP.AffExpr
        for ref in JuMP.all_constraints(model, F, S)
    ]
    sort(last.(mapping)) == collect(0:(length(mapping) - 1)) &&
        length(mapping) == HiGHS.Highs_getNumRow(optimizer.inner) ||
        error("survivor native row mapping is not a bijection")
    return mapping
end

function _survivor_validate_basis(columns, rows)
    valid = (
        HiGHS.kHighsBasisStatusLower, HiGHS.kHighsBasisStatusBasic,
        HiGHS.kHighsBasisStatusUpper, HiGHS.kHighsBasisStatusZero,
        HiGHS.kHighsBasisStatusNonbasic,
    )
    all(in(valid), columns) && all(in(valid), rows) ||
        error("HiGHS returned invalid native basis statuses")
    count(==(HiGHS.kHighsBasisStatusBasic), columns) +
        count(==(HiGHS.kHighsBasisStatusBasic), rows) == length(rows) ||
        error("HiGHS returned an incomplete basis")
    return nothing
end

function _survivor_snapshot_basis(model)
    JuMP.termination_status(model) == JuMP.MOI.OPTIMAL &&
        JuMP.primal_status(model) == JuMP.MOI.FEASIBLE_POINT &&
        JuMP.dual_status(model) == JuMP.MOI.FEASIBLE_POINT ||
        error("survivor basis snapshot requires an optimal primal/dual feasible LP")
    optimizer = _survivor_basis_backend(model)
    columns = zeros(HiGHS.HighsInt, HiGHS.Highs_getNumCol(optimizer.inner))
    rows = zeros(HiGHS.HighsInt, HiGHS.Highs_getNumRow(optimizer.inner))
    status = HiGHS.Highs_getBasis(optimizer.inner, columns, rows)
    status == HiGHS.kHighsStatusOk ||
        error("HiGHS_getBasis failed with status $status")
    _survivor_validate_basis(columns, rows)
    return SurvivorNativeBasis(
        optimizer, columns, rows, _survivor_basis_mapping(model, optimizer),
        _survivor_basis_row_mapping(model, optimizer),
    )
end

function _survivor_restore_basis!(model, basis::SurvivorNativeBasis; repair=false)
    optimizer = _survivor_basis_backend(model)
    optimizer === basis.owner ||
        error("survivor basis belongs to a different optimizer")
    length(basis.columns) == HiGHS.Highs_getNumCol(optimizer.inner) &&
        length(basis.rows) == HiGHS.Highs_getNumRow(optimizer.inner) &&
        basis.mapping == _survivor_basis_mapping(model, optimizer) &&
        basis.row_mapping == _survivor_basis_row_mapping(model, optimizer) ||
        error("survivor model structure changed since the basis snapshot")
    _survivor_validate_basis(basis.columns, basis.rows)
    status = HiGHS.Highs_setBasis(optimizer.inner, basis.columns, basis.rows)
    if status != HiGHS.kHighsStatusOk
        repair || error("HiGHS_setBasis failed with status $status")
        @warn "survivor parent basis rejected after coefficient update; using simplex crash basis" status
        HiGHS.Highs_clearSolver(optimizer.inner) == HiGHS.kHighsStatusOk ||
            error("HiGHS could not clear a rejected parent basis")
    end
    return nothing
end

struct SurvivorBranchNode
    id::Int
    region::Int
    path::Vector{Int}
    upper::Float64
    completion_lower_bound::Float64
    completion_ready::Bool
end

function _survivor_tree_plan(tree, indices; expected_objective=nothing)
    length(indices) == tree.number_of_weeks ||
        error("survivor tree witness has the wrong horizon")
    length(unique(tree.data.team[indices])) == length(indices) ||
        error("survivor tree witness reuses a team")
    sort(tree.candidate_positions[indices]) == collect(1:tree.number_of_weeks) ||
        error("survivor tree witness does not pick exactly once per week")
    eligible = _survivor_market_guard_mask(tree.data, tree.state, tree.config)
    all(eligible[indices]) ||
        error("survivor tree witness violates eligibility")
    isempty(intersect(Set(tree.data.team[indices]), Set(values(tree.state.picks_made)))) ||
        error("survivor tree witness reuses a previous pick")
    return _survivor_plan_from_selected_indices(
        tree.data, tree.state, tree.config, tree.inputs, indices,
        tree.number_of_weeks, tree.losses_to_elimination, tree.curvature_weeks,
        tree.gradient_reference_indices; expected_objective,
    )
end

function _survivor_tree_candidates(tree, path, position)
    teams = Set(tree.data.team[path])
    eligible = _survivor_market_guard_mask(tree.data, tree.state, tree.config)
    indices = [i for i in tree.candidate_indices
               if tree.candidate_positions[i] == position && eligible[i] &&
                   tree.original_pick_bounds[i].upper >= 1.0 &&
                   !(tree.data.team[i] in teams)]
    sort!(indices; by=i -> (-tree.inputs.derivatives[i].base_probability, i))
    return indices
end

function _survivor_tree_hull_interval(bounds, key, index)
    kind = first(key)
    if kind in (:probability_sum, :adjusted_sum)
        function total(source, side)
            rounding = side == :lower ? RoundDown : RoundUp
            return setprecision(BigFloat, 128) do
                setrounding(BigFloat, rounding) do
                    value = sum(BigFloat, @view(
                        getproperty(source.candidate_probability, side)[index, :]
                    ))
                    if kind == :adjusted_sum
                        value += BigFloat(0.5) * sum(BigFloat, @view(
                            getproperty(source.candidate_hessian, side)[index, :]
                        ))
                    end
                    Float64(value, rounding)
                end
            end
        end
        return (
            total(bounds.team_conditioned, :lower),
            total(bounds.team_conditioned, :upper),
            total(bounds, :lower), total(bounds, :upper),
        )
    end
    slots = (index, Base.tail(key)...)
    selected = getproperty(bounds.team_conditioned, kind)
    other = getproperty(bounds, kind)
    return (selected.lower[slots...], selected.upper[slots...],
            other.lower[slots...], other.upper[slots...])
end

function _survivor_tree_update_terms!(tree, available)
    conditioned = available === nothing ? deepcopy(tree.bounds) : _survivor_scalar_bounds(
        tree.inputs, tree.candidate_positions, tree.number_of_weeks,
        tree.losses_to_elimination;
        curvature_weeks=tree.curvature_weeks,
        gradient_reference_indices=tree.gradient_reference_indices,
        candidate_teams=tree.data.team, available,
    )
    # Absolute intersections with root intervals, never with the previous sibling.
    for kind in (:probability, :parameter_gradient, :gradient, :hessian,
                 :candidate_probability, :candidate_parameter_gradient,
                 :candidate_gradient, :candidate_hessian)
        current, root = getproperty(conditioned, kind), getproperty(tree.bounds, kind)
        current.lower .= max.(current.lower, root.lower)
        current.upper .= min.(current.upper, root.upper)
        all(current.lower .<= current.upper) ||
            error("survivor conditioned $kind intervals do not intersect root bounds")
        if startswith(String(kind), "candidate_")
            selected = getproperty(conditioned.team_conditioned, kind)
            root_selected = getproperty(tree.bounds.team_conditioned, kind)
            for index in axes(selected.lower, 1)
                if available === nothing || available[index]
                    selectdim(selected.lower, 1, index) .= max.(
                        selectdim(selected.lower, 1, index),
                        selectdim(root_selected.lower, 1, index),
                        selectdim(current.lower, 1, index),
                    )
                    selectdim(selected.upper, 1, index) .= min.(
                        selectdim(selected.upper, 1, index),
                        selectdim(root_selected.upper, 1, index),
                        selectdim(current.upper, 1, index),
                    )
                else
                    selectdim(selected.lower, 1, index) .= selectdim(current.lower, 1, index)
                    selectdim(selected.upper, 1, index) .= selectdim(current.upper, 1, index)
                end
            end
            all(selected.lower .<= selected.upper) ||
                error("survivor conditioned selected $kind intervals are inconsistent")
        end
    end
    function interval!(variable, bounds, slots...)
        JuMP.set_lower_bound(variable, bounds.lower[slots...])
        JuMP.set_upper_bound(variable, bounds.upper[slots...])
        return nothing
    end
    for position in axes(tree.probability, 1), loss in axes(tree.probability, 2)
        interval!(tree.probability[position, loss], conditioned.probability, position, loss)
    end
    for position in axes(tree.hessian, 1), loss in axes(tree.hessian, 2)
        interval!(tree.hessian[position, loss], conditioned.hessian, position, loss)
    end
    for position in eachindex(tree.parameter_gradient),
        loss in axes(tree.parameter_gradient[position], 1),
        parameter in axes(tree.parameter_gradient[position], 2)
        interval!(tree.parameter_gradient[position][loss, parameter],
                  conditioned.parameter_gradient, position, loss, parameter)
    end
    for position in eachindex(tree.gradient)
        states = tree.gradient[position]
        states === nothing && continue
        for loss in axes(states, 1),
            (slot, reference) in enumerate(tree.gradient_reference_indices[position])
            interval!(states[loss, slot], conditioned.gradient, position, loss, reference)
        end
    end
    for hull in tree.model.ext[:survivor_node_hulls]
        lower, upper, all_lower, all_upper =
            _survivor_tree_hull_interval(conditioned, hull.key, hull.index)
        all(isfinite, (lower, upper, all_lower, all_upper)) &&
            all_lower <= lower <= upper <= all_upper ||
            error("survivor conditioned hull intervals must be finite, ordered, and nested")
        JuMP.set_lower_bound(hull.dummy, min(0.0, lower))
        JuMP.set_upper_bound(hull.dummy, max(0.0, upper))
        JuMP.set_normalized_coefficient(hull.lower_row, hull.selected, -lower)
        JuMP.set_normalized_coefficient(hull.upper_row, hull.selected, -upper)
        JuMP.set_normalized_coefficient(hull.other_upper_row, hull.selected, -all_upper)
        JuMP.set_normalized_rhs(hull.other_upper_row, -all_upper)
        JuMP.set_normalized_coefficient(hull.other_lower_row, hull.selected, -all_lower)
        JuMP.set_normalized_rhs(hull.other_lower_row, -all_lower)
    end
    tree.model.ext[:survivor_node_bounds] = conditioned
    return nothing
end

function _survivor_tree_apply_path!(tree, path; tighten=true)
    positions = tree.candidate_positions[path]
    length(unique(positions)) == length(path) &&
        length(unique(tree.data.team[path])) == length(path) || return false
    fixed = Dict(positions .=> path)
    teams = Set(tree.data.team[path])
    eligible = _survivor_market_guard_mask(tree.data, tree.state, tree.config)
    all(i -> eligible[i] && tree.original_pick_bounds[i].upper >= 1.0, path) ||
        return false
    for i in tree.candidate_indices
        chosen = get(fixed, tree.candidate_positions[i], 0)
        lower = max(tree.original_pick_bounds[i].lower, chosen == i ? 1.0 : 0.0)
        upper = chosen != 0 ? (chosen == i ? 1.0 : 0.0) :
            (eligible[i] && !(tree.data.team[i] in teams) ? 1.0 : 0.0)
        upper = min(tree.original_pick_bounds[i].upper, upper)
        lower <= upper || return false
        JuMP.set_lower_bound(tree.selected[i], lower)
        JuMP.set_upper_bound(tree.selected[i], upper)
    end
    feasible = all(
        any(JuMP.upper_bound(tree.selected[i]) > 0.0
            for i in tree.candidate_indices if tree.candidate_positions[i] == p)
        for p in 1:tree.number_of_weeks
    )
    feasible || return false
    if tighten
        available = [JuMP.upper_bound(tree.selected[i]) > 0.0 for i in tree.candidate_indices]
        changed = true
        while changed
            changed = false
            for position in 1:tree.number_of_weeks
                choices = findall(i -> available[i] &&
                    tree.candidate_positions[i] == position, tree.candidate_indices)
                isempty(choices) && return false
                length(choices) == 1 || continue
                forced = only(choices)
                for i in tree.candidate_indices
                    if available[i] && tree.candidate_positions[i] != position &&
                       isequal(tree.data.team[i], tree.data.team[forced])
                        JuMP.lower_bound(tree.selected[i]) > 0.0 && return false
                        available[i] = false
                        JuMP.set_upper_bound(tree.selected[i], 0.0)
                        changed = true
                    end
                end
            end
        end
        _survivor_tree_update_terms!(tree, available)
    else
        _survivor_tree_update_terms!(tree, nothing)
    end
    return true
end

# A feasible dual status alone is not a safe numerical upper bound. Maximize
# the dual Lagrangian over the original finite variable intervals, including
# stationarity residuals; clamp inequality multipliers to their valid signs.
function _survivor_tree_upper_bound(model)
    JuMP.termination_status(model) == JuMP.MOI.OPTIMAL &&
        JuMP.primal_status(model) == JuMP.MOI.FEASIBLE_POINT &&
        JuMP.dual_status(model) == JuMP.MOI.FEASIBLE_POINT ||
        error("survivor node bound requires an optimal primal/dual feasible LP")
    objective_sign = JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? -1.0 : 1.0
    primal = objective_sign * Float64(JuMP.objective_value(model))
    dual = objective_sign * Float64(JuMP.dual_objective_value(model))
    isfinite(primal) && isfinite(dual) ||
        error("survivor LP returned nonfinite objectives")
    isapprox(primal, dual; atol=2e-6, rtol=1e-7) ||
        error("survivor LP primal and dual objectives disagree: $primal vs $dual")
    optimizer = _survivor_basis_backend(model)
    for name in ("max_primal_infeasibility", "max_dual_infeasibility")
        value = Ref{Cdouble}(0.0)
        status = HiGHS.Highs_getDoubleInfoValue(optimizer.inner, name, value)
        status == HiGHS.kHighsStatusOk && isfinite(value[]) && 0 <= value[] <= 1e-7 ||
            error("survivor LP has invalid $name: $(value[]) (status $status)")
    end
    # HiGHS.jl may filter barrier row duals using invalid basis statuses.
    rows = zeros(Cdouble, HiGHS.Highs_getNumRow(optimizer.inner))
    HiGHS.Highs_getSolution(optimizer.inner, C_NULL, C_NULL, C_NULL, rows) ==
        HiGHS.kHighsStatusOk || error("HiGHS could not read barrier row multipliers")
    sense = JuMP.objective_sense(model) == JuMP.MOI.MAX_SENSE ? -1.0 : 1.0
    upper = _survivor_tree_lagrangian_bound(
        model; row_dual=ref -> sense * rows[Int(HiGHS.row(optimizer, JuMP.index(ref))) + 1],
    )
    isfinite(upper) || error("survivor Lagrangian certificate is not finite")
    margin = 1e-8 * max(1.0, abs(upper), abs(primal))
    upper + margin >= primal ||
        error("survivor Lagrangian certificate contradicts the LP primal")
    return nextfloat(max(upper, primal) + margin)
end

function _survivor_tree_lagrangian_bound(model; row_dual::Function=JuMP.dual)
    return setprecision(BigFloat, 128) do
        objective = JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ?
                    -JuMP.objective_function(model) : JuMP.objective_function(model)
        coefficients = Dict(v => BigFloat(c) for (c, v) in JuMP.linear_terms(objective))
        constant = BigFloat(objective.constant)
        for (F, S) in JuMP.list_of_constraint_types(model)
            F == JuMP.AffExpr || continue
            for ref in JuMP.all_constraints(model, F, S)
                object = JuMP.constraint_object(ref)
                multiplier = Float64(row_dual(ref))
                isfinite(multiplier) || error("survivor LP returned a nonfinite multiplier")
                rhs = if object.set isa JuMP.MOI.LessThan
                    multiplier = min(0.0, multiplier)
                    object.set.upper
                elseif object.set isa JuMP.MOI.GreaterThan
                    multiplier = max(0.0, multiplier)
                    object.set.lower
                elseif object.set isa JuMP.MOI.EqualTo
                    object.set.value
                else
                    error("unsupported survivor LP row set $(typeof(object.set))")
                end
                d = BigFloat(multiplier)
                constant += d * (BigFloat(object.func.constant) - BigFloat(rhs))
                for (c, v) in JuMP.linear_terms(object.func)
                    coefficients[v] = get(coefficients, v, BigFloat(0)) + d * BigFloat(c)
                end
            end
        end
        for v in JuMP.all_variables(model)
            lower, higher = JuMP.lower_bound(v), JuMP.upper_bound(v)
            isfinite(lower) && isfinite(higher) ||
                error("survivor bound certificate requires finite variable intervals")
            c = get(coefficients, v, BigFloat(0))
            constant += c * BigFloat(c >= 0 ? higher : lower)
        end
        Float64(constant)
    end
end

function _survivor_tree_assignment(tree)
    values = Float64.(JuMP.value.(tree.selected))
    all(isfinite, values) || error("survivor node has nonfinite pick values")
    all(x -> -1e-7 <= x <= 1 + 1e-7, values) ||
        error("survivor node has pick values outside their bounds")
    fractional = findfirst(1:tree.number_of_weeks) do position
        any(1e-7 < values[i] < 1 - 1e-7 for i in tree.candidate_indices
            if tree.candidate_positions[i] == position)
    end
    fractional !== nothing && return (week=fractional, indices=nothing)
    indices = [i for i in tree.candidate_indices if values[i] >= 1 - 1e-7]
    return (week=nothing, indices=indices)
end

mutable struct SurvivorBranchSummary
    parent::Int
    region::Int
    upper::Float64
    children::Dict{Int,Float64}
    closed_upper::Float64
end

function _survivor_tree_summary(parent, region, upper)
    return SurvivorBranchSummary(parent, region, upper, Dict{Int,Float64}(), -Inf)
end

# Closed descendants contribute only a scalar maximum, never a path or basis.
function _survivor_tree_propagate!(summaries, roots, id; close=false)
    while true
        summary = summaries[id]
        parent = summary.parent
        upper = summary.upper
        if parent == 0
            roots[summary.region] = upper
            close && delete!(summaries, id)
            return nothing
        end
        ancestor = summaries[parent]
        if close
            delete!(ancestor.children, id)
            ancestor.closed_upper = max(ancestor.closed_upper, upper)
            delete!(summaries, id)
        else
            ancestor.children[id] = upper
        end
        ancestor.upper = min(
            ancestor.upper,
            max(ancestor.closed_upper, maximum(values(ancestor.children); init=-Inf)),
        )
        id = parent
        close = isempty(ancestor.children)
    end
end

function _survivor_tree_close!(summaries, roots, id, upper)
    summary = summaries[id]
    isempty(summary.children) || error("cannot close a survivor node with live children")
    summary.upper = min(summary.upper, upper)
    _survivor_tree_propagate!(summaries, roots, id; close=true)
    return nothing
end

function _survivor_tree_branch!(summaries, roots, id, bound, child_ids)
    summary = summaries[id]
    isempty(summary.children) || error("survivor node was already branched")
    summary.upper = min(summary.upper, bound)
    for child in child_ids
        summaries[child] = _survivor_tree_summary(id, summary.region, summary.upper)
        summary.children[child] = summary.upper
    end
    isempty(child_ids) && return _survivor_tree_close!(summaries, roots, id, -Inf)
    _survivor_tree_propagate!(summaries, roots, id)
    return nothing
end

function _survivor_tree_next_node(queue, incumbent_region, iteration, roots=nothing)
    competing = findall(n -> n.region != incumbent_region, queue)
    own = findall(n -> n.region == incumbent_region, queue)
    candidates = !isempty(own) && (iteration % 4 == 0 || isempty(competing)) ?
        own : competing
    region_upper = Dict{Int,Float64}()
    for i in candidates
        n = queue[i]
        region_upper[n.region] = max(get(region_upper, n.region, -Inf), n.upper)
    end
    if roots !== nothing
        for region in keys(region_upper)
            region_upper[region] = roots[region]
        end
    end
    return first(sort(candidates; by=i -> (
        -region_upper[queue[i].region], -queue[i].upper,
        -queue[i].completion_lower_bound, length(queue[i].path), queue[i].id,
    )))
end

function _survivor_tree_tolerance(lower, upper)
    return 1e-6 * max(1.0, abs(lower), isfinite(upper) ? abs(upper) : 1.0)
end

function _survivor_tree_best(incumbent_region, incumbent, region, plan)
    if plan.objective_value > incumbent.objective_value ||
       (plan.objective_value == incumbent.objective_value && region < incumbent_region)
        return region, plan
    end
    return incumbent_region, incumbent
end

function _survivor_tree_finish(plan, proven, reason, upper, nodes, started)
    elapsed = (time_ns() - started) / 1e9
    if proven
        @debug "survivor branch-and-bound certificate" first_pick=only(plan.current_pick.team) objective=plan.objective_value competing_upper=upper nodes elapsed_seconds=elapsed reason guarantee=:first_pick_within_tolerance
    else
        @warn "survivor branch-and-bound time limit; first pick is unproven" first_pick=only(plan.current_pick.team) objective=plan.objective_value competing_upper=upper nodes elapsed_seconds=elapsed reason
    end
    return plan
end

function _optimize_survivor_branch_and_bound!(
    tree;
    observer::Function=event -> nothing,
    remaining_time::Function=() -> _survivor_benders_remaining_time(
        tree.config, tree.budget_started_at,
    ),
    tighten_terms::Bool=true,
)
    model = tree.model
    started = tree.budget_started_at
    incumbent = _survivor_tree_plan(tree, tree.warm_start.selected)
    @debug "survivor branch-and-bound initial incumbent" objective=incumbent.objective_value first_pick=only(incumbent.current_pick.team) schedule=collect(zip(incumbent.selections.week, incumbent.selections.team))
    regions = _survivor_tree_candidates(tree, Int[], 1)
    incumbent_region = tree.warm_start.selected[1]
    remaining() = remaining_time()
    expired() = remaining() !== nothing && remaining() <= 0.0
    initial_upper = _survivor_benders_objective_upper_bound(
        tree.bounds, tree.number_of_weeks, tree.losses_to_elimination,
        tree.curvature_weeks, 0,
    )
    upper = Dict(region => initial_upper for region in regions)
    summaries = Dict{Int,SurvivorBranchSummary}()
    queue = SurvivorBranchNode[]
    function finish(plan, proven, reason, competing_upper, nodes)
        observer((; kind=:finish, proven, reason, upper=competing_upper, nodes, plan,
                  retained=length(summaries), pending=[n.id for n in queue],
                  root_bounds=copy(upper)))
        return _survivor_tree_finish(plan, proven, reason, competing_upper, nodes, started)
    end
    length(regions) == 1 &&
        return finish(incumbent, true, :forced_first_pick, -Inf, 0)
    expired() && return finish(incumbent, false, :build_timeout, initial_upper, 0)
    debug = _survivor_debug_logging_enabled()
    JuMP.set_optimizer_attribute(model, "solver", "hipo")
    JuMP.set_optimizer_attribute(model, "threads", 0)
    JuMP.set_optimizer_attribute(model, "parallel", "on")
    JuMP.set_optimizer_attribute(model, "run_crossover", "off")
    JuMP.set_optimizer_attribute(model, "presolve", "on")
    # HiGHS 1.15.1 recovers incorrect presolved HiPO duals for maximization
    # without crossover. Equivalent minimization avoids this on many models.
    if JuMP.objective_sense(model) == JuMP.MOI.MAX_SENSE
        JuMP.set_objective_function(model, -JuMP.objective_function(model))
        JuMP.set_objective_sense(model, JuMP.MOI.MIN_SENSE)
    end
    function solve_relaxation!()
        JuMP.optimize!(model)
        status = JuMP.termination_status(model)
        status in (JuMP.MOI.NUMERICAL_ERROR, JuMP.MOI.OTHER_ERROR) && expired() &&
            return JuMP.MOI.TIME_LIMIT
        if status in (JuMP.MOI.NUMERICAL_ERROR, JuMP.MOI.OTHER_ERROR) &&
           JuMP.get_optimizer_attribute(model, "presolve") == "on" && !expired()
            @warn "survivor HiPO relaxation failed with presolve; retrying HiPO without presolve" status raw_status=JuMP.raw_status(model)
            JuMP.set_optimizer_attribute(model, "presolve", "off")
            optimizer = _survivor_basis_backend(model)
            HiGHS.Highs_clearSolver(optimizer.inner) == HiGHS.kHighsStatusOk ||
                error("HiGHS could not clear a failed HiPO solve")
            retry_remaining = remaining()
            retry_remaining === nothing ||
                JuMP.set_time_limit_sec(model, max(1e-9, retry_remaining))
            JuMP.optimize!(model)
        end
        status = JuMP.termination_status(model)
        status in (JuMP.MOI.NUMERICAL_ERROR, JuMP.MOI.OTHER_ERROR) && expired() &&
            return JuMP.MOI.TIME_LIMIT
        return status
    end
    @debug "survivor branch-and-bound root" hessian_weeks=tree.curvature_weeks horizon=tree.number_of_weeks solver=:hipo crossover=:off presolve=:on variables=JuMP.num_variables(model)
    root_remaining = remaining()
    root_remaining === nothing || JuMP.set_time_limit_sec(model, max(1e-9, root_remaining))
    root_started = time_ns()
    status = solve_relaxation!()
    @debug "survivor branch-and-bound root solved" status seconds=(time_ns() - root_started) / 1e9 barrier_iterations=_survivor_benders_barrier_iterations(model) simplex_iterations=_survivor_benders_simplex_iterations(model)
    observer((; kind=:root, status, seconds=(time_ns() - started) / 1e9,
              barrier_iterations=_survivor_benders_barrier_iterations(model),
              simplex_iterations=_survivor_benders_simplex_iterations(model)))
    status == JuMP.MOI.TIME_LIMIT &&
        return finish(incumbent, false, :root_timeout, initial_upper, 0)
    status == JuMP.MOI.OPTIMAL ||
        error("survivor root LP failed with status $status despite a feasible witness")
    root_upper = _survivor_tree_upper_bound(model)
    for region in regions
        upper[region] = root_upper
    end
    root_upper + _survivor_tree_tolerance(incumbent.objective_value, root_upper) >= incumbent.objective_value ||
        error("survivor root bound contradicts the greedy witness")
    root_assignment = _survivor_tree_assignment(tree)
    observer((kind=:root_optimum, upper=root_upper, assignment=root_assignment))
    if root_assignment.week === nothing
        plan = _survivor_tree_plan(
            tree, root_assignment.indices; expected_objective=-JuMP.objective_value(model),
        )
        root_upper - plan.objective_value <= _survivor_tree_tolerance(plan.objective_value, root_upper) ||
            error("survivor integral root does not close its numerical bound")
        incumbent_region, incumbent = _survivor_tree_best(
            incumbent_region, incumbent,
            only(i for i in root_assignment.indices if tree.candidate_positions[i] == 1), plan,
        )
        return finish(incumbent, true, :integral_root, root_upper, 0)
    end
    queue = [SurvivorBranchNode(
        id, region, [region], root_upper,
        -Inf, false,
    ) for (id, region) in enumerate(regions)]
    next_id = length(queue)
    summaries = Dict(n.id => _survivor_tree_summary(0, n.region, n.upper) for n in queue)
    upper = Dict(region => root_upper for region in regions)
    observer((kind=:partition, nodes=copy(queue)))
    @debug "survivor branch-and-bound root partition" regions=length(regions) queue=length(queue) upper=root_upper child_solver=:hipo crossover=:off presolve=JuMP.get_optimizer_attribute(model, "presolve")
    nodes = 0
    while true
        competing = maximum((upper[a] for a in regions if a != incumbent_region); init=-Inf)
        tolerance = _survivor_tree_tolerance(incumbent.objective_value, competing)
        if incumbent.objective_value >= competing - tolerance
            return finish(incumbent, true, :region_certificate, competing, nodes)
        end
        isempty(queue) && error("survivor exhausted tree has inconsistent region certificates")
        expired() && return finish(incumbent, false, :node_timeout, competing, nodes)
        for i in eachindex(queue)
            node = queue[i]
            node.completion_ready && continue
            expired() && break
            if node.upper <= incumbent.objective_value +
                             _survivor_tree_tolerance(incumbent.objective_value, node.upper)
                continue
            end
            completion = _survivor_greedy_selected_indices(
                tree.data, tree.state, tree.config, tree.inputs;
                fixed_indices=node.path, expired,
            )
            completion === nothing && expired() && break
            plan = completion === nothing ? nothing : _survivor_tree_plan(tree, completion)
            node = SurvivorBranchNode(
                node.id, node.region, node.path, node.upper,
                plan === nothing ? -Inf : plan.objective_value, true,
            )
            queue[i] = node
            if plan !== nothing
                plan.objective_value <= node.upper +
                    _survivor_tree_tolerance(plan.objective_value, node.upper) ||
                    error("survivor node completion contradicts its inherited upper bound")
                improved = plan.objective_value > incumbent.objective_value
                if improved
                    @debug "survivor branch-and-bound incumbent improved" node=node.id previous_objective=incumbent.objective_value objective=plan.objective_value fixed_indices=node.path schedule=collect(zip(plan.selections.week, plan.selections.team))
                end
                incumbent_region, incumbent = _survivor_tree_best(
                    incumbent_region, incumbent, node.region, plan,
                )
                observer((; kind=:node_completion, node, plan, improved))
            end
            observer((; kind=:node_ranked, node))
        end
        competing = maximum((upper[a] for a in regions if a != incumbent_region); init=-Inf)
        tolerance = _survivor_tree_tolerance(incumbent.objective_value, competing)
        if incumbent.objective_value >= competing - tolerance
            return finish(incumbent, true, :region_certificate, competing, nodes)
        end
        expired() && return finish(incumbent, false, :node_timeout, competing, nodes)
        node_index = _survivor_tree_next_node(queue, incumbent_region, nodes + 1, upper)
        node = queue[node_index]
        deleteat!(queue, node_index)
        if node.upper <= incumbent.objective_value +
                         _survivor_tree_tolerance(incumbent.objective_value, node.upper)
            _survivor_tree_close!(summaries, upper, node.id, node.upper)
            continue
        end
        if !_survivor_tree_apply_path!(tree, node.path; tighten=tighten_terms)
            _survivor_tree_close!(summaries, upper, node.id, -Inf)
            continue
        end
        node.completion_ready || error("survivor selected node has no completion ranking")
        observer((; kind=:node_start, node))
        if expired()
            push!(queue, node)
            competing = maximum((upper[a] for a in regions if a != incumbent_region); init=-Inf)
            return finish(incumbent, false, :node_timeout, competing, nodes)
        end
        remaining() === nothing || JuMP.set_time_limit_sec(model, max(1e-9, remaining()))
        solve_started = time_ns()
        status = solve_relaxation!()
        nodes += 1
        observer((; kind=:node_solve, node, status,
                  seconds=(time_ns() - solve_started) / 1e9,
                  iterations=_survivor_benders_barrier_iterations(model),
                  simplex_iterations=_survivor_benders_simplex_iterations(model)))
        if status == JuMP.MOI.TIME_LIMIT
            push!(queue, node)
            @debug "survivor branch-and-bound interrupted HiPO" node=node.id region=tree.data.team[node.region] queue=length(queue) inherited_upper=node.upper seconds=(time_ns() - solve_started) / 1e9 iterations=_survivor_benders_barrier_iterations(model)
            competing = maximum((upper[a] for a in regions if a != incumbent_region); init=-Inf)
            return finish(incumbent, false, :interrupted_node, competing, nodes)
        elseif status == JuMP.MOI.INFEASIBLE
            for plan in (incumbent,)
                all(tree.data.team[i] == only(plan.selections.team[
                    plan.selections.week .== tree.data.week[i],
                ]) for i in node.path) &&
                    error("survivor infeasible LP contradicts a known feasible witness")
            end
            !isfinite(node.completion_lower_bound) ||
                error("survivor infeasible LP contradicts its feasible completion")
            # A completed default-rank heuristic fails only when its exact
            # week/team matching test proves no schedule extends this path.
            JuMP.dual_status(model) == JuMP.MOI.INFEASIBILITY_CERTIFICATE ||
                (node.completion_ready && node.completion_lower_bound == -Inf) ||
                error("survivor infeasible node has no supported infeasibility certificate")
            _survivor_tree_close!(summaries, upper, node.id, -Inf)
            continue
        elseif status != JuMP.MOI.OPTIMAL
            error("survivor child LP failed with status $status")
        end
        bound = min(node.upper, _survivor_tree_upper_bound(model))
        assignment = _survivor_tree_assignment(tree)
        observer((; kind=:node_optimum, node, bound, assignment))
        if assignment.week === nothing
            plan = _survivor_tree_plan(
                tree, assignment.indices; expected_objective=-JuMP.objective_value(model),
            )
            bound + tolerance >= plan.objective_value ||
                error("survivor inherited bound contradicts an integral witness")
            bound - plan.objective_value <= _survivor_tree_tolerance(plan.objective_value, bound) ||
                error("survivor integral node does not close its numerical bound")
            incumbent_region, incumbent = _survivor_tree_best(
                incumbent_region, incumbent, node.region, plan,
            )
            leaf_upper = nextfloat(plan.objective_value +
                                  1e-8 * max(1.0, abs(plan.objective_value)))
            _survivor_tree_close!(summaries, upper, node.id, leaf_upper)
        elseif bound <= incumbent.objective_value +
                        _survivor_tree_tolerance(incumbent.objective_value, bound)
            _survivor_tree_close!(summaries, upper, node.id, bound)
        else
            assignment.week in tree.candidate_positions[node.path] &&
                error("survivor LP is fractional in a fixed week")
            child_ids = Int[]
            for i in _survivor_tree_candidates(tree, node.path, assignment.week)
                next_id += 1
                push!(child_ids, next_id)
                push!(queue, SurvivorBranchNode(
                    next_id, node.region, [node.path; i], bound,
                    -Inf, false,
                ))
            end
            _survivor_tree_branch!(summaries, upper, node.id, bound, child_ids)
        end
        observer((kind=:queue, nodes=copy(queue), retained=length(summaries),
                  parents=Dict(id => s.parent for (id, s) in summaries),
                  upper=copy(upper)))
        if debug
            competing = maximum((upper[a] for a in regions if a != incumbent_region); init=-Inf)
            if nodes % 25 == 1
                println(stderr)
                header = _survivor_tree_progress_header()
                println(stderr, header)
                println(stderr, repeat("-", length(header)))
            end
            println(stderr, _survivor_tree_progress_row((
                node.id, tree.data.team[node.region], length(node.path),
                something(assignment.week, "-"), length(queue),
                only(incumbent.current_pick.team),
                _survivor_benders_progress_value(incumbent.objective_value),
                _survivor_benders_progress_value(competing),
                _survivor_benders_progress_value(bound), status,
                _survivor_benders_barrier_iterations(model),
                _survivor_benders_progress_value((time_ns() - solve_started) / 1e9),
            )))
            flush(stderr)
        end
    end
end
function _survivor_tree_progress_row(values)
    widths = (7, 6, 5, 4, 7, 5, 11, 11, 11, 18, 9, 11)
    return join((
        index in (2, 6, 10) ?
        rpad(string(value), widths[index]) :
        lpad(string(value), widths[index])
        for (index, value) in enumerate(values)
    ), " | ")
end

function _survivor_tree_progress_header()
    return _survivor_tree_progress_row((
        "node", "region", "depth", "week", "queue", "first",
        "best-LB", "other-UB", "node-UB", "status", "IPM", "seconds",
    ))
end

function _build_survivor_full_model(data, state, config, inputs; kwargs...)
    return _optimize_survivor_expected_weeks_scalar_milp(
        data, state, config, inputs; build_only=true, kwargs...,
    )
end
