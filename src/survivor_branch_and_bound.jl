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

struct SurvivorTreeDualBound
    upper::Union{Nothing,Float64}
    reason::Symbol
end

struct SurvivorTreeAssignmentGuidance
    week::Union{Nothing,Int}
    indices::Union{Nothing,Vector{Int}}
    reason::Symbol
end

function _survivor_tree_highs_backend(model)
    optimizer = JuMP.backend(model)
    optimizer isa HiGHS.Optimizer ||
        throw(ArgumentError(
            "survivor HiPO branch-and-bound requires a direct HiGHS optimizer",
        ))
    return optimizer
end

function _survivor_tree_solver_upper_bound(model, status)
    JuMP.result_count(model) > 0 ||
        return SurvivorTreeDualBound(nothing, :no_solver_result)

    JuMP.dual_status(model) == JuMP.MOI.FEASIBLE_POINT ||
        return SurvivorTreeDualBound(nothing, :dual_not_feasible)

    optimizer = _survivor_tree_highs_backend(model)
    native_dual_status = Ref{HiGHS.HighsInt}(HiGHS.kHighsSolutionStatusNone)
    dual_info_status = HiGHS.Highs_getIntInfoValue(
        optimizer.inner, "dual_solution_status", native_dual_status,
    )
    dual_info_status == HiGHS.kHighsStatusOk &&
        native_dual_status[] == HiGHS.kHighsSolutionStatusFeasible ||
        return SurvivorTreeDualBound(nothing, :native_dual_not_feasible)

    if status != JuMP.MOI.OPTIMAL
        dual_feasibility_tolerance = Ref{Cdouble}(NaN)
        dual_residual_tolerance = Ref{Cdouble}(NaN)
        feasibility_option_status = HiGHS.Highs_getDoubleOptionValue(
            optimizer.inner, "dual_feasibility_tolerance", dual_feasibility_tolerance,
        )
        residual_option_status = HiGHS.Highs_getDoubleOptionValue(
            optimizer.inner, "dual_residual_tolerance", dual_residual_tolerance,
        )
        feasibility_option_status == HiGHS.kHighsStatusOk &&
            isfinite(dual_feasibility_tolerance[]) ||
            return SurvivorTreeDualBound(nothing, :dual_feasibility_tolerance_unavailable)
        residual_option_status == HiGHS.kHighsStatusOk &&
            isfinite(dual_residual_tolerance[]) ||
            return SurvivorTreeDualBound(nothing, :dual_residual_tolerance_unavailable)

        dual_infeasibility = Ref{Cdouble}(NaN)
        dual_residual = Ref{Cdouble}(NaN)
        infeasibility_status = HiGHS.Highs_getDoubleInfoValue(
            optimizer.inner, "max_dual_infeasibility", dual_infeasibility,
        )
        residual_status = HiGHS.Highs_getDoubleInfoValue(
            optimizer.inner, "max_dual_residual_error", dual_residual,
        )
        infeasibility_status == HiGHS.kHighsStatusOk &&
            isfinite(dual_infeasibility[]) &&
            dual_infeasibility[] <= dual_feasibility_tolerance[] ||
            return SurvivorTreeDualBound(nothing, :dual_infeasibility_exceeds_tolerance)
        residual_status == HiGHS.kHighsStatusOk &&
            isfinite(dual_residual[]) &&
            dual_residual[] <= dual_residual_tolerance[] ||
            return SurvivorTreeDualBound(nothing, :dual_residual_exceeds_tolerance)
    end

    solver_bound = Float64(JuMP.objective_bound(model))
    isfinite(solver_bound) ||
        return SurvivorTreeDualBound(nothing, :solver_dual_bound_nonfinite)
    upper = JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ?
            -solver_bound : solver_bound
    return SurvivorTreeDualBound(upper, :solver_dual_bound)
end

function _survivor_tree_assignment_guidance(tree, path)
    JuMP.result_count(tree.model) > 0 ||
        return SurvivorTreeAssignmentGuidance(nothing, nothing, :no_solver_result)
    JuMP.primal_status(tree.model) == JuMP.MOI.FEASIBLE_POINT ||
        return SurvivorTreeAssignmentGuidance(nothing, nothing, :no_feasible_primal)
    values = Float64.(JuMP.value.(tree.selected))
    all(isfinite, values) ||
        return SurvivorTreeAssignmentGuidance(nothing, nothing, :nonfinite_pick_values)
    all(value -> -1e-7 <= value <= 1 + 1e-7, values) ||
        return SurvivorTreeAssignmentGuidance(nothing, nothing, :pick_value_out_of_bounds)

    fixed_positions = Set(tree.candidate_positions[path])
    for index in path
        position = tree.candidate_positions[index]
        values[index] >= 1 - 1e-7 ||
            return SurvivorTreeAssignmentGuidance(nothing, nothing, :fixed_pick_mismatch)
        all(values[other] <= 1e-7 for other in tree.candidate_indices
            if tree.candidate_positions[other] == position && other != index) ||
            return SurvivorTreeAssignmentGuidance(nothing, nothing, :fixed_week_mismatch)
    end
    for position in 1:tree.number_of_weeks
        position in fixed_positions && continue
        any(1e-7 < values[index] < 1 - 1e-7 for index in tree.candidate_indices
            if tree.candidate_positions[index] == position) &&
            return SurvivorTreeAssignmentGuidance(position, nothing, :fractional_week)
    end

    indices = Int[]
    for position in 1:tree.number_of_weeks
        selected = [index for index in tree.candidate_indices
                    if tree.candidate_positions[index] == position &&
                       values[index] >= 1 - 1e-7]
        length(selected) == 1 ||
            return SurvivorTreeAssignmentGuidance(nothing, nothing, :week_not_one_hot)
        all(values[index] <= 1e-7 for index in tree.candidate_indices
            if tree.candidate_positions[index] == position && index != only(selected)) ||
            return SurvivorTreeAssignmentGuidance(nothing, nothing, :week_not_one_hot)
        push!(indices, only(selected))
    end
    length(unique(tree.data.team[indices])) == tree.number_of_weeks ||
        return SurvivorTreeAssignmentGuidance(nothing, nothing, :team_reused)
    all(index -> index in indices, path) ||
        return SurvivorTreeAssignmentGuidance(nothing, nothing, :fixed_pick_mismatch)
    eligible = _survivor_market_guard_mask(tree.data, tree.state, tree.config)
    all(eligible[indices]) ||
        return SurvivorTreeAssignmentGuidance(nothing, nothing, :ineligible_pick)
    return SurvivorTreeAssignmentGuidance(nothing, indices, :validated_assignment)
end

function _survivor_tree_fallback_week(tree, path)
    fixed_positions = Set(tree.candidate_positions[path])
    return findfirst(position -> !(position in fixed_positions), 1:tree.number_of_weeks)
end

function _survivor_tree_can_fallback(status)
    return status in (
        JuMP.MOI.NUMERICAL_ERROR, JuMP.MOI.OTHER_ERROR,
        JuMP.MOI.ITERATION_LIMIT, JuMP.MOI.SLOW_PROGRESS,
        JuMP.MOI.INFEASIBLE_OR_UNBOUNDED,
    )
end

function _survivor_tree_must_stop(status)
    return status in (
        JuMP.MOI.TIME_LIMIT, JuMP.MOI.INTERRUPTED, JuMP.MOI.MEMORY_LIMIT,
        JuMP.MOI.NODE_LIMIT, JuMP.MOI.SOLUTION_LIMIT, JuMP.MOI.NORM_LIMIT,
        JuMP.MOI.OBJECTIVE_LIMIT, JuMP.MOI.OTHER_LIMIT,
    )
end

function _survivor_tree_merge_upper(inherited, solver_upper, feasible_values, context)
    upper = min(inherited, solver_upper === nothing ? inherited : solver_upper)
    for feasible in feasible_values
        isfinite(feasible) || continue
        upper + _survivor_tree_tolerance(feasible, upper) >= feasible ||
            error("survivor $context dual bound contradicts a feasible schedule")
        upper = min(inherited, max(upper, feasible))
    end
    return upper
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

# Closed descendants contribute only a scalar maximum, never full path state.
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

function _survivor_tree_tighten!(summaries, roots, id, upper)
    summary = summaries[id]
    isempty(summary.children) || error("cannot tighten a branched survivor node")
    summary.upper = min(summary.upper, upper)
    _survivor_tree_propagate!(summaries, roots, id)
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

function _survivor_tree_diagnostic(status, bound_reason, guidance_reason)
    parts = String[]
    status == JuMP.MOI.OPTIMAL || push!(parts, lowercase(string(status)))
    bound_reason == :solver_dual_bound || push!(parts, string(bound_reason))
    guidance_reason in (:validated_assignment, :fractional_week) ||
        push!(parts, string(guidance_reason))
    unique!(parts)
    return isempty(parts) ? "-" : join(parts, "/")
end

function _survivor_tree_progress_cell(value, width)
    text = replace(string(value), '\n' => ' ', '\r' => ' ')
    if length(text) > width
        text = width <= 3 ? first(text, width) : string(first(text, width - 3), "...")
    end
    return text
end

function _survivor_tree_progress_row(values)
    widths = (6, 8, 4, 4, 5, 7, 9, 9, 9, 11, 16, 40, 6, 6, 8)
    length(values) == length(widths) ||
        throw(ArgumentError("survivor progress row has the wrong number of columns"))
    left_aligned = (2, 6, 10, 11, 12)
    return join((
        let text = _survivor_tree_progress_cell(value, widths[index])
            index in left_aligned ? rpad(text, widths[index]) : lpad(text, widths[index])
        end
        for (index, value) in enumerate(values)
    ), " | ")
end

function _survivor_tree_progress_header()
    return _survivor_tree_progress_row((
        "node", "region", "depth", "week", "queue", "first",
        "best-LB", "other-UB", "node-UB", "action", "bound-source", "reason",
        "IPM", "XO", "seconds",
    ))
end

function _survivor_tree_emit_progress!(row_count, values)
    row_count[] += 1
    if row_count[] % 25 == 1
        row_count[] == 1 || println(stderr)
        header = _survivor_tree_progress_header()
        println(stderr, header)
        println(stderr, repeat("-", length(header)))
    end
    println(stderr, _survivor_tree_progress_row(values))
    flush(stderr)
    return nothing
end

function _survivor_tree_finish(plan, proven, reason, upper, nodes, started)
    elapsed = (time_ns() - started) / 1.0e9
    if _survivor_debug_logging_enabled()
        rows = Ref(0)
        first_pick = only(plan.current_pick.team)
        _survivor_tree_emit_progress!(rows, (
            "ROOT", first_pick, 0, "-", 0, first_pick,
            _survivor_progress_value(plan.objective_value), "-",
            _survivor_progress_value(upper), "FORCED", "matching",
            string(reason), "-", "-", 0.0,
        ))
        _survivor_tree_emit_progress!(rows, (
            "FINAL", first_pick, "-", "-", "-", first_pick,
            _survivor_progress_value(plan.objective_value),
            _survivor_progress_value(upper), _survivor_progress_value(upper),
            proven ? "PROVEN" : "UNPROVEN", "region_upper", string(reason),
            "-", "-", elapsed,
        ))
    elseif !proven
        @warn "survivor branch-and-bound stopped before proving the first pick" first_pick=only(plan.current_pick.team) objective=plan.objective_value competing_upper=upper nodes elapsed_seconds=elapsed reason
    end
    return plan
end

function _optimize_survivor_branch_and_bound!(
    tree;
    observer::Function=event -> nothing,
    remaining_time::Function=() -> _survivor_remaining_time(
        tree.config, tree.budget_started_at,
    ),
    optimize_relaxation!::Function=JuMP.optimize!,
)
    model = tree.model
    started = tree.budget_started_at
    incumbent = _survivor_tree_plan(tree, tree.warm_start.selected)
    regions = _survivor_tree_candidates(tree, Int[], 1)
    isempty(regions) && error("survivor tree has no eligible first-pick regions")
    incumbent_region = tree.warm_start.selected[1]
    remaining() = remaining_time()
    expired() = remaining() !== nothing && remaining() <= 0.0
    initial_upper = _survivor_tree_global_upper_bound(
        tree.bounds, tree.number_of_weeks, tree.losses_to_elimination,
        tree.curvature_weeks,
    )
    upper = Dict(region => initial_upper for region in regions)
    summaries = Dict{Int,SurvivorBranchSummary}()
    queue = SurvivorBranchNode[]
    nodes = 0
    debug = _survivor_debug_logging_enabled()
    table_rows = Ref(0)

    function competing_upper()
        return maximum(
            (upper[region] for region in regions if region != incumbent_region);
            init=-Inf,
        )
    end

    function emit_row(node, region, depth, week, node_upper, action, source,
                      reason, iterations, crossover_iterations, seconds;
                      queue_size=length(queue), other_upper=competing_upper())
        debug || return nothing
        _survivor_tree_emit_progress!(table_rows, (
            node, region, depth, week, queue_size,
            only(incumbent.current_pick.team),
            _survivor_progress_value(incumbent.objective_value),
            _survivor_progress_value(other_upper),
            _survivor_progress_value(node_upper), action, source, reason,
            _survivor_progress_value(iterations),
            _survivor_progress_value(crossover_iterations),
            _survivor_progress_value(seconds),
        ))
        return nothing
    end

    function finish(plan, proven, reason, competing, processed_nodes)
        elapsed = (time_ns() - started) / 1.0e9
        observer((; kind=:finish, proven, reason, upper=competing,
                  nodes=processed_nodes, plan, retained=length(summaries),
                  pending=[node.id for node in queue], root_bounds=copy(upper)))
        emit_row(
            "FINAL", only(plan.current_pick.team), "-", "-", competing,
            proven ? "PROVEN" : "UNPROVEN", "region_upper", string(reason),
            nothing, nothing, elapsed; other_upper=competing,
        )
        if !debug && !proven
            @warn "survivor branch-and-bound stopped before proving the first pick" first_pick=only(plan.current_pick.team) objective=plan.objective_value competing_upper=competing nodes=processed_nodes elapsed_seconds=elapsed reason
        end
        return plan
    end

    if length(regions) == 1
        emit_row(
            "ROOT", String(tree.data.team[only(regions)]), 0, "-", initial_upper,
            "FORCED", "matching", "forced_first_pick", nothing, nothing, 0.0;
            queue_size=0, other_upper=-Inf,
        )
        return finish(incumbent, true, :forced_first_pick, -Inf, 0)
    end
    expired() && return finish(incumbent, false, :build_timeout, initial_upper, 0)

    JuMP.set_optimizer_attribute(model, "output_flag", false)
    JuMP.set_optimizer_attribute(model, "log_to_console", false)
    JuMP.set_optimizer_attribute(model, "log_file", "")
    JuMP.set_optimizer_attribute(model, "log_dev_level", 0)
    JuMP.set_optimizer_attribute(model, "solver", "hipo")
    JuMP.set_optimizer_attribute(model, "threads", 0)
    JuMP.set_optimizer_attribute(model, "parallel", "on")
    JuMP.set_optimizer_attribute(model, "run_crossover", "on")
    JuMP.set_optimizer_attribute(model, "presolve", "choose")
    if JuMP.objective_sense(model) == JuMP.MOI.MAX_SENSE
        JuMP.set_objective_function(model, -JuMP.objective_function(model))
        JuMP.set_objective_sense(model, JuMP.MOI.MIN_SENSE)
    end

    function solve_relaxation!()
        optimize_relaxation!(model)
        status = JuMP.termination_status(model)
        return status in (JuMP.MOI.NUMERICAL_ERROR, JuMP.MOI.OTHER_ERROR) && expired() ?
            JuMP.MOI.TIME_LIMIT : status
    end

    root_remaining = remaining()
    root_remaining === nothing || JuMP.set_time_limit_sec(model, max(1e-9, root_remaining))
    root_started = time_ns()
    status = solve_relaxation!()
    root_seconds = (time_ns() - root_started) / 1.0e9
    root_iterations = _survivor_barrier_iterations(model)
    root_crossover_iterations = _survivor_crossover_iterations(model)
    observer((; kind=:root, status, seconds=root_seconds,
              barrier_iterations=root_iterations,
              crossover_iterations=root_crossover_iterations))
    status == JuMP.MOI.INFEASIBLE &&
        error("survivor root LP is infeasible despite a feasible schedule")
    status == JuMP.MOI.OPTIMAL || _survivor_tree_can_fallback(status) ||
        _survivor_tree_must_stop(status) ||
        error("survivor root LP failed with unexpected status $status")

    root_result = _survivor_tree_solver_upper_bound(model, status)
    root_assignment = _survivor_tree_assignment_guidance(tree, Int[])
    root_upper = _survivor_tree_merge_upper(
        initial_upper, root_result.upper, (incumbent.objective_value,), "root",
    )
    root_source = root_result.upper !== nothing && root_result.upper <= initial_upper ?
        :solver_dual : :global_interval
    root_plan = nothing
    previous_incumbent = incumbent.objective_value
    if root_assignment.indices !== nothing && root_assignment.week === nothing
        root_plan = _survivor_tree_plan(tree, root_assignment.indices)
        root_upper = _survivor_tree_merge_upper(
            root_upper, nothing, (root_plan.objective_value,), "root",
        )
        region = only(index for index in root_assignment.indices
                      if tree.candidate_positions[index] == 1)
        incumbent_region, incumbent = _survivor_tree_best(
            incumbent_region, incumbent, region, root_plan,
        )
    end
    for region in regions
        upper[region] = root_upper
    end
    root_upper + _survivor_tree_tolerance(incumbent.objective_value, root_upper) >=
        incumbent.objective_value ||
        error("survivor root bound contradicts the greedy witness")
    root_reason = _survivor_tree_diagnostic(
        status, root_result.reason, root_assignment.reason,
    )
    emit_row(
        "ROOT", "all", 0, something(root_assignment.week, "-"), root_upper,
        "ROOT", string(root_source), root_reason, root_iterations,
        root_crossover_iterations, root_seconds; queue_size=length(regions),
    )
    if root_plan !== nothing && incumbent.objective_value > previous_incumbent
        emit_row(
            "ROOT", String(incumbent.current_pick.team), 0, "-",
            root_plan.objective_value, "INCUMBENT", "root_primal",
            "validated_assignment", root_iterations, root_crossover_iterations,
            root_seconds; queue_size=length(regions),
        )
    end
    if !debug && (root_result.upper === nothing || status != JuMP.MOI.OPTIMAL)
        @warn "survivor HiPO root used a fallback bound or did not solve to optimality" status bound_source=root_source bound_reason=root_result.reason upper=root_upper
    end
    root_event_kind = status == JuMP.MOI.OPTIMAL &&
                      root_result.upper !== nothing ?
                      :root_optimum : :root_fallback
    observer((kind=root_event_kind, upper=root_upper,
              assignment=root_assignment, bound_source=root_source, status))

    if _survivor_tree_must_stop(status) || expired()
        if incumbent.objective_value >= root_upper -
           _survivor_tree_tolerance(incumbent.objective_value, root_upper)
            return finish(incumbent, true, :root_bound_certificate, root_upper, 0)
        end
        reason = status == JuMP.MOI.TIME_LIMIT || expired() ?
            :root_timeout : :root_interrupted
        return finish(incumbent, false, reason, root_upper, 0)
    end
    if incumbent.objective_value >= root_upper -
       _survivor_tree_tolerance(incumbent.objective_value, root_upper)
        reason = root_assignment.indices !== nothing && root_assignment.week === nothing ?
            :integral_root : :root_bound_certificate
        return finish(incumbent, true, reason, root_upper, 0)
    end

    queue = [SurvivorBranchNode(
        id, region, [region], root_upper, -Inf, false,
    ) for (id, region) in enumerate(regions)]
    next_id = length(queue)
    summaries = Dict(
        node.id => _survivor_tree_summary(0, node.region, node.upper)
        for node in queue
    )
    observer((kind=:partition, nodes=copy(queue)))

    function emit_node(node, week, bound, action, source, reason,
                       iterations, crossover_iterations, solve_started)
        emit_row(
            node.id, String(tree.data.team[node.region]), length(node.path),
            something(week, "-"), bound, action, string(source), reason,
            iterations, crossover_iterations,
            (time_ns() - solve_started) / 1.0e9,
        )
        return nothing
    end

    function branch_node!(node, bound, assignment, node_status, solve_started,
                          iterations, crossover_iterations, bound_source,
                          diagnostic)
        fixed_positions = Set(tree.candidate_positions[node.path])
        week = assignment.week
        if week === nothing || week in fixed_positions
            week = _survivor_tree_fallback_week(tree, node.path)
        end
        week === nothing && error("survivor fallback branching has no unfixed week")
        child_ids = Int[]
        for index in _survivor_tree_candidates(tree, node.path, week)
            next_id += 1
            push!(child_ids, next_id)
            push!(queue, SurvivorBranchNode(
                next_id, node.region, [node.path; index], bound, -Inf, false,
            ))
        end
        _survivor_tree_branch!(summaries, upper, node.id, bound, child_ids)
        observer((; kind=:node_branch, node, week, bound, status=node_status,
                  bound_source, children=copy(child_ids)))
        emit_node(
            node, week, bound, "BRANCH", bound_source, diagnostic,
            iterations, crossover_iterations, solve_started,
        )
        return nothing
    end

    while true
        competing = competing_upper()
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
            completion_started = time_ns()
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
                previous_incumbent = incumbent.objective_value
                incumbent_region, incumbent = _survivor_tree_best(
                    incumbent_region, incumbent, node.region, plan,
                )
                improved = incumbent.objective_value > previous_incumbent
                if improved
                    emit_row(
                        node.id, String(tree.data.team[node.region]), length(node.path),
                        "-", node.upper, "INCUMBENT", "greedy_completion",
                        "validated_schedule", nothing, nothing,
                        (time_ns() - completion_started) / 1.0e9,
                    )
                end
                observer((; kind=:node_completion, node, plan, improved))
            end
            observer((; kind=:node_ranked, node))
        end
        competing = competing_upper()
        tolerance = _survivor_tree_tolerance(incumbent.objective_value, competing)
        incumbent.objective_value >= competing - tolerance &&
            return finish(incumbent, true, :region_certificate, competing, nodes)
        expired() && return finish(incumbent, false, :node_timeout, competing, nodes)

        node_index = _survivor_tree_next_node(queue, incumbent_region, nodes + 1, upper)
        node = queue[node_index]
        deleteat!(queue, node_index)
        if node.upper <= incumbent.objective_value +
                         _survivor_tree_tolerance(incumbent.objective_value, node.upper)
            _survivor_tree_close!(summaries, upper, node.id, node.upper)
            emit_node(node, nothing, node.upper, "PRUNE", :inherited,
                      "incumbent_bound", nothing, nothing, time_ns())
            continue
        end
        if !_survivor_tree_apply_path!(tree, node.path)
            _survivor_tree_close!(summaries, upper, node.id, -Inf)
            emit_node(node, nothing, -Inf, "PRUNE", :structural,
                      "infeasible_path", nothing, nothing, time_ns())
            continue
        end
        node.completion_ready || error("survivor selected node has no completion ranking")

        if length(node.path) == tree.number_of_weeks &&
           length(Set(tree.candidate_positions[node.path])) == tree.number_of_weeks
            exact_started = time_ns()
            plan = _survivor_tree_plan(tree, node.path)
            node.upper + _survivor_tree_tolerance(plan.objective_value, node.upper) >=
                plan.objective_value ||
                error("survivor exact schedule contradicts its inherited upper bound")
            previous_incumbent = incumbent.objective_value
            incumbent_region, incumbent = _survivor_tree_best(
                incumbent_region, incumbent, node.region, plan,
            )
            if incumbent.objective_value > previous_incumbent
                emit_row(
                    node.id, String(tree.data.team[node.region]), length(node.path),
                    "-", node.upper, "INCUMBENT", "exact_schedule",
                    "validated_schedule", nothing, nothing,
                    (time_ns() - exact_started) / 1.0e9,
                )
            end
            leaf_upper = nextfloat(
                plan.objective_value + 1e-8 * max(1.0, abs(plan.objective_value)),
            )
            _survivor_tree_close!(summaries, upper, node.id, leaf_upper)
            nodes += 1
            observer((; kind=:node_exact, node, plan, bound=leaf_upper))
            observer((kind=:queue, nodes=copy(queue), retained=length(summaries),
                      parents=Dict(id => summary.parent for (id, summary) in summaries),
                      upper=copy(upper)))
            emit_node(node, nothing, min(node.upper, leaf_upper), "EXACT",
                      :exact_schedule, "closed", 0, 0, exact_started)
            continue
        end

        observer((; kind=:node_start, node))
        if expired()
            push!(queue, node)
            competing = competing_upper()
            return finish(incumbent, false, :node_timeout, competing, nodes)
        end
        remaining() === nothing ||
            JuMP.set_time_limit_sec(model, max(1e-9, remaining()))
        solve_started = time_ns()
        status = solve_relaxation!()
        nodes += 1
        solve_seconds = (time_ns() - solve_started) / 1.0e9
        node_iterations = _survivor_barrier_iterations(model)
        node_crossover_iterations = _survivor_crossover_iterations(model)
        observer((; kind=:node_solve, node, status, seconds=solve_seconds,
                  iterations=node_iterations,
                  crossover_iterations=node_crossover_iterations))

        if status == JuMP.MOI.INFEASIBLE
            matches_incumbent = all(
                tree.data.team[index] == only(incumbent.selections.team[
                    incumbent.selections.week .== tree.data.week[index],
                ]) for index in node.path
            )
            matches_incumbent &&
                error("survivor infeasible LP contradicts a known feasible witness")
            !isfinite(node.completion_lower_bound) ||
                error("survivor infeasible LP contradicts its feasible completion")
            JuMP.dual_status(model) == JuMP.MOI.INFEASIBILITY_CERTIFICATE ||
                (node.completion_ready && node.completion_lower_bound == -Inf) ||
                error("survivor infeasible node has no supported infeasibility certificate")
            _survivor_tree_close!(summaries, upper, node.id, -Inf)
            observer((; kind=:node_infeasible, node, status))
            emit_node(node, nothing, -Inf, "INFEASIBLE", :certificate,
                      "infeasibility_proven", node_iterations,
                      node_crossover_iterations, solve_started)
            continue
        end
        status == JuMP.MOI.OPTIMAL || _survivor_tree_can_fallback(status) ||
            _survivor_tree_must_stop(status) ||
            error("survivor child LP failed with unexpected status $status")

        solver_bound = _survivor_tree_solver_upper_bound(model, status)
        assignment = _survivor_tree_assignment_guidance(tree, node.path)
        if !debug && assignment.indices === nothing &&
           assignment.reason in (
               :nonfinite_pick_values, :pick_value_out_of_bounds,
               :fixed_pick_mismatch, :fixed_week_mismatch, :week_not_one_hot,
               :team_reused, :ineligible_pick,
           )
            @warn "survivor HiPO primal assignment is inconsistent and will not guide the tree" node=node.id status reason=assignment.reason
        end
        schedule = assignment.indices !== nothing && assignment.week === nothing ?
            _survivor_tree_plan(tree, assignment.indices) : nothing
        feasible_values = schedule === nothing ?
            (node.completion_lower_bound,) :
            (node.completion_lower_bound, schedule.objective_value)
        bound = _survivor_tree_merge_upper(
            node.upper, solver_bound.upper, feasible_values, "node",
        )
        bound_source = solver_bound.upper !== nothing && solver_bound.upper <= node.upper ?
            :solver_dual : :inherited
        diagnostic = _survivor_tree_diagnostic(
            status, solver_bound.reason, assignment.reason,
        )
        if !debug && (solver_bound.upper === nothing || status != JuMP.MOI.OPTIMAL)
            @warn "survivor HiPO node used its inherited bound or stopped before optimality" node=node.id status bound_source bound_reason=solver_bound.reason upper=bound
        end
        event_kind = status == JuMP.MOI.OPTIMAL &&
                     bound_source == :solver_dual &&
                     (assignment.week !== nothing || assignment.indices !== nothing) ?
                     :node_optimum : :node_fallback
        observer((; kind=event_kind, node, bound, assignment, status,
                  bound_source, bound_rejection=solver_bound.reason))
        known_node_lower = max(
            node.completion_lower_bound,
            schedule === nothing ? -Inf : schedule.objective_value,
        )

        if bound <= incumbent.objective_value +
                    _survivor_tree_tolerance(incumbent.objective_value, bound)
            _survivor_tree_close!(summaries, upper, node.id, bound)
            observer((; kind=:node_pruned, node, bound, source=bound_source,
                      reason=:incumbent_bound))
            emit_node(node, nothing, bound, "PRUNE", bound_source,
                      string("incumbent_bound/", diagnostic), node_iterations,
                      node_crossover_iterations, solve_started)
        elseif isfinite(known_node_lower) &&
               bound <= known_node_lower +
                        _survivor_tree_tolerance(known_node_lower, bound)
            _survivor_tree_close!(summaries, upper, node.id, bound)
            observer((; kind=:node_pruned, node, bound, source=bound_source,
                      reason=:feasible_schedule_bound))
            emit_node(node, nothing, bound, "SCHED_CLOSE", bound_source,
                      string("schedule_bound/", diagnostic), node_iterations,
                      node_crossover_iterations, solve_started)
        elseif _survivor_tree_must_stop(status) || expired()
            bound < node.upper &&
                _survivor_tree_tighten!(summaries, upper, node.id, bound)
            interrupted_node = SurvivorBranchNode(
                node.id, node.region, node.path, bound,
                node.completion_lower_bound, node.completion_ready,
            )
            push!(queue, interrupted_node)
            competing = competing_upper()
            tolerance = _survivor_tree_tolerance(
                incumbent.objective_value, competing,
            )
            if incumbent.objective_value >= competing - tolerance
                emit_node(node, nothing, bound, "PRUNE", bound_source,
                          "region_certificate", node_iterations,
                          node_crossover_iterations, solve_started)
                return finish(incumbent, true, :region_certificate, competing, nodes)
            end
            reason = expired() || status == JuMP.MOI.TIME_LIMIT ?
                :interrupted_node : :solver_interrupted
            emit_node(
                node, nothing, bound,
                status == JuMP.MOI.TIME_LIMIT || expired() ? "TIMEOUT" : "STOPPED",
                bound_source, diagnostic, node_iterations,
                node_crossover_iterations, solve_started,
            )
            return finish(incumbent, false, reason, competing, nodes)
        else
            branch_node!(node, bound, assignment, status, solve_started,
                         node_iterations, node_crossover_iterations,
                         bound_source, diagnostic)
        end
        observer((; kind=:queue, nodes=copy(queue), retained=length(summaries),
                  parents=Dict(id => summary.parent for (id, summary) in summaries),
                  upper=copy(upper)))
    end
end

function _build_survivor_full_model(data, state, config, inputs; kwargs...)
    return _optimize_survivor_expected_weeks_scalar_milp(
        data, state, config, inputs; build_only=true, kwargs...,
    )
end
