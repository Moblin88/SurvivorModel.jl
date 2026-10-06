using Test
using SurvivorModel
using DataFrames

const BNB = SurvivorModel

function _bnb_numeric_snapshot(tree)
    m = tree.model
    variables = [(BNB.JuMP.lower_bound(v), BNB.JuMP.upper_bound(v))
                 for v in BNB.JuMP.all_variables(m)]
    rows = [(BNB.JuMP.constraint_object(ref).func,
             BNB.JuMP.constraint_object(ref).set)
            for (F, S) in BNB.JuMP.list_of_constraint_types(m) if F == BNB.JuMP.AffExpr
            for ref in BNB.JuMP.all_constraints(m, F, S)]
    return (variables, rows)
end

function _bnb_fixture(; current_week=1, repeated=true)
    keys = [(:td, "A"), (:td, "B")]
    parameters = BNB.SurvivorParameterSystem(
        keys, Dict(k => i for (i, k) in enumerate(keys)), [0.0, 0.0], [1.0, 1.0],
    )
    values = [
        (0.90, [0.3, -0.1], 0.02), (0.70, [-0.2, 0.3], -0.03),
        (0.85, [0.1, 0.4], 0.01), (0.80, [-0.3, -0.1], 0.04),
        (0.88, [0.3, 0.3], -0.05), (0.75, [-0.1, 0.2], 0.02),
    ]
    inputs = BNB.SurvivorObjectiveInputs(
        parameters, [BNB.SurvivorCandidateDerivatives(p, g, h) for (p, g, h) in values],
    )
    data = DataFrame(
        game_id=["w1a", "w1b", "w2a", "w2b", "w3a", "w3b"],
        week=current_week .+ [0, 0, 1, 1, 2, 2],
        team=repeated ? ["A", "B", "A", "C", "B", "C"] : ["A", "B", "C", "D", "E", "F"],
        opponent=["X", "Y", "Z", "Q", "R", "S"],
        is_home=fill(true, 6), win_probability=first.(values),
        market_spread=fill(3.0, 6),
    )
    return data, inputs
end

function _bnb_config(; through_week=3, hessian_weeks=3, kwargs...)
    return SurvivorSelectionConfig(
        ;
        minimum_favorite_spread=nothing, market_guard_weeks=0,
        through_week, hessian_weeks, branch_and_bound=true, kwargs...,
    )
end

@testset "Greedy completion preserves its own first pick" begin
    for repeated in (false, true), losses in 0:2, H in 0:3
        data, inputs = _bnb_fixture(; repeated)
        state = SurvivorPoolState(2025, 1; strikes_remaining=losses)
        config = _bnb_config(; hessian_weeks=H)
        greedy = BNB._survivor_greedy_selected_indices(data, state, config, inputs)
        completion = BNB._survivor_greedy_selected_indices(
            data, state, config, inputs; fixed_indices=[first(greedy)],
        )
        @test completion == greedy
    end
end

function _bnb_exhaustive(data, inputs, state, config)
    horizon = config.through_week - state.current_week + 1
    positions = Int.(data.week) .- state.current_week .+ 1
    eligible = BNB._survivor_market_guard_mask(data, state, config)
    choices = [[i for i in 1:nrow(data) if positions[i] == p && eligible[i]]
               for p in 1:horizon]
    objectives = Dict{Int,Float64}()
    for raw in Iterators.product(choices...)
        indices = collect(raw)
        length(unique(data.team[indices])) == horizon || continue
        isempty(intersect(Set(data.team[indices]), Set(values(state.picks_made)))) || continue
        objective = _bnb_independent_objective(indices, inputs, state, config)
        objectives[indices[1]] = max(get(objectives, indices[1], -Inf), objective)
    end
    return objectives
end

function _bnb_independent_objective(indices, inputs, state, config)
    losses = max(1, state.strikes_remaining)
    probability = [1.0; zeros(losses - 1)]
    gradient = zeros(length(inputs.parameters.keys), losses)
    hessian = zeros(losses)
    objective = 0.0
    for (week, i) in enumerate(indices)
        derivative = inputs.derivatives[i]
        p, g, h = derivative.base_probability, derivative.gradient, derivative.hessian_covariance
        previous_p = [0.0; probability[1:end-1]]
        previous_g = hcat(zeros(size(gradient, 1)), gradient[:, 1:end-1])
        previous_h = [0.0; hessian[1:end-1]]
        difference = probability - previous_p
        next_h = p .* hessian + (1 - p) .* previous_h + h .* difference +
            2 .* vec(sum((inputs.parameters.variance .* g) .* (gradient - previous_g); dims=1))
        gradient = p .* gradient + (1 - p) .* previous_g + g * difference'
        probability = p .* probability + (1 - p) .* previous_p
        hessian = next_h
        objective += sum(probability) + (week <= config.hessian_weeks ? 0.5 * sum(hessian) : 0.0)
    end
    return objective
end

@testset "branch-conditioned recurrence intervals and full hull updates" begin
    for repeated in (false, true), losses in (0, 2, 3), H in (0, 1, 2, 3),
        zero_gradient in (false, true)
        data, original = _bnb_fixture(; repeated)
        inputs = zero_gradient ? BNB.SurvivorObjectiveInputs(
            original.parameters,
            [BNB.SurvivorCandidateDerivatives(d.base_probability, zeros(2), 0.0)
             for d in original.derivatives],
        ) : original
        state = SurvivorPoolState(2025, 1; strikes_remaining=losses)
        config = _bnb_config(; hessian_weeks=H)
        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        variables = BNB.JuMP.num_variables(tree.model)
        row_mapping = BNB._survivor_basis_row_mapping(tree.model, BNB.JuMP.backend(tree.model))
        schedules = [collect(raw) for raw in Iterators.product([1, 2], [3, 4], [5, 6])
                     if length(unique(data.team[collect(raw)])) == 3]
        for path in (Int[], [1], [2], [3], [5], [2, 5], first(schedules))
            completions = filter(s -> all(in(s), path), schedules)
            isempty(completions) && continue
            @test BNB._survivor_tree_apply_path!(tree, path)
            bounds = tree.model.ext[:survivor_node_bounds]
            @test BNB.JuMP.num_variables(tree.model) == variables
            @test BNB._survivor_basis_row_mapping(
                tree.model, BNB.JuMP.backend(tree.model),
            ) == row_mapping
            for schedule in completions
                values = BNB._survivor_scalar_forward_values(
                    schedule, inputs, 3, max(1, losses);
                    curvature_weeks=H,
                    gradient_reference_indices=tree.gradient_reference_indices,
                )
                @test all(bounds.probability.lower .<= values.probability .+ 1e-14)
                @test all(values.probability .<= bounds.probability.upper .+ 1e-14)
                @test all(bounds.hessian.lower .<= values.hessian[1:(H + 1), :] .+ 1e-14)
                @test all(values.hessian[1:(H + 1), :] .<= bounds.hessian.upper .+ 1e-14)
                for position in 1:H, reference in tree.gradient_reference_indices[position]
                    @test all(bounds.gradient.lower[position, :, reference] .<=
                              values.gradient[position, :, reference] .+ 1e-14)
                    @test all(values.gradient[position, :, reference] .<=
                              bounds.gradient.upper[position, :, reference] .+ 1e-14)
                end
                for position in 1:H
                    @test all(bounds.parameter_gradient.lower[position, :, :] .<=
                              values.parameter_gradient[position, :, :] .+ 1e-14)
                    @test all(values.parameter_gradient[position, :, :] .<=
                              bounds.parameter_gradient.upper[position, :, :] .+ 1e-14)
                end
                for (position, index) in enumerate(schedule), loss in 1:max(1, losses)
                    @test bounds.team_conditioned.candidate_probability.lower[index, loss] <=
                          values.probability[position + 1, loss] + 1e-14
                    @test values.probability[position + 1, loss] <=
                          bounds.team_conditioned.candidate_probability.upper[index, loss] + 1e-14
                    if position <= H
                        @test bounds.team_conditioned.candidate_hessian.lower[index, loss] <=
                              values.hessian[position + 1, loss] + 1e-14
                        @test values.hessian[position + 1, loss] <=
                              bounds.team_conditioned.candidate_hessian.upper[index, loss] + 1e-14
                    end
                    if position < H
                        for parameter in eachindex(inputs.parameters.keys)
                            value = values.parameter_gradient[position + 1, loss, parameter]
                            @test bounds.team_conditioned.candidate_parameter_gradient.lower[
                                index, loss, parameter,
                            ] <= value <= bounds.team_conditioned.candidate_parameter_gradient.upper[
                                index, loss, parameter,
                            ]
                        end
                        for reference in tree.gradient_reference_indices[position + 1]
                            value = values.gradient[position + 1, loss, reference]
                            @test bounds.team_conditioned.candidate_gradient.lower[
                                index, loss, reference,
                            ] <= value <= bounds.team_conditioned.candidate_gradient.upper[
                                index, loss, reference,
                            ]
                        end
                    end
                end
            end
            for hull in tree.model.ext[:survivor_node_hulls]
                lower, upper, all_lower, all_upper =
                    BNB._survivor_tree_hull_interval(bounds, hull.key, hull.index)
                # HiGHS drops matrix entries below its small-matrix threshold.
                @test BNB.JuMP.normalized_coefficient(hull.lower_row, hull.selected) ≈ -lower atol=1e-12
                @test BNB.JuMP.normalized_coefficient(hull.upper_row, hull.selected) ≈ -upper atol=1e-12
                @test BNB.JuMP.normalized_coefficient(hull.other_upper_row, hull.selected) ≈ -all_upper atol=1e-12
                @test BNB.JuMP.normalized_rhs(hull.other_upper_row) == -all_upper
                @test BNB.JuMP.normalized_coefficient(hull.other_lower_row, hull.selected) ≈ -all_lower atol=1e-12
                @test BNB.JuMP.normalized_rhs(hull.other_lower_row) == -all_lower
            end
        end
        complete = first(schedules)
        @test BNB._survivor_tree_apply_path!(tree, complete)
        bounds = tree.model.ext[:survivor_node_bounds]
        @test maximum(bounds.probability.upper - bounds.probability.lower) < 1e-10
        @test maximum(bounds.hessian.upper - bounds.hessian.lower) < 1e-10
        @test BNB._survivor_tree_apply_path!(tree, [1])
        snapshot = _bnb_numeric_snapshot(tree)
        @test BNB._survivor_tree_apply_path!(tree, [2])
        @test BNB._survivor_tree_apply_path!(tree, [1])
        @test _bnb_numeric_snapshot(tree) == snapshot
    end

    data, inputs = _bnb_fixture(; repeated=false)
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
    m = tree.model
    BNB.JuMP.optimize!(m)
    root = BNB._survivor_snapshot_basis(m)
    BNB.JuMP.set_optimizer_attribute(m, "solver", "simplex")
    BNB.JuMP.set_optimizer_attribute(m, "presolve", "off")
    @test BNB._survivor_tree_apply_path!(tree, [2]; tighten=false)
    BNB._survivor_restore_basis!(m, root; repair=true)
    baseline_seconds = @elapsed BNB.JuMP.optimize!(m)
    @test BNB.JuMP.termination_status(m) == BNB.JuMP.MOI.OPTIMAL
    baseline = BNB.JuMP.objective_value(m)
    tightening_seconds = @elapsed BNB._survivor_tree_apply_path!(tree, [2])
    BNB._survivor_restore_basis!(m, root; repair=true)
    tightened_seconds = @elapsed BNB.JuMP.optimize!(m)
    @test BNB.JuMP.termination_status(m) == BNB.JuMP.MOI.OPTIMAL
    tightened = BNB.JuMP.objective_value(m)
    optimum = _bnb_exhaustive(data, inputs, state, _bnb_config())[2]
    @test baseline - tightened > 1e-4
    @test tightened >= optimum - 1e-8
    @info "branch term bound benchmark" baseline tightened optimum baseline_seconds tightening_seconds tightened_seconds
    @test BNB._survivor_tree_apply_path!(tree, [2]; tighten=false)
    BNB._survivor_restore_basis!(m, root; repair=true)
    BNB.JuMP.optimize!(m)
    @test BNB.JuMP.objective_value(m) ≈ baseline atol=1e-10

    singular = BNB._survivor_direct_milp_model(_bnb_config(); lp_relaxation=true)
    BNB.JuMP.@variable(singular, 0 <= x <= 1)
    BNB.JuMP.@variable(singular, 0 <= y <= 1)
    BNB.JuMP.@constraint(singular, x + y == 1)
    row = BNB.JuMP.@constraint(singular, x - y == 0)
    BNB.JuMP.@objective(singular, Max, 1.0 * x)
    BNB.JuMP.optimize!(singular)
    basis = BNB._survivor_snapshot_basis(singular)
    @test count(==(BNB.HiGHS.kHighsBasisStatusBasic), basis.columns) == 2
    BNB.JuMP.set_normalized_coefficient(row, y, 1.0)
    BNB.JuMP.set_normalized_rhs(row, 1.0)
    BNB.JuMP.set_optimizer_attribute(singular, "solver", "simplex")
    BNB.JuMP.set_optimizer_attribute(singular, "presolve", "off")
    BNB._survivor_restore_basis!(singular, basis; repair=true)
    BNB.JuMP.optimize!(singular)
    @test BNB.JuMP.termination_status(singular) == BNB.JuMP.MOI.OPTIMAL
    @test BNB.JuMP.objective_value(singular) ≈ 1.0
    @test BNB._survivor_snapshot_basis(singular).row_mapping == basis.row_mapping
end

@testset "external full-LP survivor tree" begin
    @testset "shared local curvature ranking and fixed completions" begin
        data, original = _bnb_fixture(; repeated=false)
        derivatives = [
            BNB.SurvivorCandidateDerivatives(0.9, [0.4, -0.2], -0.4),
            BNB.SurvivorCandidateDerivatives(0.8, [-0.3, 0.1], 0.2),
            BNB.SurvivorCandidateDerivatives(0.85, [0.5, -0.2], 0.0),
            BNB.SurvivorCandidateDerivatives(0.8, [-0.5, 0.2], 0.0),
            BNB.SurvivorCandidateDerivatives(0.9, [0.0, 0.0], 0.0),
            BNB.SurvivorCandidateDerivatives(0.7, [0.0, 0.0], 0.0),
        ]
        inputs = BNB.SurvivorObjectiveInputs(original.parameters, derivatives)
        for mode in (:milp, :benders, :tree), losses in (0, 2), H in (0, 1, 2, 3, 20)
            config = SurvivorSelectionConfig(
                minimum_favorite_spread=nothing, market_guard_weeks=0, through_week=3,
                hessian_weeks=H, branch_and_bound=mode == :tree,
                benders_weeks=mode == :benders ? 1 : nothing,
            )
            state = SurvivorPoolState(2025, 1; strikes_remaining=losses)
            selected = BNB._survivor_greedy_selected_indices(data, state, config, inputs)
            @test selected[1] == (H > 0 && losses == 0 ? 2 : 1)
            for p in 1:3
                scores = Dict(i => begin
                    prefix = [selected[1:p-1]; i]
                    _bnb_independent_objective(prefix, inputs, state, config) -
                        _bnb_independent_objective(prefix[1:end-1], inputs, state, config)
                end for i in (2p-1):2p)
                @test scores[selected[p]] ≈ maximum(values(scores))
            end
            second = BNB._survivor_greedy_selected_indices(
                data, state, config, inputs; first_pick_rank=2,
            )
            @test second[1] != selected[1]
            fixed = BNB._survivor_greedy_selected_indices(
                data, state, config, inputs; fixed_indices=[2, 5],
            )
            @test fixed[[1, 3]] == [2, 5]
            @test BNB._survivor_greedy_selected_indices(
                data, state, config, inputs; fixed_indices=[2], expired=() -> true,
            ) === nothing
        end
        state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        @test 0.9 - 0.4 / 2 < 0.8 + 0.2 / 2
        @test BNB._survivor_greedy_selected_indices(
            data, state, _bnb_config(hessian_weeks=2), inputs,
        ) == [2, 4, 5]
        # Under pick 2, the week-two scores include signed gradient cross terms.
        @test _bnb_independent_objective([2, 3], inputs, state, _bnb_config()) -
            _bnb_independent_objective([2], inputs, state, _bnb_config()) ≈ 0.595
        @test _bnb_independent_objective([2, 4], inputs, state, _bnb_config()) -
            _bnb_independent_objective([2], inputs, state, _bnb_config()) ≈ 0.89
        zero_inputs = BNB.SurvivorObjectiveInputs(
            original.parameters,
            [BNB.SurvivorCandidateDerivatives(d.base_probability, zeros(2), 0.0)
             for d in derivatives],
        )
        for config in (
            _bnb_config(),
            SurvivorSelectionConfig(through_week=3, minimum_favorite_spread=nothing, market_guard_weeks=0),
            SurvivorSelectionConfig(through_week=3, minimum_favorite_spread=nothing, market_guard_weeks=0, benders_weeks=1),
        )
            @test BNB._survivor_greedy_selected_indices(data, state, config, zero_inputs) == [1, 3, 5]
            @test BNB._survivor_greedy_selected_indices(
                data, state, config, zero_inputs; first_pick_rank=2,
            ) == [2, 3, 5]
        end
        @test BNB._survivor_greedy_selected_indices(
            data, state, _bnb_config(banned_first_pick_teams=["B"]), inputs,
        )[1] == 1
        data.team = ["A", "B", "C", "D", "A", "A"]
        @test BNB._survivor_greedy_selected_indices(
            data, state, _bnb_config(), inputs; fixed_indices=[5],
        )[1] == 2
        tree_data, tree_inputs = _bnb_fixture()
        improvement_values = [
            (0.9, [0.19856583836352895, 0.4444417803697178], -0.002936583258071421),
            (0.7, [-0.21735497624481478, 0.18077375607802246], -0.1379466981478844),
            (0.85, [0.7117928058592533, 0.1165757479123653], 0.05421875122694457),
            (0.8, [0.0878530418803952, 0.2784927467022817], 0.2022415405969966),
            (0.88, [0.19996274067723946, -0.11289074118238332], -0.028367815579687546),
            (0.75, [-0.4812073739055828, 0.2746490193906272], 0.1827535115973055),
        ]
        tree_inputs = BNB.SurvivorObjectiveInputs(
            tree_inputs.parameters,
            [BNB.SurvivorCandidateDerivatives(p, g, h) for (p, g, h) in improvement_values],
        )
        events = []
        tree = BNB._build_survivor_full_model(
            tree_data, SurvivorPoolState(2025, 1; strikes_remaining=2),
            _bnb_config(), tree_inputs,
        )
        BNB._optimize_survivor_branch_and_bound!(tree; observer=e -> push!(events, e))
        initial_objective = BNB._survivor_tree_plan(tree, tree.warm_start.selected).objective_value
        completions = filter(e -> e.kind == :node_completion, events)
        @test !isempty(completions)
        @test any(e -> e.improved, completions)
        @test any(e -> e.plan.objective_value > initial_objective + 1e-6, completions)
        @test length(unique(e.node.id for e in completions)) == length(completions)
        for e in completions
            @test e.node.completion_ready
            @test e.node.completion_lower_bound == e.plan.objective_value
            @test all(tree_data.team[i] == only(e.plan.selections.team[
                e.plan.selections.week .== tree_data.week[i],
            ]) for i in e.node.path)
            @test e.plan.objective_value <= e.node.upper + 2e-6
        end
        for (position, e) in enumerate(events)
            e.kind == :node_start || continue
            @test e.node.completion_ready
            if isfinite(e.node.completion_lower_bound)
                completed = only(filter(c -> c.node.id == e.node.id, completions))
                @test completed.plan.objective_value == e.node.completion_lower_bound
                @test any(c -> c.kind == :node_completion && c.node.id == e.node.id,
                          events[1:(position - 1)])
            end
            ranked = filter(e -> e.kind == :node_ranked, events)
            @test length(unique(e.node.id for e in ranked)) == length(ranked)
            @test all(e -> e.node.completion_ready, ranked)
            @test fieldtype(BNB.SurvivorBranchNode, :completion_lower_bound) == Float64
            @test !(:completion in fieldnames(BNB.SurvivorBranchNode))
            @test !(:basis in fieldnames(BNB.SurvivorBranchNode))
            for e in filter(e -> e.kind == :node_start, events)
                @test count(r -> r.node.id == e.node.id, ranked) == 1
            end
        end
    end
    @test_throws ArgumentError SurvivorSelectionConfig(branch_and_bound=true, benders_weeks=0)
    @test !SurvivorSelectionConfig().branch_and_bound
    @testset "residual-safe numerical upper certificates" begin
        m = BNB._survivor_direct_milp_model(_bnb_config(); lp_relaxation=true)
        BNB.JuMP.@variable(m, 0 <= x <= 1)
        BNB.JuMP.@variable(m, 0 <= y <= 1)
        BNB.JuMP.@constraint(m, x + y <= 1)
        BNB.JuMP.@constraint(m, x + y >= 0)
        BNB.JuMP.@objective(m, Max, x + y)
        BNB.JuMP.optimize!(m)
        @test BNB._survivor_tree_upper_bound(m) >= 1
        @test BNB._survivor_tree_lagrangian_bound(m; row_dual=_ -> 0.0) == 2
        for multiplier in (-100.0, -1.000001, -0.999999, 0.999999, 100.0)
            @test BNB._survivor_tree_lagrangian_bound(m; row_dual=_ -> multiplier) >= 1
        end
        @test_throws ErrorException BNB._survivor_tree_lagrangian_bound(m; row_dual=_ -> NaN)
        BNB.JuMP.delete_upper_bound(y)
        @test_throws ErrorException BNB._survivor_tree_lagrangian_bound(m)
    end

    @testset "H horizon and exact first-pick certificates" begin
        for current in (1, 16), losses in (0, 2), H in (0, 1, 2, 3, 20)
            data, inputs = _bnb_fixture(; current_week=current)
            state = SurvivorPoolState(2025, current; strikes_remaining=losses)
            config = _bnb_config(; through_week=current + 2, hessian_weeks=H)
            exhaustive = _bnb_exhaustive(data, inputs, state, config)
            events = []
            tree = BNB._build_survivor_full_model(data, state, config, inputs)
            @test tree.curvature_weeks == min(H, 3)
            @test length(tree.gradient) == min(H, 3)
            @test length(tree.parameter_gradient) <= min(H, 3)
            if H == 0
                @test isempty(tree.gradient_reference_indices)
                @test isempty(tree.parameter_gradient)
                @test size(tree.hessian, 1) == 1
            end
            function observe_solver(event)
                push!(events, event)
                if event.kind in (:root, :node_start, :node_solve)
                    for (option, value) in (
                        ("solver", "hipo"), ("run_crossover", "off"),
                        ("parallel", "on"), ("threads", 0),
                    )
                        @test BNB.JuMP.get_optimizer_attribute(tree.model, option) == value
                    end
                end
                if event.kind in (:root, :node_solve)
                    @test event.simplex_iterations == 0
                    count = Ref{BNB.HiGHS.HighsInt}(0)
                    @test BNB.HiGHS.Highs_getIntInfoValue(
                        BNB.JuMP.backend(tree.model).inner, "crossover_iteration_count", count,
                    ) == BNB.HiGHS.kHighsStatusOk
                    @test count[] == 0
                end
            end
            plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=observe_solver)
            first_index = only(findall((data.week .== current) .&
                                      (data.team .== only(plan.current_pick.team))))
            @test exhaustive[first_index] >= maximum(values(exhaustive)) - 5e-6
            @test plan.objective_value <= exhaustive[first_index] + 2e-6
            @test sum(plan.selections.objective_contribution) ≈ plan.objective_value
            indices = [only(findall((data.week .== w) .& (data.team .== t)))
                       for (w, t) in zip(plan.selections.week, plan.selections.team)]
            @test plan.objective_value ≈ _bnb_independent_objective(indices, inputs, state, config) atol=2e-6
            @test length(unique(plan.selections.team)) == 3
            @test last(events).proven
            for event in events
                if event.kind == :node_optimum
                    @test event.bound <= event.node.upper
                elseif event.kind == :queue
                    for a in keys(exhaustive)
                        @test event.upper[a] >= exhaustive[a] - 5e-6
                    end
                end
            end
            extensive = BNB._optimize_survivor_expected_weeks_scalar_milp(
                data, state, SurvivorSelectionConfig(
                    minimum_favorite_spread=nothing, market_guard_weeks=0,
                    through_week=current + 2, hessian_weeks=H,
                ), inputs,
            )
            @test extensive.objective_value ≈ maximum(values(exhaustive)) atol=2e-6
        end
    end

    @testset "exhaustive partition, earliest fractional week, interleaving" begin
        for losses in (0, 2)
            data, inputs = _bnb_fixture(; repeated=false)
            state = SurvivorPoolState(2025, 1; strikes_remaining=losses)
            config = _bnb_config()
            tree = BNB._build_survivor_full_model(data, state, config, inputs)
            events = []
            seen_branch = Ref(false)
            function observe(event)
                push!(events, event)
                if event.kind == :root_optimum && losses == 0
                    @test event.assignment.week == 3
                    @test BNB.JuMP.value(tree.selected[1]) ≈ 1
                    @test abs(BNB.JuMP.value(tree.selected[2])) <= 1e-7
                elseif event.kind == :node_optimum && event.assignment.week !== nothing
                    fixed = tree.candidate_positions[event.node.path]
                    week = event.assignment.week
                    @test !(week in fixed)
                    for p in 1:(week - 1), i in tree.candidate_indices
                        tree.candidate_positions[i] == p || continue
                        value = BNB.JuMP.value(tree.selected[i])
                        @test value <= 1e-7 || value >= 1 - 1e-7
                    end
                elseif event.kind == :queue
                    necessary = Set{Int}()
                    for node in event.nodes
                        id = node.id
                        while id != 0
                            push!(necessary, id)
                            id = event.parents[id]
                        end
                    end
                    @test Set(keys(event.parents)) == necessary
                    @test event.retained == length(necessary)
                    previous = events[end - 1]
                    if previous.kind == :node_optimum
                        children = filter(n -> n.path[1:end-1] == previous.node.path, event.nodes)
                        if !isempty(children)
                            seen_branch[] = true
                            @test sort(last.(getproperty.(children, :path))) ==
                                  sort(BNB._survivor_tree_candidates(
                                      tree, previous.node.path, previous.assignment.week,
                                  ))
                            @test all(n -> n.upper <= previous.node.upper, children)
                        end
                    end
                    exhaustive = _bnb_exhaustive(data, inputs, state, config)
                    for a in keys(exhaustive)
                        @test event.upper[a] >= exhaustive[a] - 2e-6
                    end
                end
            end
            plan = BNB._optimize_survivor_branch_and_bound!(
                tree; observer=observe, tighten_terms=false,
            )
            partition = only(filter(e -> e.kind == :partition, events))
            @test sort(getproperty.(partition.nodes, :region)) == [1, 2]
            @test all(n -> length(n.path) == 1, partition.nodes)
            @test last(events).proven
            @test plan.objective_value <= maximum(values(_bnb_exhaustive(data, inputs, state, config))) + 2e-6
            losses == 2 && @test seen_branch[]
        end
        data, inputs = _bnb_fixture()
        tree = BNB._build_survivor_full_model(
            data, SurvivorPoolState(2025, 1), _bnb_config(), inputs,
        )
        queue = [
            BNB.SurvivorBranchNode(1, 1, [1], 4.0, -Inf, false),
            BNB.SurvivorBranchNode(2, 2, [2], 3.9, -Inf, false),
            BNB.SurvivorBranchNode(3, 2, [2, 3], 3.8, -Inf, false),
        ]
        @test BNB._survivor_tree_next_node(queue, 1, 1) == 2
        @test BNB._survivor_tree_next_node(queue, 1, 4) == 1
        plan = BNB._survivor_tree_plan(tree, tree.warm_start.selected)
        @test BNB._survivor_tree_best(2, plan, 1, plan)[1] == 1
        @test BNB._survivor_tree_best(1, plan, 2, plan)[1] == 1
        # Synthetic objectives isolate the scheduler from heuristic quality.
        tied = [
            BNB.SurvivorBranchNode(10, 2, [2], 4.0, 2.0, true),
            BNB.SurvivorBranchNode(11, 2, [2, 3], 4.0, 3.0, true),
        ]
        @test BNB._survivor_tree_next_node(tied, 1, 1) == 2
        tied[1] = BNB.SurvivorBranchNode(10, 2, [2], 4.1, 2.0, true)
        @test BNB._survivor_tree_next_node(tied, 1, 1) == 1
        tied[1] = BNB.SurvivorBranchNode(10, 3, [3], 4.0, 2.0, true)
        @test BNB._survivor_tree_next_node(tied, 1, 1, Dict(2 => 4.0, 3 => 4.2)) == 1
        tied[1] = BNB.SurvivorBranchNode(10, 2, [2], 4.0, -Inf, true)
        tied[2] = BNB.SurvivorBranchNode(11, 2, [2, 3], 4.0, -2.0, true)
        @test BNB._survivor_tree_next_node(tied, 1, 1) == 2
        tied[1] = BNB.SurvivorBranchNode(10, 2, [2], 4.0, -2.0, true)
        @test BNB._survivor_tree_next_node(tied, 1, 1) == 1
    end

    @testset "compact parent certificates and cascading closure" begin
        roots = Dict(1 => 10.0, 2 => 12.0)
        summaries = Dict(
            1 => BNB._survivor_tree_summary(0, 1, 10.0),
            2 => BNB._survivor_tree_summary(0, 2, 12.0),
        )
        BNB._survivor_tree_branch!(summaries, roots, 1, 9.0, [3, 4])
        @test roots[1] == 9.0
        @test summaries[4].upper == 9.0
        BNB._survivor_tree_branch!(summaries, roots, 3, 7.0, [5, 6])
        @test roots[1] == 9.0 # The unsolved sibling still carries its inherited cap.
        BNB._survivor_tree_close!(summaries, roots, 5, -Inf)
        @test !haskey(summaries, 5)
        @test summaries[3].upper == 7.0
        BNB._survivor_tree_close!(summaries, roots, 6, 6.0) # Bound-pruned, not infeasible.
        @test !haskey(summaries, 3)
        @test !haskey(summaries, 6)
        @test summaries[1].closed_upper == 6.0
        @test roots[1] == 9.0
        BNB._survivor_tree_close!(summaries, roots, 4, 5.0) # Exact leaf.
        @test roots[1] == 6.0
        @test collect(keys(summaries)) == [2]
        # Closing a former incumbent's root must retain its bound after a switch.
        BNB._survivor_tree_branch!(summaries, roots, 2, 8.0, [7])
        BNB._survivor_tree_branch!(summaries, roots, 7, 8.0, Int[])
        @test roots == Dict(1 => 6.0, 2 => -Inf)
        @test isempty(summaries)
        @test all(field -> !(field in (:path, :basis)), fieldnames(BNB.SurvivorBranchSummary))
        # Repeated completed subtrees do not accumulate historical records.
        for id in 10:1000
            summaries[id] = BNB._survivor_tree_summary(0, 2, 8.0)
            BNB._survivor_tree_branch!(summaries, roots, id, 7.0, [id + 1000])
            BNB._survivor_tree_close!(summaries, roots, id + 1000, 4.0)
            @test isempty(summaries)
            @test roots[2] == 4.0
        end
        data, inputs = _bnb_fixture()
        tree = BNB._build_survivor_full_model(
            data, SurvivorPoolState(2025, 1; strikes_remaining=2), _bnb_config(), inputs,
        )
        plans = [BNB._survivor_tree_plan(tree, path) for path in ([1, 4, 5], [2, 3, 6])]
        lower_region = argmin(getproperty.(plans, :objective_value))
        higher_region = argmax(getproperty.(plans, :objective_value))
        @test lower_region != higher_region
        roots = Dict(a => plans[a].objective_value + 1.0 for a in 1:2)
        summaries = Dict(a => BNB._survivor_tree_summary(0, a, roots[a]) for a in 1:2)
        BNB._survivor_tree_close!(
            summaries, roots, lower_region, plans[lower_region].objective_value,
        )
        region, incumbent = BNB._survivor_tree_best(
            lower_region, plans[lower_region], higher_region, plans[higher_region],
        )
        @test region == higher_region
        @test incumbent === plans[higher_region]
        @test !haskey(summaries, lower_region)
        @test roots[lower_region] == plans[lower_region].objective_value
        @test roots[lower_region] <= incumbent.objective_value
    end

    @testset "root and interrupted-child timeouts retain region bounds" begin
        data, inputs = _bnb_fixture()
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        for phase in (:root, :child)
            tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
            events = []
            phase == :root && BNB.JuMP.set_time_limit_sec(tree.model, 1e-9)
            function observe(event)
                push!(events, event)
                if phase == :child && event.kind == :node_start
                    BNB.JuMP.set_time_limit_sec(tree.model, 1e-9)
                end
            end
            @test_logs (:warn, r"first pick is unproven") match_mode=:any begin
                plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=observe)
                @test nrow(plan.selections) == 3
            end
            final = last(events)
            @test !final.proven
            @test final.reason == (phase == :root ? :root_timeout : :interrupted_node)
            @test final.upper >= maximum(values(_bnb_exhaustive(data, inputs, state, _bnb_config())))
            if phase == :child
                partition = only(filter(e -> e.kind == :partition, events))
                @test final.upper == first(partition.nodes).upper
                @test final.nodes == 1
                @test length(final.pending) == length(partition.nodes)
                @test final.retained == length(partition.nodes)
                @test all(==(final.upper), values(final.root_bounds))
            end
        end
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        expired = Ref(false)
        events = []
        function expire_after_root(event)
            push!(events, event)
            if event.kind == :root_optimum
                expired[] = true
            end
        end
        @test_logs (:warn, r"first pick is unproven") match_mode=:any BNB._optimize_survivor_branch_and_bound!(
            tree; observer=expire_after_root,
            remaining_time=() -> expired[] ? 0.0 : nothing,
        )
        @test last(events).reason == :node_timeout
        @test last(events).nodes == 0
        @test last(events).upper == first(only(filter(e -> e.kind == :partition, events)).nodes).upper
        @test length(last(events).pending) == 2
        @test last(events).retained == 2
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        expired[] = false
        events = []
        function expire_after_completion(event)
            push!(events, event)
            event.kind == :node_completion && (expired[] = true)
        end
        @test_logs (:warn, r"first pick is unproven") match_mode=:any BNB._optimize_survivor_branch_and_bound!(
            tree; observer=expire_after_completion,
            remaining_time=() -> expired[] ? 0.0 : nothing,
        )
        @test count(e -> e.kind == :node_completion, events) == 1
        @test !any(e -> e.kind == :node_start, events)
        @test last(events).reason == :node_timeout
        @test last(events).nodes == 0
        partition = only(filter(e -> e.kind == :partition, events))
        @test sort(last(events).pending) == sort([n.id for n in partition.nodes])
        @test last(events).retained == length(partition.nodes)
        @test all(==(first(partition.nodes).upper), values(last(events).root_bounds))
        @test last(events).plan.objective_value >=
              only(filter(e -> e.kind == :node_completion, events)).plan.objective_value
    end

    @testset "zero-valued first pick and Hall-infeasible region" begin
        _, original = _bnb_fixture()
        teams = ["A", "D", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
        probabilities = [0.95, 0.7, 0.8, 0.75, 0.7, 0.85, 0.8, 0.75, 0.9, 0.85, 0.8]
        data = DataFrame(
            game_id=string.(1:11), week=[1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            team=teams, opponent=fill("X", 11), is_home=fill(true, 11),
            win_probability=probabilities, market_spread=fill(3.0, 11),
        )
        inputs = BNB.SurvivorObjectiveInputs(
            original.parameters,
            [BNB.SurvivorCandidateDerivatives(
                p, original.derivatives[mod1(i, 6)].gradient,
                original.derivatives[mod1(i, 6)].hessian_covariance,
            ) for (i, p) in enumerate(probabilities)],
        )
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        config = _bnb_config(; through_week=4, hessian_weeks=4)
        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        events = []
        function observe_hall(event)
            push!(events, event)
            if event.kind == :root_optimum
                @test event.assignment.week !== nothing
                @test abs(BNB.JuMP.value(tree.selected[1])) <= 1e-7
            end
        end
        plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=observe_hall)
        @test only(plan.current_pick.team) == "D"
        partition = only(filter(e -> e.kind == :partition, events))
        @test sort(getproperty.(partition.nodes, :region)) == [1, 2]
        @test any(e -> e.kind == :node_solve && e.node.region == 1 &&
                       e.status == BNB.JuMP.MOI.INFEASIBLE, events)
        infeasible_ranking = only(filter(
            e -> e.kind == :node_ranked && e.node.region == 1, events,
        ))
        @test infeasible_ranking.node.completion_ready
        @test infeasible_ranking.node.completion_lower_bound == -Inf
        @test last(events).proven
        @test last(events).upper == -Inf
        @test last(events).nodes == 1
    end

    @testset "debug output and unexpected HiPO status" begin
        data, inputs = _bnb_fixture(; repeated=false)
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        logs = IOBuffer()
        mktempdir() do directory
            stdout_path = joinpath(directory, "stdout")
            stderr_path = joinpath(directory, "stderr")
            open(stdout_path, "w") do output
                open(stderr_path, "w") do progress
                    redirect_stdout(output) do
                        redirect_stderr(progress) do
                            BNB.Logging.with_logger(BNB.Logging.ConsoleLogger(logs, BNB.Logging.Debug)) do
                                plan = BNB._optimize_survivor_branch_and_bound!(tree)
                                @test nrow(plan.selections) == 3
                            end
                        end
                    end
                end
            end
            @test isempty(read(stdout_path, String))
            progress = read(stderr_path, String)
            header = BNB._survivor_tree_progress_header()
            @test occursin(header, progress)
            @test occursin(repeat("-", length(header)), progress)
            header_separators = findall(==('|'), header)
            rows = filter(line -> occursin('|', line), split(progress, '\n'))
            @test length(rows) > 1
            @test all(row -> findall(==('|'), row) == header_separators, rows)
            @test all(row -> length(row) == length(header), rows)
            @test !occursin("Running HiGHS", progress)
        end
        text = String(take!(logs))
        @test occursin("hessian_weeks = 3", text)
        @test occursin("child_solver = :hipo", text)
        @test occursin("crossover = :off", text)
        @test occursin("first_pick_within_tolerance", text)
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        function limit_iterations(event)
            event.kind == :node_start &&
                BNB.JuMP.set_optimizer_attribute(tree.model, "ipm_iteration_limit", 0)
        end
        @test_throws ErrorException BNB._optimize_survivor_branch_and_bound!(
            tree; observer=limit_iterations,
        )
        @test BNB.JuMP.termination_status(tree.model) == BNB.JuMP.MOI.ITERATION_LIMIT
    end

    @testset "fixed early picks retain the original Hessian prefix" begin
        data, inputs = _bnb_fixture(; current_week=16)
        state = SurvivorPoolState(2025, 16; strikes_remaining=2)
        for H in (0, 1, 2, 3, 18)
            config = _bnb_config(; through_week=18, hessian_weeks=H)
            tree = BNB._build_survivor_full_model(data, state, config, inputs)
            BNB.JuMP.optimize!(tree.model)
            root = BNB._survivor_snapshot_basis(tree.model)
            BNB.JuMP.set_optimizer_attribute(tree.model, "solver", "simplex")
            BNB.JuMP.set_optimizer_attribute(tree.model, "presolve", "off")
            for path in ([1, 4, 5], [2, 3, 6])
                @test BNB._survivor_tree_apply_path!(tree, path)
                BNB._survivor_restore_basis!(tree.model, root)
                BNB.JuMP.optimize!(tree.model)
                @test BNB.JuMP.termination_status(tree.model) == BNB.JuMP.MOI.OPTIMAL
                @test BNB.JuMP.objective_value(tree.model) ≈
                      _bnb_independent_objective(path, inputs, state, config) atol=2e-6
                @test BNB._survivor_tree_upper_bound(tree.model) >=
                      _bnb_independent_objective(path, inputs, state, config) - 1e-8
                plan = BNB._survivor_tree_plan(tree, path)
                @test all(iszero, plan.selections.parameter_variance_adjustment[(min(H, 3) + 1):end])
            end
        end
        @test_throws ArgumentError BNB._optimize_survivor_expected_weeks_scalar_milp(
            data, state, _bnb_config(; through_week=18), inputs;
            lp_output_file="unsupported.lp",
        )
    end

    @testset "original native basis across siblings and grandchildren" begin
        data, inputs = _bnb_fixture()
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        m = tree.model
        @test all(v -> BNB.JuMP.start_value(v) === nothing, BNB.JuMP.all_variables(m))
        BNB.JuMP.optimize!(m)
        root = BNB._survivor_snapshot_basis(m)
        @test length(root.columns) == BNB.JuMP.num_variables(m)
        @test root.mapping == BNB._survivor_basis_mapping(m, BNB.JuMP.backend(m))
        BNB.JuMP.set_optimizer_attribute(m, "solver", "simplex")
        BNB.JuMP.set_optimizer_attribute(m, "presolve", "off")
        bounds = []
        iterations = Float64[]
        for path in ([1], [2], [1, 4], [2, 3])
            @test BNB._survivor_tree_apply_path!(tree, path)
            parent = length(path) == 1 ? root : bounds[path[1]].basis
            BNB._survivor_restore_basis!(m, parent)
            optimizer = BNB.JuMP.backend(m)
            columns = similar(parent.columns)
            rows = similar(parent.rows)
            @test BNB.HiGHS.Highs_getBasis(optimizer.inner, columns, rows) == BNB.HiGHS.kHighsStatusOk
            @test columns == parent.columns
            @test rows == parent.rows
            BNB.JuMP.optimize!(m)
            @test BNB.JuMP.termination_status(m) == BNB.JuMP.MOI.OPTIMAL
            upper = BNB._survivor_tree_upper_bound(m)
            @test isfinite(BNB._survivor_benders_simplex_iterations(m))
            push!(iterations, BNB._survivor_benders_simplex_iterations(m))
            if length(path) == 1
                push!(bounds, (upper=upper, basis=BNB._survivor_snapshot_basis(m)))
            else
                @test upper <= bounds[path[1]].upper + 1e-6
            end

        end
        @test any(>(0), iterations)
        saved = root.columns[1]
        root.columns[1] = -1
        @test_throws ErrorException BNB._survivor_restore_basis!(m, root)
        root.columns[1] = saved
        other = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        @test_throws ErrorException BNB._survivor_restore_basis!(other.model, root)
        ref = first(BNB.JuMP.all_constraints(m, BNB.JuMP.AffExpr, BNB.JuMP.MOI.EqualTo{Float64}))
        object = BNB.JuMP.constraint_object(ref)
        BNB.JuMP.delete(m, ref)
        BNB.JuMP.add_constraint(m, object)
        @test_throws ErrorException BNB._survivor_restore_basis!(m, root)
        BNB.JuMP.@variable(m, extra_column)
        @test_throws ErrorException BNB._survivor_restore_basis!(m, root)
    end

    @testset "presolve remains enabled on supported HiPO models" begin
        data, inputs = _bnb_fixture(; repeated=false)
        tree = BNB._build_survivor_full_model(
            data, SurvivorPoolState(2025, 1; strikes_remaining=2), _bnb_config(), inputs,
        )
        events = []
        function observe_presolve(event)
            push!(events, event)
            if event.kind in (:root, :node_start, :node_solve)
                @test BNB.JuMP.get_optimizer_attribute(tree.model, "presolve") == "on"
                @test BNB.JuMP.get_optimizer_attribute(tree.model, "run_crossover") == "off"
            end
        end
        BNB._optimize_survivor_branch_and_bound!(tree; observer=observe_presolve)
        @test any(e -> e.kind == :node_solve, events)
        @test last(events).proven
    end

    @testset "crossover-free certificates agree with parent-basis simplex" begin
        data, inputs = _bnb_fixture()
        for losses in 0:2, H in (0, 1, 3)
            tree = BNB._build_survivor_full_model(
                data, SurvivorPoolState(2025, 1; strikes_remaining=losses),
                _bnb_config(; hessian_weeks=H), inputs,
            )
            m = tree.model
            BNB.JuMP.optimize!(m)
            parent = BNB._survivor_snapshot_basis(m)
            for path in ([1], [2], [1, 4], [2, 3])
                @test BNB._survivor_tree_apply_path!(tree, path)
                BNB.JuMP.set_optimizer_attribute(m, "solver", "simplex")
                BNB.JuMP.set_optimizer_attribute(m, "presolve", "off")
                BNB._survivor_restore_basis!(m, parent; repair=true)
                BNB.JuMP.optimize!(m)
                @test BNB.JuMP.termination_status(m) == BNB.JuMP.MOI.OPTIMAL
                objective = BNB.JuMP.objective_value(m)
                certificate = BNB._survivor_tree_upper_bound(m)
                BNB.JuMP.set_optimizer_attribute(m, "solver", "hipo")
                BNB.JuMP.set_optimizer_attribute(m, "run_crossover", "off")
                BNB.JuMP.set_optimizer_attribute(m, "hipo_system", "normaleq")
                @test BNB.HiGHS.Highs_clearSolver(BNB.JuMP.backend(m).inner) ==
                      BNB.HiGHS.kHighsStatusOk
                BNB.JuMP.optimize!(m)
                @test BNB.JuMP.termination_status(m) == BNB.JuMP.MOI.OPTIMAL
                @test BNB.JuMP.objective_value(m) ≈ objective atol=2e-6
                @test BNB._survivor_tree_upper_bound(m) ≈ certificate atol=2e-6
                @test BNB._survivor_tree_upper_bound(m) >= objective - 1e-8
                @test BNB._survivor_benders_simplex_iterations(m) == 0
            end
        end
    end

    @testset "eligibility, structural conflicts, feasibility look-ahead" begin
        data, inputs = _bnb_fixture()
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        @test !BNB._survivor_tree_apply_path!(tree, [1, 3])
        @test !BNB._survivor_tree_apply_path!(tree, [1, 2])
        @test BNB._survivor_tree_candidates(tree, [1], 2) == [4]
        @test_throws ErrorException BNB._survivor_tree_plan(tree, [1, 3, 5])
        guarded = DataFrame(data)
        guarded.market_spread = [1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        config = SurvivorSelectionConfig(
            branch_and_bound=true, through_week=3, market_guard_weeks=1,
        )
        plan = BNB._optimize_survivor_expected_weeks_scalar_milp(guarded, state, config, inputs)
        @test only(plan.current_pick.team) == "B"
        constant = BNB._optimize_survivor_expected_weeks_scalar_milp(
            data, SurvivorPoolState(2025, 1; strikes_remaining=4), _bnb_config(), inputs,
        )
        @test constant.objective_value == 3
        @test all(iszero, constant.selections.parameter_variance_adjustment)
        @test constant.selection_config.branch_and_bound
        blocked = DataFrame(data)
        blocked.team .= "A"
        @test_throws ArgumentError BNB._build_survivor_full_model(blocked, state, _bnb_config(), inputs)
        trap = DataFrame(data)
        trap.team = ["A", "B", "A", "A", "C", "D"]
        indices = BNB._survivor_greedy_selected_indices(trap, state, _bnb_config(), inputs)
        @test indices[1] == 2
        @test length(unique(trap.team[indices])) == 3
    end

    @testset "constant objective bypasses MILP even with an expired budget" begin
        _, original = _bnb_fixture()
        data = DataFrame(
            game_id=string.(1:96), week=repeat(1:3; inner=32),
            team=repeat(["T$i" for i in 1:32], 3),
            opponent=fill("X", 96), is_home=fill(true, 96),
            win_probability=repeat(collect(range(0.95, 0.55; length=32)), 3),
            market_spread=fill(3.0, 96),
        )
        inputs = BNB.SurvivorObjectiveInputs(
            original.parameters,
            [BNB.SurvivorCandidateDerivatives(p, [0.3, -0.1], 0.02)
             for p in data.win_probability],
        )
        state = SurvivorPoolState(2025, 1; strikes_remaining=4)
        config = _bnb_config(; timeout_seconds=1e-9)
        plan = BNB._optimize_survivor_expected_weeks_scalar_milp(data, state, config, inputs)
        @test plan.selections.team == ["T1", "T2", "T3"]
        @test plan.selections.week == [1, 2, 3]
        @test plan.current_pick == plan.selections[1:1, :]
        @test plan.objective_value == 3.0
        @test plan.selection_config === config
        @test plan.state === state
        @test plan.selections.survival_probability == ones(3)
        @test plan.selections.elimination_probability == zeros(3)
        @test plan.selections.parameter_variance_adjustment == zeros(3)
        @test plan.selections.variance_adjusted_survival_probability == ones(3)
        @test plan.selections.objective_contribution == ones(3)
        @test plan.selections[:, names(data)] == data[[1, 34, 67], :]
        @test BNB._optimize_survivor_expected_weeks_scalar_milp(
            data, state, config, inputs,
        ).selections == plan.selections

        trap, trap_inputs = _bnb_fixture()
        trap.team = ["A", "B", "A", "A", "C", "D"]
        matched = BNB._optimize_survivor_expected_weeks_scalar_milp(
            trap, state, config, trap_inputs,
        )
        @test matched.selections.team == ["B", "A", "C"]
        used_state = SurvivorPoolState(2025, 2; strikes_remaining=4, picks_made=Dict(1 => "C"))
        used_trap = DataFrame(trap)
        used_trap.week .+= 1
        @test BNB._optimize_survivor_expected_weeks_scalar_milp(
            used_trap, used_state, _bnb_config(; through_week=4, timeout_seconds=1e-9), trap_inputs,
        ).selections.team == ["B", "A", "D"]
        banned_config = _bnb_config(; banned_first_pick_teams=["T1"], timeout_seconds=1e-9)
        filtered = BNB._normalize_survivor_candidates(
            data, state, 3; banned_first_pick_teams=banned_config.banned_first_pick_teams,
        )
        filtered_inputs = BNB.SurvivorObjectiveInputs(
            inputs.parameters, inputs.derivatives[parse.(Int, filtered.game_id)],
        )
        @test BNB._optimize_survivor_expected_weeks_scalar_milp(
            filtered, state, banned_config, filtered_inputs,
        ).selections.team == ["T2", "T1", "T3"]
        trap.market_spread[1] = 1.0
        guarded_config = SurvivorSelectionConfig(
            branch_and_bound=true, through_week=3, market_guard_weeks=1,
            timeout_seconds=1e-9,
        )
        @test only(BNB._optimize_survivor_expected_weeks_scalar_milp(
            trap, state, guarded_config, trap_inputs,
        ).current_pick.team) == "B"
        trap.team .= "A"
        @test_throws ArgumentError BNB._optimize_survivor_expected_weeks_scalar_milp(
            trap, state, config, trap_inputs,
        )
        mktempdir() do directory
            path = joinpath(directory, "constant.lp")
            @test BNB._optimize_survivor_expected_weeks_scalar_milp(
                data, state, config, inputs; lp_output_file=path, export_lp_only=true,
            ) == path
            @test occursin("binary", lowercase(read(path, String)))
            @test_throws ArgumentError BNB._optimize_survivor_expected_weeks_scalar_milp(
                data, state, config, inputs; export_lp_only=true,
            )
        end
    end

    @testset "build timeout retains exact feasible incumbent" begin
        data, inputs = _bnb_fixture()
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        config = _bnb_config(; timeout_seconds=1e-9)
        events = []
        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        @test_logs (:warn, r"first pick is unproven") begin
            plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=e -> push!(events, e))
            @test nrow(plan.selections) == 3
            @test plan.objective_value == BNB._survivor_tree_plan(tree, tree.warm_start.selected).objective_value
        end
        @test last(events).reason == :build_timeout
        @test !last(events).proven
        @test isfinite(last(events).upper)
    end
end
