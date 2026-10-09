using Test
using Random
using SurvivorModel
using DataFrames

const BNB = SurvivorModel

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

function _bnb_config(;
    through_week=3,
    hessian_weeks=3,
    branch_and_bound_workers=1,
    kwargs...,
)
    return SurvivorSelectionConfig(
        ;
        minimum_favorite_spread=nothing, market_guard_weeks=0,
        through_week, hessian_weeks, branch_and_bound=true,
        branch_and_bound_workers, kwargs...,
    )
end

function _bnb_optimize_single_threaded!(model)
    BNB.HiGHS.Highs_resetGlobalScheduler(1)
    try
        return BNB.JuMP.optimize!(model)
    finally
        BNB.HiGHS.Highs_resetGlobalScheduler(1)
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

function _bnb_numeric_snapshot(tree)
    model = tree.model
    variables = [(BNB.JuMP.lower_bound(v), BNB.JuMP.upper_bound(v))
                 for v in BNB.JuMP.all_variables(model)]
    rows = [(BNB.JuMP.constraint_object(ref).func,
             BNB.JuMP.constraint_object(ref).set)
            for (F, S) in BNB.JuMP.list_of_constraint_types(model) if F == BNB.JuMP.AffExpr
            for ref in BNB.JuMP.all_constraints(model, F, S)]
    return (variables, rows)
end

@testset "Greedy completion preserves fixed picks" begin
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

@testset "Branch-conditioned recurrence intervals and hull updates" begin
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
        schedules = [collect(raw) for raw in Iterators.product([1, 2], [3, 4], [5, 6])
                     if length(unique(data.team[collect(raw)])) == 3]
        for path in (Int[], [1], [2], [3], [5], [2, 5], first(schedules))
            completions = filter(schedule -> all(in(schedule), path), schedules)
            isempty(completions) && continue
            @test BNB._survivor_tree_apply_path!(tree, path)
            bounds = tree.model.ext[:survivor_node_bounds]
            @test BNB.JuMP.num_variables(tree.model) == variables
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
end

@testset "HiPO bounds require current dual-valid results" begin
    rng = MersenneTwister(88)
    matrix = rand(rng, 80, 120)
    rhs = 0.44 .* vec(sum(matrix; dims=2))
    costs = rand(rng, 120)
    accepted_nonoptimal = false
    invalid_finite_bound = NaN
    optimum = NaN
    for limit in (4, 8, 12, 40)
        model = BNB._survivor_direct_milp_model(_bnb_config(); lp_relaxation=true)
        BNB.JuMP.@variable(model, 0 <= x[1:120] <= 1)
        BNB.JuMP.@constraint(model, matrix * x .>= rhs)
        BNB.JuMP.@objective(model, Min, sum(costs .* x) + 7.25)
        for (option, value) in (
            ("solver", "hipo"), ("run_crossover", "off"),
            ("presolve", "off"), ("ipm_iteration_limit", limit),
        )
            BNB.JuMP.set_optimizer_attribute(model, option, value)
        end
        _bnb_optimize_single_threaded!(model)
        status = BNB.JuMP.termination_status(model)
        raw_bound = BNB.JuMP.objective_bound(model)
        @test isfinite(raw_bound)
        result = BNB._survivor_tree_solver_upper_bound(model, status)
        if result.upper !== nothing
            @test BNB.JuMP.dual_status(model) == BNB.JuMP.MOI.FEASIBLE_POINT
            @test result.upper ≈ -raw_bound
            @test result.reason == :solver_dual_bound
            accepted_nonoptimal |= status != BNB.JuMP.MOI.OPTIMAL
        else
            @test result.reason in (
                :no_solver_result, :dual_not_feasible, :native_dual_not_feasible,
                :dual_feasibility_tolerance_unavailable,
                :dual_residual_tolerance_unavailable,
                :dual_infeasibility_exceeds_tolerance,
                :dual_residual_exceeds_tolerance,
            )
        end
        if limit == 4
            @test result.upper === nothing
            invalid_finite_bound = raw_bound
        elseif limit == 40
            @test status == BNB.JuMP.MOI.OPTIMAL
            optimum = BNB.JuMP.objective_value(model)
        end
    end
    @test accepted_nonoptimal
    @test invalid_finite_bound > optimum + 1e-3

    model = BNB._survivor_direct_milp_model(_bnb_config(); lp_relaxation=true)
    BNB.JuMP.@variable(model, 0 <= x <= 1)
    BNB.JuMP.@variable(model, 0 <= y <= 1)
    BNB.JuMP.@constraint(model, x + y <= 1)
    BNB.JuMP.@objective(model, Max, 2x + y + 11)
    for (option, value) in (
        ("solver", "hipo"), ("run_crossover", "on"), ("presolve", "off"),
    )
        BNB.JuMP.set_optimizer_attribute(model, option, value)
    end
    _bnb_optimize_single_threaded!(model)
    result = BNB._survivor_tree_solver_upper_bound(
        model, BNB.JuMP.termination_status(model),
    )
    @test result.upper ≈ BNB.JuMP.objective_bound(model)
    @test result.upper >= 13
end

@testset "Tree search isolates the HiGHS scheduler" begin
    data, inputs = _bnb_fixture()
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    tree = BNB._build_survivor_full_model(
        data, state, _bnb_config(; branch_and_bound_workers=2), inputs,
    )

    function solve_default_model()
        model = BNB.JuMP.Model(BNB.HiGHS.Optimizer)
        BNB.JuMP.set_silent(model)
        BNB.JuMP.@variable(model, selected[1:3], Bin)
        BNB.JuMP.@constraint(model, sum(selected) <= 2)
        BNB.JuMP.@objective(model, Max, sum(selected))
        BNB._survivor_with_default_highs_scheduler() do
            BNB.JuMP.optimize!(model)
        end
        return BNB.JuMP.termination_status(model)
    end

    @test solve_default_model() == BNB.JuMP.MOI.OPTIMAL
    events = []
    plan = BNB._optimize_survivor_branch_and_bound!(
        tree; observer=event -> push!(events, event),
    )
    @test only(plan.current_pick.team) in data.team[data.week .== 1]
    @test last(events).proven
    @test solve_default_model() == BNB.JuMP.MOI.OPTIMAL
end

@testset "Worker HiGHS models own remapped tree references" begin
    data, inputs = _bnb_fixture()
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
    BNB.JuMP.set_objective_function(tree.model, -BNB.JuMP.objective_function(tree.model))
    BNB.JuMP.set_objective_sense(tree.model, BNB.JuMP.MOI.MIN_SENSE)
    original_snapshot = _bnb_numeric_snapshot(tree)
    worker = BNB._survivor_tree_copy_worker(tree)
    JuMP = BNB.JuMP
    @test JuMP.num_variables(worker.model) == JuMP.num_variables(tree.model)
    @test JuMP.num_constraints(worker.model; count_variable_in_set_constraints=true) ==
          JuMP.num_constraints(tree.model; count_variable_in_set_constraints=true)
    @test JuMP.objective_sense(worker.model) == JuMP.MOI.MIN_SENSE
    @test [JuMP.get_optimizer_attribute(worker.model, option)
           for option in ("solver", "threads", "parallel", "run_crossover")] ==
          ["hipo", 1, "off", "on"]
    for variable in worker.selected
        @test JuMP.owner_model(variable) === worker.model
    end
    for variable in worker.probability
        @test JuMP.owner_model(variable) === worker.model
    end
    for variable in worker.hessian
        @test JuMP.owner_model(variable) === worker.model
    end
    for states in worker.parameter_gradient, variable in states
        @test JuMP.owner_model(variable) === worker.model
    end
    for states in worker.gradient
        states === nothing && continue
        for variable in states
            @test JuMP.owner_model(variable) === worker.model
        end
    end
    for hull in worker.model.ext[:survivor_node_hulls]
        @test JuMP.owner_model(hull.dummy) === worker.model
        @test JuMP.owner_model(hull.selected) === worker.model
        for row in (
            hull.lower_row, hull.upper_row,
            hull.other_upper_row, hull.other_lower_row,
        )
            @test JuMP.owner_model(row) === worker.model
        end
    end
    @test BNB._survivor_tree_apply_path!(worker, [1])
    first_path_snapshot = _bnb_numeric_snapshot(worker)
    @test BNB._survivor_tree_apply_path!(worker, [2])
    @test BNB._survivor_tree_apply_path!(worker, [1])
    @test _bnb_numeric_snapshot(worker) == first_path_snapshot
    @test _bnb_numeric_snapshot(tree) == original_snapshot
end

@testset "Parallel HiPO tree certifies the exhaustive first pick" begin
    data, inputs = _bnb_fixture()
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    config = _bnb_config(; branch_and_bound_workers=2)
    exhaustive = _bnb_exhaustive(data, inputs, state, config)
    tree = BNB._build_survivor_full_model(data, state, config, inputs)
    events = []
    active = Set{Int}()
    max_active = Ref(0)
    function observe(event)
        push!(events, event)
        if event.kind == :node_start
            push!(active, hasproperty(event, :worker) ? event.worker : 1)
            max_active[] = max(max_active[], length(active))
        elseif event.kind in (
            :node_solve, :node_structural, :node_preparation_timeout,
        )
            worker = hasproperty(event, :worker) ? event.worker : 1
            @test worker in active
            delete!(active, worker)
        elseif event.kind == :workers_joined
            @test event.tasks_terminated
            empty!(active)
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
        end
    end
    plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=observe)
    first_index = only(findall((data.week .== 1) .&
                              (data.team .== only(plan.current_pick.team))))
    @test exhaustive[first_index] >= maximum(values(exhaustive)) - 5e-6
    @test plan.objective_value <= exhaustive[first_index] + 2e-6
    @test last(events).proven
    @test isempty(active)
    if Threads.nthreads(:default) >= 2
        @test max_active[] >= 2
        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        calls = Threads.Atomic{Int}(0)
        barrier = Threads.Condition()
        arrived = 0
        entering_solver = 0
        max_entering_solver = 0
        function synchronize_worker_solves(model)
            call = Threads.atomic_add!(calls, 1) + 1
            call == 1 && return BNB.JuMP.optimize!(model)
            lock(barrier) do
                arrived += 1
                if arrived == 2
                    notify(barrier, all=true)
                else
                    while arrived < 2
                        wait(barrier)
                    end
                end
                entering_solver += 1
                max_entering_solver = max(
                    max_entering_solver, entering_solver,
                )
                if entering_solver == 2
                    notify(barrier, all=true)
                else
                    while entering_solver < 2
                        wait(barrier)
                    end
                end
            end
            status = BNB.JuMP.optimize!(model)
            lock(barrier) do
                entering_solver -= 1
                notify(barrier, all=true)
            end
            return status
        end
        parallel_events = []
        BNB._optimize_survivor_branch_and_bound!(
            tree;
            observer=event -> push!(parallel_events, event),
            optimize_relaxation! = synchronize_worker_solves,
        )
        parallel_finish = only(filter(
            event -> event.kind == :finish, parallel_events,
        ))
        @test arrived == 2
        @test max_entering_solver == 2
        @test parallel_finish.nodes >= calls[] - 1
    else
        @test max_active[] <= 1
    end
end

if Threads.nthreads(:default) >= 2
    @testset "Parallel worker failures and shared timeout shut down cleanly" begin
        data, inputs = _bnb_fixture()
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        config = _bnb_config(; branch_and_bound_workers=2)

        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        calls = Threads.Atomic{Int}(0)
        function fail_in_worker(model)
            call = Threads.atomic_add!(calls, 1) + 1
            call == 1 && return BNB.JuMP.optimize!(model)
            error("injected worker failure")
        end
        failure = try
            BNB._optimize_survivor_branch_and_bound!(
                tree; optimize_relaxation! = fail_in_worker,
            )
            nothing
        catch error
            error
        end
        @test failure isa ErrorException
        @test occursin("worker", sprint(showerror, failure))
        @test occursin("injected worker failure", sprint(showerror, failure))

        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        force_timeout = Threads.Atomic{Bool}(false)
        events = []
        function observe_timeout(event)
            push!(events, event)
            event.kind == :node_start && (force_timeout[] = true)
        end
        remaining_time = () -> force_timeout[] ? 0.0 : nothing
        @test_logs (:warn, r"stopped before proving the first pick") match_mode=:any begin
            plan = BNB._optimize_survivor_branch_and_bound!(
                tree;
                observer=observe_timeout,
                remaining_time,
            )
            @test nrow(plan.selections) == 3
        end
        final = only(filter(event -> event.kind == :finish, events))
        joined = only(filter(event -> event.kind == :workers_joined, events))
        @test !final.proven
        @test final.reason == :node_timeout
        @test joined.tasks_terminated
        @test final.nodes == 0
        @test !isempty(final.pending)
    end
end

@testset "HiPO tree matches exhaustive schedules over horizons" begin
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
                    ("solver", "hipo"), ("run_crossover", "on"),
                    ("parallel", "off"), ("threads", 1),
                )
                    @test BNB.JuMP.get_optimizer_attribute(tree.model, option) == value
                end
            end
            if event.kind in (:root, :node_start, :node_solve)
                @test BNB.JuMP.get_optimizer_attribute(tree.model, "presolve") == "choose"
            end
        end
        plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=observe_solver)
        first_index = only(findall((data.week .== current) .&
                                  (data.team .== only(plan.current_pick.team))))
        @test exhaustive[first_index] >= maximum(values(exhaustive)) - 5e-6
        @test plan.objective_value <= exhaustive[first_index] + 2e-6
        @test sum(plan.selections.objective_contribution) ≈ plan.objective_value
        indices = [only(findall((data.week .== week) .& (data.team .== team)))
                   for (week, team) in zip(plan.selections.week, plan.selections.team)]
        @test plan.objective_value ≈
              _bnb_independent_objective(indices, inputs, state, config) atol=2e-6
        @test length(unique(plan.selections.team)) == 3
        @test last(events).proven
        for event in events
            if event.kind == :node_optimum
                @test event.bound <= event.node.upper
            elseif event.kind == :queue
                for region in keys(exhaustive)
                    @test event.upper[region] >= exhaustive[region] - 5e-6
                end
            end
        end
        extensive_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing, market_guard_weeks=0,
            through_week=current + 2, hessian_weeks=H,
        )
        extensive = BNB._optimize_survivor_expected_weeks_scalar_milp(
            data, state, extensive_config, inputs,
        )
        @test extensive.objective_value ≈ maximum(values(exhaustive)) atol=2e-6
    end
end

@testset "HiPO crossover and dual bounds agree" begin
    data, inputs = _bnb_fixture()
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    config = _bnb_config(; hessian_weeks=2)
    for path in ([1], [1, 4])
        results = Dict{String,NamedTuple}()
        for crossover in ("off", "on")
            tree = BNB._build_survivor_full_model(data, state, config, inputs)
            @test BNB._survivor_tree_apply_path!(tree, path)
            model = tree.model
            JuMP = BNB.JuMP
            JuMP.set_objective_function(model, -JuMP.objective_function(model))
            JuMP.set_objective_sense(model, JuMP.MOI.MIN_SENSE)
            for (option, value) in (
                ("solver", "hipo"), ("run_crossover", crossover),
                ("parallel", "off"), ("threads", 1), ("presolve", "off"),
            )
                JuMP.set_optimizer_attribute(model, option, value)
            end
            seconds = @elapsed _bnb_optimize_single_threaded!(model)
            @test JuMP.termination_status(model) == JuMP.MOI.OPTIMAL
            @test JuMP.primal_status(model) == JuMP.MOI.FEASIBLE_POINT
            @test JuMP.dual_status(model) == JuMP.MOI.FEASIBLE_POINT
            objective = -JuMP.objective_value(model)
            result = BNB._survivor_tree_solver_upper_bound(model, JuMP.termination_status(model))
            @test result.upper !== nothing
            upper = result.upper
            @test upper >= objective - 2e-6
            results[crossover] = (; seconds, objective, upper)
        end
        @test results["on"].objective ≈ results["off"].objective atol=2e-6
        @test results["on"].upper >= results["on"].objective - 2e-6
        @test results["off"].upper >= results["off"].objective - 2e-6
    end
end

@testset "Tree partitioning, ranking, and proof propagation" begin
    for losses in (0, 2)
        data, inputs = _bnb_fixture(; repeated=false)
        state = SurvivorPoolState(2025, 1; strikes_remaining=losses)
        config = _bnb_config()
        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        events = []
        function observe(event)
            push!(events, event)
            if event.kind == :root_optimum && losses == 0
                @test event.assignment.week == 3
                @test BNB.JuMP.value(tree.selected[1]) ≈ 1
                @test abs(BNB.JuMP.value(tree.selected[2])) <= 1e-7
            elseif event.kind == :node_optimum && event.assignment.week !== nothing
                fixed = tree.candidate_positions[event.node.path]
                @test !(event.assignment.week in fixed)
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
                if previous.kind == :node_branch
                    children = filter(node -> node.id in previous.children, event.nodes)
                    if !isempty(children)
                        @test sort(last.(getproperty.(children, :path))) ==
                              sort(BNB._survivor_tree_candidates(
                                  tree, previous.node.path, previous.week,
                              ))
                        @test all(node -> node.upper <= previous.node.upper, children)
                    end
                end
                exhaustive = _bnb_exhaustive(data, inputs, state, config)
                for region in keys(exhaustive)
                    @test event.upper[region] >= exhaustive[region] - 2e-6
                end
            end
        end
        plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=observe)
        partition = only(filter(event -> event.kind == :partition, events))
        @test sort(getproperty.(partition.nodes, :region)) == [1, 2]
        @test all(node -> length(node.path) == 1, partition.nodes)
        @test last(events).proven
        @test plan.objective_value <= maximum(values(_bnb_exhaustive(data, inputs, state, config))) + 2e-6
        @test !(:parent_basis in fieldnames(BNB.SurvivorBranchNode))
        @test !(:simplex in fieldnames(typeof(config)))

    end
    data, inputs = _bnb_fixture()
    tree = BNB._build_survivor_full_model(data, SurvivorPoolState(2025, 1), _bnb_config(), inputs)
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
    tied = [
        BNB.SurvivorBranchNode(10, 2, [2], 4.0, 2.0, true),
        BNB.SurvivorBranchNode(11, 2, [2, 3], 4.0, 3.0, true),
    ]
    @test BNB._survivor_tree_next_node(tied, 1, 1) == 2
    tied[1] = BNB.SurvivorBranchNode(10, 2, [2], 4.1, 2.0, true)
    @test BNB._survivor_tree_next_node(tied, 1, 1) == 1
    tied[1] = BNB.SurvivorBranchNode(10, 3, [3], 4.0, 2.0, true)
    @test BNB._survivor_tree_next_node(tied, 1, 1, Dict(2 => 4.0, 3 => 4.2)) == 1

    roots = Dict(1 => 10.0, 2 => 12.0)
    summaries = Dict(
        1 => BNB._survivor_tree_summary(0, 1, 10.0),
        2 => BNB._survivor_tree_summary(0, 2, 12.0),
    )
    BNB._survivor_tree_branch!(summaries, roots, 1, 9.0, [3, 4])
    BNB._survivor_tree_branch!(summaries, roots, 3, 7.0, [5, 6])
    BNB._survivor_tree_close!(summaries, roots, 5, -Inf)
    BNB._survivor_tree_close!(summaries, roots, 6, 6.0)
    @test summaries[1].closed_upper == 6.0
    BNB._survivor_tree_close!(summaries, roots, 4, 5.0)
    @test roots[1] == 6.0
    @test collect(keys(summaries)) == [2]
    @test all(field -> !(field in (:path, :basis)), fieldnames(BNB.SurvivorBranchSummary))
end

@testset "HiPO failures branch without application retries" begin
    data, inputs = _bnb_fixture(; repeated=false)
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    config = _bnb_config()
    exhaustive = _bnb_exhaustive(data, inputs, state, config)
    for phase in (:root, :child)
        tree = BNB._build_survivor_full_model(data, state, config, inputs)
        events = []
        calls = Ref(0)
        target_call = phase == :root ? 1 : 2
        function fail_once!(model)
            calls[] += 1
            BNB.JuMP.optimize!(model)
            if calls[] == target_call
                solution = BNB.JuMP.backend(model).solution
                solution.model_status = BNB.HiGHS.kHighsModelStatusUnknown
                solution.primal_solution_status = BNB.HiGHS.kHighsSolutionStatusNone
                solution.dual_solution_status = BNB.HiGHS.kHighsSolutionStatusNone
                @test BNB.JuMP.termination_status(model) == BNB.JuMP.MOI.OTHER_ERROR
            end
            return nothing
        end
        plan = BNB._optimize_survivor_branch_and_bound!(
            tree; observer=event -> push!(events, event),
            optimize_relaxation! = fail_once!,
        )
        @test last(events).proven
        first_index = only(findall((data.week .== 1) .&
                                  (data.team .== only(plan.current_pick.team))))
        @test exhaustive[first_index] >= maximum(values(exhaustive)) - 5e-6
        @test plan.objective_value <= exhaustive[first_index] + 2e-6
        @test calls[] == count(event -> event.kind == :root, events) +
                         count(event -> event.kind == :node_solve, events)
        @test !any(event -> event.kind == :retry, events)
        if phase == :root
            fallback = only(filter(event -> event.kind == :root_fallback, events))
            @test fallback.status == BNB.JuMP.MOI.OTHER_ERROR
            @test fallback.bound_source == :global_interval
            @test fallback.assignment.indices === nothing
            @test fallback.assignment.reason in (:no_solver_result, :no_feasible_primal)
        else
            fallback = only(filter(
                event -> event.kind == :node_fallback &&
                         event.status == BNB.JuMP.MOI.OTHER_ERROR, events,
            ))
            @test fallback.bound_source == :inherited
            branch = only(filter(
                event -> event.kind == :node_branch &&
                         event.status == BNB.JuMP.MOI.OTHER_ERROR, events,
            ))
            @test branch.week == BNB._survivor_tree_fallback_week(tree, branch.node.path)
            @test length(branch.children) ==
                  length(BNB._survivor_tree_candidates(tree, branch.node.path, branch.week))
        end
    end
end

@testset "Timeouts retain feasible incumbents and bounds" begin
    data, inputs = _bnb_fixture()
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    for phase in (:root, :child)
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        events = []
        phase == :root && BNB.JuMP.set_time_limit_sec(tree.model, 1e-9)
        function observe(event)
            push!(events, event)
            phase == :child && event.kind == :node_start &&
                BNB.JuMP.set_time_limit_sec(tree.model, 1e-9)
        end
        @test_logs (:warn, r"stopped before proving the first pick") match_mode=:any begin
            plan = BNB._optimize_survivor_branch_and_bound!(tree; observer=observe)
            @test nrow(plan.selections) == 3
        end
        final = last(events)
        @test !final.proven
        @test final.reason == (phase == :root ? :root_timeout : :interrupted_node)
        @test final.upper >= maximum(values(_bnb_exhaustive(data, inputs, state, _bnb_config())))
        if phase == :child
            partition = only(filter(event -> event.kind == :partition, events))
            @test final.upper == first(partition.nodes).upper
            @test final.nodes == 1
            @test length(final.pending) == length(partition.nodes)
            @test final.retained == length(partition.nodes)
            @test all(==(final.upper), values(final.root_bounds))
        end
    end
    tree = BNB._build_survivor_full_model(
        data, state, _bnb_config(; timeout_seconds=1e-9), inputs,
    )
    events = []
    @test_logs (:warn, r"stopped before proving the first pick") begin
        BNB._optimize_survivor_branch_and_bound!(tree; observer=event -> push!(events, event))
    end
    @test last(events).reason == :build_timeout
    @test !last(events).proven
    @test isfinite(last(events).upper)
end

@testset "Eligibility, zero-valued picks, and structural infeasibility" begin
    data, inputs = _bnb_fixture()
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
    @test !BNB._survivor_tree_apply_path!(tree, [1, 3])
    @test !BNB._survivor_tree_apply_path!(tree, [1, 2])
    @test BNB._survivor_tree_candidates(tree, [1], 2) == [4]
    @test_throws ErrorException BNB._survivor_tree_plan(tree, [1, 3, 5])

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
    @test sort(getproperty.(only(filter(e -> e.kind == :partition, events)).nodes, :region)) == [1, 2]
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

    guarded_data, guarded_inputs = _bnb_fixture()
    guarded = DataFrame(guarded_data)
    guarded.market_spread = [1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    config = SurvivorSelectionConfig(
        branch_and_bound=true, through_week=3, market_guard_weeks=1,
    )
    @test only(BNB._optimize_survivor_expected_weeks_scalar_milp(
        guarded, SurvivorPoolState(2025, 1; strikes_remaining=2), config,
        guarded_inputs,
    ).current_pick.team) == "B"
end

@testset "Debug progress is table-only and aligned" begin
    data, inputs = _bnb_fixture(; repeated=false)
    state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
    logger_output = IOBuffer()
    mktempdir() do directory
        stdout_path = joinpath(directory, "stdout")
        stderr_path = joinpath(directory, "stderr")
        open(stdout_path, "w") do output
            open(stderr_path, "w") do stderr
                redirect_stdout(output) do
                    redirect_stderr(stderr) do
                        BNB.Logging.with_logger(
                            BNB.Logging.ConsoleLogger(logger_output, BNB.Logging.Debug),
                        ) do
                            @test BNB._survivor_debug_logging_enabled()
                            BNB._optimize_survivor_branch_and_bound!(tree)
                        end
                    end
                end
            end
        end
        @test isempty(read(stdout_path, String))
        progress = read(stderr_path, String)
        header = BNB._survivor_tree_progress_header()
        @test occursin(header, progress)
        @test occursin("ROOT", progress)
        @test occursin("FINAL", progress)
        @test occursin("PROVEN", progress)
        @test !occursin("HiGHS relaxation", progress)
        @test !occursin("survivor branch-and-bound", String(take!(logger_output)))
        header_separators = findall(==('|'), header)
        rows = filter(line -> !isempty(line) && occursin('|', line), split(progress, '\n'))
        @test length(rows) > 2
        @test all(row -> findall(==('|'), row) == header_separators, rows)
        @test all(row -> length(row) == length(header), rows)
    end

    for phase in (:root, :child)
        data, inputs = _bnb_fixture(; repeated=false)
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        tree = BNB._build_survivor_full_model(data, state, _bnb_config(), inputs)
        logger_output = IOBuffer()
        events = []
        calls = Ref(0)
        target_call = phase == :root ? 1 : 2
        function fail_once!(model)
            calls[] += 1
            BNB.JuMP.optimize!(model)
            if calls[] == target_call
                solution = BNB.JuMP.backend(model).solution
                solution.model_status = BNB.HiGHS.kHighsModelStatusUnknown
                solution.primal_solution_status = BNB.HiGHS.kHighsSolutionStatusNone
                solution.dual_solution_status = BNB.HiGHS.kHighsSolutionStatusNone
            end
            return nothing
        end
        progress = mktemp() do _, stdout
            progress = mktemp() do _, stderr
                redirect_stdout(stdout) do
                    redirect_stderr(stderr) do
                        BNB.Logging.with_logger(
                            BNB.Logging.ConsoleLogger(logger_output, BNB.Logging.Debug),
                        ) do
                            BNB._optimize_survivor_branch_and_bound!(
                                tree;
                                observer=event -> push!(events, event),
                                optimize_relaxation! = fail_once!,
                            )
                        end
                    end
                end
                flush(stderr)
                seekstart(stderr)
                read(stderr, String)
            end
            flush(stdout)
            seekstart(stdout)
            @test isempty(read(stdout, String))
            progress
        end
        @test isempty(strip(String(take!(logger_output))))
        @test occursin("ROOT", progress)
        if phase == :root
            @test occursin("global_interval", progress)
            @test occursin("no_solver_result", progress) ||
                  occursin("no_feasible_primal", progress)
        else
            @test occursin("inherited", progress)
            @test occursin("BRANCH", progress)
            @test occursin("no_solver_result", progress) ||
                  occursin("no_feasible_primal", progress)
        end
        header = BNB._survivor_tree_progress_header()
        separators = findall(==('|'), header)
        rows = filter(line -> !isempty(line) && occursin('|', line), split(progress, '\n'))
        @test all(row -> findall(==('|'), row) == separators, rows)
        @test all(row -> length(row) == length(header), rows)
    end

    forced_data, forced_inputs = _bnb_fixture()
    forced_data.market_spread = [1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    forced_config = SurvivorSelectionConfig(
        branch_and_bound=true, through_week=3, market_guard_weeks=1,
    )
    forced_state = SurvivorPoolState(2025, 1; strikes_remaining=2)
    forced_logger = IOBuffer()
    forced_progress = mktemp() do _, stdout
        progress = mktemp() do _, stderr
            redirect_stdout(stdout) do
                redirect_stderr(stderr) do
                    BNB.Logging.with_logger(
                        BNB.Logging.ConsoleLogger(forced_logger, BNB.Logging.Debug),
                    ) do
                        plan = BNB._optimize_survivor_expected_weeks_scalar_milp(
                            forced_data, forced_state, forced_config, forced_inputs,
                        )
                        @test only(plan.current_pick.team) == "B"
                    end
                end
            end
            flush(stderr)
            seekstart(stderr)
            read(stderr, String)
        end
        flush(stdout)
        seekstart(stdout)
        @test isempty(read(stdout, String))
        progress
    end
    @test isempty(strip(String(take!(forced_logger))))
    @test occursin("ROOT", forced_progress)
    @test occursin("FINAL", forced_progress)
    @test occursin("PROVEN", forced_progress)
    forced_header = BNB._survivor_tree_progress_header()
    forced_rows = filter(
        line -> !isempty(line) && occursin('|', line), split(forced_progress, '\n'),
    )
    @test length(forced_rows) == 3
    @test all(row -> length(row) == length(forced_header), forced_rows)

    dummy_row = (
        1, "TEAM", 0, "-", 1, "FIRST", 1.0, 0.5, 1.0, "BRANCH",
        "inherited", "iter_limit/no_primal", 2, 0, 0.01,
    )
    output = mktemp() do _, io
        rows = Ref(0)
        redirect_stderr(io) do
            for _ in 1:26
                BNB._survivor_tree_emit_progress!(rows, dummy_row)
            end
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
    @test count(==(BNB._survivor_tree_progress_header()), split(output, '\n')) == 2
end

@testset "Retired solver options are rejected" begin
    config = SurvivorSelectionConfig()
    @test !config.branch_and_bound
    @test !(:simplex in fieldnames(typeof(config)))
    @test !(:benders_weeks in fieldnames(typeof(config)))
    @test_throws MethodError SurvivorSelectionConfig(simplex=true)
    @test_throws MethodError SurvivorSelectionConfig(benders_weeks=1)
    @test_throws ArgumentError SurvivorSelectionConfig(branch_and_bound=true, hessian_weeks=-1)
end
