using DataFrames
using Dates
using Test
using SurvivorModel

function _survivor_forecast_fixture()
    return DataFrame(
        game_id=["week1_ab", "week1_cd", "week2_ac", "week2_de"],
        week=[1, 1, 2, 2],
        away_team=["A", "C", "A", "D"],
        home_team=["B", "D", "C", "E"],
        away_win_probability=[0.5, 0.7, 0.95, 0.6],
        home_win_probability=[0.8, 0.6, 0.4, 0.3],
        game_completed=[false, false, false, false],
    )
end

function _survivor_context_fixture()
    schedule = DataFrame(
        game_id=["2022_01_AB", "2023_01_AB", "2023_02_CD"],
        season=[2022, 2023, 2023],
        game_type=["REG", "REG", "REG"],
        week=[1, 1, 2],
        away_team=["A", "A", "C"],
        home_team=["B", "B", "D"],
        away_score=Union{Missing,Int}[14, 17, missing],
        home_score=Union{Missing,Int}[21, 24, missing],
        result=Union{Missing,Int}[7, 7, missing],
    )
    historical = DataFrame(
        game_id=["2022_01_AB", "2022_01_AB", "2022_01_AB", "2022_01_AB"],
        fixed_drive=[1, 2, 3, 4],
        posteam=["B", "A", "B", "A"],
        defteam=["A", "B", "A", "B"],
        posteam_home=[true, false, true, false],
        defteam_home=[false, true, false, true],
        drive_result=["Touchdown", "Punt", "Field goal", "Punt"],
        time_of_possession=Second.([60, 60, 60, 60]),
        home_spread_change=[7.0, 0.0, 3.0, 0.0],
    )
    current = DataFrame(
        game_id=["2023_01_AB", "2023_01_AB"],
        fixed_drive=[1, 2],
        posteam=["B", "A"],
        defteam=["A", "B"],
        posteam_home=[true, false],
        defteam_home=[false, true],
        drive_result=["Touchdown", "Punt"],
        time_of_possession=Second.([60, 60]),
        home_spread_change=[7.0, 0.0],
    )
    return schedule, historical, current
end

function _market_guard_candidates()
    return DataFrame(
        game_id=[
            "week1_guard", "week1_guard",
            "week2_guard", "week2_guard",
            "week3_guard", "week3_guard",
        ],
        week=[1, 1, 2, 2, 3, 3],
        team=["A", "B", "C", "D", "E", "F"],
        opponent=["X", "Y", "Z", "W", "V", "U"],
        is_home=[false, true, false, true, false, true],
        win_probability=[0.99, 0.8, 0.98, 0.7, 0.97, 0.6],
        market_spread=[1.0, 2.0, 1.5, 3.0, -10.0, -3.0],
    )
end

function _capture_stderr_output(f::Function)
    return mktemp() do _, output
        result = redirect_stderr(output) do
            f()
        end
        flush(output)
        seekstart(output)
        return result, read(output, String)
    end
end

function _test_benders_progress_rows(output::AbstractString)
    rows = Vector{Vector{String}}()
    for line in split(output, '\n')
        fields = strip.(split(line, " | "))
        length(fields) == 13 || continue
        tryparse(Int, fields[1]) === nothing && continue
        push!(rows, String.(fields))
    end
    return rows
end

@testset "survivor pool optimization" begin
    @testset "candidate expansion and filtering" begin
        forecast = _survivor_forecast_fixture()
        candidates = build_survivor_candidates(forecast)
        @test nrow(candidates) == 6
        @test Set(candidates.team) == Set(["A", "B", "C", "D"])
        @test all(
            candidates.win_probability .>=
            DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY,
        )
        @test candidates.win_probability[
            (candidates.week .== 1) .& (candidates.team .== "B")
        ][1] == 0.8

        filtered = build_survivor_candidates(
            forecast;
            picks_made=Dict(1 => "A"),
        )
        @test !any(filtered.team .== "A")

        malformed = vcat(
            forecast,
            DataFrame(
                game_id=["week1_extra"],
                week=[1],
                away_team=["A"],
                home_team=["E"],
                away_win_probability=[0.5],
                home_win_probability=[0.5],
                game_completed=[false],
            );
            cols=:union,
        )
        @test_throws ArgumentError build_survivor_candidates(malformed)

        forecast_with_spreads = _survivor_forecast_fixture()
        forecast_with_spreads.spread_line = [2.5, -3.0, 4.0, -1.5]
        spread_candidates = build_survivor_candidates(forecast_with_spreads)
        @test spread_candidates.market_spread[
            (spread_candidates.week .== 1) .&
            (spread_candidates.team .== "B")
        ][1] == 2.5
        @test spread_candidates.market_spread[
            (spread_candidates.week .== 1) .&
            (spread_candidates.team .== "A")
        ][1] == -2.5
        @test spread_candidates.market_spread[
            (spread_candidates.week .== 1) .&
            (spread_candidates.team .== "C")
        ][1] == 3.0
    end

    @testset "near-term market favorite guard" begin
        candidates = _market_guard_candidates()
        state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        data = SurvivorModel._normalize_survivor_candidates(
            candidates,
            state,
            3,
        )
        eligibility = SurvivorModel._survivor_market_guard_mask(
            data,
            state,
            SurvivorSelectionConfig(through_week=3),
        )
        @test eligibility == [false, true, false, true, true, true]
        @test DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD == 2.0
        @test DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY == 0.5
        model = SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
        SurvivorModel.JuMP.set_silent(model)
        SurvivorModel.JuMP.@variable(model, selected[1:nrow(data)], Bin)
        SurvivorModel._add_survivor_assignment_constraints!(
            model,
            data,
            state,
            SurvivorSelectionConfig(through_week=3),
            selected,
        )
        SurvivorModel.JuMP.@objective(
            model,
            Max,
            sum(data.win_probability[index] * selected[index] for index in 1:nrow(data)),
        )
        SurvivorModel.JuMP.optimize!(model)
        market_guard_plan = sort(
            data[findall(SurvivorModel.JuMP.value.(selected) .> 0.5), :],
            [:week, :team],
        )
        @test market_guard_plan.team == ["B", "D", "E"]
        @test market_guard_plan.market_spread == [2.0, 3.0, -10.0]

        missing_line = DataFrame(
            game_id=["missing_line", "missing_line"],
            week=[1, 1],
            team=["A", "B"],
            opponent=["C", "D"],
            is_home=[true, false],
            win_probability=[0.95, 0.8],
            market_spread=Union{Missing,Float64}[missing, 2.5],
        )
        missing_line_data = SurvivorModel._normalize_survivor_candidates(
            missing_line,
            SurvivorPoolState(2025, 1; strikes_remaining=0),
            1,
        )
        @test SurvivorModel._survivor_market_guard_mask(
            missing_line_data,
            state,
            SurvivorSelectionConfig(through_week=1),
        ) == [true, true]
        @test SurvivorModel._survivor_market_guard_mask(
            missing_line_data,
            state,
            SurvivorSelectionConfig(
                missing_market_policy=:exclude,
                through_week=1,
            ),
        ) == [false, true]
    end

    @testset "state validation and infeasible inputs" begin
        @test_throws ArgumentError SurvivorPoolState(
            2025,
            1;
            picks_made=Dict(1 => "A"),
        )
        @test_throws ArgumentError SurvivorPoolState(
            2025,
            1;
            picks_made=Dict(0 => "A", 1 => "A"),
        )

        candidates = build_survivor_candidates(_survivor_forecast_fixture())
        state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        @test_throws ArgumentError SurvivorModel._normalize_survivor_candidates(
            candidates,
            state,
            3,
        )
    end

    @testset "fitted context integration" begin
        schedule, historical, current = _survivor_context_fixture()
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        forecast = forecast_win_probabilities(
            context;
            include_completed=true,
            full_schedule=true,
        )
        plan = optimize_survivor_pool(
            context;
            selection_config=SurvivorSelectionConfig(through_week=2),
            include_completed=true,
            strikes_remaining=0,
        )
        @test nrow(plan.selections) == 1
        @test plan.selections.game_id[1] == forecast.game_id[1]
        @test plan.selections.win_probability[1] ≈ max(
            forecast.home_win_probability[1],
            forecast.away_win_probability[1],
        )
        @test plan.selection_config.through_week == 2
    end

    @testset "analytic posterior derivatives" begin
        schedule, historical, current = _survivor_context_fixture()
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        candidates = DataFrame(
            game_id=["derivative_game"],
            week=[2],
            team=["C"],
            opponent=["D"],
            is_home=[false],
            win_probability=[0.9],
        )
        state = SurvivorPoolState(2023, 2; strikes_remaining=0)
        data = SurvivorModel._normalize_survivor_candidates(
            candidates,
            state,
            2,
        )
        inputs = SurvivorModel._survivor_objective_inputs(
            context.model,
            context.marks,
            data;
            horizon=GAME_CLOCK_SECONDS,
        )
        row = first(eachrow(data))
        local_mean, global_indices =
            SurvivorModel._survivor_candidate_local_parameters(
                context.model,
                row,
                inputs.parameters,
            )
        home_derivatives = SurvivorModel._game_probability_derivatives(
            local_mean,
            context.model.prior,
            context.marks;
            horizon=GAME_CLOCK_SECONDS,
        )
        orientation = Bool(row.is_home) ? 1.0 : -1.0
        derivative = only(inputs.derivatives)
        expected_gradient = zeros(length(inputs.parameters.keys))
        for (local_index, global_index) in enumerate(global_indices)
            expected_gradient[global_index] +=
                orientation * home_derivatives.gradient[local_index]
        end
        expected_probability = Bool(row.is_home) ?
            home_derivatives.probability :
            1.0 - home_derivatives.probability
        expected_hessian_covariance = orientation * sum(
            home_derivatives.hessian[first_local, second_local] *
                inputs.parameters.variance[global_indices[first_local]]
            for first_local in eachindex(global_indices),
                second_local in eachindex(global_indices)
            if global_indices[first_local] == global_indices[second_local]
        )
        @test derivative.base_probability ≈ expected_probability
        @test derivative.gradient ≈ expected_gradient
        @test inputs.covariance_gradient_gram[1, 1] ≈ sum(
            expected_gradient[index]^2 * inputs.parameters.variance[index]
            for index in eachindex(expected_gradient)
        )
        @test derivative.hessian_covariance ≈ expected_hessian_covariance
    end

    @testset "whole-plan scalar covariance recursion" begin
        schedule, historical, current = _survivor_context_fixture()
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        candidates = DataFrame(
            game_id=["shared_game_1", "shared_game_2"],
            week=[1, 2],
            team=["B", "C"],
            opponent=["A", "A"],
            is_home=[true, true],
            win_probability=[0.9, 0.9],
        )
        state = SurvivorPoolState(2023, 1; strikes_remaining=0)
        data = SurvivorModel._normalize_survivor_candidates(
            candidates,
            state,
            2,
        )
        config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
        )
        inputs = SurvivorModel._survivor_objective_inputs(
            context.model,
            context.marks,
            data;
            horizon=GAME_CLOCK_SECONDS,
        )
        @test size(inputs.covariance_gradient_gram) == (nrow(data), nrow(data))
        @test all(isfinite, inputs.covariance_gradient_gram)
        defensive_a_index = findfirst(
            ==((:defensive, "A")),
            inputs.parameters.keys,
        )
        @test defensive_a_index !== nothing
        first_local_mean, first_indices =
            SurvivorModel._survivor_candidate_local_parameters(
                context.model,
                first(eachrow(data)),
                inputs.parameters,
            )
        second_local_mean, second_indices =
            SurvivorModel._survivor_candidate_local_parameters(
                context.model,
                last(eachrow(data)),
                inputs.parameters,
            )
        @test first_indices[2] == defensive_a_index
        @test second_indices[2] == defensive_a_index

        covariance_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                config,
                inputs,
            )
        first_derivatives, second_derivatives = inputs.derivatives
        base_objective = first_derivatives.base_probability *
            (1.0 + second_derivatives.base_probability)
        objective_covariance = (
            (1.0 + second_derivatives.base_probability) *
                first_derivatives.hessian_covariance +
            first_derivatives.base_probability *
                second_derivatives.hessian_covariance +
            2.0 * inputs.covariance_gradient_gram[1, 2]
        )
        expected_objective = base_objective + 0.5 * objective_covariance
        @test covariance_plan.objective_value ≈ expected_objective
        @test sum(covariance_plan.selections.objective_contribution) ≈
            covariance_plan.objective_value

        zero_parameters = SurvivorModel.SurvivorParameterSystem(
            inputs.parameters.keys,
            inputs.parameters.indices,
            inputs.parameters.log_mean,
            zeros(length(inputs.parameters.variance)),
        )
        zero_derivatives = [
            SurvivorModel.SurvivorCandidateDerivatives(
                derivative.base_probability,
                derivative.gradient,
                0.0,
            )
            for derivative in inputs.derivatives
        ]
        zero_inputs = SurvivorModel.SurvivorObjectiveInputs(
            zero_parameters,
            zero_derivatives,
        )
        @test all(zero_inputs.covariance_gradient_gram .== 0.0)
        zero_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                config,
                zero_inputs,
            )
        @test zero_plan.objective_value ≈ base_objective
        constant_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                SurvivorPoolState(2023, 1; strikes_remaining=3),
                config,
                zero_inputs,
            )
        @test constant_plan.objective_value == 2.0
        @test all(constant_plan.selections.survival_probability .== 1.0)
        linear_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                SurvivorSelectionConfig(
                    minimum_favorite_spread=nothing,
                    market_guard_weeks=0,
                    through_week=2,
                    hessian_weeks=0,
                ),
                inputs,
            )
        @test linear_plan.objective_value ≈ base_objective
        @test all(linear_plan.selections.parameter_variance_adjustment .== 0.0)
    end

    @testset "scalar recurrence bounds" begin
        gradient_references =
            SurvivorModel._survivor_gradient_reference_indices(
                [1, 1, 2, 2],
                2,
            )
        @test gradient_references == [[1, 2, 3, 4], [3, 4]]
        @test sum(length, gradient_references) == 6
        sparse_gradient_references =
            SurvivorModel._survivor_gradient_reference_indices(
                [1, 1, 2, 2],
                2,
                [
                    0.0 0.0 1.0 0.0
                    0.0 0.0 0.0 0.0
                    1.0 0.0 0.0 0.0
                    0.0 0.0 0.0 0.0
                ],
            )
        @test sparse_gradient_references == [Int[], [3]]

        keys = [
            (:td, "A"),
            (:td, "B"),
        ]
        parameters = SurvivorModel.SurvivorParameterSystem(
            keys,
            Dict(key => index for (index, key) in enumerate(keys)),
            [0.0, 0.0],
            [1.0, 1.0],
        )
        derivatives = [
            SurvivorModel.SurvivorCandidateDerivatives(
                0.8,
                [1.0, -2.0],
                0.4,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.7,
                [-1.0, 1.0],
                -0.3,
            ),
        ]
        inputs = SurvivorModel.SurvivorObjectiveInputs(
            parameters,
            derivatives,
        )
        @test inputs.covariance_gradient_gram ≈
            [5.0 -3.0; -3.0 2.0]
        prefix_references =
            SurvivorModel._survivor_gradient_reference_indices(
                [1, 2],
                2,
                inputs.covariance_gradient_gram;
                maximum_reference_position=1,
            )
        @test prefix_references == [Int[]]
        bounds = SurvivorModel._survivor_scalar_bounds(
            inputs,
            [1, 2],
            2,
            2,
        )
        prefix_bounds = SurvivorModel._survivor_scalar_bounds(
            inputs,
            [1, 2],
            2,
            2;
            curvature_weeks=1,
            gradient_reference_indices=prefix_references,
        )
        values = SurvivorModel._survivor_scalar_forward_values(
            [1, 2],
            inputs,
            2,
            2,
        )
        prefix_values = SurvivorModel._survivor_scalar_forward_values(
            [1, 2],
            inputs,
            2,
            2;
            curvature_weeks=1,
            gradient_reference_indices=prefix_references,
        )
        @test all(values.probability .>= bounds.probability.lower)
        @test all(values.probability .<= bounds.probability.upper)
        @test all(values.gradient .>= bounds.gradient.lower)
        @test all(values.gradient .<= bounds.gradient.upper)
        @test all(values.hessian .>= bounds.hessian.lower)
        @test all(values.hessian .<= bounds.hessian.upper)
        @test size(prefix_bounds.gradient.lower, 1) == 2
        @test size(prefix_bounds.hessian.lower, 1) == 2
        @test prefix_values.probability == values.probability
        @test all(prefix_values.hessian[3:end, :] .== 0.0)
        @test all(prefix_values.hessian[1:2, :] .>= prefix_bounds.hessian.lower)
        @test all(prefix_values.hessian[1:2, :] .<= prefix_bounds.hessian.upper)
        single_bounds = SurvivorModel._survivor_scalar_bounds(
            SurvivorModel.SurvivorObjectiveInputs(
                parameters,
                derivatives[1:1],
            ),
            [1],
            1,
            1,
        )
        @test single_bounds.candidate_probability.lower[1, 1] <= 0.8
        @test single_bounds.candidate_probability.upper[1, 1] >= 0.8

        exact_zero_derivatives = [
            SurvivorModel.SurvivorCandidateDerivatives(
                1.0,
                [0.0, 0.0],
                0.0,
            )
            for _ in 1:2
        ]
        exact_zero_bounds = SurvivorModel._survivor_scalar_bounds(
            SurvivorModel.SurvivorObjectiveInputs(
                parameters,
                exact_zero_derivatives,
            ),
            [1, 2],
            2,
            2,
        )
        @test all(iszero, exact_zero_bounds.probability.lower[:, 2])
        @test all(iszero, exact_zero_bounds.probability.upper[:, 2])
        @test all(iszero, exact_zero_bounds.candidate_probability.lower[:, 2])
        @test all(iszero, exact_zero_bounds.candidate_probability.upper[:, 2])
        @test all(iszero, exact_zero_bounds.gradient.lower)
        @test all(iszero, exact_zero_bounds.gradient.upper)
        @test all(iszero, exact_zero_bounds.parameter_gradient.lower)
        @test all(iszero, exact_zero_bounds.parameter_gradient.upper)
        @test all(iszero, exact_zero_bounds.hessian.lower)
        @test all(iszero, exact_zero_bounds.hessian.upper)
    end

    @testset "team-leave-out recurrence bounds" begin
        keys = [(:td, "A"), (:td, "B"), (:td, "C")]
        parameters = SurvivorModel.SurvivorParameterSystem(
            keys,
            Dict(key => index for (index, key) in enumerate(keys)),
            zeros(3),
            [0.7, 1.2, 0.9],
        )
        derivatives = [
            SurvivorModel.SurvivorCandidateDerivatives(
                0.9,
                [0.9, 0.0, 0.0],
                0.2,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.6,
                [0.0, 0.7, 0.0],
                -0.1,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.4,
                [0.0, 0.0, 0.5],
                0.3,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.8,
                [0.8, 0.1, -0.2],
                0.1,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.7,
                [-0.4, 0.5, 0.1],
                0.15,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.5,
                [0.3, -0.8, 0.6],
                -0.2,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.75,
                [0.2, 0.3, 0.4],
                -0.12,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.65,
                [0.1, -0.3, 0.5],
                0.08,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.55,
                [-0.2, 0.6, 0.1],
                0.18,
            ),
        ]
        inputs = SurvivorModel.SurvivorObjectiveInputs(parameters, derivatives)
        positions = repeat(1:3, inner=3)
        teams = repeat(["A", "B", "C"], 3)
        references = SurvivorModel._survivor_gradient_reference_indices(
            positions,
            3,
            inputs.covariance_gradient_gram;
            maximum_reference_position=3,
        )
        bounds = SurvivorModel._survivor_scalar_bounds(
            inputs,
            positions,
            3,
            2;
            curvature_weeks=3,
            gradient_reference_indices=references,
            candidate_teams=teams,
        )
        team_bounds = bounds.team_conditioned
        all_history_bounds = SurvivorModel._survivor_scalar_bounds(
            inputs,
            positions,
            3,
            2;
            curvature_weeks=3,
            gradient_reference_indices=references,
        )

        candidate = 4
        @test (
            team_bounds.candidate_probability.upper[candidate, 1] -
            team_bounds.candidate_probability.lower[candidate, 1]
        ) < (
            all_history_bounds.candidate_probability.upper[candidate, 1] -
            all_history_bounds.candidate_probability.lower[candidate, 1]
        )
        @test (
            team_bounds.candidate_parameter_gradient.upper[
                candidate,
                1,
                1,
            ] -
            team_bounds.candidate_parameter_gradient.lower[
                candidate,
                1,
                1,
            ]
        ) < (
            all_history_bounds.candidate_parameter_gradient.upper[
                candidate,
                1,
                1,
            ] -
            all_history_bounds.candidate_parameter_gradient.lower[
                candidate,
                1,
                1,
            ]
        )
        @test (
            team_bounds.candidate_gradient.upper[
                candidate,
                1,
                7,
            ] -
            team_bounds.candidate_gradient.lower[
                candidate,
                1,
                7,
            ]
        ) < (
            all_history_bounds.candidate_gradient.upper[
                candidate,
                1,
                7,
            ] -
            all_history_bounds.candidate_gradient.lower[
                candidate,
                1,
                7,
            ]
        )
        @test (
            team_bounds.candidate_hessian.upper[candidate, 1] -
            team_bounds.candidate_hessian.lower[candidate, 1]
        ) < (
            all_history_bounds.candidate_hessian.upper[candidate, 1] -
            all_history_bounds.candidate_hessian.lower[candidate, 1]
        )

        exhaustive_objectives = Float64[]
        for first_team in 1:3, second_team in 1:3, third_team in 1:3
            length(unique((first_team, second_team, third_team))) == 3 ||
                continue
            selected = [first_team, 3 + second_team, 6 + third_team]
            values = SurvivorModel._survivor_scalar_forward_values(
                selected,
                inputs,
                3,
                2;
                curvature_weeks=3,
                gradient_reference_indices=references,
            )
            @test all(values.probability .>= bounds.probability.lower)
            @test all(values.probability .<= bounds.probability.upper)
            @test all(
                values.parameter_gradient .>=
                bounds.parameter_gradient.lower,
            )
            @test all(
                values.parameter_gradient .<=
                bounds.parameter_gradient.upper,
            )
            @test all(values.gradient .>= bounds.gradient.lower)
            @test all(values.gradient .<= bounds.gradient.upper)
            @test all(values.hessian .>= bounds.hessian.lower)
            @test all(values.hessian .<= bounds.hessian.upper)
            push!(
                exhaustive_objectives,
                sum(
                    values.probability[position + 1, loss_state] +
                    0.5 * values.hessian[position + 1, loss_state]
                    for position in 1:3,
                    loss_state in 1:2
                ),
            )

            for position in 1:3
                index = selected[position]
                derivative = inputs.derivatives[index]
                for loss_state in 1:2
                    previous_probability = loss_state == 1 ?
                        0.0 :
                        values.probability[position, loss_state - 1]
                    probability_difference =
                        values.probability[position, loss_state] -
                        previous_probability
                    candidate_probability =
                        derivative.base_probability *
                        values.probability[position, loss_state] +
                        (1.0 - derivative.base_probability) *
                        previous_probability
                    @test team_bounds.candidate_probability.lower[
                        index,
                        loss_state,
                    ] <= candidate_probability <=
                    team_bounds.candidate_probability.upper[
                        index,
                        loss_state,
                    ]

                    if position < 3
                        for parameter in 1:3
                            previous_gradient = loss_state == 1 ?
                                0.0 :
                                values.parameter_gradient[
                                    position,
                                    loss_state - 1,
                                    parameter,
                                ]
                            candidate_gradient =
                                derivative.base_probability *
                                values.parameter_gradient[
                                    position,
                                    loss_state,
                                    parameter,
                                ] +
                                (1.0 - derivative.base_probability) *
                                previous_gradient +
                                derivative.gradient[parameter] *
                                probability_difference
                            @test team_bounds.candidate_parameter_gradient.lower[
                                    index,
                                    loss_state,
                                    parameter,
                                ] <= candidate_gradient <=
                            team_bounds.candidate_parameter_gradient.upper[
                                    index,
                                    loss_state,
                                    parameter,
                                ]
                        end
                        for reference in references[position + 1]
                            previous_projected_gradient = loss_state == 1 ?
                                0.0 :
                                values.gradient[
                                    position,
                                    loss_state - 1,
                                    reference,
                                ]
                            candidate_projected_gradient =
                                derivative.base_probability *
                                values.gradient[
                                    position,
                                    loss_state,
                                    reference,
                                ] +
                                (1.0 - derivative.base_probability) *
                                previous_projected_gradient +
                                inputs.covariance_gradient_gram[
                                    index,
                                    reference,
                                ] * probability_difference
                            @test team_bounds.candidate_gradient.lower[
                                index,
                                loss_state,
                                reference,
                            ] <=
                                candidate_projected_gradient <=
                            team_bounds.candidate_gradient.upper[
                                index,
                                loss_state,
                                reference,
                            ]
                        end
                    end

                    previous_hessian = loss_state == 1 ?
                        0.0 :
                        values.hessian[position, loss_state - 1]
                    candidate_gradient_reference =
                        index in references[position] ? index : nothing
                    current_projected_gradient =
                        candidate_gradient_reference === nothing ?
                        0.0 :
                        values.gradient[position, loss_state, index]
                    previous_projected_gradient =
                        loss_state == 1 ||
                                candidate_gradient_reference === nothing ?
                        0.0 :
                        values.gradient[position, loss_state - 1, index]
                    candidate_hessian =
                        derivative.base_probability *
                        values.hessian[position, loss_state] +
                        (1.0 - derivative.base_probability) *
                        previous_hessian +
                        derivative.hessian_covariance *
                        probability_difference +
                        2.0 * (
                            current_projected_gradient -
                            previous_projected_gradient
                        )
                    @test team_bounds.candidate_hessian.lower[
                        index,
                        loss_state,
                    ] <= candidate_hessian <=
                    team_bounds.candidate_hessian.upper[
                        index,
                        loss_state,
                    ]
                end
                candidate_probability_sum =
                    sum(values.probability[position + 1, :])
                @test sum(team_bounds.candidate_probability.lower[index, :]) <=
                    candidate_probability_sum <=
                    sum(team_bounds.candidate_probability.upper[index, :])
                candidate_adjusted_sum =
                    candidate_probability_sum +
                    0.5 * sum(values.hessian[position + 1, :])
                @test (
                    sum(team_bounds.candidate_probability.lower[index, :]) +
                    0.5 * sum(team_bounds.candidate_hessian.lower[index, :])
                ) <= candidate_adjusted_sum <= (
                    sum(team_bounds.candidate_probability.upper[index, :]) +
                    0.5 * sum(team_bounds.candidate_hessian.upper[index, :])
                )
            end
        end

        candidates = DataFrame(
            game_id=["week-$position" for position in positions],
            week=positions,
            team=teams,
            opponent=fill("OPP", length(positions)),
            is_home=fill(true, length(positions)),
            win_probability=[
                derivative.base_probability for derivative in derivatives
            ],
            market_spread=fill(missing, length(positions)),
        )
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=3,
            hessian_weeks=3,
        )
        plan = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
            candidates,
            state,
            config,
            inputs,
        )
        @test plan.objective_value ≈ maximum(exhaustive_objectives)
    end

    @testset "hybrid gradient state selection" begin
        @test SurvivorModel._survivor_gradient_switch_position(
            [[1, 2], [3], [1, 2, 3], [2, 4]],
            2,
        ) == 4
        @test SurvivorModel._survivor_gradient_switch_position(
            [[1], [1, 2, 3]],
            2,
        ) == 3
        @test SurvivorModel._survivor_gradient_switch_position(
            [Int[], [1, 2]],
            2,
        ) == 1
        @test SurvivorModel._survivor_gradient_switch_position(
            Vector{Vector{Int}}(),
            2,
        ) == 1

        parameter_keys = [(:td, "A"), (:td, "B")]
        parameters = SurvivorModel.SurvivorParameterSystem(
            parameter_keys,
            Dict(key => index for (index, key) in enumerate(parameter_keys)),
            [0.0, 0.0],
            [0.7, 1.2],
        )
        derivatives = [
            SurvivorModel.SurvivorCandidateDerivatives(
                probability,
                gradient,
                hessian_covariance,
            )
            for (probability, gradient, hessian_covariance) in (
                (0.82, [1.0, 1.0], 0.03),
                (0.71, [1.1, 0.9], -0.02),
                (0.77, [0.9, 1.2], 0.01),
                (0.63, [1.2, 1.1], -0.01),
                (0.88, [0.8, 1.3], 0.02),
                (0.68, [1.3, 0.8], -0.03),
            )
        ]
        inputs = SurvivorModel.SurvivorObjectiveInputs(
            parameters,
            derivatives,
        )
        candidate_positions = [1, 1, 2, 2, 3, 3]
        references = SurvivorModel._survivor_gradient_reference_indices(
            candidate_positions,
            3,
            inputs.covariance_gradient_gram;
            maximum_reference_position=3,
        )
        @test length.(references) == [0, 4, 2]
        switch_position =
            SurvivorModel._survivor_gradient_switch_position(references, 2)
        @test switch_position == 3
        tracked_widths = [
            position < switch_position ? 2 : length(references[position])
            for position in eachindex(references)
        ]
        @test tracked_widths == [2, 2, 2]
        @test all(width -> width <= 2, tracked_widths)

        selected = [1, 3, 5]
        values = SurvivorModel._survivor_scalar_forward_values(
            selected,
            inputs,
            3,
            2;
            curvature_weeks=3,
            gradient_reference_indices=references,
        )
        for position in eachindex(references),
            loss_state in 1:2,
            reference in references[position]
            projected_gradient =
                SurvivorModel._survivor_parameter_gradient_contraction(
                    @view(values.parameter_gradient[position, :, :]),
                    loss_state,
                    reference,
                    inputs,
                )
            @test values.gradient[position, loss_state, reference] ≈
                  projected_gradient
        end
        bounds = SurvivorModel._survivor_scalar_bounds(
            inputs,
            candidate_positions,
            3,
            2;
            curvature_weeks=3,
            gradient_reference_indices=references,
        )
        @test all(
            values.parameter_gradient .>= bounds.parameter_gradient.lower,
        )
        @test all(
            values.parameter_gradient .<= bounds.parameter_gradient.upper,
        )

        candidates = DataFrame(
            game_id=["w1", "w1", "w2", "w2", "w3", "w3"],
            week=candidate_positions,
            team=["A", "B", "C", "D", "E", "F"],
            opponent=["X", "Y", "U", "V", "W", "Z"],
            is_home=[true, false, true, false, true, false],
            win_probability=[derivative.base_probability for derivative in derivatives],
        )
        state = SurvivorPoolState(2025, 1; strikes_remaining=2)
        data = SurvivorModel._normalize_survivor_candidates(
            candidates,
            state,
            3,
        )
        config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=3,
            hessian_weeks=3,
        )
        plan = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
            data,
            state,
            config,
            inputs,
        )
        selected_by_position =
            SurvivorModel._survivor_fixed_selected_indices(
                data,
                plan.selections,
                state,
                3,
            )
        selected_values = SurvivorModel._survivor_scalar_forward_values(
            selected_by_position,
            inputs,
            3,
            2;
            curvature_weeks=3,
            gradient_reference_indices=references,
        )
        objective(values) = sum(
            values.probability[position + 1, loss_state] +
            0.5 * values.hessian[position + 1, loss_state]
            for position in 1:3,
            loss_state in 1:2
        )
        exhaustive_objective = maximum(
            objective(
                SurvivorModel._survivor_scalar_forward_values(
                    [first, second, third],
                    inputs,
                    3,
                    2;
                    curvature_weeks=3,
                    gradient_reference_indices=references,
                ),
            )
            for first in 1:2, second in 3:4, third in 5:6
        )
        @test plan.objective_value ≈ objective(selected_values)
        @test plan.objective_value ≈ exhaustive_objective

        parameter_only_positions = [1, 1, 1, 2, 2, 2]
        parameter_only_references =
            SurvivorModel._survivor_gradient_reference_indices(
                parameter_only_positions,
                2,
                inputs.covariance_gradient_gram;
                maximum_reference_position=2,
            )
        @test length.(parameter_only_references) == [0, 3]
        @test SurvivorModel._survivor_gradient_switch_position(
            parameter_only_references,
            2,
        ) == 3
        parameter_only_candidates = DataFrame(
            game_id=["p1", "p1", "p1", "p2", "p2", "p2"],
            week=parameter_only_positions,
            team=["A", "B", "C", "D", "E", "F"],
            opponent=["X", "Y", "Z", "U", "V", "W"],
            is_home=[true, false, true, false, true, false],
            win_probability=[
                derivative.base_probability for derivative in derivatives
            ],
        )
        parameter_only_data =
            SurvivorModel._normalize_survivor_candidates(
                parameter_only_candidates,
                state,
                2,
            )
        parameter_only_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
            hessian_weeks=2,
        )
        parameter_only_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                parameter_only_data,
                state,
                parameter_only_config,
                inputs,
            )
        parameter_only_selected =
            SurvivorModel._survivor_fixed_selected_indices(
                parameter_only_data,
                parameter_only_plan.selections,
                state,
                2,
            )
        parameter_only_objective(values) = sum(
            values.probability[position + 1, loss_state] +
            0.5 * values.hessian[position + 1, loss_state]
            for position in 1:2,
            loss_state in 1:2
        )
        parameter_only_values = SurvivorModel._survivor_scalar_forward_values(
            parameter_only_selected,
            inputs,
            2,
            2;
            curvature_weeks=2,
            gradient_reference_indices=parameter_only_references,
        )
        parameter_only_exhaustive = maximum(
            parameter_only_objective(
                SurvivorModel._survivor_scalar_forward_values(
                    [first, second],
                    inputs,
                    2,
                    2;
                    curvature_weeks=2,
                    gradient_reference_indices=parameter_only_references,
                ),
            )
            for first in 1:3, second in 4:6
        )
        @test parameter_only_plan.objective_value ≈
              parameter_only_objective(parameter_only_values)
        @test parameter_only_plan.objective_value ≈
              parameter_only_exhaustive
    end

    @testset "one-hot dummy product hull" begin
        model = SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
        SurvivorModel.JuMP.set_silent(model)
        SurvivorModel.JuMP.@variable(model, 0 <= selected[1:2] <= 1)
        SurvivorModel.JuMP.@variable(model, aggregate)
        SurvivorModel.JuMP.@constraint(
            model,
            sum(selected[index] for index in 1:2) == 1,
        )
        SurvivorModel.JuMP.@constraint(model, selected[1] == 0.5)
        SurvivorModel.JuMP.@constraint(model, selected[2] == 0.5)
        @test SurvivorModel.JuMP.num_variables(model) == 3
        @test SurvivorModel.JuMP.num_constraints(
            model;
            count_variable_in_set_constraints=false,
        ) == 3

        dummies = SurvivorModel._survivor_add_one_hot_dummies!(
            model,
            aggregate,
            Dict(1 => 0.8, 2 => -0.6),
            selected,
            [0.0, -1.0],
            [1.0, 0.0],
            1:2,
        )
        @test SurvivorModel.JuMP.num_variables(model) == 5
        @test SurvivorModel.JuMP.num_constraints(
            model;
            count_variable_in_set_constraints=false,
        ) == 10
        SurvivorModel.JuMP.@objective(model, Max, aggregate)
        SurvivorModel.JuMP.optimize!(model)
        @test SurvivorModel.JuMP.termination_status(model) ==
            SurvivorModel.JuMP.MOI.OPTIMAL
        @test SurvivorModel.JuMP.value(aggregate) ≈ 0.4
        @test SurvivorModel.JuMP.value(dummies[1]) ≈ 0.5
        @test SurvivorModel.JuMP.value(dummies[2]) ≈ -0.1

        SurvivorModel.JuMP.set_objective_sense(
            model,
            SurvivorModel.JuMP.MOI.MIN_SENSE,
        )
        SurvivorModel.JuMP.optimize!(model)
        @test SurvivorModel.JuMP.termination_status(model) ==
            SurvivorModel.JuMP.MOI.OPTIMAL
        @test SurvivorModel.JuMP.value(aggregate) ≈ -0.2
        @test SurvivorModel.JuMP.value(dummies[1]) ≈ 0.3
        @test SurvivorModel.JuMP.value(dummies[2]) ≈ -0.5

        for selected_values in ((1.0, 0.0), (0.0, 1.0))
            integer_model =
                SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
            SurvivorModel.JuMP.set_silent(integer_model)
            SurvivorModel.JuMP.@variable(
                integer_model,
                0 <= integer_selected[1:2] <= 1,
            )
            SurvivorModel.JuMP.@variable(integer_model, integer_aggregate)
            SurvivorModel.JuMP.@constraint(
                integer_model,
                sum(integer_selected[index] for index in 1:2) == 1,
            )
            for index in 1:2
                SurvivorModel.JuMP.@constraint(
                    integer_model,
                    integer_selected[index] == selected_values[index],
                )
            end
            integer_dummies = SurvivorModel._survivor_add_one_hot_dummies!(
                integer_model,
                integer_aggregate,
                Dict(1 => 0.8, 2 => -0.6),
                integer_selected,
                [0.0, -1.0],
                [1.0, 0.0],
                1:2,
            )
            SurvivorModel.JuMP.@objective(
                integer_model,
                Max,
                integer_aggregate,
            )
            SurvivorModel.JuMP.optimize!(integer_model)
            @test SurvivorModel.JuMP.value(integer_aggregate) ≈
                selected_values[1] * 0.8 +
                selected_values[2] * -0.6
            @test SurvivorModel.JuMP.value(integer_dummies[1]) ≈
                selected_values[1] * 0.8
            @test SurvivorModel.JuMP.value(integer_dummies[2]) ≈
                selected_values[2] * -0.6
        end
    end

    @testset "fixed-bound one-hot dummy substitution" begin
        model = SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
        SurvivorModel.JuMP.set_silent(model)
        SurvivorModel.JuMP.@variable(model, 0 <= selected[1:3] <= 1)
        SurvivorModel.JuMP.@variable(model, aggregate)
        SurvivorModel.JuMP.@constraint(
            model,
            sum(selected[index] for index in 1:3) == 1,
        )
        for (index, selected_value) in enumerate((0.25, 0.25, 0.5))
            SurvivorModel.JuMP.@constraint(
                model,
                selected[index] == selected_value,
            )
        end
        dummies = SurvivorModel._survivor_add_one_hot_dummies!(
            model,
            aggregate,
            Dict(1 => 0.5, 2 => -0.25, 3 => 0.0),
            selected,
            [0.5, -0.25, 0.0],
            [0.5, -0.25, 0.0],
            1:3,
            dummy_start_values=Dict(
                1 => 0.125,
                2 => -0.0625,
                3 => 0.0,
            ),
        )
        @test SurvivorModel.JuMP.num_variables(model) == 4
        @test SurvivorModel.JuMP.num_constraints(
            model;
            count_variable_in_set_constraints=false,
        ) == 5

        SurvivorModel.JuMP.@objective(model, Max, aggregate)
        SurvivorModel.JuMP.optimize!(model)
        @test SurvivorModel.JuMP.termination_status(model) ==
            SurvivorModel.JuMP.MOI.OPTIMAL
        @test SurvivorModel.JuMP.value(aggregate) ≈ 0.0625
        @test SurvivorModel.JuMP.value(dummies[1]) ≈ 0.125
        @test SurvivorModel.JuMP.value(dummies[2]) ≈ -0.0625
        @test SurvivorModel.JuMP.value(dummies[3]) ≈ 0.0

        near_bound_model =
            SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
        SurvivorModel.JuMP.set_silent(near_bound_model)
        SurvivorModel.JuMP.@variable(
            near_bound_model,
            0 <= near_selected <= 1,
        )
        SurvivorModel.JuMP.@variable(near_bound_model, near_aggregate)
        SurvivorModel.JuMP.@constraint(near_bound_model, near_selected == 0.5)
        SurvivorModel._survivor_add_one_hot_dummies!(
            near_bound_model,
            near_aggregate,
            Dict(1 => 1.0),
            [near_selected],
            [1.0],
            [nextfloat(1.0)],
            1:1,
        )
        @test SurvivorModel.JuMP.num_variables(near_bound_model) == 3
        @test SurvivorModel.JuMP.num_constraints(
            near_bound_model;
            count_variable_in_set_constraints=false,
        ) == 6
    end

    @testset "conditional one-hot product hull bounds" begin
        for (selected_value, recurrence_value) in ((0.0, 0.0), (1.0, 0.5))
            model = SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
            SurvivorModel.JuMP.set_silent(model)
            SurvivorModel.JuMP.@variable(model, 0 <= selected <= 1)
            SurvivorModel.JuMP.@variable(model, recurrence)
            SurvivorModel.JuMP.@variable(model, aggregate)
            SurvivorModel.JuMP.@constraint(model, selected == selected_value)
            SurvivorModel.JuMP.@constraint(
                model,
                recurrence == recurrence_value,
            )
            constraints_before_hull = SurvivorModel.JuMP.num_constraints(
                model;
                count_variable_in_set_constraints=false,
            )
            dummies = SurvivorModel._survivor_add_one_hot_dummies!(
                model,
                aggregate,
                Dict(1 => recurrence),
                [selected],
                [0.2],
                [0.8],
                1:1,
                recurrence_lower=[0.0],
                recurrence_upper=[1.0],
            )
            @test SurvivorModel.JuMP.num_constraints(
                model;
                count_variable_in_set_constraints=false,
            ) - constraints_before_hull == 5
            SurvivorModel.JuMP.@objective(model, Min, aggregate)
            SurvivorModel.JuMP.optimize!(model)
            @test SurvivorModel.JuMP.termination_status(model) ==
                SurvivorModel.JuMP.MOI.OPTIMAL
            @test SurvivorModel.JuMP.value(dummies[1]) ≈
                selected_value * recurrence_value
        end

        for (
            selected_lower,
            selected_upper,
            selected_value,
            recurrence_value,
            expected_status,
            expected_dummy,
        ) in (
            (
                0.2,
                0.2,
                0.0,
                0.5,
                SurvivorModel.JuMP.MOI.OPTIMAL,
                0.0,
            ),
            (
                0.2,
                0.2,
                1.0,
                0.2,
                SurvivorModel.JuMP.MOI.OPTIMAL,
                0.2,
            ),
            (
                0.2,
                0.2,
                1.0,
                0.3,
                SurvivorModel.JuMP.MOI.INFEASIBLE,
                nothing,
            ),
            (
                0.0,
                0.0,
                1.0,
                0.2,
                SurvivorModel.JuMP.MOI.INFEASIBLE,
                nothing,
            ),
        )
            model = SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
            SurvivorModel.JuMP.set_silent(model)
            SurvivorModel.JuMP.@variable(model, 0 <= selected <= 1)
            SurvivorModel.JuMP.@variable(model, recurrence)
            SurvivorModel.JuMP.@variable(model, aggregate)
            SurvivorModel.JuMP.@constraint(model, selected == selected_value)
            SurvivorModel.JuMP.@constraint(
                model,
                recurrence == recurrence_value,
            )
            constraints_before_hull = SurvivorModel.JuMP.num_constraints(
                model;
                count_variable_in_set_constraints=false,
            )
            dummies = SurvivorModel._survivor_add_one_hot_dummies!(
                model,
                aggregate,
                Dict(1 => recurrence),
                [selected],
                [selected_lower],
                [selected_upper],
                1:1,
                recurrence_lower=[0.0],
                recurrence_upper=[1.0],
            )
            @test SurvivorModel.JuMP.num_constraints(
                model;
                count_variable_in_set_constraints=false,
            ) - constraints_before_hull == 3
            SurvivorModel.JuMP.@objective(model, Min, aggregate)
            SurvivorModel.JuMP.optimize!(model)
            @test SurvivorModel.JuMP.termination_status(model) == expected_status
            if expected_status == SurvivorModel.JuMP.MOI.OPTIMAL
                @test SurvivorModel.JuMP.value(dummies[1]) ≈ expected_dummy
            end
        end

        infeasible_model =
            SurvivorModel.JuMP.Model(SurvivorModel.HiGHS.Optimizer)
        SurvivorModel.JuMP.set_silent(infeasible_model)
        SurvivorModel.JuMP.@variable(infeasible_model, 0 <= selected <= 1)
        SurvivorModel.JuMP.@variable(infeasible_model, recurrence)
        SurvivorModel.JuMP.@variable(infeasible_model, aggregate)
        SurvivorModel.JuMP.@constraint(infeasible_model, selected == 1.0)
        SurvivorModel.JuMP.@constraint(infeasible_model, recurrence == 0.1)
        SurvivorModel._survivor_add_one_hot_dummies!(
            infeasible_model,
            aggregate,
            Dict(1 => recurrence),
            [selected],
            [0.2],
            [0.8],
            1:1,
            recurrence_lower=[0.0],
            recurrence_upper=[1.0],
        )
        SurvivorModel.JuMP.optimize!(infeasible_model)
        @test SurvivorModel.JuMP.termination_status(infeasible_model) ==
            SurvivorModel.JuMP.MOI.INFEASIBLE
    end

    @testset "scalar one-hot gating" begin
        keys = [
            (:td, "A"),
            (:td, "B"),
        ]
        parameters = SurvivorModel.SurvivorParameterSystem(
            keys,
            Dict(key => index for (index, key) in enumerate(keys)),
            [0.0, 0.0],
            [1.0, 1.0],
        )
        derivatives = [
            SurvivorModel.SurvivorCandidateDerivatives(
                0.85,
                [0.4, 0.0],
                0.1,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.60,
                [-0.2, 0.1],
                -0.05,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.80,
                [0.1, 0.3],
                0.02,
            ),
            SurvivorModel.SurvivorCandidateDerivatives(
                0.70,
                [-0.1, 0.4],
                -0.01,
            ),
        ]
        inputs = SurvivorModel.SurvivorObjectiveInputs(
            parameters,
            derivatives,
        )
        candidates = DataFrame(
            game_id=["week1", "week1", "week2", "week2"],
            week=[1, 1, 2, 2],
            team=["A", "B", "C", "D"],
            opponent=["X", "Y", "Z", "W"],
            is_home=[true, false, true, false],
            win_probability=[0.85, 0.60, 0.80, 0.70],
        )
        state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        data = SurvivorModel._normalize_survivor_candidates(
            candidates,
            state,
            2,
        )
        config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
        )
        greedy_data = DataFrame(data)
        greedy_data.win_probability = [0.51, 0.99, 0.51, 0.99]
        greedy_data.team = ["A", "B", "A", "C"]
        greedy_indices = SurvivorModel._survivor_greedy_selected_indices(
            greedy_data,
            state,
            config,
            inputs,
        )
        @test greedy_indices == [1, 4]
        plan = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
            data,
            state,
            config,
            inputs,
        )
        expected_objective = maximum(
            begin
                values = SurvivorModel._survivor_scalar_forward_values(
                    [first_index, second_index],
                    inputs,
                    2,
                    1,
                )
                sum(
                    values.probability[position + 1, 1] +
                    0.5 * values.hessian[position + 1, 1]
                    for position in 1:2
                )
            end
            for first_index in 1:2,
            second_index in 3:4
        )
        @test plan.objective_value ≈ expected_objective
        @test nrow(plan.selections) == 2
        @test plan.selections.week == [1, 2]

        benders_plan = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
            data,
            state,
            SurvivorSelectionConfig(
                minimum_favorite_spread=nothing,
                market_guard_weeks=0,
                through_week=2,
                benders_weeks=1,
                timeout_seconds=60.0,
            ),
            inputs,
        )
        @test benders_plan.objective_value ≈ plan.objective_value
        @test nrow(benders_plan.selections) == 2
        @test length(unique(benders_plan.selections.team)) == 2
        benders_zero_prefix_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                SurvivorSelectionConfig(
                    minimum_favorite_spread=nothing,
                    market_guard_weeks=0,
                    through_week=2,
                    benders_weeks=0,
                    timeout_seconds=60.0,
                ),
                inputs,
            )
        @test benders_zero_prefix_plan.objective_value ≈
              plan.objective_value

        prefix_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
            hessian_weeks=1,
        )
        prefix_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                prefix_config,
                inputs,
            )
        prefix_selected_by_position =
            SurvivorModel._survivor_fixed_selected_indices(
                data,
                prefix_plan.selections,
                state,
                2,
            )
        prefix_selected_values =
            SurvivorModel._survivor_scalar_forward_values(
                prefix_selected_by_position,
                inputs,
                2,
                1;
                curvature_weeks=1,
                gradient_reference_indices=SurvivorModel._survivor_gradient_reference_indices(
                    [1, 1, 2, 2],
                    2,
                    inputs.covariance_gradient_gram;
                    maximum_reference_position=1,
                ),
            )
        prefix_expected_objective = sum(
            prefix_selected_values.probability[position + 1, 1]
            for position in 1:2
        ) + 0.5 * prefix_selected_values.hessian[2, 1]
        @test prefix_plan.objective_value ≈ prefix_expected_objective
        @test prefix_plan.selections.parameter_variance_adjustment[2] ≈ 0.0

        benders_prefix_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                SurvivorSelectionConfig(
                    minimum_favorite_spread=nothing,
                    market_guard_weeks=0,
                    through_week=2,
                    hessian_weeks=1,
                    benders_weeks=1,
                ),
                inputs,
            )
        @test benders_prefix_plan.objective_value ≈
              prefix_plan.objective_value
        benders_clamped_prefix_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                SurvivorSelectionConfig(
                    minimum_favorite_spread=nothing,
                    market_guard_weeks=0,
                    through_week=2,
                    hessian_weeks=1,
                    benders_weeks=20,
                ),
                inputs,
            )
        @test benders_clamped_prefix_plan.objective_value ≈
              prefix_plan.objective_value

        debug_derivative_values = [
            (0.90, [0.3, -0.1], 0.02),
            (0.70, [-0.2, 0.3], -0.03),
            (0.85, [0.1, 0.4], 0.01),
            (0.80, [-0.3, -0.1], 0.04),
            (0.88, [0.3, 0.3], -0.05),
            (0.75, [-0.1, 0.2], 0.02),
        ]
        debug_inputs = SurvivorModel.SurvivorObjectiveInputs(
            parameters,
            [
                SurvivorModel.SurvivorCandidateDerivatives(
                    probability,
                    gradient,
                    hessian,
                )
                for (probability, gradient, hessian) in debug_derivative_values
            ],
        )
        debug_candidates = DataFrame(
            game_id=["w1a", "w1b", "w2a", "w2b", "w3a", "w3b"],
            week=[1, 1, 2, 2, 3, 3],
            team=["A", "B", "A", "C", "B", "C"],
            opponent=["X", "Y", "Z", "Q", "R", "S"],
            is_home=[true, false, true, false, true, false],
            win_probability=[
                values[1] for values in debug_derivative_values
            ],
        )
        debug_state = SurvivorPoolState(
            2025,
            1;
            strikes_remaining=2,
        )
        debug_data = SurvivorModel._normalize_survivor_candidates(
            debug_candidates,
            debug_state,
            3,
        )
        debug_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=3,
            hessian_weeks=3,
            benders_weeks=1,
        )
        debug_number_of_weeks = 3
        debug_curvature_weeks = min(
            debug_config.hessian_weeks,
            debug_number_of_weeks,
        )
        debug_losses_to_elimination =
            SurvivorModel._survivor_loss_threshold(debug_state)
        debug_candidate_indices = 1:nrow(debug_data)
        debug_candidate_positions = [
            Int(debug_data.week[index]) - debug_state.current_week + 1
            for index in debug_candidate_indices
        ]
        debug_week_indices = [
            findall(==(position), debug_candidate_positions)
            for position in 1:debug_number_of_weeks
        ]
        debug_gradient_reference_indices =
            SurvivorModel._survivor_gradient_reference_indices(
                debug_candidate_positions,
                debug_number_of_weeks,
                debug_inputs.covariance_gradient_gram;
                maximum_reference_position=debug_curvature_weeks,
            )
        debug_gradient_reference_slots = [
            Dict(
                reference => slot
                for (slot, reference) in enumerate(references)
            )
            for references in debug_gradient_reference_indices
        ]
        debug_gradient_switch_position =
            SurvivorModel._survivor_gradient_switch_position(
                debug_gradient_reference_indices,
                length(debug_inputs.parameters.keys),
            )
        debug_parameter_state_count = min(
            debug_curvature_weeks,
            debug_gradient_switch_position - 1,
        )
        debug_bounds = SurvivorModel._survivor_scalar_bounds(
            debug_inputs,
            debug_candidate_positions,
            debug_number_of_weeks,
            debug_losses_to_elimination;
            curvature_weeks=debug_curvature_weeks,
            gradient_reference_indices=debug_gradient_reference_indices,
            candidate_teams=debug_data.team,
        )
        debug_feasible_schedules = [
            collect(schedule) for schedule in
            Iterators.product(debug_week_indices...) if
            length(unique(debug_data.team[collect(schedule)])) ==
            debug_number_of_weeks
        ]
        @test !isempty(debug_feasible_schedules)
        debug_lp_relaxation =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                debug_data,
                debug_state,
                debug_config,
                debug_inputs;
                lp_relaxation=true,
            )
        @test debug_lp_relaxation.termination_status ==
              SurvivorModel.JuMP.MOI.OPTIMAL
        @test debug_lp_relaxation.dual_cut.margin > 0.0
        for selected_by_position in debug_feasible_schedules
            selected_values =
                SurvivorModel._survivor_scalar_forward_values(
                    selected_by_position,
                    debug_inputs,
                    debug_number_of_weeks,
                    debug_losses_to_elimination;
                    curvature_weeks=debug_curvature_weeks,
                    gradient_reference_indices=
                        debug_gradient_reference_indices,
                )
            full_objective = sum(
                selected_values.probability[position + 1, loss_state]
                for position in 1:debug_number_of_weeks,
                loss_state in 1:debug_losses_to_elimination
            ) + 0.5 * sum(
                selected_values.hessian[position + 1, loss_state]
                for position in 1:debug_curvature_weeks,
                loss_state in 1:debug_losses_to_elimination;
                init=0.0,
            )
            prefix_objective = SurvivorModel._survivor_benders_prefix_expression(
                selected_values.probability, 1, debug_losses_to_elimination,
            )
            recourse_objective = full_objective - prefix_objective
            lp_cut_value = SurvivorModel._survivor_benders_lp_cut_value(
                debug_lp_relaxation.dual_cut,
                selected_by_position,
                selected_values.probability,
                1,
                debug_losses_to_elimination,
            )
            @test lp_cut_value + 1e-7 >= full_objective
            @test propertynames(debug_lp_relaxation.dual_cut) ==
                (:intercept, :pick_slopes, :probability_slopes, :margin, :stationarity_residual)
            @test debug_lp_relaxation.dual_cut.stationarity_residual <= 2e-6
            cut = SurvivorModel._survivor_benders_analytic_cut_components(
                selected_by_position,
                recourse_objective,
                debug_inputs,
                debug_candidate_indices,
                debug_week_indices,
                debug_number_of_weeks,
                debug_losses_to_elimination,
                debug_curvature_weeks,
                1,
                debug_gradient_reference_indices,
                debug_gradient_reference_slots,
                debug_gradient_switch_position,
                debug_parameter_state_count,
                debug_bounds,
            )
            for objective_shift in (-1e-6, 1e-6)
                perturbed_cut =
                    SurvivorModel._survivor_benders_analytic_cut_components(
                        selected_by_position,
                        recourse_objective + objective_shift,
                        debug_inputs,
                        debug_candidate_indices,
                        debug_week_indices,
                        debug_number_of_weeks,
                        debug_losses_to_elimination,
                        debug_curvature_weeks,
                        1,
                        debug_gradient_reference_indices,
                        debug_gradient_reference_slots,
                        debug_gradient_switch_position,
                        debug_parameter_state_count,
                        debug_bounds,
                    )
                @test perturbed_cut == cut
            end
            @test_throws ArgumentError SurvivorModel._survivor_benders_analytic_cut_components(
                selected_by_position,
                recourse_objective - 0.01,
                debug_inputs,
                debug_candidate_indices,
                debug_week_indices,
                debug_number_of_weeks,
                debug_losses_to_elimination,
                debug_curvature_weeks,
                1,
                debug_gradient_reference_indices,
                debug_gradient_reference_slots,
                debug_gradient_switch_position,
                debug_parameter_state_count,
                debug_bounds,
            )
            for other_schedule in debug_feasible_schedules
                other_values =
                    SurvivorModel._survivor_scalar_forward_values(
                        other_schedule,
                        debug_inputs,
                        debug_number_of_weeks,
                        debug_losses_to_elimination;
                        curvature_weeks=debug_curvature_weeks,
                        gradient_reference_indices=
                            debug_gradient_reference_indices,
                    )
                other_full_objective = sum(
                    other_values.probability[position + 1, loss_state]
                    for position in 1:debug_number_of_weeks,
                    loss_state in 1:debug_losses_to_elimination
                ) + 0.5 * sum(
                    other_values.hessian[position + 1, loss_state]
                    for position in 1:debug_curvature_weeks,
                    loss_state in 1:debug_losses_to_elimination;
                    init=0.0,
                )
                other_prefix_objective = SurvivorModel._survivor_benders_prefix_expression(
                    other_values.probability, 1, debug_losses_to_elimination,
                )
                other_recourse_objective =
                    other_full_objective - other_prefix_objective
                cut_value = cut.intercept +
                    sum(
                        cut.pick_slopes[index]
                        for index in other_schedule
                    ) +
                    sum(
                        cut.probability_slopes[position, loss_state] *
                        other_values.probability[position + 1, loss_state]
                        for position in 1:1,
                        loss_state in 1:debug_losses_to_elimination;
                        init=0.0,
                    )
                @test hasproperty(cut, :probability_slopes)
                if other_schedule == selected_by_position
                    @test cut_value ≈ other_recourse_objective atol=1e-7
                else
                    @test cut_value + 1e-7 >= other_recourse_objective
                end
            end
        end
        @testset "probability-only master across correction horizons" begin
            for losses in (1, 2), hessian_weeks in 0:3
                matrix_state = SurvivorPoolState(2025, 1; strikes_remaining=losses)
                references = SurvivorModel._survivor_gradient_reference_indices(
                    debug_data.week, 3, debug_inputs.covariance_gradient_gram;
                    maximum_reference_position=hessian_weeks,
                )
                objectives = Dict(
                    Tuple(schedule) => begin
                        forward = SurvivorModel._survivor_scalar_forward_values(
                            schedule, debug_inputs, 3, losses;
                            curvature_weeks=hessian_weeks,
                            gradient_reference_indices=references,
                        )
                        SurvivorModel._survivor_benders_prefix_expression(
                            forward.probability, 3, losses,
                        ) + 0.5 * sum(forward.hessian)
                    end
                    for schedule in debug_feasible_schedules
                )
                optimum = maximum(values(objectives))
                slots = [
                    Dict(reference => slot for (slot, reference) in enumerate(refs))
                    for refs in references
                ]
                bounds = SurvivorModel._survivor_scalar_bounds(
                    debug_inputs, debug_data.week, 3, losses;
                    curvature_weeks=hessian_weeks,
                    gradient_reference_indices=references,
                    candidate_teams=debug_data.team,
                )
                for prefix in 0:3
                    upper = SurvivorModel._survivor_benders_objective_upper_bound(
                        bounds, 3, losses, hessian_weeks, prefix,
                    )
                    expected_upper = sum(
                        bounds.probability.upper[position + 1, loss_state]
                        for position in (prefix + 1):3, loss_state in 1:losses;
                        init=0.0,
                    ) + 0.5 * sum(
                        bounds.hessian.upper[position + 1, loss_state]
                        for position in 1:hessian_weeks, loss_state in 1:losses;
                        init=0.0,
                    )
                    @test upper == expected_upper
                    for switch in 1:(hessian_weeks + 1),
                        schedule in debug_feasible_schedules
                        forward = SurvivorModel._survivor_scalar_forward_values(
                            schedule, debug_inputs, 3, losses;
                            curvature_weeks=hessian_weeks,
                            gradient_reference_indices=references,
                        )
                        recourse = objectives[Tuple(schedule)] -
                            SurvivorModel._survivor_benders_prefix_expression(
                                forward.probability, prefix, losses,
                            )
                        @test recourse <= upper + 1e-8
                        cut = SurvivorModel._survivor_benders_analytic_cut_components(
                            schedule, recourse, debug_inputs, debug_candidate_indices,
                            debug_week_indices, 3, losses, hessian_weeks, prefix,
                            references, slots, switch,
                            min(hessian_weeks, switch - 1), bounds,
                        )
                        @test propertynames(cut) ==
                            (:intercept, :pick_slopes, :probability_slopes)
                        @test size(cut.probability_slopes) == (prefix, losses)
                        for other_schedule in debug_feasible_schedules
                            other = SurvivorModel._survivor_scalar_forward_values(
                                other_schedule, debug_inputs, 3, losses;
                                curvature_weeks=hessian_weeks,
                                gradient_reference_indices=references,
                            )
                            other_recourse = objectives[Tuple(other_schedule)] -
                                SurvivorModel._survivor_benders_prefix_expression(
                                    other.probability, prefix, losses,
                                )
                            cut_value = cut.intercept +
                                sum(cut.pick_slopes[index] for index in other_schedule) +
                                sum(
                                    cut.probability_slopes[position, loss_state] *
                                    other.probability[position + 1, loss_state]
                                    for position in 1:prefix, loss_state in 1:losses;
                                    init=0.0,
                                )
                            @test cut_value + 1e-8 >= other_recourse
                            if schedule == other_schedule
                                @test cut_value ≈ other_recourse atol=2e-10
                            end
                        end
                    end
                    matrix_config = SurvivorSelectionConfig(
                        minimum_favorite_spread=nothing, market_guard_weeks=0,
                        through_week=3, hessian_weeks=hessian_weeks,
                        benders_weeks=prefix, timeout_seconds=60.0,
                    )
                    result = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                        debug_data, matrix_state, matrix_config, debug_inputs,
                    )
                    schedule = SurvivorModel._survivor_fixed_selected_indices(
                        debug_data, result.selections, matrix_state, 3,
                    )
                    @test result.objective_value ≈ objectives[Tuple(schedule)] atol=2e-8
                    best_first_pick = maximum(
                        objective for (other, objective) in objectives
                        if first(other) == first(schedule)
                    )
                    @test best_first_pick ≈ optimum atol=2e-6
                end
            end
        end
        debug_reference_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                debug_data,
                debug_state,
                SurvivorSelectionConfig(
                    minimum_favorite_spread=nothing,
                    market_guard_weeks=0,
                    through_week=3,
                ),
                debug_inputs,
            )
        benders_debug_buffer = IOBuffer()
        benders_debug_logger = SurvivorModel.Logging.ConsoleLogger(
            benders_debug_buffer,
            SurvivorModel.Logging.Debug,
        )
        benders_debug_plan, benders_debug_stderr_output =
            _capture_stderr_output() do
                SurvivorModel.Logging.with_logger(benders_debug_logger) do
                    SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                        debug_data,
                        debug_state,
                        debug_config,
                        debug_inputs,
                    )
                end
            end
        benders_debug_output = String(take!(benders_debug_buffer))
        @test !occursin("Running HiGHS", benders_debug_stderr_output)
        @test !occursin("Simplex", benders_debug_stderr_output)
        benders_progress_rows =
            _test_benders_progress_rows(benders_debug_stderr_output)
        progress_header = SurvivorModel._survivor_benders_progress_header()
        progress_header_fields = strip.(split(progress_header, " | "))
        @test length(progress_header_fields) == 13
        @test count(==("cuts"), progress_header_fields) == 1
        @test benders_debug_plan.objective_value ≈
              debug_reference_plan.objective_value
        @test nrow(benders_debug_plan.selections) == 3
        @test length(unique(benders_debug_plan.selections.team)) == 3
        debug_selected_teams = sort(
            benders_debug_plan.selections,
            :week,
        ).team
        debug_team_list = "[" *
            join(String.(debug_selected_teams), ", ") *
            "]"
        @test occursin(
            "selected_teams = \"$debug_team_list\"",
            benders_debug_output,
        )
        @test !occursin("selected_schedule", benders_debug_output)
        @test occursin("survivor Benders model built", benders_debug_output)
        @test !occursin("survivor Benders progress", benders_debug_output)
        @test occursin(
            progress_header,
            benders_debug_stderr_output,
        )
        @test occursin("best-LB", benders_debug_stderr_output)
        @test occursin("rec-gap", benders_debug_stderr_output)
        @test any(row -> row[2] == "master", benders_progress_rows)
        @test all(
            row -> !(row[2] in ("master-cut", "alt-cut")),
            benders_progress_rows,
        )
        @test all(
            length(row) == length(progress_header_fields)
            for row in benders_progress_rows
        )
        sample_progress_row = SurvivorModel._survivor_benders_progress_row(
            21,
            "master",
            SurvivorModel.JuMP.MOI.OPTIMAL,
            "SEA",
            "-",
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.0,
            4,
            0.1,
        )
        @test length(progress_header) == length(sample_progress_row)
        repeated_header_row_count, repeated_header_output =
            _capture_stderr_output() do
                SurvivorModel._survivor_benders_print_progress_row(
                    20,
                    sample_progress_row,
                )
            end
        @test repeated_header_row_count == 21
        @test repeated_header_output ==
            "\n" * progress_header * "\n" *
            repeat("-", length(progress_header)) * "\n" *
            sample_progress_row * "\n"
        @test occursin(
            "survivor Benders recourse evaluated analytically",
            benders_debug_output,
        )
        @test !occursin(
            "survivor Benders recourse solve complete",
            benders_debug_output,
        )
        @test occursin(
            "survivor Benders cut added",
            benders_debug_output,
        )
        @test occursin("elapsed_seconds", benders_debug_output)
        @test occursin("theta", benders_debug_stderr_output)
        @test !occursin("probability_objective", benders_debug_output)
        @test occursin("relative_gap", benders_debug_output)
        @test occursin("node_count", benders_debug_output)
        @test occursin("cut_count", benders_debug_output)
        @test occursin("recourse_objective", benders_debug_output)
        @test occursin("selected_teams", benders_debug_output)
        @test occursin("variables", benders_debug_output)
        @test occursin("prefix_weeks = 1", benders_debug_output)
        @test occursin("constraints", benders_debug_output)
        @test occursin("nonzeros", benders_debug_output)
        @test occursin("simplex_iterations", benders_debug_output)
        @test occursin("intercept", benders_debug_output)
        @test occursin("slope_min", benders_debug_output)
        @test occursin("slope_max", benders_debug_output)
        @test occursin(
            "survivor LP relaxation solver statistics",
            benders_debug_output,
        )
        @test occursin("solver = :hipo", benders_debug_output)
        @test occursin("run_crossover = :on", benders_debug_output)
        @test occursin("barrier_iterations", benders_debug_output)
        @test occursin(
            "objective_component = :tail_and_covariance_correction",
            benders_debug_output,
        )
        @test occursin("probability_slope_min", benders_debug_output)
        @test occursin(
            "method = :full_model_lp_dual_recourse_cut",
            benders_debug_output,
        )
        @test !occursin("simplex_iterations = nothing", benders_debug_output)
        initial_recourse_position = findfirst(
            "survivor Benders recourse evaluated analytically",
            benders_debug_output,
        )
        initial_cut_position = findfirst(
            "survivor Benders cut added",
            benders_debug_output,
        )
        @test initial_recourse_position !== nothing
        @test initial_cut_position !== nothing
        if initial_recourse_position !== nothing &&
           initial_cut_position !== nothing
            @test first(initial_recourse_position) <
                  first(initial_cut_position)
        end
        debug_greedy_selected_indices =
            SurvivorModel._survivor_greedy_selected_indices(
                debug_data,
                debug_state,
                debug_config,
                debug_inputs,
            )
        debug_second_greedy_selected_indices =
            SurvivorModel._survivor_greedy_selected_indices(
                debug_data,
                debug_state,
                debug_config,
                debug_inputs;
                first_pick_rank=2,
            )
        @test debug_greedy_selected_indices[1] !=
              debug_second_greedy_selected_indices[1]
        debug_greedy_teams = sort(
            debug_data[debug_greedy_selected_indices, :],
            :week,
        ).team
        debug_greedy_team_list = "[" *
            join(String.(debug_greedy_teams), ", ") *
            "]"
        @test occursin(
            "source = :greedy_warm_start",
            benders_debug_output,
        )
        @test occursin(
            "source = :second_greedy_warm_start",
            benders_debug_output,
        )
        @test occursin(
            "cut_method = :analytic_recourse_dual",
            benders_debug_output,
        )
        @test occursin("iteration = 0", benders_debug_output)
        @test occursin("cut_count = 2", benders_debug_output)
        @test occursin("cut_count = 3", benders_debug_output)
        @test occursin(
            "selected_teams = \"$debug_greedy_team_list\"",
            benders_debug_output,
        )

        linear_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
            hessian_weeks=0,
        )
        linear_reference_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                linear_config,
                inputs,
            )
        linear_benders_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
            hessian_weeks=0,
            benders_weeks=1,
        )
        linear_benders_buffer = IOBuffer()
        linear_benders_logger = SurvivorModel.Logging.ConsoleLogger(
            linear_benders_buffer,
            SurvivorModel.Logging.Debug,
        )
        linear_benders_plan =
            SurvivorModel.Logging.with_logger(linear_benders_logger) do
                SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                    data,
                    state,
                    linear_benders_config,
                    inputs,
                )
            end
        linear_benders_output = String(take!(linear_benders_buffer))
        @test linear_benders_plan.objective_value ≈
              linear_reference_plan.objective_value
        @test occursin(
            "objective_component = :tail_and_covariance_correction",
            linear_benders_output,
        )

        alternative_week_count = 5
        alternative_choices = 4
        alternative_pairs = [
            (week, choice)
            for week in 1:alternative_week_count
            for choice in 1:alternative_choices
        ]
        alternative_derivative_values = [
            begin
                index = (week - 1) * alternative_choices + choice
                (
                    probability=0.77 + 0.21 * sin(1.234 * index),
                    gradient=[
                        1.6 * sin(1.71 * index),
                        1.4 * cos(0.93 * index),
                    ],
                    hessian=0.8 * sin(2.11 * index),
                )
            end
            for (week, choice) in alternative_pairs
        ]
        alternative_parameter_keys = [(:td, "A"), (:td, "B")]
        alternative_parameters =
            SurvivorModel.SurvivorParameterSystem(
                alternative_parameter_keys,
                Dict(
                    key => index
                    for (index, key) in enumerate(alternative_parameter_keys)
                ),
                [0.0, 0.0],
                [0.7, 0.4],
            )
        alternative_inputs = SurvivorModel.SurvivorObjectiveInputs(
            alternative_parameters,
            [
                SurvivorModel.SurvivorCandidateDerivatives(
                    values.probability,
                    values.gradient,
                    values.hessian,
                )
                for values in alternative_derivative_values
            ],
        )
        alternative_candidates = DataFrame(
            game_id=[
                "alternative_w$(week)_g$(choice)"
                for (week, choice) in alternative_pairs
            ],
            week=[week for (week, _) in alternative_pairs],
            team=[
                "ALT$(week)_$(choice)"
                for (week, choice) in alternative_pairs
            ],
            opponent=[
                "OPP$(week)_$(choice)"
                for (week, choice) in alternative_pairs
            ],
            is_home=[iseven(choice) for (_, choice) in alternative_pairs],
            win_probability=[
                values.probability
                for values in alternative_derivative_values
            ],
        )
        alternative_state =
            SurvivorPoolState(2025, 1; strikes_remaining=2)
        alternative_data = SurvivorModel._normalize_survivor_candidates(
            alternative_candidates,
            alternative_state,
            alternative_week_count,
        )
        alternative_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=alternative_week_count,
            hessian_weeks=0,
            benders_weeks=1,
            timeout_seconds=60.0,
        )
        alternative_benders_buffer = IOBuffer()
        alternative_benders_logger = SurvivorModel.Logging.ConsoleLogger(
            alternative_benders_buffer,
            SurvivorModel.Logging.Debug,
        )
        alternative_benders_plan, alternative_benders_stderr_output =
            _capture_stderr_output() do
                SurvivorModel.Logging.with_logger(
                    alternative_benders_logger,
                ) do
                    SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                        alternative_data,
                        alternative_state,
                        alternative_config,
                        alternative_inputs,
                    )
                end
            end
        alternative_benders_output = String(take!(alternative_benders_buffer))
        @test occursin(
            SurvivorModel._survivor_benders_progress_header(),
            alternative_benders_stderr_output,
        )
        @test !occursin("survivor Benders progress", alternative_benders_output)
        @test occursin(
            "survivor Benders solve complete",
            alternative_benders_output,
        )
        alternative_progress_rows =
            _test_benders_progress_rows(alternative_benders_stderr_output)
        alternative_solve_line = findfirst(
            row -> length(row) >= 13 && row[2] == "alt",
            alternative_progress_rows,
        )
        alternative_master_bound = nothing
        alternative_forbidden_first_pick = nothing
        alternative_selected_first_pick = nothing
        if alternative_solve_line !== nothing
            progress_fields = alternative_progress_rows[alternative_solve_line]
            if length(progress_fields) >= 13
                alternative_selected_first_pick = progress_fields[5]
                alternative_forbidden_first_pick = progress_fields[6]
                alternative_master_bound =
                    tryparse(Float64, progress_fields[9])
            end
        end
        @test alternative_master_bound !== nothing
        if alternative_master_bound !== nothing
            @test alternative_master_bound > 0.0
        end
        @test alternative_forbidden_first_pick !== nothing
        @test alternative_selected_first_pick !== nothing
        if alternative_selected_first_pick !== nothing &&
           alternative_forbidden_first_pick !== nothing
            @test alternative_selected_first_pick !=
                  alternative_forbidden_first_pick
        end
        alternative_master_lines = findall(
            row -> length(row) == 13 && row[2] == "master",
            alternative_progress_rows,
        )
        @test !isempty(alternative_master_lines)
        for master_line in alternative_master_lines
            master_row = alternative_progress_rows[master_line]
            master_line < length(alternative_progress_rows) || continue
            alternative_row = alternative_progress_rows[master_line + 1]
            alternative_row[2] == "alt" || continue
            @test parse(Int, alternative_row[4]) ==
                  parse(Int, master_row[4]) + 1
            if master_line + 2 <= length(alternative_progress_rows)
                next_master_row =
                    alternative_progress_rows[master_line + 2]
                if next_master_row[2] == "master"
                    @test parse(Int, next_master_row[1]) ==
                          parse(Int, alternative_row[1]) + 1
                    @test parse(Int, next_master_row[4]) ==
                          parse(Int, alternative_row[4]) + 1
                end
            end
        end
        alternative_candidate_positions = [
            Int(alternative_data.week[index]) -
            alternative_state.current_week + 1
            for index in 1:nrow(alternative_data)
        ]
        alternative_week_indices = [
            findall(==(position), alternative_candidate_positions)
            for position in 1:alternative_week_count
        ]
        alternative_losses =
            SurvivorModel._survivor_loss_threshold(alternative_state)
        alternative_gradient_references =
            SurvivorModel._survivor_gradient_reference_indices(
                alternative_candidate_positions,
                alternative_week_count,
                alternative_inputs.covariance_gradient_gram;
                maximum_reference_position=alternative_config.hessian_weeks,
            )
        alternative_best_objective = -Inf
        alternative_best_first_picks = Set{String}()
        alternative_best_other_pick_objective = -Inf
        for schedule in Iterators.product(alternative_week_indices...)
            selected_by_position = collect(schedule)
            values = SurvivorModel._survivor_scalar_forward_values(
                selected_by_position,
                alternative_inputs,
                alternative_week_count,
                alternative_losses;
                curvature_weeks=alternative_config.hessian_weeks,
                gradient_reference_indices=
                    alternative_gradient_references,
            )
            objective = sum(
                values.probability[position + 1, loss_state]
                for position in 1:alternative_week_count,
                loss_state in 1:alternative_losses
            ) + 0.5 * sum(
                values.hessian[position + 1, loss_state]
                for position in 1:alternative_config.hessian_weeks,
                loss_state in 1:alternative_losses;
                init=0.0,
            )
            if alternative_forbidden_first_pick !== nothing &&
               String(alternative_data.team[selected_by_position[1]]) !=
               alternative_forbidden_first_pick
                alternative_best_other_pick_objective = max(
                    alternative_best_other_pick_objective,
                    objective,
                )
            end
            if objective > alternative_best_objective + 1e-10
                alternative_best_objective = objective
                empty!(alternative_best_first_picks)
                push!(
                    alternative_best_first_picks,
                    String(alternative_data.team[selected_by_position[1]]),
                )
            elseif isapprox(
                objective,
                alternative_best_objective;
                rtol=0.0,
                atol=1e-10,
            )
                push!(
                    alternative_best_first_picks,
                    String(alternative_data.team[selected_by_position[1]]),
                )
            end
        end
        @test length(alternative_best_first_picks) == 1
        @test alternative_benders_plan.objective_value ≈
              alternative_best_objective
        @test only(alternative_benders_plan.current_pick.team) in
              alternative_best_first_picks
        @test isfinite(alternative_best_other_pick_objective)
        if alternative_master_bound !== nothing &&
           alternative_forbidden_first_pick !== nothing
            @test alternative_best_other_pick_objective <=
                  alternative_master_bound +
                  SurvivorModel._survivor_benders_gap_tolerance(
                      alternative_best_other_pick_objective,
                      alternative_master_bound,
                  )
        end

        benders_timeout_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
            benders_weeks=1,
            timeout_seconds=1e-12,
        )
        benders_timeout_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                benders_timeout_config,
                inputs,
            )
        benders_timeout_greedy =
            SurvivorModel._survivor_greedy_selected_indices(
                data,
                state,
                benders_timeout_config,
                inputs,
            )
        benders_timeout_values =
            SurvivorModel._survivor_scalar_forward_values(
                benders_timeout_greedy,
                inputs,
                2,
                1;
                curvature_weeks=2,
                gradient_reference_indices=SurvivorModel._survivor_gradient_reference_indices(
                    [1, 1, 2, 2],
                    2,
                    inputs.covariance_gradient_gram;
                    maximum_reference_position=2,
                ),
            )
        benders_timeout_objective = sum(
            benders_timeout_values.probability[position + 1, 1] +
            0.5 * benders_timeout_values.hessian[position + 1, 1]
            for position in 1:2
        )
        @test benders_timeout_plan.objective_value ≈
              benders_timeout_objective

        clamped_plan = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
            data,
            state,
            SurvivorSelectionConfig(
                minimum_favorite_spread=nothing,
                market_guard_weeks=0,
                through_week=2,
                hessian_weeks=19,
            ),
            inputs,
        )
        @test clamped_plan.objective_value ≈ plan.objective_value

    end

    @testset "debug solver progress logging" begin
        config = SurvivorSelectionConfig()
        SurvivorModel.Logging.with_logger(SurvivorModel.Logging.NullLogger()) do
            @test !SurvivorModel._survivor_debug_logging_enabled()
            model = SurvivorModel._survivor_milp_model(config)
            @test SurvivorModel.JuMP.MOI.get(
                SurvivorModel.JuMP.backend(model),
                SurvivorModel.JuMP.MOI.Silent(),
            )
        end

        debug_logger = SurvivorModel.Logging.ConsoleLogger(
            IOBuffer(),
            SurvivorModel.Logging.Debug,
        )
        SurvivorModel.Logging.with_logger(debug_logger) do
            @test SurvivorModel._survivor_debug_logging_enabled()
            model = SurvivorModel._survivor_milp_model(config)
            @test !SurvivorModel.JuMP.MOI.get(
                SurvivorModel.JuMP.backend(model),
                SurvivorModel.JuMP.MOI.Silent(),
            )
            benders_config = SurvivorSelectionConfig(benders_weeks=1)
            benders_model = SurvivorModel._survivor_milp_model(
                benders_config,
            )
            @test SurvivorModel.JuMP.MOI.get(
                SurvivorModel.JuMP.backend(benders_model),
                SurvivorModel.JuMP.MOI.Silent(),
            )
            benders_direct_model =
                SurvivorModel._survivor_direct_milp_model(
                    benders_config,
                )
            @test SurvivorModel.JuMP.MOI.get(
                SurvivorModel.JuMP.backend(benders_direct_model),
                SurvivorModel.JuMP.MOI.Silent(),
            )
        end
    end

    @testset "Benders initial LP solver options" begin
        moi = SurvivorModel.JuMP.MOI
        config = SurvivorSelectionConfig(benders_weeks=1)
        lp_optimizer = moi.instantiate(
            SurvivorModel._survivor_optimizer(
                config;
                lp_relaxation=true,
            ),
        )
        master_optimizer = moi.instantiate(
            SurvivorModel._survivor_optimizer(config),
        )
        raw_attribute(name) = moi.RawOptimizerAttribute(name)
        @test moi.get(lp_optimizer, raw_attribute("solver")) == "hipo"
        @test moi.get(lp_optimizer, raw_attribute("parallel")) == "on"
        @test moi.get(lp_optimizer, raw_attribute("threads")) == 0
        @test moi.get(lp_optimizer, raw_attribute("run_crossover")) == "on"
        @test moi.get(master_optimizer, raw_attribute("solver")) == "choose"
        @test moi.get(master_optimizer, raw_attribute("mip_lp_solver")) == "hipo"
        @test moi.get(master_optimizer, raw_attribute("run_crossover")) == "on"
    end

    @testset "selection configuration" begin
        @test SurvivorSelectionConfig().hessian_weeks == 3
        @test SurvivorSelectionConfig(hessian_weeks=0).hessian_weeks == 0
        @test SurvivorSelectionConfig(hessian_weeks=19).hessian_weeks == 19
        @test SurvivorSelectionConfig().benders_weeks === nothing
        @test SurvivorSelectionConfig(benders_weeks=0).benders_weeks == 0
        @test SurvivorSelectionConfig(benders_weeks=20).benders_weeks == 20
        @test_throws ArgumentError SurvivorSelectionConfig(benders_weeks=-1)
        @test SurvivorSelectionConfig().timeout_seconds === nothing
        @test SurvivorSelectionConfig(timeout_seconds=12.5).timeout_seconds == 12.5
        @test SurvivorSelectionConfig().banned_first_pick_teams == String[]
        ban_config = SurvivorSelectionConfig(
            banned_first_pick_teams=[" kc ", "sf", "KC"],
        )
        @test ban_config.banned_first_pick_teams == ["KC", "SF"]
        @test_throws ArgumentError SurvivorSelectionConfig(
            banned_first_pick_teams=["KC", " "],
        )

        @testset "Benders direct master" begin
            model = SurvivorModel._survivor_direct_milp_model(
                SurvivorSelectionConfig(),
            )
            @test SurvivorModel.JuMP.backend(model) isa
                  SurvivorModel.HiGHS.Optimizer

            SurvivorModel.JuMP.@variable(model, selected, Bin)
            SurvivorModel.JuMP.@variable(model, theta <= 1.0)
            SurvivorModel.JuMP.@objective(model, Max, 1.0 * theta)
            SurvivorModel.JuMP.optimize!(model)
            @test SurvivorModel.JuMP.objective_value(model) ≈ 1.0

            SurvivorModel.JuMP.@constraint(model, theta <= 0.5)
            SurvivorModel.JuMP.optimize!(model)
            @test SurvivorModel.JuMP.termination_status(model) ==
                  SurvivorModel.JuMP.MOI.OPTIMAL
            @test SurvivorModel.JuMP.objective_value(model) ≈ 0.5
        end

        ban_candidates = DataFrame(
            game_id=["week1", "week1", "week2", "week2"],
            week=[1, 1, 2, 2],
            team=["A", "B", "A", "C"],
            opponent=["X", "Y", "D", "E"],
            is_home=[true, false, true, false],
            win_probability=[0.9, 0.8, 0.75, 0.65],
        )
        ban_state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        filtered_ban_candidates =
            SurvivorModel._normalize_survivor_candidates(
                ban_candidates,
                ban_state,
                2;
                banned_first_pick_teams=[" a "],
            )
        @test filtered_ban_candidates.week == [1, 2, 2]
        @test filtered_ban_candidates.team == ["B", "A", "C"]
        @test_throws ArgumentError SurvivorModel._normalize_survivor_candidates(
            ban_candidates,
            ban_state,
            2;
            banned_first_pick_teams=["A", "B"],
        )

        no_guard_config = SurvivorSelectionConfig(
            minimum_favorite_spread=nothing,
            missing_market_policy=:exclude,
            market_guard_weeks=0,
            through_week=3,
        )
        @test no_guard_config.minimum_favorite_spread === nothing
        @test_throws ArgumentError SurvivorSelectionConfig(
            missing_market_policy=:unknown,
        )
        @test_throws ArgumentError SurvivorSelectionConfig(
            minimum_favorite_spread=-1.0,
        )
        @test_throws ArgumentError SurvivorSelectionConfig(
            timeout_seconds=0.0,
        )
        @test_throws ArgumentError SurvivorSelectionConfig(
            timeout_seconds=-1.0,
        )
        @test_throws ArgumentError SurvivorSelectionConfig(
            timeout_seconds=NaN,
        )
        @test_throws ArgumentError SurvivorSelectionConfig(
            timeout_seconds=Inf,
        )
        @test_throws ArgumentError SurvivorSelectionConfig(
            hessian_weeks=-1,
        )

        @testset "timed feasible incumbent" begin
            timeout_seconds = 5.0
            model = SurvivorModel.JuMP.Model(
                SurvivorModel.JuMP.optimizer_with_attributes(
                    SurvivorModel.HiGHS.Optimizer,
                    "time_limit" => timeout_seconds,
                ),
            )
            SurvivorModel.JuMP.set_silent(model)
            SurvivorModel.JuMP.@variable(model, selected[1:600], Bin)
            for offset in 1:30
                SurvivorModel.JuMP.@constraint(
                    model,
                    [index=1:20],
                    sum(
                        selected[mod1(index + offset * step, 600)]
                        for step in 0:30
                    ) <= 4,
                )
            end
            SurvivorModel.JuMP.@objective(
                model,
                Max,
                sum((index % 101 + 1) * selected[index] for index in 1:600),
            )
            for variable in selected
                SurvivorModel.JuMP.set_start_value(variable, 0.0)
            end
            SurvivorModel.JuMP.optimize!(model)
            @test SurvivorModel.JuMP.termination_status(model) ==
                SurvivorModel.JuMP.MOI.TIME_LIMIT
            @test SurvivorModel.JuMP.primal_status(model) ==
                SurvivorModel.JuMP.MOI.FEASIBLE_POINT
            @test SurvivorModel._survivor_has_feasible_incumbent(model)
            diagnostics = SurvivorModel._survivor_log_milp_result(
                model,
                :test,
                timeout_seconds,
            )
            @test diagnostics.termination_status ==
                SurvivorModel.JuMP.MOI.TIME_LIMIT
            @test diagnostics.primal_status ==
                SurvivorModel.JuMP.MOI.FEASIBLE_POINT
            @test diagnostics.has_values
            @test diagnostics.incumbent_objective !== nothing
            @test diagnostics.objective_bound !== nothing
            @test diagnostics.relative_gap !== nothing
            @test diagnostics.node_count !== nothing
        end

    end
end
