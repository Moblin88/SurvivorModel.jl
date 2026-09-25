using DataFrames
using Dates
using ForwardDiff
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

function _survivor_expected_weeks_bruteforce(
    probabilities::AbstractVector{<:Real},
    losses_to_elimination::Integer,
)
    total = Ref(0.0)
    number_of_weeks = length(probabilities)
    visit = function(week, losses, probability)
        if week > number_of_weeks
            total[] += number_of_weeks * probability
            return
        end
        win_probability = Float64(probabilities[week])
        visit(week + 1, losses, probability * win_probability)
        loss_probability = probability * (1.0 - win_probability)
        if losses + 1 >= losses_to_elimination
            total[] += (week - 1) * loss_probability
        else
            visit(week + 1, losses + 1, loss_probability)
        end
    end
    visit(1, 0, 1.0)
    return total[]
end

@testset "survivor pool optimization" begin
    @testset "reach discounts" begin
        @test survivor_reach_discounts(
            4;
            weekly_survival_probability=0.65,
            strikes_remaining=0,
        ) ≈ [1.0, 0.65, 0.4225, 0.274625]
        @test survivor_reach_discounts(
            5;
            weekly_survival_probability=0.65,
            strikes_remaining=1,
        ) ≈ [1.0, 0.65, 0.4225, 0.274625, 0.17850625]
        @test survivor_reach_discounts(
            4;
            weekly_survival_probability=0.65,
            strikes_remaining=2,
        ) ≈ [1.0, 1.0, 0.8775, 0.71825]
        @test survivor_reach_discounts(
            2,
            4;
            weekly_survival_probability=0.65,
            strikes_remaining=1,
        ) ≈ [1.0, 0.65, 0.4225]
        @test survivor_reach_discounts(0) == Float64[]
        @test_throws ArgumentError survivor_reach_discounts(
            2;
            weekly_survival_probability=1.1,
        )
    end

    @testset "candidate expansion and filtering" begin
        forecast = _survivor_forecast_fixture()
        candidates = build_survivor_candidates(forecast)
        @test nrow(candidates) == 6
        @test Set(candidates.team) == Set(["A", "B", "C", "D"])
        @test all(
            candidates.win_probability .>=
            DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY,
        )
        direct_candidates = DataFrame(
            game_id=["direct", "direct"],
            week=[1, 1],
            team=["Underdog", "Favorite"],
            opponent=["X", "Y"],
            is_home=[false, true],
            win_probability=[0.49, 0.51],
        )
        direct_plan = optimize_survivor_pool(
            direct_candidates,
            SurvivorPoolState(2025, 1; strikes_remaining=0);
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                through_week=1,
            ),
        )
        @test direct_plan.current_pick.team == ["Favorite"]
        @test_throws ArgumentError optimize_survivor_pool(
            direct_candidates,
            SurvivorPoolState(2025, 1; strikes_remaining=0);
            selection_config=SurvivorSelectionConfig(
                objective=:exact_milp,
                through_week=1,
            ),
        )
        @test_throws ArgumentError optimize_survivor_pool(
            direct_candidates,
            SurvivorPoolState(2025, 1; strikes_remaining=0);
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                prove_first_pick=true,
                through_week=1,
            ),
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

    @testset "binary assignment and current pick" begin
        state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        plan = optimize_survivor_pool(
            build_survivor_candidates(_survivor_forecast_fixture()),
            state;
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                through_week=2,
                weekly_survival_probability=0.65,
            ),
        )
        @test plan.selections.week == [1, 2]
        @test plan.selections.team == ["B", "A"]
        @test plan.current_pick.team == ["B"]
        @test length(unique(plan.selections.team)) == 2
        @test plan.discounts.discount ≈ [1.0, 0.65]
        @test plan.objective_value ≈ 0.8 + 0.8 * 0.95
        @test plan.objective_value ≈ sum(plan.selections.objective_contribution)
    end

    @testset "exact expected-weeks objective" begin
        candidates = DataFrame(
            game_id=["week1", "week1", "week2", "week2"],
            week=[1, 1, 2, 2],
            team=["A", "B", "C", "D"],
            opponent=["X", "Y", "Z", "W"],
            is_home=[false, true, false, true],
            win_probability=[0.6, 0.8, 0.9, 0.7],
        )
        config = SurvivorSelectionConfig(
            objective=:fixed_exact_milp,
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
        )
        two_loss_allowance_plan = optimize_survivor_pool(
            candidates,
            SurvivorPoolState(2025, 1; strikes_remaining=2);
            selection_config=config,
        )
        @test nrow(two_loss_allowance_plan.selections) == 2
        @test two_loss_allowance_plan.selections.team == ["B", "C"]
        @test two_loss_allowance_plan.objective_value ≈
            _survivor_expected_weeks_bruteforce([0.8, 0.9], 2)
        @test two_loss_allowance_plan.objective_value ≈
            sum(two_loss_allowance_plan.selections.objective_contribution)
        @test two_loss_allowance_plan.selections.survival_probability ≈ [1.0, 0.98]
        @test two_loss_allowance_plan.selections.elimination_probability ≈ [0.0, 0.02]

        one_loss_allowance_plan = optimize_survivor_pool(
            candidates,
            SurvivorPoolState(2025, 1; strikes_remaining=1);
            selection_config=config,
        )
        @test one_loss_allowance_plan.selections.team == ["B", "C"]
        @test one_loss_allowance_plan.objective_value ≈
            _survivor_expected_weeks_bruteforce([0.8, 0.9], 1)
        @test one_loss_allowance_plan.selections.survival_probability ≈ [0.8, 0.72]

        constant_plan = optimize_survivor_pool(
            candidates,
            SurvivorPoolState(2025, 1; strikes_remaining=3);
            selection_config=config,
        )
        @test constant_plan.objective_value ≈ 2.0
        @test all(constant_plan.selections.survival_probability .== 1.0)
    end

    @testset "expected-weeks probability validation" begin
        endpoint_candidates = DataFrame(
            game_id=["endpoint", "endpoint"],
            week=[1, 1],
            team=["A", "B"],
            opponent=["C", "D"],
            is_home=[true, false],
            win_probability=[1.0, 0.8],
        )
        exact_endpoint_plan = optimize_survivor_pool(
            endpoint_candidates,
            SurvivorPoolState(2025, 1; strikes_remaining=1);
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                minimum_favorite_spread=nothing,
                market_guard_weeks=0,
                through_week=1,
            ),
        )
        @test exact_endpoint_plan.current_pick.team == ["A"]
        @test exact_endpoint_plan.objective_value ≈ 1.0
    end

    @testset "near-term market favorite guard" begin
        state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        plan = optimize_survivor_pool(
            _market_guard_candidates(),
            state;
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                through_week=3,
                weekly_survival_probability=0.65,
            ),
        )
        @test plan.selections.team == ["B", "D", "E"]
        @test plan.selections.market_spread == [2.0, 3.0, -10.0]
        @test plan.current_pick.team == ["B"]
        @test DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD == 2.0
        @test DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY == 0.5

        missing_line = DataFrame(
            game_id=["missing_line", "missing_line"],
            week=[1, 1],
            team=["A", "B"],
            opponent=["C", "D"],
            is_home=[true, false],
            win_probability=[0.95, 0.8],
            market_spread=Union{Missing,Float64}[missing, 1.0],
        )
        missing_line_plan = optimize_survivor_pool(
            missing_line,
            SurvivorPoolState(2025, 1; strikes_remaining=0);
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                through_week=1,
            ),
        )
        @test missing_line_plan.current_pick.team == ["A"]

        @test_throws ArgumentError optimize_survivor_pool(
            DataFrame(
                game_id=["infeasible", "infeasible"],
                week=[1, 1],
                team=["A", "B"],
                opponent=["C", "D"],
                is_home=[true, false],
                win_probability=[0.95, 0.8],
                market_spread=[1.0, 1.5],
            ),
            SurvivorPoolState(2025, 1; strikes_remaining=0);
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                through_week=1,
            ),
        )
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
        state = SurvivorPoolState(2025, 1; picks_made=Dict(), strikes_remaining=0)
        @test_throws ArgumentError optimize_survivor_pool(
            candidates,
            state;
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                through_week=3,
            ),
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
            time_edges=[0, Inf],
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
            time_edges=[0, Inf],
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
        probability_function = SurvivorModel._survivor_candidate_win_function(
            context.model,
            context.marks,
            row;
            horizon=GAME_CLOCK_SECONDS,
        )
        forward_gradient = ForwardDiff.gradient(
            probability_function,
            local_mean,
        )
        forward_hessian = ForwardDiff.hessian(
            probability_function,
            local_mean,
        )
        derivative = only(inputs.derivatives)
        expected_gradient = zeros(length(inputs.parameters.keys))
        for (local_index, global_index) in enumerate(global_indices)
            expected_gradient[global_index] += forward_gradient[local_index]
        end
        @test derivative.base_probability ≈ probability_function(local_mean)
        @test derivative.gradient ≈ expected_gradient
        @test inputs.covariance_gradient_gram[1, 1] ≈ sum(
            expected_gradient[index]^2 * inputs.parameters.variance[index]
            for index in eachindex(expected_gradient)
        )
        @test derivative.hessian_covariance ≈ sum(
            forward_hessian[local_index, local_index] *
            inputs.parameters.variance[global_index]
            for (local_index, global_index) in enumerate(global_indices)
        )
    end

    @testset "whole-plan scalar covariance recursion" begin
        schedule, historical, current = _survivor_context_fixture()
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            time_edges=[0, Inf],
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
            objective=:exact_milp,
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
            ==((:defensive, "A", 1)),
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

        discount_table = SurvivorModel._survivor_discount_table(state, config)
        covariance_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                config,
                discount_table,
                inputs,
            )
        parameter_mean = inputs.parameters.log_mean
        probability_functions = [
            SurvivorModel._survivor_candidate_win_function(
                context.model,
                context.marks,
                row;
                horizon=GAME_CLOCK_SECONDS,
            )
            for row in eachrow(data)
        ]
        local_means = [first_local_mean, second_local_mean]
        local_indices = [first_indices, second_indices]
        objective_function = global_theta -> begin
            probabilities = [
                probability_functions[index](
                    [
                        global_theta[local_indices[index][local_position]] +
                        (
                            local_means[index][local_position] -
                            parameter_mean[local_indices[index][local_position]]
                        )
                        for local_position in eachindex(local_indices[index])
                    ],
                )
                for index in eachindex(probability_functions)
            ]
            probabilities[1] + probabilities[1] * probabilities[2]
        end
        objective_hessian = ForwardDiff.hessian(
            objective_function,
            parameter_mean,
        )
        expected_objective =
            objective_function(parameter_mean) +
            0.5 * sum(
                objective_hessian[index, index] *
                inputs.parameters.variance[index]
                for index in eachindex(parameter_mean)
            )
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
                discount_table,
                zero_inputs,
            )
        fixed_data = DataFrame(data)
        fixed_data.win_probability = [
            derivative.base_probability for derivative in inputs.derivatives
        ]
        fixed_plan =
            SurvivorModel._optimize_survivor_expected_weeks_fixed_milp(
                fixed_data,
                state,
                SurvivorSelectionConfig(
                    objective=:fixed_exact_milp,
                    minimum_favorite_spread=nothing,
                    market_guard_weeks=0,
                    through_week=2,
                ),
                discount_table,
            )
        @test zero_plan.objective_value ≈ fixed_plan.objective_value
        @test zero_plan.selections.team == fixed_plan.selections.team
        linear_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                SurvivorSelectionConfig(
                    objective=:exact_milp,
                    minimum_favorite_spread=nothing,
                    market_guard_weeks=0,
                    through_week=2,
                    hessian_weeks=0,
                ),
                discount_table,
                inputs,
            )
        @test linear_plan.objective_value ≈ fixed_plan.objective_value
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
            (:td, "A", 1),
            (:td, "B", 1),
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

    @testset "scalar one-hot gating" begin
        keys = [
            (:td, "A", 1),
            (:td, "B", 1),
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
            objective=:exact_milp,
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
        discount_table = SurvivorModel._survivor_discount_table(state, config)
        plan = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
            data,
            state,
            config,
            discount_table,
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

        prefix_config = SurvivorSelectionConfig(
            objective=:exact_milp,
            minimum_favorite_spread=nothing,
            market_guard_weeks=0,
            through_week=2,
            hessian_weeks=1,
        )
        prefix_discount_table =
            SurvivorModel._survivor_discount_table(state, prefix_config)
        prefix_plan =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                prefix_config,
                prefix_discount_table,
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

        clamped_plan = SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
            data,
            state,
            SurvivorSelectionConfig(
                objective=:exact_milp,
                minimum_favorite_spread=nothing,
                market_guard_weeks=0,
                through_week=2,
                hessian_weeks=19,
            ),
            discount_table,
            inputs,
        )
        @test clamped_plan.objective_value ≈ plan.objective_value

        proof_inputs = SurvivorModel.SurvivorObjectiveInputs(
            parameters,
            [
                SurvivorModel.SurvivorCandidateDerivatives(
                    0.95,
                    [0.0, 0.0],
                    0.2,
                ),
                SurvivorModel.SurvivorCandidateDerivatives(
                    0.60,
                    [0.0, 0.0],
                    -0.1,
                ),
                SurvivorModel.SurvivorCandidateDerivatives(
                    0.90,
                    [0.0, 0.0],
                    0.3,
                ),
                SurvivorModel.SurvivorCandidateDerivatives(
                    0.70,
                    [0.0, 0.0],
                    -0.05,
                ),
            ],
        )
        first_selection = sort(data[[1, 3], :], [:week, :team])
        alternate_selection = sort(data[[2, 3], :], [:week, :team])
        first_evaluation = SurvivorModel._survivor_full_scalar_objective(
            data,
            state,
            proof_inputs,
            first_selection,
        )
        alternate_evaluation = SurvivorModel._survivor_full_scalar_objective(
            data,
            state,
            proof_inputs,
            alternate_selection,
        )
        @test first_evaluation.objective > alternate_evaluation.objective
        @test any(abs.(first_evaluation.values.hessian[2:end, :]) .> 0.0)
        infeasible_proof =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                config,
                discount_table,
                proof_inputs;
                use_warm_start=false,
                curvature_weeks_override=2,
                objective_lower_bound=first_evaluation.objective,
                forbidden_first_pick_index=1,
                proof_mode=true,
                phase=:first_pick_proof,
                return_solver_diagnostics=true,
            )
        @test infeasible_proof.proof_status === :infeasible
        feasible_proof =
            SurvivorModel._optimize_survivor_expected_weeks_scalar_milp(
                data,
                state,
                config,
                discount_table,
                proof_inputs;
                use_warm_start=false,
                curvature_weeks_override=2,
                objective_lower_bound=alternate_evaluation.objective,
                forbidden_first_pick_index=1,
                proof_mode=true,
                phase=:first_pick_proof,
                return_solver_diagnostics=true,
            )
        @test feasible_proof.proof_status === :feasible
        @test only(feasible_proof.proof_first_pick).index == 2
        fake_plan = SurvivorPoolPlan(
            state,
            alternate_selection,
            alternate_selection[alternate_selection.week .== 1, :],
            discount_table,
            alternate_evaluation.objective,
            config,
        )
        @test_throws ArgumentError SurvivorModel._survivor_prove_first_pick(
            data,
            state,
            config,
            discount_table,
            proof_inputs,
            fake_plan,
        )
    end

    @testset "selection configuration" begin
        @test SurvivorSelectionConfig().objective ===
            :exact_milp
        @test SurvivorSelectionConfig().hessian_weeks == 3
        @test SurvivorSelectionConfig(hessian_weeks=0).hessian_weeks == 0
        @test SurvivorSelectionConfig(hessian_weeks=19).hessian_weeks == 19
        @test !SurvivorSelectionConfig().prove_first_pick
        @test SurvivorSelectionConfig(prove_first_pick=true).prove_first_pick
        @test SurvivorSelectionConfig().timeout_seconds === nothing
        @test SurvivorSelectionConfig(timeout_seconds=12.5).timeout_seconds == 12.5
        @test SurvivorSelectionConfig(objective=:exact_milp).objective ===
            :exact_milp
        @test SurvivorSelectionConfig(objective=:fixed_exact_milp).objective ===
            :fixed_exact_milp
        no_guard_config = SurvivorSelectionConfig(
            objective=:milp,
            minimum_favorite_spread=nothing,
            missing_market_policy=:exclude,
            market_guard_weeks=0,
            through_week=3,
        )
        @test no_guard_config.minimum_favorite_spread === nothing
        @test_throws ArgumentError SurvivorSelectionConfig(
            objective=:unknown,
        )
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

        missing_line = DataFrame(
            game_id=["missing_line", "missing_line"],
            week=[1, 1],
            team=["A", "B"],
            opponent=["C", "D"],
            is_home=[true, false],
            win_probability=[0.95, 0.8],
            market_spread=Union{Missing,Float64}[missing, 1.0],
        )
        @test_throws ArgumentError optimize_survivor_pool(
            missing_line,
            SurvivorPoolState(2025, 1; strikes_remaining=0);
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                through_week=1,
                missing_market_policy=:exclude,
            ),
        )

        no_guard_plan = optimize_survivor_pool(
            missing_line,
            SurvivorPoolState(2025, 1; strikes_remaining=0);
            selection_config=SurvivorSelectionConfig(
                objective=:fixed_exact_milp,
                minimum_favorite_spread=nothing,
                market_guard_weeks=0,
                through_week=1,
            ),
        )
        @test no_guard_plan.selection_config.minimum_favorite_spread === nothing
        @test no_guard_plan.current_pick.team == ["A"]
        @test isfinite(no_guard_plan.objective_value)
    end
end
