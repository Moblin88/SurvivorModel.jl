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
        bounds = SurvivorModel._survivor_scalar_bounds(
            inputs,
            [1, 2],
            2,
            2,
        )
        values = SurvivorModel._survivor_scalar_forward_values(
            [1, 2],
            inputs,
            2,
            2,
        )
        @test all(values.probability .>= bounds.probability.lower)
        @test all(values.probability .<= bounds.probability.upper)
        @test all(values.gradient .>= bounds.gradient.lower)
        @test all(values.gradient .<= bounds.gradient.upper)
        @test all(values.hessian .>= bounds.hessian.lower)
        @test all(values.hessian .<= bounds.hessian.upper)
        @test SurvivorModel._survivor_other_interval(
            bounds.candidate_probability.lower,
            bounds.candidate_probability.upper,
            [1],
            1,
            1,
        ) === nothing

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
    end

    @testset "selection configuration" begin
        @test SurvivorSelectionConfig().objective ===
            :exact_milp
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

        @testset "timed feasible incumbent" begin
            model = SurvivorModel.JuMP.Model(
                SurvivorModel.JuMP.optimizer_with_attributes(
                    SurvivorModel.HiGHS.Optimizer,
                    "time_limit" => 0.01,
                    "threads" => 4,
                    "parallel" => "on",
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
                0.01,
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
