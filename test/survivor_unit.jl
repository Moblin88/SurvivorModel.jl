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
        ) ≈ [1.0, 1.0, 0.8775, 0.71825, 0.56298125]
        @test survivor_reach_discounts(
            4;
            weekly_survival_probability=0.65,
            strikes_remaining=2,
        ) ≈ [1.0, 1.0, 1.0, 0.957125]
        @test survivor_reach_discounts(
            2,
            4;
            weekly_survival_probability=0.65,
            strikes_remaining=1,
        ) ≈ [1.0, 1.0, 0.8775]
        @test survivor_reach_discounts(0) == Float64[]
        @test_throws ArgumentError survivor_reach_discounts(
            2;
            weekly_survival_probability=1.1,
        )
    end

    @testset "candidate expansion and filtering" begin
        forecast = _survivor_forecast_fixture()
        candidates = build_survivor_candidates(forecast)
        @test nrow(candidates) == 8
        @test Set(candidates.team) == Set(["A", "B", "C", "D", "E"])
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
            through_week=2,
            weekly_survival_probability=0.65,
        )
        @test plan.selections.week == [1, 2]
        @test plan.selections.team == ["B", "A"]
        @test plan.current_pick.team == ["B"]
        @test length(unique(plan.selections.team)) == 2
        @test plan.discounts.discount ≈ [1.0, 0.65]
        @test plan.objective_value ≈ 0.8 + 0.95 * 0.65
        @test plan.objective_value ≈ sum(plan.selections.objective_contribution)
    end

    @testset "near-term market favorite guard" begin
        state = SurvivorPoolState(2025, 1; strikes_remaining=0)
        plan = optimize_survivor_pool(
            _market_guard_candidates(),
            state;
            through_week=3,
            weekly_survival_probability=0.65,
        )
        @test plan.selections.team == ["B", "D", "E"]
        @test plan.selections.market_spread == [2.0, 3.0, -10.0]
        @test plan.current_pick.team == ["B"]
        @test DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD == 2.0

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
            through_week=1,
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
            through_week=1,
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
            through_week=3,
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
            through_week=2,
            include_completed=true,
            strikes_remaining=0,
        )
        @test nrow(plan.selections) == 1
        @test plan.selections.game_id[1] == forecast.game_id[1]
        @test plan.selections.win_probability[1] ≈ max(
            forecast.home_win_probability[1],
            forecast.away_win_probability[1],
        )
    end
end
