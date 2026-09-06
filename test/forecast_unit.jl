using SurvivorModel
using DataFrames
using Dates
using Test

function _forecast_fixture()
    schedule = DataFrame(
        game_id=[
            "2022_01_AWAY_HOME",
            "2022_01_HOME_AWAY",
            "2023_01_AWAY_HOME",
            "2023_02_HOME_AWAY",
            "2023_03_AWAY_HOME",
            "2023_POST_HOME_AWAY",
        ],
        season=[2022, 2022, 2023, 2023, 2023, 2023],
        game_type=["REG", "REG", "REG", "REG", "REG", "SB"],
        week=[1, 1, 1, 2, 3, 1],
        gameday=[
            Date(2022, 9, 11),
            Date(2022, 9, 12),
            Date(2023, 9, 10),
            Date(2023, 9, 17),
            Date(2023, 9, 24),
            Date(2024, 2, 11),
        ],
        away_team=["AWAY", "HOME", "AWAY", "HOME", "AWAY", "HOME"],
        home_team=["HOME", "AWAY", "HOME", "AWAY", "HOME", "AWAY"],
        away_score=Union{Missing,Int}[17, 20, 17, 24, missing, missing],
        home_score=Union{Missing,Int}[24, 17, 24, 21, missing, missing],
        result=Union{Missing,Int}[7, -3, 7, -3, missing, missing],
    )

    historical = DataFrame(
        game_id=[
            "2022_01_AWAY_HOME",
            "2022_01_AWAY_HOME",
            "2022_01_HOME_AWAY",
            "2022_01_HOME_AWAY",
        ],
        fixed_drive=[1, 2, 1, 2],
        posteam=["HOME", "AWAY", "AWAY", "HOME"],
        defteam=["AWAY", "HOME", "HOME", "AWAY"],
        posteam_home=[true, false, false, true],
        defteam_home=[false, true, true, false],
        drive_result=["Touchdown", "Punt", "Field goal", "Punt"],
        time_of_possession=Second.([60, 60, 120, 120]),
        home_spread_change=[7.0, 0.0, -3.0, 0.0],
    )

    current = DataFrame(
        game_id=[
            "2023_01_AWAY_HOME",
            "2023_01_AWAY_HOME",
            "2023_02_HOME_AWAY",
            "2023_02_HOME_AWAY",
        ],
        fixed_drive=[1, 2, 1, 2],
        posteam=["HOME", "AWAY", "AWAY", "HOME"],
        defteam=["AWAY", "HOME", "HOME", "AWAY"],
        posteam_home=[true, false, false, true],
        defteam_home=[false, true, true, false],
        drive_result=["Touchdown", "Punt", "Touchdown", "Field goal"],
        time_of_possession=Second.([60, 60, 60, 60]),
        home_spread_change=[7.0, 0.0, -7.0, 3.0],
    )

    return schedule, historical, current
end

@testset "regular-season forecast" begin
    schedule, historical, current = _forecast_fixture()

    @testset "schedule normalization" begin
        normalized = load_schedule(schedule)
        @test normalized.game_id == schedule.game_id
        @test normalized.season == schedule.season
        @test_throws ArgumentError load_schedule(
            vcat(schedule, schedule[1:1, :]; cols=:union),
        )
    end

    @testset "fixed pre-week cutoff and output filtering" begin
        forecast = forecast_regular_season(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            time_edges=[0, Inf],
        )
        @test nrow(forecast) == 2
        @test all(forecast.game_type .== "REG")
        @test all(forecast.week .>= 2)
        @test forecast.game_id == [
            "2023_02_HOME_AWAY",
            "2023_03_AWAY_HOME",
        ]
        @test forecast.game_completed == [true, false]
        @test forecast.result[1] == -3
        @test all(0 .<= forecast.home_win_probability .<= 1)
        @test all(0 .<= forecast.away_win_probability .<= 1)
        @test all(
            forecast.home_win_probability .+
            forecast.away_win_probability .≈ 1.0,
        )
        @test all(isfinite, forecast.expected_spread)
        @test all(isfinite, forecast.predictive_spread_variance)

        unplayed = forecast_regular_season(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            include_completed=false,
            time_edges=[0, Inf],
        )
        @test nrow(unplayed) == 1
        @test only(unplayed.game_id) == "2023_03_AWAY_HOME"
        @test only(unplayed.game_completed) == false
    end

    @testset "reusable context and split outputs" begin
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            time_edges=[0, Inf],
        )
        probabilities = forecast_win_probabilities(context)
        spreads = forecast_spreads(context)
        full = forecast_regular_season(context)

        @test probabilities.home_win_probability ≈
            full.home_win_probability
        @test probabilities.away_win_probability ≈
            full.away_win_probability
        @test !(:expected_spread in propertynames(probabilities))
        @test !(:predictive_spread_variance in propertynames(probabilities))
        @test !(:home_win_probability in propertynames(spreads))
        @test !(:away_win_probability in propertynames(spreads))
        @test spreads.expected_spread ≈ full.expected_spread
        @test spreads.predictive_spread_variance ≈
            full.predictive_spread_variance

        first_game = first(eachrow(context.games))
        direct_probability = expected_game_win_probability(
            context.model,
            context.marks,
            first_game.home_team,
            first_game.away_team;
            horizon=60.0,
        )
        metrics = expected_game_metrics(
            context.model,
            context.marks,
            first_game.home_team,
            first_game.away_team;
            horizon=60.0,
        )
        @test direct_probability ≈ metrics.expected_win_probability

        reused_prior_context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            prior=context.model.prior,
            time_edges=[0, Inf],
        )
        reused_prior = forecast_win_probabilities(reused_prior_context)
        @test reused_prior.home_win_probability ≈
            probabilities.home_win_probability
        @test reused_prior.away_win_probability ≈
            probabilities.away_win_probability
    end

    @testset "schedule-only historical results" begin
        results = regular_season_results(
            2023;
            schedule=schedule,
            from_week=2,
            through_week=3,
        )
        @test results.game_id == [
            "2023_02_HOME_AWAY",
            "2023_03_AWAY_HOME",
        ]
        @test !(:home_win_probability in propertynames(results))
        @test results.game_completed == [true, false]

        completed = regular_season_results(
            2023;
            schedule=schedule,
            from_week=2,
            through_week=3,
            include_unplayed=false,
        )
        @test only(completed.game_id) == "2023_02_HOME_AWAY"
    end

    @testset "future target-season drives do not leak" begin
        without_future = current[1:2, :]
        with_future = forecast_regular_season(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            time_edges=[0, Inf],
        )
        without_future_forecast = forecast_regular_season(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=without_future,
            time_edges=[0, Inf],
        )
        @test with_future.home_win_probability ≈
            without_future_forecast.home_win_probability
        @test with_future.expected_spread ≈
            without_future_forecast.expected_spread
    end

    @testset "week one uses historical data only" begin
        empty_current = current[1:0, :]
        with_current = forecast_regular_season(
            2023;
            as_of_week=1,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            time_edges=[0, Inf],
        )
        without_current = forecast_regular_season(
            2023;
            as_of_week=1,
            schedule=schedule,
            historical_drives=historical,
            current_drives=empty_current,
            time_edges=[0, Inf],
        )
        @test with_current.home_win_probability ≈
            without_current.home_win_probability
        @test with_current.away_win_probability ≈
            without_current.away_win_probability
    end

    @testset "input validation" begin
        @test_throws ArgumentError forecast_regular_season(
            2023;
            as_of_week=0,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        @test_throws ArgumentError forecast_regular_season(
            2024;
            as_of_week=1,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
    end
end
