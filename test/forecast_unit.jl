using SurvivorModel
using DataFrames
using Dates
using Test
import SurvivorModel: hazard_theta

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
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        forecast = forecast_win_probabilities(context)
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

        unplayed = forecast_win_probabilities(
            context;
            include_completed=false,
        )
        @test nrow(unplayed) == 1
        @test only(unplayed.game_id) == "2023_03_AWAY_HOME"
        @test only(unplayed.game_completed) == false
    end

    @testset "reusable context and probability forecasts" begin
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        probabilities = forecast_win_probabilities(context)

        first_game = first(eachrow(context.games))
        direct_probability = expected_game_win_probability(
            context.model,
            context.marks,
            first_game.home_team,
            first_game.away_team;
            horizon=60.0,
        )
        @test direct_probability ≈
            forecast_win_probabilities(context; horizon=60.0).home_win_probability[1]

        reused_prior_context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
            prior=context.model.prior,
        )
        reused_prior = forecast_win_probabilities(reused_prior_context)
        @test reused_prior.home_win_probability ≈
            probabilities.home_win_probability
        @test reused_prior.away_win_probability ≈
            probabilities.away_win_probability

        normalized_context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=load_schedule(schedule),
            historical_drives=historical,
            current_drives=current,
            prior=context.model.prior,
            _normalized_schedule=true,
        )
        normalized_probabilities = forecast_win_probabilities(normalized_context)
        @test normalized_probabilities.home_win_probability ≈
            probabilities.home_win_probability
        @test normalized_probabilities.away_win_probability ≈
            probabilities.away_win_probability

        indexed_context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=load_schedule(schedule),
            historical_drives=SurvivorModel._regular_season_drives(
                historical,
                schedule,
            ),
            current_drives=SurvivorModel._regular_season_drives(
                current,
                schedule,
            ),
            prior=context.model.prior,
            _normalized_schedule=true,
            _schedule_indexed_drives=true,
        )
        indexed_probabilities = forecast_win_probabilities(indexed_context)
        @test indexed_probabilities.home_win_probability ≈
            probabilities.home_win_probability
        @test indexed_probabilities.away_win_probability ≈
            probabilities.away_win_probability
    end

    @testset "forecast posterior log-moment cache" begin
        context = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        first_game = first(eachrow(context.games))
        cache = SurvivorModel._HazardLogMomentCache(context.model)

        uncached_theta = hazard_theta(
            context.model,
            first_game.home_team,
            first_game.away_team,
        )
        cached_theta = SurvivorModel._hazard_theta_with_cache(
            context.model,
            first_game.home_team,
            first_game.away_team,
            cache,
        )
        @test cached_theta.log_mean == uncached_theta.log_mean
        @test cached_theta.covariance == uncached_theta.covariance
        @test cache.misses == 4
        @test cache.hits == 0

        repeated_theta = SurvivorModel._hazard_theta_with_cache(
            context.model,
            first_game.home_team,
            first_game.away_team,
            cache,
        )
        @test repeated_theta.log_mean == cached_theta.log_mean
        @test repeated_theta.covariance == cached_theta.covariance
        @test cache.misses == 4
        @test cache.hits == 4
        @test length(cache.posteriors) == 4
        @test length(cache.log_moments) == 4

        forecast = forecast_win_probabilities(context)
        for (index, row) in enumerate(eachrow(context.games))
            direct_probability = expected_game_win_probability(
                context.model,
                context.marks,
                row.home_team,
                row.away_team,
            )
            @test forecast.home_win_probability[index] ≈
                direct_probability
        end
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
        with_future = forecast_win_probabilities(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        without_future_forecast = forecast_win_probabilities(
            2023;
            as_of_week=2,
            schedule=schedule,
            historical_drives=historical,
            current_drives=without_future,
        )
        @test with_future.home_win_probability ≈
            without_future_forecast.home_win_probability
    end

    @testset "week one uses historical data only" begin
        empty_current = current[1:0, :]
        with_current = forecast_win_probabilities(
            2023;
            as_of_week=1,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        without_current = forecast_win_probabilities(
            2023;
            as_of_week=1,
            schedule=schedule,
            historical_drives=historical,
            current_drives=empty_current,
        )
        @test with_current.home_win_probability ≈
            without_current.home_win_probability
        @test with_current.away_win_probability ≈
            without_current.away_win_probability
    end

    @testset "future opening week without current PBP" begin
        future_season = typemax(Int)
        loaded_historical, empty_current = SurvivorModel._load_forecast_drives(
            future_season,
            3,
            historical,
            nothing;
            allow_missing_current=true,
        )
        @test loaded_historical == historical
        @test nrow(empty_current) == 0
        @test propertynames(empty_current) == propertynames(historical)
        @test_throws ArgumentError SurvivorModel._load_forecast_drives(
            future_season,
            3,
            historical,
            nothing,
        )
    end

    @testset "input validation" begin
        @test_throws ArgumentError forecast_win_probabilities(
            2023;
            as_of_week=0,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
        @test_throws ArgumentError forecast_win_probabilities(
            2024;
            as_of_week=1,
            schedule=schedule,
            historical_drives=historical,
            current_drives=current,
        )
    end
end
