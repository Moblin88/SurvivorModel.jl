using SurvivorModel
using DataFrames
using Dates
using Statistics
using Test

function _calibration_fixture()
    schedule = DataFrame(
        game_id=[
            "2022_01_AWAY_HOME",
            "2023_01_AWAY_HOME",
            "2023_02_HOME_AWAY",
            "2023_03_AWAY_HOME",
        ],
        season=[2022, 2023, 2023, 2023],
        game_type=fill("REG", 4),
        week=[1, 1, 2, 3],
        gameday=[
            Date(2022, 9, 11),
            Date(2023, 9, 10),
            Date(2023, 9, 17),
            Date(2023, 9, 24),
        ],
        away_team=["AWAY", "AWAY", "HOME", "AWAY"],
        home_team=["HOME", "HOME", "AWAY", "HOME"],
        result=Union{Missing,Int}[7, 7, -3, missing],
    )

    historical = DataFrame(
        game_id=["2022_01_AWAY_HOME", "2022_01_AWAY_HOME"],
        fixed_drive=[1, 2],
        posteam=["HOME", "AWAY"],
        defteam=["AWAY", "HOME"],
        posteam_home=[true, false],
        defteam_home=[false, true],
        drive_result=["Touchdown", "Punt"],
        time_of_possession=Second.([60, 60]),
        home_spread_change=[7.0, 0.0],
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
        drive_result=["Touchdown", "Punt", "Field goal", "Punt"],
        time_of_possession=Second.([60, 60, 60, 60]),
        home_spread_change=[7.0, 0.0, -3.0, 0.0],
    )

    return schedule, historical, current
end

function _recent_season_fixture()
    schedule = DataFrame(
        game_id=[
            "2021_01_AWAY_HOME",
            "2022_01_AWAY_HOME",
            "2022_02_HOME_AWAY",
        ],
        season=[2021, 2022, 2022],
        game_type=fill("REG", 3),
        week=[1, 1, 2],
        gameday=[
            Date(2021, 9, 12),
            Date(2022, 9, 11),
            Date(2022, 9, 18),
        ],
        away_team=["AWAY", "AWAY", "HOME"],
        home_team=["HOME", "HOME", "AWAY"],
        result=[3, 7, -3],
    )

    drives = DataFrame(
        game_id=[
            "2021_01_AWAY_HOME",
            "2021_01_AWAY_HOME",
        ],
        fixed_drive=[1, 2],
        posteam=["HOME", "AWAY"],
        defteam=["AWAY", "HOME"],
        posteam_home=[true, false],
        defteam_home=[false, true],
        drive_result=["Touchdown", "Punt"],
        time_of_possession=Second.([60, 60]),
        home_spread_change=[3.0, 0.0],
    )

    return schedule, drives
end

@testset "calibration metrics" begin
    @testset "scores and ties" begin
        predicted = [0.1, 0.6, 0.9, 0.5]
        actual = [0.0, 1.0, 1.0, 0.5]
        @test brier_score(predicted, actual) ≈
            mean((predicted .- actual) .^ 2)
        @test log_loss([0.5], [0.5]) ≈ log(2.0)
        @test_throws ArgumentError brier_score([0.5], [0.0, 1.0])
        @test_throws ArgumentError brier_score([1.1], [1.0])
    end

    @testset "reliability bins" begin
        reliability = reliability_bins(
            [0.05, 0.15, 0.95, 1.0],
            [0.0, 1.0, 0.5, 1.0];
            probability_bins=[0.0, 0.5, 1.0],
        )
        @test reliability.n_games == [2, 2]
        @test sum(reliability.n_games) == 4
        @test reliability.mean_predicted[1] ≈ 0.1
        @test reliability.observed_rate[1] ≈ 0.5
        @test reliability.mean_predicted[2] ≈ 0.975
        @test reliability.observed_rate[2] ≈ 0.75
    end

    @testset "historical snapshot evaluation" begin
        schedule, historical, current = _calibration_fixture()
        report = evaluate_calibration(
            2023;
            cutoff_weeks=[1, 2],
            schedule=schedule,
            drives=vcat(historical, current; cols=:union),
            time_edges=[0, Inf],
        )

        @test report.summary.season == [2023, 2023]
        @test report.summary.as_of_week == [1, 2]
        @test report.summary.forecasted_games == [3, 2]
        @test report.summary.scored_games == [2, 1]
        @test nrow(report.games) == 3
        @test all(
            0.0 .<= report.games.actual_home_outcome .<= 1.0
        )
        @test all(0.0 .<= report.summary.brier_score .<= 1.0)
        @test all(report.summary.log_loss .>= 0.0)
        @test all(isfinite, report.summary.mean_predicted)
        @test all(isfinite, report.summary.observed_rate)
        @test sum(report.reliability.n_games) == nrow(report.games)

        without_future = evaluate_calibration(
            2023;
            cutoff_weeks=[1, 2],
            schedule=schedule,
            drives=vcat(historical, current[1:2, :]; cols=:union),
            time_edges=[0, Inf],
        )
        @test report.games.home_win_probability ≈
            without_future.games.home_win_probability

        no_scored_games = evaluate_calibration(
            2023;
            cutoff_weeks=[18],
            schedule=schedule,
            drives=vcat(historical, current; cols=:union),
            time_edges=[0, Inf],
        )
        @test no_scored_games.summary.scored_games == [0]
        @test all(iszero, no_scored_games.reliability.n_games)
    end

    @testset "recent completed season selection" begin
        schedule, drives = _recent_season_fixture()
        report = evaluate_calibration(
            ;
            cutoff_weeks=[1],
            recent_seasons=1,
            schedule=schedule,
            drives=drives,
            time_edges=[0, Inf],
        )

        @test report.summary.season == [2022]
        @test report.summary.scored_games == [2]
        @test_throws ArgumentError evaluate_calibration(
            ;
            cutoff_weeks=[1],
            recent_seasons=2,
            schedule=schedule,
            drives=drives,
            time_edges=[0, Inf],
        )
    end
end
