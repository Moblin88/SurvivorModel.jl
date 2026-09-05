using SurvivorModel
using DataFrames
using Dates
using Test

@testset "live drive data sanity" begin
    drives = load_drive_pbp(2023)

    possession_seconds = Dates.value.(drives.time_of_possession)
    complete_columns = setdiff(names(drives), ["drive_start_yards_to_goal"])

    @testset "completeness and consistency" begin
        @test nrow(drives) > 5_000
        @test all(name -> !any(ismissing, drives[!, name]), complete_columns)
        @test count(ismissing, drives.drive_start_yards_to_goal) / nrow(drives) < 0.01
        @test all(!isempty, drives.game_id)
        @test all(!isempty, drives.posteam)
        @test all(!isempty, drives.defteam)
        @test all(!isempty, drives.drive_result)
        @test all(>(0), drives.fixed_drive)
        @test all(drives.posteam .!= drives.defteam)
        @test all(drives.posteam_home .!= drives.defteam_home)
    end

    @testset "physically plausible values" begin
        @test all(x -> ismissing(x) || isfinite(x), drives.drive_start_yards_to_goal)
        @test all(isfinite, drives.yards_gained)
        @test all(isfinite, drives.home_spread_change)
        @test all(0 .<= possession_seconds .<= 900)
        @test all(x -> ismissing(x) || 0 <= x <= 100, drives.drive_start_yards_to_goal)
        @test all(-125 .<= drives.yards_gained .<= 125)
        @test all(-10 .<= drives.home_spread_change .<= 10)
    end

    @testset "score changes agree with drive results" begin
        scoring_magnitudes = Dict(
            "Field goal" => [3],
            "Safety" => [2],
            "Touchdown" => [4, 6, 7, 8],
            "Opp touchdown" => [4, 6, 7, 8],
        )
        non_scoring_results = [
            "End of half",
            "Missed field goal",
            "Punt",
            "Turnover",
            "Turnover on downs",
        ]

        for (result, allowed_magnitudes) in scoring_magnitudes
            changes = drives.home_spread_change[drives.drive_result .== result]
            @test !isempty(changes)
            @test all(change -> abs(change) in allowed_magnitudes, changes)
        end

        for result in non_scoring_results
            changes = drives.home_spread_change[drives.drive_result .== result]
            @test !isempty(changes)
            @test all(iszero, changes)
        end
    end

    @testset "representative NFL drive distributions" begin
        valid_start_yards = collect(skipmissing(drives.drive_start_yards_to_goal))
        @test count(valid_start_yards .>= 35) / length(valid_start_yards) > 0.70
        @test count(abs.(drives.yards_gained) .<= 85) / nrow(drives) > 0.98
        @test count(possession_seconds .<= 420) / nrow(drives) > 0.95
    end
end
