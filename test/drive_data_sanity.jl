using SurvivorModel
using DataFrames
using Dates
using Test

@testset "live drive data sanity" begin
    drives = load_drive_pbp(2023)

    possession_seconds = Dates.value.(drives.time_of_possession)

    @testset "completeness and consistency" begin
        @test nrow(drives) > 5_000
        @test all(name -> !any(ismissing, drives[!, name]), names(drives))
        @test all(!isempty, drives.game_id)
        @test all(!isempty, drives.posteam)
        @test all(!isempty, drives.defteam)
        @test all(!isempty, drives.drive_result)
        @test all(>(0), drives.fixed_drive)
        @test all(drives.posteam .!= drives.defteam)
        @test all(drives.posteam_home .!= drives.defteam_home)
    end

    @testset "physically plausible values" begin
        @test all(isfinite, drives.yardline_100)
        @test all(isfinite, drives.yards_gained)
        @test all(isfinite, drives.home_spread_change)
        @test all(0 .<= possession_seconds .<= 900)
        @test all(0 .<= drives.yardline_100 .<= 100)
        @test all(-125 .<= drives.yards_gained .<= 125)
        @test all(-10 .<= drives.home_spread_change .<= 10)
    end

    @testset "representative NFL drive distributions" begin
        @test count(drives.yardline_100 .>= 35) / nrow(drives) > 0.70
        @test count(abs.(drives.yards_gained) .<= 85) / nrow(drives) > 0.98
        @test count(possession_seconds .<= 420) / nrow(drives) > 0.99
    end
end
