using SurvivorModel
using DataFrames
using Dates
using Test

@testset "live drive data sanity" begin
    drives = load_drive_pbp(seasons=[2023])

    @test nrow(drives) > 5_000
    @test all(name -> !any(ismissing, drives[!, name]), names(drives))
    @test all(!isempty, drives.game_id)
    @test all(!isempty, drives.posteam)
    @test all(!isempty, drives.defteam)
    @test all(!isempty, drives.drive_result)
    @test all(drives.posteam .!= drives.defteam)
    @test all(drives.posteam_home .!= drives.defteam_home)

    possession_seconds = Dates.value.(drives.time_of_possession)

    @test all(0 .<= possession_seconds .<= 900)
    @test all(0 .<= drives.yardline_100 .<= 100)
    @test all(-100 .<= drives.yards_gained .<= 100)
    @test all(-10 .<= drives.home_spread_change .<= 10)

    @test count(drives.yardline_100 .>= 20) / nrow(drives) > 0.85
    @test count(abs.(drives.yards_gained) .<= 60) / nrow(drives) > 0.95
    @test count(possession_seconds .<= 420) / nrow(drives) > 0.99
end
