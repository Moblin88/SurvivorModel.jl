using SurvivorModel
using DataFrames
using Distributions
using Statistics
using Test

@testset "live drive model sanity" begin
    drives = load_drive_pbp(2023:2025)

    model = fit_hazard_model(drives)
    marks = fit_score_marks(drives)

    @testset "hazard rates are positive and finite" begin
        teams = unique(drives.posteam)
        for team in first(teams, 5)
            for time_bin in 1:(length(model.time_edges) - 1)
                td_rate = hazard_rate(model, :td, team, time_bin)
                defensive_rate = hazard_rate(model, :defensive, team, time_bin)
                @test td_rate > 0 && isfinite(td_rate)
                @test defensive_rate > 0 && isfinite(defensive_rate)
            end
        end
    end

    @testset "score marks are plausible" begin
        @test 5 < marks.mean_td < 8
        @test marks.var_td > 0
        @test isfinite(marks.mean_defensive)
        @test marks.var_defensive >= 0
    end

    @testset "per-drive moments are plausible" begin
        teams = unique(drives.posteam)
        home, away = teams[1], teams[2]
        moments = drive_moments(model, marks, home, away; posteam_home=true)
        @test 0 < moments.p_td < 1
        @test 0 < moments.p_defensive < 1
        @test moments.p_td + moments.p_defensive ≈ 1.0
        @test 60 < moments.mean_T < 400
        @test moments.var_T > 0
        @test moments.var_S >= 0
    end

    @testset "game spread distribution is in a plausible NFL range" begin
        teams = unique(drives.posteam)
        home, away = teams[1], teams[2]
        home_moments = drive_moments(
            model,
            marks,
            home,
            away;
            posteam_home=true,
        )
        away_moments = drive_moments(
            model,
            marks,
            away,
            home;
            posteam_home=false,
        )
        dist = game_spread_distribution(home_moments, away_moments)

        @test dist isa Normal
        @test -30 < mean(dist) < 30
        @test 5 < std(dist) < 25
    end
end
