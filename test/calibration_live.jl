using SurvivorModel
using DataFrames
using Test

@testset "live calibration report" begin
    report = evaluate_calibration(
        ;
        cutoff_weeks=(1, 5, 10, 15),
        recent_seasons=3,
    )

    @test nrow(report.summary) == 12
    @test all(report.summary.scored_games .> 0)
    @test all(isfinite, skipmissing(report.summary.brier_score))
    @test all(isfinite, skipmissing(report.summary.log_loss))
    @test nrow(report.spread_summary) == 12
    @test all(report.spread_summary.scored_games .> 0)
    @test all(isfinite, skipmissing(report.spread_summary.mean_error))
    @test all(isfinite, skipmissing(report.spread_summary.rmse))
    @test all(0.0 .<= report.spread_coverage.coverage .<= 1.0)

    println("Calibration summary:")
    show(stdout, report.summary; allrows=true, allcols=true)
    println()
    println("Calibration reliability bins:")
    show(stdout, report.reliability; allrows=true, allcols=true)
    println()
    println("Score-difference calibration summary:")
    show(stdout, report.spread_summary; allrows=true, allcols=true)
    println()
    println("Score-difference reliability bins:")
    show(stdout, report.spread_reliability; allrows=true, allcols=true)
    println()
    println("Score-difference predictive-interval coverage:")
    show(stdout, report.spread_coverage; allrows=true, allcols=true)
    println()
end
