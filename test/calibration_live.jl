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

    println("Calibration summary:")
    show(stdout, report.summary; allrows=true, allcols=true)
    println()
    println("Calibration reliability bins:")
    show(stdout, report.reliability; allrows=true, allcols=true)
    println()
end
