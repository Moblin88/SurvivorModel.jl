module SurvivorModel

using DataFrames
using Dates
using Distributions
using ForwardDiff
using HiGHS
using LinearAlgebra
using NFLData
using SpecialFunctions
using Statistics
using JuMP
using Optim
using Printf
using Random
using Scratch
using Serialization

export load_drive_pbp, summarize_drives
export DEFAULT_TIME_EDGES, GAME_CLOCK_SECONDS
export MAX_HISTORICAL_SEASONS
export DEFAULT_HISTORICAL_SEASONS, DEFAULT_PRIOR_FIT_METHOD
export GammaParams, GammaMixture, HazardPrior, HazardModel, ScoreMarks, DriveMoments
export LikelihoodFitDiagnostics
export PriorFitMethod, EMECMEFit, EMLBFGSFit, DirectLBFGSFit, DirectBFGSFit
export MomentFit, MomentLBFGSFit, HybridFit, BlockNewtonFit, SchurNewtonFit
export prior_fit_method_name
export HazardTheta, ExpectedGameMetrics, ExpectedGameSpreadMetrics
export build_exposure_data, fit_empirical_bayes_prior, fit_hazard_model
export clear_historical_prior_cache!
export update_hazard_model!, hazard_posterior, hazard_rate, home_multiplier
export hazard_persistence
export likelihood_fit_diagnostics
export fit_score_marks, drive_moments, hazard_theta
export simulate_renewal_schedule
export game_spread_distribution
export expected_game_win_probability, expected_game_spread_metrics
export expected_game_metrics
export load_schedule, regular_season_results
export RegularSeasonForecastContext, fit_regular_season_forecast
export forecast_win_probabilities, forecast_spreads, forecast_regular_season
export DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY
export DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD
export SurvivorPoolState, SurvivorPoolPlan
export build_survivor_candidates, survivor_reach_discounts
export optimize_survivor_pool
export CalibrationReport, brier_score, log_loss, reliability_bins
export score_difference_metrics, spread_reliability_bins
export spread_interval_coverage
export evaluate_calibration
export synthetic_fit_benchmark_drives, synthetic_fit_recovery_drives
export fit_benchmark, fit_recovery_benchmark

include("drives.jl")
include("renewal_model.jl")
include("prior_fitting.jl")
include("historical_cache.jl")
include("game_forecast.jl")
include("renewal_simulation.jl")
include("survivor.jl")
include("calibration.jl")
include("fit_benchmark.jl")
include("cli.jl")

end
