module SurvivorModel

using DataFrames
using Dates
using Distributions
using ForwardDiff
using HiGHS
using NFLData
using Optim
using SpecialFunctions
using Statistics
using JuMP

export load_drive_pbp, summarize_drives
export DEFAULT_TIME_EDGES, GAME_CLOCK_SECONDS
export DEFAULT_RECENCY_HALF_LIFE, MAX_HISTORICAL_SEASONS
export GammaParams, HazardPrior, HazardModel, ScoreMarks, DriveMoments
export HazardTheta, ExpectedGameMetrics, ExpectedGameSpreadMetrics
export build_exposure_data, fit_empirical_bayes_prior, fit_hazard_model
export update_hazard_model!, hazard_posterior, hazard_rate, home_multiplier
export recency_weights
export fit_score_marks, drive_moments, hazard_theta
export game_spread_distribution
export expected_game_win_probability, expected_game_spread_metrics
export expected_game_metrics
export load_schedule, regular_season_results
export RegularSeasonForecastContext, fit_regular_season_forecast
export forecast_win_probabilities, forecast_spreads, forecast_regular_season
export DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY
export SurvivorPoolState, SurvivorPoolPlan
export build_survivor_candidates, survivor_reach_discounts
export optimize_survivor_pool
export CalibrationReport, brier_score, log_loss, reliability_bins
export score_difference_metrics, spread_reliability_bins
export spread_interval_coverage
export evaluate_calibration

include("drives.jl")
include("model.jl")
include("schedule.jl")
include("survivor.jl")
include("calibration.jl")

end
