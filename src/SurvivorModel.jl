module SurvivorModel

using DataFrames
using Dates
using Distributions
using ForwardDiff
using NFLData
using Optim
using SpecialFunctions
using Statistics

export load_drive_pbp, summarize_drives
export DEFAULT_TIME_EDGES, GAME_CLOCK_SECONDS
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

include("drives.jl")
include("model.jl")
include("schedule.jl")

end
