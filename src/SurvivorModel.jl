module SurvivorModel

using DataFrames
using Dates
using Distributions
using ForwardDiff
using HiGHS
using LinearAlgebra
using Logging
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
export GammaParams, GammaMixture, HazardPrior, HazardModel
export LikelihoodFitDiagnostics
export PriorFitMethod, EMLBFGSFit
export build_exposure_data, fit_empirical_bayes_prior, fit_hazard_model
export update_hazard_model!, hazard_posterior, hazard_rate, home_multiplier
export hazard_persistence
export likelihood_fit_diagnostics
export load_schedule, regular_season_results
export RegularSeasonForecastContext, fit_regular_season_forecast
export forecast_win_probabilities
export expected_game_win_probability
export DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY
export DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD
export DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY
export DEFAULT_SURVIVOR_OBJECTIVE, DEFAULT_SURVIVOR_REACH_DISCOUNT_POLICY
export DEFAULT_SURVIVOR_MARKET_GUARD_WEEKS
export DEFAULT_SURVIVOR_MISSING_MARKET_POLICY
export SurvivorSelectionConfig
export SurvivorPoolState, SurvivorPoolPlan
export build_survivor_candidates, survivor_reach_discounts
export optimize_survivor_pool
export CalibrationReport, brier_score, log_loss, reliability_bins
export evaluate_calibration

include("drives.jl")
include("drive_cache.jl")
include("renewal_model.jl")
include("prior_fitting.jl")
include("historical_cache.jl")
include("game_forecast.jl")
include("survivor_uncertainty.jl")
include("renewal_simulation.jl")
include("survivor.jl")
include("calibration.jl")
include("fit_benchmark.jl")
include("cli.jl")

end
