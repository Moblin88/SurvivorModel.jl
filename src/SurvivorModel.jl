module SurvivorModel

using DataFrames
using Dates
using Distributions
using NFLData
using Optim
using Statistics

export load_drive_pbp, summarize_drives
export DEFAULT_TIME_EDGES, GAME_CLOCK_SECONDS
export GammaParams, HazardPrior, HazardModel, ScoreMarks, DriveMoments
export build_exposure_data, fit_empirical_bayes_prior, fit_hazard_model
export update_hazard_model!, hazard_posterior, hazard_rate, home_multiplier
export recency_weights
export fit_score_marks, drive_moments
export game_spread_distribution

include("drives.jl")
include("model.jl")

end
