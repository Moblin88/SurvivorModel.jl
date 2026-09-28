"""
Cause-specific Weibull competing-risk drive model.

Conditional on the team rates, a drive is a race between an offensive
touchdown process and a defensive-event process. Each cause has a
team-shared Weibull shape, with one Gamma-distributed cumulative-hazard rate
per team. End-of-half drives are right-censored.
"""

const MAX_HISTORICAL_SEASONS = typemax(Int)
const DEFAULT_HISTORICAL_SEASONS = 5
const SECONDS_PER_MINUTE = 60.0
struct WeibullEmpiricalBayesFit end
const DEFAULT_PRIOR_FIT_METHOD = WeibullEmpiricalBayesFit()

"""
    GammaParams

Shape-rate parameters for a Gamma distribution.
"""
struct GammaParams
    shape::Float64
    rate::Float64

    function GammaParams(shape::Real, rate::Real)
        isfinite(shape) && shape > 0 ||
            throw(ArgumentError("Gamma shape must be finite and positive"))
        isfinite(rate) && rate > 0 ||
            throw(ArgumentError("Gamma rate must be finite and positive"))
        return new(Float64(shape), Float64(rate))
    end
end

"""
    GammaMixture

Finite mixture of Gamma distributions in shape-rate parameterization. The
component source seasons identify the last reset season represented by each
component.
"""
struct GammaMixture
    weights::Vector{Float64}
    components::Vector{GammaParams}
    source_seasons::Vector{Int}

    function GammaMixture(
        weights::AbstractVector{<:Real},
        components::AbstractVector{<:GammaParams};
        source_seasons=fill(0, length(components)),
    )
        length(weights) == length(components) ||
            throw(ArgumentError("mixture weights and components must have equal lengths"))
        isempty(components) &&
            throw(ArgumentError("Gamma mixtures must contain at least one component"))
        normalized_weights = Float64.(weights)
        all(isfinite, normalized_weights) &&
            all(>=(0.0), normalized_weights) ||
            throw(ArgumentError("mixture weights must be finite and nonnegative"))
        total_weight = sum(normalized_weights)
        isfinite(total_weight) && total_weight > 0.0 ||
            throw(ArgumentError("Gamma mixtures must have positive finite mass"))
        sources = Int.(collect(source_seasons))
        length(sources) == length(components) ||
            throw(ArgumentError("mixture source seasons must match components"))
        return new(
            normalized_weights ./ total_weight,
            GammaParams[components...],
            sources,
        )
    end
end

function _gamma_mixture_log_moments(mixture::GammaMixture)
    component_log_means = [
        SpecialFunctions.digamma(component.shape) - log(component.rate)
        for component in mixture.components
    ]
    mean_log = sum(
        weight * value
        for (weight, value) in zip(mixture.weights, component_log_means)
    )
    second_log = sum(
        weight * (
            SpecialFunctions.trigamma(component.shape) + value^2
        )
        for (weight, component, value) in zip(
            mixture.weights,
            mixture.components,
            component_log_means,
        )
    )
    return mean_log, second_log - mean_log^2
end

function _gamma_mixture_mean(mixture::GammaMixture)
    return sum(
        weight * component.shape / component.rate
        for (weight, component) in zip(mixture.weights, mixture.components)
    )
end

function _gamma_mixture_variance(mixture::GammaMixture)
    second_moment = sum(
        weight * component.shape * (component.shape + 1.0) /
            component.rate^2
        for (weight, component) in zip(mixture.weights, mixture.components)
    )
    return second_moment - _gamma_mixture_mean(mixture)^2
end

function _gamma_mixture_home_adjusted(
    mixture::GammaMixture,
    multiplier::Real,
)
    multiplier_value = Float64(multiplier)
    isfinite(multiplier_value) && multiplier_value > 0.0 ||
        throw(ArgumentError("home multiplier must be finite and positive"))
    return GammaMixture(
        mixture.weights,
        [
            GammaParams(component.shape, component.rate / multiplier_value)
            for component in mixture.components
        ];
        source_seasons=mixture.source_seasons,
    )
end

function _log_gamma_poisson_predictive(
    component::GammaParams,
    count::Real,
    exposure::Real,
)
    count_value = Float64(count)
    exposure_value = Float64(exposure)
    isfinite(count_value) && count_value >= 0.0 ||
        throw(ArgumentError("event counts must be finite and nonnegative"))
    isfinite(exposure_value) && exposure_value >= 0.0 ||
        throw(ArgumentError("exposures must be finite and nonnegative"))
    count_integer = round(Int, count_value)
    isapprox(count_value, count_integer; atol=1e-10) ||
        throw(ArgumentError("event counts must be integer-valued"))
    exposure_value == 0.0 &&
        return count_integer == 0 ? 0.0 : -Inf
    probability = component.rate / (component.rate + exposure_value)
    return logpdf(
        NegativeBinomial(component.shape, probability),
        count_integer,
    )
end

function _update_gamma_mixture(
    mixture::GammaMixture,
    count::Real,
    exposure::Real,
)
    count_value = Float64(count)
    exposure_value = Float64(exposure)
    updated_components = [
        GammaParams(
            component.shape + count_value,
            component.rate + exposure_value,
        )
        for component in mixture.components
    ]
    log_weights = [
        log(weight) + _log_gamma_poisson_predictive(
            component,
            count_value,
            exposure_value,
        )
        for (weight, component) in zip(mixture.weights, mixture.components)
    ]
    maximum_log_weight = maximum(log_weights)
    isfinite(maximum_log_weight) ||
        throw(ArgumentError("Gamma mixture update has no finite component likelihood"))
    updated_weights = exp.(log_weights .- maximum_log_weight)
    return GammaMixture(
        updated_weights,
        updated_components;
        source_seasons=mixture.source_seasons,
    )
end

"""
    LikelihoodFitDiagnostics

Diagnostics from a historical event-process likelihood fit. The likelihood is
reported without data-only point-process constants.
"""
struct LikelihoodFitDiagnostics
    log_likelihood::Float64
    converged::Bool
    iterations::Int
    function_evaluations::Int
    status::Symbol
    boundary_parameters::Vector{Symbol}
end

function _classify_event(drive_result::AbstractString)
    drive_result == "Touchdown" && return :td
    drive_result == "End of half" && return :censored
    return :defensive
end

function _season_from_game_id(game_id)
    match_result = match(r"^(\d{4})", string(game_id))
    return isnothing(match_result) ? missing : parse(Int, match_result.captures[1])
end

function _season_value(value)
    ismissing(value) && return missing
    value isa Integer && return Int(value)
    return parse(Int, string(value))
end

function _drive_season(drives::AbstractDataFrame, index::Integer)
    if :season in propertynames(drives)
        return _season_value(drives.season[index])
    end
    :game_id in propertynames(drives) ||
        throw(ArgumentError("drives must include either season or game_id"))
    return _season_from_game_id(drives.game_id[index])
end

function _latest_observed_season(drives::AbstractDataFrame)
    seasons = Int[]
    for index in 1:nrow(drives)
        season = _drive_season(drives, index)
        ismissing(season) || push!(seasons, Int(season))
    end
    return isempty(seasons) ? nothing : maximum(seasons)
end

function _drive_duration_minutes(duration)
    seconds = Float64(Dates.value(Second(duration)))
    isfinite(seconds) && seconds >= 0.0 ||
        throw(ArgumentError("drive durations must be finite and nonnegative"))
    return seconds / SECONDS_PER_MINUTE
end

"""
    build_drive_data(drives) -> DataFrame

Build one continuous-duration record per complete drive. End-of-half drives
remain in the data as right-censored records with `event == :censored`.
Durations are represented in minutes; zero-second observed event times are
floored to one second, the source data's time resolution.
"""
function build_drive_data(drives::AbstractDataFrame)
    required = (
        :drive_result,
        :time_of_possession,
        :posteam,
        :defteam,
        :posteam_home,
        :defteam_home,
    )
    missing_columns = filter(column -> !(column in propertynames(drives)), required)
    isempty(missing_columns) ||
        throw(ArgumentError("drives are missing required columns: $missing_columns"))
    complete = subset(
        drives,
        :drive_result => ByRow(!ismissing),
        :time_of_possession => ByRow(!ismissing),
        :posteam => ByRow(!ismissing),
        :defteam => ByRow(!ismissing),
        :posteam_home => ByRow(!ismissing),
        :defteam_home => ByRow(!ismissing);
        skipmissing=true,
    )
    seasons = Union{Missing,Int}[
        _drive_season(complete, index)
        for index in 1:nrow(complete)
    ]
    events = Symbol[
        _classify_event(String(complete.drive_result[index]))
        for index in 1:nrow(complete)
    ]
    durations = Float64[
        _drive_duration_minutes(complete.time_of_possession[index])
        for index in 1:nrow(complete)
    ]
    for index in eachindex(durations)
        events[index] === :censored && continue
        durations[index] == 0.0 && (durations[index] = 1.0 / SECONDS_PER_MINUTE)
    end
    return DataFrame(
        season=seasons,
        posteam=String.(complete.posteam),
        defteam=String.(complete.defteam),
        posteam_home=Bool.(complete.posteam_home),
        defteam_home=Bool.(complete.defteam_home),
        duration=durations,
        event=events,
    )
end

mutable struct OutcomeStats
    counts::Dict{String,Int}
    home_exposure::Dict{String,Float64}
    away_exposure::Dict{String,Float64}
end

OutcomeStats() = OutcomeStats(
    Dict{String,Int}(),
    Dict{String,Float64}(),
    Dict{String,Float64}(),
)

mutable struct HazardSufficientStats
    td::OutcomeStats
    defensive::OutcomeStats
end

HazardSufficientStats() = HazardSufficientStats(OutcomeStats(), OutcomeStats())

function _outcome_stats(stats::HazardSufficientStats, kind::Symbol)
    kind === :td && return stats.td
    kind === :defensive && return stats.defensive
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end

function _add_stat!(
    values::Dict{String,Float64},
    key::String,
    amount::Real,
)
    values[key] = get(values, key, 0.0) + Float64(amount)
    return nothing
end

function _record_cause_exposure!(
    stats::OutcomeStats,
    team::String,
    is_home::Bool,
    exposure::Float64,
    is_event::Bool,
)
    _add_stat!(is_home ? stats.home_exposure : stats.away_exposure, team, exposure)
    if is_event
        stats.counts[team] = get(stats.counts, team, 0) + 1
    end
    return nothing
end

function _add_drive_stats!(
    stats::HazardSufficientStats,
    data::AbstractDataFrame,
    td_shape::Real,
    defensive_shape::Real,
)
    for row in eachrow(data)
        td_exposure = row.duration^td_shape
        defensive_exposure = row.duration^defensive_shape
        _record_cause_exposure!(
            stats.td,
            row.posteam,
            row.posteam_home,
            td_exposure,
            row.event === :td,
        )
        _record_cause_exposure!(
            stats.defensive,
            row.defteam,
            row.defteam_home,
            defensive_exposure,
            row.event === :defensive,
        )
    end
    return stats
end

"""
    HazardPrior

Empirical-Bayes Weibull shapes, Gamma hyperparameters, home multipliers,
season-reset probabilities, and team-specific finite-mixture priors for
touchdown and defensive-event cumulative-hazard rates.
"""
struct HazardPrior
    td_shape::Float64
    defensive_shape::Float64
    td_hyperparameters::GammaParams
    defensive_hyperparameters::GammaParams
    td_team_mixtures::Dict{String,GammaMixture}
    defensive_team_mixtures::Dict{String,GammaMixture}
    td_home_multiplier::Float64
    defensive_home_multiplier::Float64
    td_persistence::Float64
    defensive_persistence::Float64
    historical_seasons::Vector{Int}
    td_fit_diagnostics::Union{Nothing,LikelihoodFitDiagnostics}
    defensive_fit_diagnostics::Union{Nothing,LikelihoodFitDiagnostics}
end

function _default_hazard_prior()
    default_rate_prior = GammaParams(1.0, 10.0)
    return HazardPrior(
        1.0,
        1.0,
        default_rate_prior,
        default_rate_prior,
        Dict{String,GammaMixture}(),
        Dict{String,GammaMixture}(),
        1.0,
        1.0,
        0.5,
        0.5,
        Int[],
        nothing,
        nothing,
    )
end

function _validate_cause(kind::Symbol)
    kind === :td && return nothing
    kind === :defensive && return nothing
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end

"""
    weibull_shape(prior, kind) -> Float64

Return the fitted Weibull shape for `:td` or `:defensive`.
"""
function weibull_shape(prior::HazardPrior, kind::Symbol)
    _validate_cause(kind)
    return kind === :td ? prior.td_shape : prior.defensive_shape
end

function home_multiplier(prior::HazardPrior, kind::Symbol)
    _validate_cause(kind)
    return kind === :td ?
        prior.td_home_multiplier : prior.defensive_home_multiplier
end

function hazard_persistence(prior::HazardPrior, kind::Symbol)
    _validate_cause(kind)
    return kind === :td ? prior.td_persistence : prior.defensive_persistence
end

function likelihood_fit_diagnostics(prior::HazardPrior, kind::Symbol)
    _validate_cause(kind)
    return kind === :td ?
        prior.td_fit_diagnostics : prior.defensive_fit_diagnostics
end

"""
    HazardModel

Mutable current-season Weibull rate state. The prior stores historical
empirical-Bayes mixtures; sufficient statistics contain only current-season
event counts and transformed exposures.
"""
mutable struct HazardModel
    prior::HazardPrior
    stats::HazardSufficientStats
end

home_multiplier(model::HazardModel, kind::Symbol) =
    home_multiplier(model.prior, kind)

function _prior_mixture(prior::HazardPrior, kind::Symbol, team::String)
    _validate_cause(kind)
    mixtures = kind === :td ? prior.td_team_mixtures : prior.defensive_team_mixtures
    haskey(mixtures, team) && return mixtures[team]
    hyperparameter = kind === :td ?
        prior.td_hyperparameters : prior.defensive_hyperparameters
    return GammaMixture([1.0], [hyperparameter]; source_seasons=[0])
end

"""
    update_hazard_model!(model, drives) -> HazardModel

Add current-season drive durations, event counts, and transformed exposures.
"""
function update_hazard_model!(model::HazardModel, drives::AbstractDataFrame)
    data = build_drive_data(drives)
    _add_drive_stats!(
        model.stats,
        data,
        model.prior.td_shape,
        model.prior.defensive_shape,
    )
    return model
end

"""
    hazard_posterior(model, kind, team; home=false) -> GammaMixture

Return the finite Gamma-mixture posterior for a team's Weibull
cumulative-hazard rate. With `home=true`, return the mixture for the
home-adjusted rate.
"""
function hazard_posterior(
    model::HazardModel,
    kind::Symbol,
    team;
    home::Bool=false,
)
    _validate_cause(kind)
    team_name = string(team)
    prior = _prior_mixture(model.prior, kind, team_name)
    stats = _outcome_stats(model.stats, kind)
    multiplier = home_multiplier(model.prior, kind)
    posterior = _update_gamma_mixture(
        prior,
        get(stats.counts, team_name, 0.0),
        get(stats.away_exposure, team_name, 0.0) +
            multiplier * get(stats.home_exposure, team_name, 0.0),
    )
    return home ? _gamma_mixture_home_adjusted(posterior, multiplier) : posterior
end

"""
    hazard_rate(model, kind, team, elapsed_minutes; home=false) -> Float64

Return the posterior-mean instantaneous Weibull hazard at `elapsed_minutes`.
The elapsed time is measured in minutes.
"""
function hazard_rate(
    model::HazardModel,
    kind::Symbol,
    team,
    elapsed_minutes::Real;
    home::Bool=false,
)
    time_value = Float64(elapsed_minutes)
    isfinite(time_value) && time_value >= 0.0 ||
        throw(ArgumentError("elapsed_minutes must be finite and nonnegative"))
    shape = weibull_shape(model.prior, kind)
    rate_mean = _gamma_mixture_mean(
        hazard_posterior(model, kind, team; home=home),
    )
    if time_value == 0.0
        shape < 1.0 && return Inf
        shape > 1.0 && return 0.0
        return rate_mean
    end
    return rate_mean * shape * time_value^(shape - 1.0)
end

mutable struct _HazardLogMomentCache
    model::HazardModel
    posteriors::Dict{Tuple{Symbol,String},GammaMixture}
    log_moments::Dict{Tuple{Symbol,String,Bool},Tuple{Float64,Float64}}
    hits::Int
    misses::Int
end

_HazardLogMomentCache(model::HazardModel) = _HazardLogMomentCache(
    model,
    Dict{Tuple{Symbol,String},GammaMixture}(),
    Dict{Tuple{Symbol,String,Bool},Tuple{Float64,Float64}}(),
    0,
    0,
)

function _cached_hazard_posterior(
    cache::_HazardLogMomentCache,
    model::HazardModel,
    kind::Symbol,
    team,
)
    cache.model === model ||
        throw(ArgumentError("hazard log-moment cache belongs to a different model"))
    team_name = string(team)
    key = (kind, team_name)
    haskey(cache.posteriors, key) && return cache.posteriors[key]
    posterior = hazard_posterior(model, kind, team_name)
    cache.posteriors[key] = posterior
    return posterior
end

function _hazard_log_moments(
    model::HazardModel,
    kind::Symbol,
    team;
    home::Bool=false,
    cache::Union{Nothing,_HazardLogMomentCache}=nothing,
)
    isnothing(cache) &&
        return _gamma_mixture_log_moments(
            hazard_posterior(model, kind, team; home=home),
        )
    cache.model === model ||
        throw(ArgumentError("hazard log-moment cache belongs to a different model"))
    team_name = string(team)
    key = (kind, team_name, home)
    if haskey(cache.log_moments, key)
        cache.hits += 1
        return cache.log_moments[key]
    end
    cache.misses += 1
    posterior = _cached_hazard_posterior(cache, model, kind, team_name)
    adjusted_posterior = home ?
        _gamma_mixture_home_adjusted(
            posterior,
            home_multiplier(model.prior, kind),
        ) : posterior
    moments = _gamma_mixture_log_moments(adjusted_posterior)
    cache.log_moments[key] = moments
    return moments
end

struct HazardTheta
    log_mean::Vector{Float64}
    covariance::Matrix{Float64}
    labels::Vector{Symbol}

    function HazardTheta(
        log_mean::AbstractVector{<:Real},
        covariance::AbstractMatrix{<:Real},
        labels::AbstractVector{<:Symbol},
    )
        n = length(log_mean)
        size(covariance) == (n, n) ||
            throw(ArgumentError("theta covariance must be square and match theta length"))
        length(labels) == n ||
            throw(ArgumentError("theta labels must match theta length"))
        all(isfinite, log_mean) ||
            throw(ArgumentError("theta log means must be finite"))
        all(isfinite, covariance) ||
            throw(ArgumentError("theta covariance must be finite"))
        return new(Float64.(log_mean), Float64.(covariance), Symbol.(labels))
    end
end

function _matchup_theta_posteriors(model::HazardModel, home_team, away_team)
    requests = (
        (:td, home_team, true, :home_td),
        (:defensive, away_team, false, :away_defensive),
        (:td, away_team, false, :away_td),
        (:defensive, home_team, true, :home_defensive),
    )
    posteriors = GammaMixture[]
    posterior_keys = Tuple{Symbol,String}[]
    labels = Symbol[]
    for (kind, team, home, label) in requests
        push!(posteriors, hazard_posterior(model, kind, team; home=home))
        push!(posterior_keys, (kind, string(team)))
        push!(labels, label)
    end
    return posteriors, posterior_keys, labels
end

function hazard_theta(model::HazardModel, home_team, away_team)
    posteriors, posterior_keys, labels =
        _matchup_theta_posteriors(model, home_team, away_team)
    log_moments = [_gamma_mixture_log_moments(posterior) for posterior in posteriors]
    covariance = zeros(Float64, length(posteriors), length(posteriors))
    for i in eachindex(posteriors), j in eachindex(posteriors)
        posterior_keys[i] == posterior_keys[j] || continue
        covariance[i, j] = log_moments[i][2]
    end
    return HazardTheta(first.(log_moments), covariance, labels)
end

function _hazard_theta_with_cache(
    model::HazardModel,
    home_team,
    away_team,
    cache::_HazardLogMomentCache,
)
    cache.model === model ||
        throw(ArgumentError("hazard log-moment cache belongs to a different model"))
    requests = (
        (:td, home_team, true, :home_td),
        (:defensive, away_team, false, :away_defensive),
        (:td, away_team, false, :away_td),
        (:defensive, home_team, true, :home_defensive),
    )
    moments = Tuple{Float64,Float64}[]
    posterior_keys = Tuple{Symbol,String}[]
    labels = Symbol[]
    for (kind, team, home, label) in requests
        push!(
            moments,
            _hazard_log_moments(
                model,
                kind,
                team;
                home=home,
                cache=cache,
            ),
        )
        push!(posterior_keys, (kind, string(team)))
        push!(labels, label)
    end
    covariance = zeros(Float64, length(moments), length(moments))
    for i in eachindex(moments), j in eachindex(moments)
        posterior_keys[i] == posterior_keys[j] || continue
        covariance[i, j] = moments[i][2]
    end
    return HazardTheta(first.(moments), covariance, labels)
end

"""
    ScoreMarks

Empirical mean and variance of the possessing team's point differential,
conditional on a touchdown or defensive event.
"""
struct ScoreMarks
    mean_td::Float64
    var_td::Float64
    mean_defensive::Float64
    var_defensive::Float64
end

"""
    fit_score_marks(drives) -> ScoreMarks

Estimate outcome-conditional score moments from uncensored training drives.
"""
function fit_score_marks(drives::AbstractDataFrame)
    :home_spread_change in propertynames(drives) ||
        throw(ArgumentError("drives are missing required column :home_spread_change"))
    :posteam_home in propertynames(drives) ||
        throw(ArgumentError("drives are missing required column :posteam_home"))
    complete = subset(
        drives,
        :drive_result => ByRow(!ismissing),
        :home_spread_change => ByRow(!ismissing),
        :posteam_home => ByRow(!ismissing);
        skipmissing=true,
    )
    events = _classify_event.(String.(complete.drive_result))
    points = ifelse.(
        Bool.(complete.posteam_home),
        Float64.(complete.home_spread_change),
        -Float64.(complete.home_spread_change),
    )
    td_points = points[events .=== :td]
    defensive_points = points[events .=== :defensive]
    isempty(td_points) &&
        throw(ArgumentError("no touchdown drives available for score marks"))
    isempty(defensive_points) &&
        throw(ArgumentError("no defensive-event drives available for score marks"))
    return ScoreMarks(
        mean(td_points),
        var(td_points; corrected=false),
        mean(defensive_points),
        var(defensive_points; corrected=false),
    )
end

struct DriveMoments{T<:Real}
    p_td::T
    p_defensive::T
    mean_T::T
    var_T::T
    mean_S::T
    var_S::T
    cov_TS::T
end

struct WeibullRaceIntegrals
    values::Vector{Float64}
    jacobian::Matrix{Float64}
    hessians::Array{Float64,3}
end

function _integral_value_index(cause::Int, moment_order::Int)
    return (cause - 1) * 3 + moment_order + 1
end

function _integral_gradient_index(value_index::Int, parameter::Int)
    return 6 + (value_index - 1) * 2 + parameter
end

function _integral_hessian_index(
    value_index::Int,
    first_parameter::Int,
    second_parameter::Int,
)
    return 18 + (value_index - 1) * 4 +
        (first_parameter - 1) * 2 + second_parameter
end

function _weibull_race_integrand(
    x::Float64,
    rho_td::Float64,
    td_shape::Float64,
    rho_defensive::Float64,
    defensive_shape::Float64,
    q::Float64,
    log_time_scale::Float64,
)
    result = zeros(Float64, 42)
    if x == 0.0
        for cause in 1:2
            shape = cause == 1 ? td_shape : defensive_shape
            rho = cause == 1 ? rho_td : rho_defensive
            shape == q || continue
            value_index = _integral_value_index(cause, 0)
            limit = rho * exp(q * log_time_scale)
            result[value_index] = limit
            for parameter in 1:2
                score = parameter == cause ? 1.0 : 0.0
                result[_integral_gradient_index(value_index, parameter)] =
                    limit * score
                for second_parameter in 1:2
                    result[_integral_hessian_index(
                        value_index,
                        parameter,
                        second_parameter,
                    )] = limit * score *
                        (second_parameter == cause ? 1.0 : 0.0)
                end
            end
        end
        return result
    end

    log_x = log(x)
    log_time = log_time_scale + log_x / q
    if log_time > log(floatmax(Float64))
        return result
    end
    time = exp(log_time)
    log_jacobian = log_time_scale - log(q) + (1.0 / q - 1.0) * log_x
    log_exposure_td = log(rho_td) + td_shape * log_time
    log_exposure_defensive = log(rho_defensive) +
        defensive_shape * log_time
    exposure_td = log_exposure_td > log(floatmax(Float64)) ?
        Inf : exp(log_exposure_td)
    exposure_defensive = log_exposure_defensive > log(floatmax(Float64)) ?
        Inf : exp(log_exposure_defensive)
    total_exposure = exposure_td + exposure_defensive
    total_exposure > 745.0 && return result

    for cause in 1:2
        rho = cause == 1 ? rho_td : rho_defensive
        shape = cause == 1 ? td_shape : defensive_shape
        cause_exposure = cause == 1 ? exposure_td : exposure_defensive
        log_density_base = log(rho) + log(shape) +
            (shape - 1.0) * log_time + log_jacobian - total_exposure
        for moment_order in 0:2
            log_integrand = log_density_base + moment_order * log_time
            value = log_integrand < -745.0 ? 0.0 : exp(log_integrand)
            value_index = _integral_value_index(cause, moment_order)
            result[value_index] = value
            scores = (
                (cause == 1 ? 1.0 : 0.0) - exposure_td,
                (cause == 2 ? 1.0 : 0.0) - exposure_defensive,
            )
            for parameter in 1:2
                result[_integral_gradient_index(value_index, parameter)] =
                    value * scores[parameter]
                for second_parameter in 1:2
                    log_hessian = parameter == second_parameter ?
                        -(second_parameter == 1 ?
                            exposure_td : exposure_defensive) : 0.0
                    result[_integral_hessian_index(
                        value_index,
                        parameter,
                        second_parameter,
                    )] = value * (
                        scores[parameter] * scores[second_parameter] +
                        log_hessian
                    )
                end
            end
        end
    end
    return result
end

"""
    _weibull_race_integrals(rho_td, td_shape, rho_def, def_shape)

Integrate the two cause densities and their first two time moments. The
gradient and Hessian arrays are with respect to the two log rates.
"""
function _weibull_race_integrals(
    rho_td::Real,
    td_shape::Real,
    rho_defensive::Real,
    defensive_shape::Real;
    rtol::Real=1e-9,
)
    values = Float64.((rho_td, td_shape, rho_defensive, defensive_shape))
    rho_td_value, td_shape_value, rho_defensive_value,
        defensive_shape_value = values
    all(isfinite, values) ||
        throw(ArgumentError("Weibull rates and shapes must be finite"))
    rho_td_value > 0.0 && rho_defensive_value > 0.0 ||
        throw(ArgumentError("Weibull rates must be positive"))
    td_shape_value > 0.0 && defensive_shape_value > 0.0 ||
        throw(ArgumentError("Weibull shapes must be positive"))
    isfinite(rtol) && rtol > 0.0 ||
        throw(ArgumentError("quadrature rtol must be finite and positive"))

    q = min(td_shape_value, defensive_shape_value)
    log_time_scale = min(
        -log(rho_td_value) / td_shape_value,
        -log(rho_defensive_value) / defensive_shape_value,
    )
    isfinite(log_time_scale) ||
        throw(ArgumentError("Weibull time scale is not finite"))
    log_time_scale = clamp(
        log_time_scale,
        log(floatmin(Float64)) + 10.0,
        log(floatmax(Float64)) - 10.0,
    )
    integral, _ = QuadGK.quadgk(
        x -> _weibull_race_integrand(
            Float64(x),
            rho_td_value,
            td_shape_value,
            rho_defensive_value,
            defensive_shape_value,
            q,
            log_time_scale,
        ),
        0.0,
        Inf;
        rtol=Float64(rtol),
    )
    integral_values = integral[1:6]
    all(isfinite, integral_values) ||
        throw(ArgumentError("Weibull race integration returned non-finite moments"))
    probability_sum = integral_values[1] + integral_values[4]
    abs(probability_sum - 1.0) <= 1e-6 ||
        throw(ArgumentError(
            "Weibull race probabilities do not sum to one: $probability_sum",
        ))
    jacobian = Matrix{Float64}(undef, 6, 2)
    hessians = Array{Float64}(undef, 6, 2, 2)
    for value_index in 1:6
        for parameter in 1:2
            jacobian[value_index, parameter] =
                integral[_integral_gradient_index(value_index, parameter)]
            for second_parameter in 1:2
                hessians[value_index, parameter, second_parameter] =
                    integral[_integral_hessian_index(
                        value_index,
                        parameter,
                        second_parameter,
                    )]
            end
        end
    end
    return WeibullRaceIntegrals(integral_values, jacobian, hessians)
end

function _drive_moment_values(integrals, marks::ScoreMarks)
    p_td, mean_time_td, second_time_td,
        p_defensive, mean_time_defensive, second_time_defensive = integrals
    mean_time = mean_time_td + mean_time_defensive
    variance_time = second_time_td + second_time_defensive - mean_time^2
    mean_score = p_td * marks.mean_td +
        p_defensive * marks.mean_defensive
    second_score = p_td * (marks.var_td + marks.mean_td^2) +
        p_defensive * (marks.var_defensive + marks.mean_defensive^2)
    variance_score = second_score - mean_score^2
    mean_time_score = mean_time_td * marks.mean_td +
        mean_time_defensive * marks.mean_defensive
    covariance_time_score = mean_time_score - mean_time * mean_score
    result = similar(integrals, 7)
    result .= (
        p_td,
        p_defensive,
        mean_time,
        variance_time,
        mean_score,
        variance_score,
        covariance_time_score,
    )
    return result
end

function _drive_moments_from_weibull(
    rho_td::Real,
    td_shape::Real,
    rho_defensive::Real,
    defensive_shape::Real,
    marks::ScoreMarks,
)
    integrals = _weibull_race_integrals(
        rho_td,
        td_shape,
        rho_defensive,
        defensive_shape,
    )
    values = _drive_moment_values(integrals.values, marks)
    return DriveMoments(values...)
end

"""
    fit_hazard_model(drives; historical_drives=nothing, prior=nothing, kwargs...) -> HazardModel

Initialize a current-season Weibull competing-risk model. Supply either a
precomputed prior or historical drives from which to fit one.
"""
function fit_hazard_model(
    drives::AbstractDataFrame;
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    prior::Union{Nothing,HazardPrior}=nothing,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    current_season::Union{Nothing,Integer}=nothing,
    method=DEFAULT_PRIOR_FIT_METHOD,
)
    fitted_prior = if prior !== nothing
        prior
    elseif historical_drives !== nothing
        reference_season = current_season === nothing ?
            _latest_observed_season(drives) : Int(current_season)
        fit_empirical_bayes_prior(
            historical_drives;
            max_seasons=max_seasons,
            current_season=reference_season,
            method=method,
        )
    else
        _default_hazard_prior()
    end
    model = HazardModel(fitted_prior, HazardSufficientStats())
    return update_hazard_model!(model, drives)
end
