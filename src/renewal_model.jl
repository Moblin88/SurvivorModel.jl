"""
Piecewise-constant two-outcome drive model.

Each drive is modeled as a race between an offensive touchdown hazard and a
defensive-event hazard. The hazards are independent conditional on their
team-specific rates, piecewise constant in elapsed time since drive start,
and do not depend on field position.

Team-specific hazards use Gamma posteriors. An historical empirical-Bayes prior
can be fitted from any positive number of supplied seasons and updated with
current-season exposure and event counts as data arrive. Historical
hyperparameters are fit
with the event-process marginal likelihood, including the competing-risk
exposure term and the season-to-season reset transition. The historical fit
can also estimate global offensive and defensive home multipliers, which
remain fixed during current-season updates.
"""

# ----------------------------------------------------------------------
# 1. Time bins and exposure records
# ----------------------------------------------------------------------

"""
    DEFAULT_TIME_EDGES

Default elapsed-drive-time edges in seconds: 0-2, 2-4, 4-6, and 6+ minutes.
"""
const DEFAULT_TIME_EDGES = (0.0, 120.0, 240.0, 360.0, Inf)
# Compatibility export; historical fitting itself accepts any positive window.
const MAX_HISTORICAL_SEASONS = typemax(Int)
const RESET_EM_MAX_ITERATIONS = 100
const RESET_EM_ABSOLUTE_TOLERANCE = 1.0e-8
const RESET_EM_RELATIVE_TOLERANCE = 1.0e-8
const RESET_GAMMA_SHAPE_LOWER = 0.05
const RESET_GAMMA_SHAPE_UPPER = 1.0e8
const RESET_NEWTON_MAX_ITERATIONS = 80
const RESET_NEWTON_SCORE_TOLERANCE = 1.0e-5
const RESET_NEWTON_STEP_TOLERANCE = 1.0e-8
const RESET_NEWTON_MAX_BACKTRACKS = 20
const RESET_BLOCK_NEWTON_MAX_SWEEPS = 40
const RESET_MOMENT_MAX_ITERATIONS = 32
const RESET_MOMENT_TOLERANCE = 1.0e-6
const RESET_MOMENT_DAMPING = 0.75

"""
    _validate_time_edges(edges) -> Vector{Float64}

Validate and copy a piecewise-constant elapsed-time partition. The first edge
must be zero, all finite edges must be strictly increasing, and the final
edge must be `Inf`.
"""
function _validate_time_edges(edges)
    values = Float64.(collect(edges))
    length(values) >= 2 || throw(ArgumentError("time_edges must contain at least two values"))
    values[1] == 0.0 || throw(ArgumentError("time_edges must start at 0.0"))
    isinf(values[end]) || throw(ArgumentError("time_edges must end at Inf"))
    all(isfinite, values[1:(end - 1)]) ||
        throw(ArgumentError("only the final time edge may be infinite"))
    all(diff(values) .> 0) || throw(ArgumentError("time_edges must be strictly increasing"))
    return values
end

"""
    _classify_event(drive_result::AbstractString) -> Symbol

Classify a drive result into the two hazard-model outcomes:

- `:td` — the offense scored a touchdown.
- `:defensive` — every other non-censored drive-ending result.
- `:censored` — the game or half clock ended the observation.
"""
function _classify_event(drive_result::AbstractString)
    drive_result == "Touchdown" && return :td
    drive_result == "End of half" && return :censored
    return :defensive
end

"""
    _drive_exposure_records(T, edges, event) -> Vector{NamedTuple}

Expand one drive into one risk-exposure record per elapsed-time bin. A
censored drive contributes exposure but no event.
"""
function _drive_exposure_records(T::Real, edges::AbstractVector{<:Real}, event::Symbol)
    records = NamedTuple{(:time_bin, :exposure, :event),Tuple{Int,Float64,Symbol}}[]
    n = length(edges) - 1
    for k in 1:n
        lo, hi = edges[k], edges[k + 1]
        T <= lo && break
        exposure = min(T, hi) - lo
        exposure <= 0 && continue
        terminal = T <= hi || k == n
        ev = terminal && event !== :censored ? event : :none
        push!(records, (time_bin=k, exposure=exposure, event=ev))
        terminal && break
    end
    return records
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

function _drive_season(drives::AbstractDataFrame, i::Integer)
    if :season in propertynames(drives)
        return _season_value(drives.season[i])
    end
    return _season_from_game_id(drives.game_id[i])
end

function _latest_observed_season(drives::AbstractDataFrame)
    seasons = Int[]
    for i in 1:nrow(drives)
        season = _drive_season(drives, i)
        ismissing(season) || push!(seasons, Int(season))
    end
    return isempty(seasons) ? nothing : maximum(seasons)
end

"""
    build_exposure_data(drives; time_edges=DEFAULT_TIME_EDGES) -> (data, time_edges)

Build the long-format at-risk data used by the hazard model. The returned
`DataFrame` has one row per `(drive, time_bin)` risk interval and columns
`game_id`, `fixed_drive`, `season`, `posteam`, `defteam`, `posteam_home`,
`defteam_home`, `time_bin`, `exposure`, `td`, and `defensive`.

Rows with missing event result, duration, or teams are dropped. Field position
is not required because it is not a covariate in this model.
"""
function build_exposure_data(
    drives::AbstractDataFrame;
    time_edges=DEFAULT_TIME_EDGES,
)
    edges = _validate_time_edges(time_edges)
    complete = subset(
        drives,
        :drive_result => ByRow(!ismissing),
        :time_of_possession => ByRow(!ismissing),
        :posteam => ByRow(!ismissing),
        :defteam => ByRow(!ismissing),
        :posteam_home => ByRow(!ismissing),
        :defteam_home => ByRow(!ismissing),
        skipmissing=true,
    )

    durations = Float64[Dates.value(Second(t)) for t in complete.time_of_possession]
    events = _classify_event.(complete.drive_result)

    game_ids = eltype(complete.game_id)[]
    fixed_drives = eltype(complete.fixed_drive)[]
    seasons = Union{Missing,Int}[]
    posteams = eltype(complete.posteam)[]
    defteams = eltype(complete.defteam)[]
    posteam_home = Bool[]
    defteam_home = Bool[]
    time_bins = Int[]
    exposures = Float64[]
    touchdowns = Int[]
    defensive_events = Int[]
    for i in 1:nrow(complete)
        season = _drive_season(complete, i)
        for rec in _drive_exposure_records(durations[i], edges, events[i])
            push!(game_ids, complete.game_id[i])
            push!(fixed_drives, complete.fixed_drive[i])
            push!(seasons, season)
            push!(posteams, complete.posteam[i])
            push!(defteams, complete.defteam[i])
            push!(posteam_home, Bool(complete.posteam_home[i]))
            push!(defteam_home, Bool(complete.defteam_home[i]))
            push!(time_bins, rec.time_bin)
            push!(exposures, rec.exposure)
            push!(touchdowns, rec.event === :td ? 1 : 0)
            push!(defensive_events, rec.event === :defensive ? 1 : 0)
        end
    end

    return DataFrame(
        game_id=game_ids,
        fixed_drive=fixed_drives,
        season=seasons,
        posteam=posteams,
        defteam=defteams,
        posteam_home=posteam_home,
        defteam_home=defteam_home,
        time_bin=time_bins,
        exposure=exposures,
        td=touchdowns,
        defensive=defensive_events,
    ), edges
end

# ----------------------------------------------------------------------
# 2. Sufficient statistics and empirical-Bayes priors
# ----------------------------------------------------------------------

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
            throw(ArgumentError("Gamma mixture weights must have positive mass"))
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
    log_means = [
        SpecialFunctions.digamma(component.shape) - log(component.rate)
        for component in mixture.components
    ]
    mean_log = sum(weight * value for (weight, value) in zip(
        mixture.weights,
        log_means,
    ))
    second_log = sum(
        weight * (
            SpecialFunctions.trigamma(component.shape) + value^2
        )
        for (weight, component, value) in zip(
            mixture.weights,
            mixture.components,
            log_means,
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
    updated_components = [
        GammaParams(
            component.shape + Float64(count),
            component.rate + Float64(exposure),
        )
        for component in mixture.components
    ]
    log_weights = [
        log(weight) + _log_gamma_poisson_predictive(
            component,
            count,
            exposure,
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
reported without data-only counting-process constants, so it is suitable for
comparing parameter values fitted to the same data.
"""
struct LikelihoodFitDiagnostics
    log_likelihood::Float64
    converged::Bool
    iterations::Int
    function_evaluations::Int
    status::Symbol
    boundary_parameters::Vector{Symbol}
end
mutable struct OutcomeStats
    counts::Dict{Tuple{String,Int},Float64}
    exposure::Dict{Tuple{String,Int},Float64}
    home_counts::Dict{Tuple{String,Int},Float64}
    away_counts::Dict{Tuple{String,Int},Float64}
    home_exposure::Dict{Tuple{String,Int},Float64}
    away_exposure::Dict{Tuple{String,Int},Float64}
end

OutcomeStats() = OutcomeStats(
    Dict{Tuple{String,Int},Float64}(),
    Dict{Tuple{String,Int},Float64}(),
    Dict{Tuple{String,Int},Float64}(),
    Dict{Tuple{String,Int},Float64}(),
    Dict{Tuple{String,Int},Float64}(),
    Dict{Tuple{String,Int},Float64}(),
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
"""
    HazardPrior

Historical empirical-Bayes hyperparameters and team-specific finite-mixture
season-opening priors for the touchdown and defensive-event hazards. The
fitted home multipliers are shared across teams and time bins, and each
outcome has one persistence probability shared across its hazard curve.
"""
struct HazardPrior
    time_edges::Vector{Float64}
    td_hyperparameters::Vector{GammaParams}
    defensive_hyperparameters::Vector{GammaParams}
    td_team_mixtures::Dict{String,Vector{GammaMixture}}
    defensive_team_mixtures::Dict{String,Vector{GammaMixture}}
    td_home_multiplier::Float64
    defensive_home_multiplier::Float64
    td_persistence::Float64
    defensive_persistence::Float64
    historical_seasons::Vector{Int}
    td_fit_diagnostics::Union{Nothing,LikelihoodFitDiagnostics}
    defensive_fit_diagnostics::Union{Nothing,LikelihoodFitDiagnostics}
end

function _default_hazard_prior(time_edges::AbstractVector{<:Real})
    n_bins = length(time_edges) - 1
    td = [GammaParams(1.0, 100.0) for _ in 1:n_bins]
    defensive = [GammaParams(1.0, 100.0) for _ in 1:n_bins]
    return HazardPrior(
        Float64.(time_edges),
        td,
        defensive,
        Dict{String,Vector{GammaMixture}}(),
        Dict{String,Vector{GammaMixture}}(),
        1.0,
        1.0,
        0.5,
        0.5,
        Int[],
        nothing,
        nothing,
    )
end

"""
    home_multiplier(prior::HazardPrior, kind::Symbol) -> Float64

Return the empirical-Bayes home multiplier for `:td` or `:defensive`.
"""
function home_multiplier(prior::HazardPrior, kind::Symbol)
    kind === :td && return prior.td_home_multiplier
    kind === :defensive && return prior.defensive_home_multiplier
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end

"""
    hazard_persistence(prior::HazardPrior, kind::Symbol) -> Float64

Return the season-to-season persistence probability for `:td` or
`:defensive`.
"""
function hazard_persistence(prior::HazardPrior, kind::Symbol)
    kind === :td && return prior.td_persistence
    kind === :defensive && return prior.defensive_persistence
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end

function likelihood_fit_diagnostics(
    prior::HazardPrior,
    kind::Symbol,
)
    kind === :td && return prior.td_fit_diagnostics
    kind === :defensive && return prior.defensive_fit_diagnostics
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end
# ----------------------------------------------------------------------
# 3. Hazard model and posterior updates
# ----------------------------------------------------------------------

"""
    HazardModel

Mutable current-season hazard state. The prior stores historical
empirical-Bayes information; the sufficient statistics store only observations
added since that prior was initialized.
"""
mutable struct HazardModel
    time_edges::Vector{Float64}
    prior::HazardPrior
    stats::HazardSufficientStats
end

home_multiplier(model::HazardModel, kind::Symbol) =
    home_multiplier(model.prior, kind)

function _prior_mixture(
    prior::HazardPrior,
    kind::Symbol,
    team::String,
    time_bin::Int,
)
    1 <= time_bin <= length(prior.time_edges) - 1 ||
        throw(BoundsError(prior.time_edges, time_bin))
    if kind === :td
        mixtures = get(prior.td_team_mixtures, team, nothing)
        isnothing(mixtures) ||
            return mixtures[time_bin]
        return GammaMixture(
            [1.0],
            [prior.td_hyperparameters[time_bin]];
            source_seasons=[0],
        )
    elseif kind === :defensive
        mixtures = get(prior.defensive_team_mixtures, team, nothing)
        isnothing(mixtures) ||
            return mixtures[time_bin]
        return GammaMixture(
            [1.0],
            [prior.defensive_hyperparameters[time_bin]];
            source_seasons=[0],
        )
    end
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end

"""
    update_hazard_model!(model, drives) -> HazardModel

Add new drive exposure and event counts to the model and update its Gamma
posteriors. This is the intended in-season update operation.
"""
function update_hazard_model!(model::HazardModel, drives::AbstractDataFrame)
    data, edges = build_exposure_data(drives; time_edges=model.time_edges)
    edges == model.time_edges ||
        throw(ArgumentError("new exposure data uses incompatible time_edges"))
    _add_exposure_data!(model.stats, data)
    return model
end

"""
    hazard_posterior(model, kind, team, time_bin; home=false) -> GammaMixture

Return the exact finite Gamma-mixture posterior for one team's hazard in one
elapsed-time bin. When `home=true`, return the mixture of home-adjusted
hazards rather than the baseline (away) hazards.
"""
function hazard_posterior(
    model::HazardModel,
    kind::Symbol,
    team,
    time_bin::Integer,
    ;
    home::Bool=false,
)
    team_name = string(team)
    prior = _prior_mixture(model.prior, kind, team_name, Int(time_bin))
    outcome = _outcome_stats(model.stats, kind)
    key = (team_name, Int(time_bin))
    multiplier = home_multiplier(model.prior, kind)
    posterior = _update_gamma_mixture(
        prior,
        get(outcome.home_counts, key, 0.0) +
            get(outcome.away_counts, key, 0.0),
        get(outcome.away_exposure, key, 0.0) +
            multiplier * get(outcome.home_exposure, key, 0.0),
    )
    return home ? _gamma_mixture_home_adjusted(posterior, multiplier) : posterior
end

"""
    hazard_rate(model, kind, team, time_bin; home=false) -> Float64

Return the posterior-mean instantaneous hazard rate. `kind` is `:td` for an
offensive touchdown hazard or `:defensive` for a defensive-event hazard.
When `home=true`, apply the fitted home multiplier.
"""
function hazard_rate(
    model::HazardModel,
    kind::Symbol,
    team,
    time_bin::Integer,
    ;
    home::Bool=false,
)
    posterior = hazard_posterior(model, kind, team, time_bin; home=home)
    return _gamma_mixture_mean(posterior)
end

"""
    HazardTheta

Posterior moments of a matchup's log-hazard parameter vector. The vector is
ordered by time-bin blocks:

1. home team's offensive touchdown hazards;
2. away team's defensive-event hazards;
3. away team's offensive touchdown hazards;
4. home team's defensive-event hazards.

Each block contains one value per elapsed-time bin. `log_mean` and
`covariance` describe the posterior mean and covariance of the log hazards.
The covariance is conditional on the fitted empirical-Bayes prior and home
multipliers.
"""
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
    n_bins = length(model.time_edges) - 1
    requests = (
        (:td, home_team, true, "home_td"),
        (:defensive, away_team, false, "away_defensive"),
        (:td, away_team, false, "away_td"),
        (:defensive, home_team, true, "home_defensive"),
    )
    posteriors = GammaMixture[]
    posterior_keys = Tuple{Symbol,String,Int}[]
    labels = Symbol[]

    for (kind, team, home, label_prefix) in requests
        for time_bin in 1:n_bins
            push!(
                posteriors,
                hazard_posterior(model, kind, team, time_bin; home=home),
            )
            push!(posterior_keys, (kind, string(team), time_bin))
            push!(labels, Symbol(label_prefix, "_", time_bin))
        end
    end

    return posteriors, posterior_keys, labels
end

"""
    hazard_theta(model, home_team, away_team) -> HazardTheta

Return the ordered posterior mean and covariance of the matchup log-hazard
vector. The four blocks correspond to home offense, away defense, away
offense, and home defense, respectively, with one entry per model time bin.
The finite-mixture posteriors are transformed exactly to their mixture
log-hazard moments.
"""
function hazard_theta(model::HazardModel, home_team, away_team)
    posteriors, posterior_keys, labels =
        _matchup_theta_posteriors(model, home_team, away_team)
    log_mean = [_gamma_mixture_log_moments(posterior)[1] for posterior in posteriors]
    covariance = zeros(Float64, length(posteriors), length(posteriors))
    for i in eachindex(posteriors), j in eachindex(posteriors)
        posterior_keys[i] == posterior_keys[j] || continue
        covariance[i, j] = _gamma_mixture_log_moments(posteriors[i])[2]
    end
    return HazardTheta(log_mean, covariance, labels)
end

# ----------------------------------------------------------------------
# 4. Conditional score marks
# ----------------------------------------------------------------------

"""
    ScoreMarks

Empirical mean/variance of the possessing team's point differential,
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

Estimate outcome-conditional score moments. Censored drives are excluded.
All non-touchdown, non-censored outcomes are included in the defensive-event
mark.
"""
function fit_score_marks(drives::AbstractDataFrame)
    complete = subset(
        drives,
        :drive_result => ByRow(x -> _classify_event(x) !== :censored);
        skipmissing=true,
    )
    offense_points = ifelse.(
        complete.posteam_home,
        complete.home_spread_change,
        -complete.home_spread_change,
    )
    events = _classify_event.(complete.drive_result)

    td_points = offense_points[events .=== :td]
    defensive_points = offense_points[events .=== :defensive]
    isempty(td_points) && throw(ArgumentError("no touchdown drives available for score marks"))
    isempty(defensive_points) &&
        throw(ArgumentError("no defensive-event drives available for score marks"))

    return ScoreMarks(
        mean(td_points),
        var(td_points; corrected=false),
        mean(defensive_points),
        var(defensive_points; corrected=false),
    )
end

# ----------------------------------------------------------------------
# 5. Per-drive moments
# ----------------------------------------------------------------------

"""
    DriveMoments

Moments of one drive's duration and possessing-team score under the
two-hazard race.
"""
struct DriveMoments{T<:Real}
    p_td::T
    p_defensive::T
    mean_T::T
    var_T::T
    mean_S::T
    var_S::T
    cov_TS::T
end

_bin_widths(edges::AbstractVector{<:Real}) = diff(edges)

function _drive_moments_from_hazards(
    edges::AbstractVector{<:Real},
    marks::ScoreMarks,
    lambda_td::AbstractVector{<:Real},
    lambda_defensive::AbstractVector{<:Real},
)
    widths = _bin_widths(edges)
    n = length(widths)
    length(lambda_td) == n && length(lambda_defensive) == n ||
        throw(ArgumentError("hazard vectors must contain one value per time bin"))
    T = promote_type(eltype(lambda_td), eltype(lambda_defensive), Float64)

    lambda = lambda_td .+ lambda_defensive
    S_prev = one(T)
    p_event = zeros(T, n)
    e_time = zeros(T, n)
    contrib_ET = zeros(T, n)
    contrib_ET2 = zeros(T, n)

    for k in 1:n
        lo, width, total_rate = edges[k], widths[k], lambda[k]
        if isinf(width)
            p_event[k] = S_prev
            e_time[k] = lo + 1 / total_rate
            contrib_ET[k] = S_prev / total_rate
            contrib_ET2[k] = S_prev * (2 * lo / total_rate + 2 / total_rate^2)
        else
            decay = exp(-total_rate * width)
            p_event[k] = S_prev * (1 - decay)
            e_local = (1 - decay * (1 + total_rate * width)) /
                (total_rate * (1 - decay))
            e_time[k] = lo + e_local
            contrib_ET[k] = S_prev * (1 - decay) / total_rate
            local_sq = (1 - decay * (1 + total_rate * width)) / total_rate^2
            contrib_ET2[k] = S_prev * (
                2 * lo * (1 - decay) / total_rate + 2 * local_sq
            )
            S_prev *= decay
        end
    end

    td_weights = [
        p_event[k] * (lambda_td[k] / lambda[k]) for k in 1:n
    ]
    defensive_weights = [
        p_event[k] * (lambda_defensive[k] / lambda[k]) for k in 1:n
    ]
    p_td = sum(td_weights)
    p_defensive = sum(defensive_weights)

    conditional_mean_time = (weights, probability) ->
        probability > 0 ?
            sum(weights[k] * e_time[k] for k in 1:n) / probability :
            zero(T)
    mean_T_td = conditional_mean_time(td_weights, p_td)
    mean_T_defensive = conditional_mean_time(defensive_weights, p_defensive)

    mean_T = sum(contrib_ET)
    mean_T2 = sum(contrib_ET2)
    var_T = mean_T2 - mean_T^2

    mean_S = p_td * marks.mean_td + p_defensive * marks.mean_defensive
    mean_S2 = p_td * (marks.var_td + marks.mean_td^2) +
        p_defensive * (marks.var_defensive + marks.mean_defensive^2)
    var_S = mean_S2 - mean_S^2
    mean_TS = p_td * mean_T_td * marks.mean_td +
        p_defensive * mean_T_defensive * marks.mean_defensive
    cov_TS = mean_TS - mean_T * mean_S

    return DriveMoments(
        p_td,
        p_defensive,
        mean_T,
        var_T,
        mean_S,
        var_S,
        cov_TS,
    )
end

"""
    drive_moments(model, marks, posteam, defteam; posteam_home=false) -> DriveMoments

Compute closed-form duration, score, and time/score covariance moments for a
drive with the given offensive and defensive teams. `posteam_home` selects the
home-adjusted offensive hazard; the defensive team is assigned the
complementary away/home status.
"""
function drive_moments(
    model::HazardModel,
    marks::ScoreMarks,
    posteam,
    defteam,
    ;
    posteam_home::Bool=false,
)
    edges = model.time_edges
    widths = _bin_widths(edges)
    n = length(widths)

    lambda_td = [
        hazard_rate(model, :td, posteam, k; home=posteam_home)
        for k in 1:n
    ]
    lambda_defensive = [
        hazard_rate(model, :defensive, defteam, k; home=!posteam_home)
        for k in 1:n
    ]
    return _drive_moments_from_hazards(edges, marks, lambda_td, lambda_defensive)
end

drive_moments(
    model::HazardModel,
    marks::ScoreMarks,
    posteam,
    defteam,
    posteam_home::Bool,
) = drive_moments(model, marks, posteam, defteam; posteam_home=posteam_home)
