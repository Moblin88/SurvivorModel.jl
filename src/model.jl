"""
Piecewise-constant two-outcome drive model.

Each drive is modeled as a race between an offensive touchdown hazard and a
defensive-event hazard. The hazards are independent conditional on their
team-specific rates, piecewise constant in elapsed time since drive start,
and do not depend on field position.

Team-specific hazards use Gamma posteriors. A historical empirical-Bayes prior
can be fitted from the previous three seasons and updated with current-season
exposure and event counts as data arrive. The historical fit can also estimate
global offensive and defensive home multipliers, which remain fixed during
current-season updates.
"""

# ----------------------------------------------------------------------
# 1. Time bins and exposure records
# ----------------------------------------------------------------------

"""
    DEFAULT_TIME_EDGES

Default elapsed-drive-time edges in seconds: 0-2, 2-4, 4-6, and 6+ minutes.
"""
const DEFAULT_TIME_EDGES = (0.0, 120.0, 240.0, 360.0, Inf)

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

    rows = NamedTuple[]
    for i in 1:nrow(complete)
        season = _drive_season(complete, i)
        for rec in _drive_exposure_records(durations[i], edges, events[i])
            push!(rows, (
                game_id=complete.game_id[i],
                fixed_drive=complete.fixed_drive[i],
                season=season,
                posteam=complete.posteam[i],
                defteam=complete.defteam[i],
                posteam_home=Bool(complete.posteam_home[i]),
                defteam_home=Bool(complete.defteam_home[i]),
                time_bin=rec.time_bin,
                exposure=rec.exposure,
                td=rec.event === :td ? 1 : 0,
                defensive=rec.event === :defensive ? 1 : 0,
            ))
        end
    end

    return DataFrame(rows), edges
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

function _add_stat!(values::Dict{Tuple{String,Int},Float64}, key, amount::Real)
    values[key] = get(values, key, 0.0) + Float64(amount)
    return nothing
end

function _add_exposure_record!(stats::HazardSufficientStats, row)
    offensive_key = (string(row.posteam), Int(row.time_bin))
    defensive_key = (string(row.defteam), Int(row.time_bin))
    offensive_home = Bool(row.posteam_home)
    defensive_home = Bool(row.defteam_home)

    _add_stat!(stats.td.exposure, offensive_key, row.exposure)
    _add_stat!(stats.defensive.exposure, defensive_key, row.exposure)
    _add_stat!(
        offensive_home ? stats.td.home_exposure : stats.td.away_exposure,
        offensive_key,
        row.exposure,
    )
    _add_stat!(
        defensive_home ? stats.defensive.home_exposure : stats.defensive.away_exposure,
        defensive_key,
        row.exposure,
    )

    if row.td == 1
        _add_stat!(stats.td.counts, offensive_key, 1.0)
        _add_stat!(
            offensive_home ? stats.td.home_counts : stats.td.away_counts,
            offensive_key,
            1.0,
        )
    end
    if row.defensive == 1
        _add_stat!(stats.defensive.counts, defensive_key, 1.0)
        _add_stat!(
            defensive_home ? stats.defensive.home_counts : stats.defensive.away_counts,
            defensive_key,
            1.0,
        )
    end
    return nothing
end

function _add_exposure_data!(stats::HazardSufficientStats, data::AbstractDataFrame)
    for row in eachrow(data)
        _add_exposure_record!(stats, row)
    end
    return stats
end

function _season_stats(
    data::AbstractDataFrame,
)
    result = Dict{Int,HazardSufficientStats}()
    for row in eachrow(data)
        season = ismissing(row.season) ? 0 : Int(row.season)
        stats = get!(result, season, HazardSufficientStats())
        _add_exposure_record!(stats, row)
    end
    return result
end

function _historical_seasons(byseason::Dict{Int,HazardSufficientStats}, max_seasons::Int)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    seasons = sort!(collect(filter(!=(0), keys(byseason))))
    if isempty(seasons)
        return haskey(byseason, 0) ? [0] : Int[]
    end
    return seasons[max(1, end - max_seasons + 1):end]
end

function _season_weight(season::Int, reference::Int, half_life::Real)
    half_life == Inf && return 1.0
    half_life > 0 || throw(ArgumentError("recency half-life must be positive"))
    return exp(-(reference - season) / half_life)
end

function _cell_observations(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    time_bin::Int,
    reference::Int,
    half_life::Real,
)
    home_counts = Float64[]
    away_counts = Float64[]
    home_exposures = Float64[]
    away_exposures = Float64[]
    weights = Float64[]

    for season in seasons
        stats = byseason[Int(season)]
        outcome = _outcome_stats(stats, kind)
        keys_for_bin = Set{Tuple{String,Int}}()
        for key in keys(outcome.home_exposure)
            key[2] == time_bin || continue
            push!(keys_for_bin, key)
        end
        for key in keys(outcome.away_exposure)
            key[2] == time_bin || continue
            push!(keys_for_bin, key)
        end
        for key in keys_for_bin
            home_exposure = get(outcome.home_exposure, key, 0.0)
            away_exposure = get(outcome.away_exposure, key, 0.0)
            home_exposure + away_exposure > 0 || continue
            push!(home_counts, get(outcome.home_counts, key, 0.0))
            push!(away_counts, get(outcome.away_counts, key, 0.0))
            push!(home_exposures, home_exposure)
            push!(away_exposures, away_exposure)
            push!(weights, _season_weight(Int(season), reference, half_life))
        end
    end

    return home_counts, away_counts, home_exposures, away_exposures, weights
end

function _weighted_team_stats(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    team::String,
    time_bin::Int,
    reference::Int,
    half_life::Real,
    home_multiplier::Real,
)
    count = 0.0
    exposure = 0.0
    for season in seasons
        stats = byseason[Int(season)]
        outcome = _outcome_stats(stats, kind)
        key = (team, time_bin)
        weight = _season_weight(Int(season), reference, half_life)
        count += weight * (
            get(outcome.home_counts, key, 0.0) +
            get(outcome.away_counts, key, 0.0)
        )
        exposure += weight * (
            get(outcome.away_exposure, key, 0.0) +
            home_multiplier * get(outcome.home_exposure, key, 0.0)
        )
    end
    return count, exposure
end

function _fit_gamma_poisson_hyperparameters(
    counts::AbstractVector{<:Real},
    exposures::AbstractVector{<:Real},
    weights::AbstractVector{<:Real},
)
    length(counts) == length(exposures) == length(weights) ||
        throw(ArgumentError("Gamma-Poisson observations must have matching lengths"))
    valid = findall(i -> exposures[i] > 0 && weights[i] > 0, eachindex(exposures))
    isempty(valid) && return GammaParams(1.0, 1.0)

    weighted_events = sum(weights[i] * counts[i] for i in valid)
    weighted_exposure = sum(weights[i] * exposures[i] for i in valid)
    mean_rate = weighted_events > 0 ? weighted_events / weighted_exposure : 1e-3
    initial_shape = 1.0
    initial_rate = max(initial_shape / mean_rate, 1e-8)

    objective(x) = begin
        shape = exp(clamp(x[1], -12.0, 12.0))
        rate = exp(clamp(x[2], -20.0, 20.0))
        total = 0.0
        for i in valid
            n = round(Int, counts[i])
            p = rate / (rate + exposures[i])
            total -= weights[i] * logpdf(NegativeBinomial(shape, p), n)
        end
        return total
    end

    result = Optim.optimize(
        objective,
        [log(initial_shape), log(initial_rate)],
        Optim.NelderMead(),
        Optim.Options(iterations=400),
    )
    minimizer = Optim.minimizer(result)
    shape = exp(clamp(minimizer[1], -12.0, 12.0))
    rate = exp(clamp(minimizer[2], -20.0, 20.0))
    return GammaParams(shape, rate)
end

function _log_gamma_poisson_home_kernel(
    home_count::Real,
    away_count::Real,
    home_exposure::Real,
    away_exposure::Real,
    prior::GammaParams,
    home_multiplier::Real,
)
    effective_exposure = away_exposure + home_multiplier * home_exposure
    effective_exposure > 0 || throw(ArgumentError("exposure must be positive"))
    home_count_int = round(Int, home_count)
    away_count_int = round(Int, away_count)
    total_count = home_count_int + away_count_int
    probability = prior.rate / (prior.rate + effective_exposure)
    value = logpdf(NegativeBinomial(prior.shape, probability), total_count)
    total_count > 0 && (value -= total_count * log(effective_exposure))
    home_count_int > 0 && (
        value += home_count_int * log(home_multiplier * home_exposure)
    )
    away_count_int > 0 && (value += away_count_int * log(away_exposure))
    return value
end

function _fit_outcome_hyperparameters(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    kind::Symbol,
    reference::Int,
    half_life::Real,
)
    n_bins = length(time_edges) - 1
    observations = [
        _cell_observations(byseason, seasons, kind, k, reference, half_life)
        for k in 1:n_bins
    ]
    initial_parameters = GammaParams[]
    initial_values = Float64[]
    for (home_counts, away_counts, home_exposures, away_exposures, weights) in observations
        counts = home_counts .+ away_counts
        exposures = home_exposures .+ away_exposures
        parameters = _fit_gamma_poisson_hyperparameters(counts, exposures, weights)
        push!(initial_parameters, parameters)
        push!(initial_values, log(parameters.shape), log(parameters.rate))
    end
    push!(initial_values, 0.0)

    objective(values) = begin
        home_multiplier = exp(clamp(values[end], -12.0, 12.0))
        total = 0.0
        for k in 1:n_bins
            shape = exp(clamp(values[2k - 1], -12.0, 12.0))
            rate = exp(clamp(values[2k], -20.0, 20.0))
            prior = GammaParams(shape, rate)
            home_counts, away_counts, home_exposures, away_exposures, weights =
                observations[k]
            for i in eachindex(weights)
                total -= weights[i] * _log_gamma_poisson_home_kernel(
                    home_counts[i],
                    away_counts[i],
                    home_exposures[i],
                    away_exposures[i],
                    prior,
                    home_multiplier,
                )
            end
        end
        return total
    end

    result = Optim.optimize(
        objective,
        initial_values,
        Optim.NelderMead(),
        Optim.Options(iterations=800),
    )
    minimizer = Optim.minimizer(result)
    parameters = [
        GammaParams(
            exp(clamp(minimizer[2k - 1], -12.0, 12.0)),
            exp(clamp(minimizer[2k], -20.0, 20.0)),
        )
        for k in 1:n_bins
    ]
    home_multiplier = exp(clamp(minimizer[end], -12.0, 12.0))
    return parameters, home_multiplier
end

function _fit_hyperparameter_vectors(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    reference::Int,
    half_life::Real,
)
    td, td_home_multiplier = _fit_outcome_hyperparameters(
        byseason, seasons, time_edges, :td, reference, half_life,
    )
    defensive, defensive_home_multiplier = _fit_outcome_hyperparameters(
        byseason, seasons, time_edges, :defensive, reference, half_life,
    )
    return td, defensive, td_home_multiplier, defensive_home_multiplier
end

function _teams(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
)
    result = Set{String}()
    for season in seasons
        for kind in (:td, :defensive)
            outcome = _outcome_stats(byseason[Int(season)], kind)
            union!(result, (key[1] for key in keys(outcome.exposure)))
        end
    end
    return sort!(collect(result))
end

function _build_team_priors(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    reference::Int,
    half_life::Real,
    td_hyperparameters::AbstractVector{GammaParams},
    defensive_hyperparameters::AbstractVector{GammaParams},
    td_home_multiplier::Real,
    defensive_home_multiplier::Real,
)
    td_priors = Dict{String,Vector{GammaParams}}()
    defensive_priors = Dict{String,Vector{GammaParams}}()
    n_bins = length(time_edges) - 1

    for team in _teams(byseason, seasons)
        td_team = GammaParams[]
        defensive_team = GammaParams[]
        for time_bin in 1:n_bins
            td_count, td_exposure = _weighted_team_stats(
                byseason,
                seasons,
                :td,
                team,
                time_bin,
                reference,
                half_life,
                td_home_multiplier,
            )
            defensive_count, defensive_exposure = _weighted_team_stats(
                byseason,
                seasons,
                :defensive,
                team,
                time_bin,
                reference,
                half_life,
                defensive_home_multiplier,
            )
            td_hyper = td_hyperparameters[time_bin]
            defensive_hyper = defensive_hyperparameters[time_bin]
            push!(
                td_team,
                GammaParams(td_hyper.shape + td_count, td_hyper.rate + td_exposure),
            )
            push!(
                defensive_team,
                GammaParams(
                    defensive_hyper.shape + defensive_count,
                    defensive_hyper.rate + defensive_exposure,
                ),
            )
        end
        td_priors[team] = td_team
        defensive_priors[team] = defensive_team
    end

    return td_priors, defensive_priors
end

function _predictive_loglikelihood(
    validation::HazardSufficientStats,
    td_hyperparameters::AbstractVector{GammaParams},
    defensive_hyperparameters::AbstractVector{GammaParams},
    td_priors::Dict{String,Vector{GammaParams}},
    defensive_priors::Dict{String,Vector{GammaParams}},
    td_home_multiplier::Real,
    defensive_home_multiplier::Real,
)
    total = 0.0
    for (kind, hyperparameters, team_priors, home_multiplier) in (
        (:td, td_hyperparameters, td_priors, td_home_multiplier),
        (:defensive, defensive_hyperparameters, defensive_priors, defensive_home_multiplier),
    )
        outcome = _outcome_stats(validation, kind)
        for time_bin in 1:length(hyperparameters)
            keys_for_bin = Set{Tuple{String,Int}}()
            for key in keys(outcome.home_exposure)
                key[2] == time_bin || continue
                push!(keys_for_bin, key)
            end
            for key in keys(outcome.away_exposure)
                key[2] == time_bin || continue
                push!(keys_for_bin, key)
            end
            for (team, _) in keys_for_bin
                parameters = get(team_priors, team, hyperparameters)
                prior = parameters[time_bin]
                key = (team, time_bin)
                total += _log_gamma_poisson_home_kernel(
                    get(outcome.home_counts, key, 0.0),
                    get(outcome.away_counts, key, 0.0),
                    get(outcome.home_exposure, key, 0.0),
                    get(outcome.away_exposure, key, 0.0),
                    prior,
                    home_multiplier,
                )
            end
        end
    end
    return total
end

function _calibrate_recency_half_life(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real};
    candidates=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, Inf),
)
    length(seasons) >= 3 || return 1.0
    scores = fill(-Inf, length(candidates))

    for (candidate_index, candidate) in enumerate(candidates)
        score = 0.0
        folds = 0
        for target_index in 2:length(seasons)
            train_seasons = seasons[1:(target_index - 1)]
            target_season = Int(seasons[target_index])
            reference = maximum(train_seasons)
            td_hyper, defensive_hyper, td_home_multiplier, defensive_home_multiplier =
                _fit_hyperparameter_vectors(
                byseason, train_seasons, time_edges, reference, candidate,
            )
            td_priors, defensive_priors = _build_team_priors(
                byseason,
                train_seasons,
                time_edges,
                reference,
                candidate,
                td_hyper,
                defensive_hyper,
                td_home_multiplier,
                defensive_home_multiplier,
            )
            score += _predictive_loglikelihood(
                byseason[target_season],
                td_hyper,
                defensive_hyper,
                td_priors,
                defensive_priors,
                td_home_multiplier,
                defensive_home_multiplier,
            )
            folds += 1
        end
        folds > 0 && (scores[candidate_index] = score)
    end

    best_index = argmax(scores)
    return candidates[best_index]
end

"""
    HazardPrior

Historical empirical-Bayes hyperparameters and team-specific season-opening
Gamma priors for the touchdown and defensive-event hazards. The fitted
offensive and defensive home multipliers are shared across teams and time
bins.
"""
struct HazardPrior
    time_edges::Vector{Float64}
    td_hyperparameters::Vector{GammaParams}
    defensive_hyperparameters::Vector{GammaParams}
    td_team_priors::Dict{String,Vector{GammaParams}}
    defensive_team_priors::Dict{String,Vector{GammaParams}}
    td_home_multiplier::Float64
    defensive_home_multiplier::Float64
    recency_half_life::Float64
    recency_reference_season::Union{Nothing,Int}
    historical_seasons::Vector{Int}
    historical_weights::Dict{Int,Float64}
end

function _default_hazard_prior(time_edges::AbstractVector{<:Real})
    n_bins = length(time_edges) - 1
    td = [GammaParams(1.0, 100.0) for _ in 1:n_bins]
    defensive = [GammaParams(1.0, 100.0) for _ in 1:n_bins]
    return HazardPrior(
        Float64.(time_edges),
        td,
        defensive,
        Dict{String,Vector{GammaParams}}(),
        Dict{String,Vector{GammaParams}}(),
        1.0,
        1.0,
        Inf,
        nothing,
        Int[],
        Dict{Int,Float64}(),
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
    recency_weights(prior::HazardPrior) -> Dict{Int,Float64}

Return the historical season weights used by an empirical-Bayes prior. The
reference current season, when known, is included with weight `1.0`.
"""
function recency_weights(prior::HazardPrior)
    weights = copy(prior.historical_weights)
    isnothing(prior.recency_reference_season) ||
        (weights[prior.recency_reference_season] = 1.0)
    return weights
end

"""
    fit_empirical_bayes_prior(historical_drives; kwargs...) -> HazardPrior

Fit separate league-level Gamma-Poisson hyperparameters for each outcome and
elapsed-time bin using historical drives, along with one global offensive and
one global defensive home multiplier. Team-specific priors are formed from
the most recent `max_seasons` seasons. If `recency_half_life` is omitted, it
is selected by chronological predictive validation over the available
historical seasons. `current_season` is the reference point for the final
historical weights; when omitted, it defaults to one season after the latest
historical season.
"""
function fit_empirical_bayes_prior(
    historical_drives::AbstractDataFrame;
    time_edges=DEFAULT_TIME_EDGES,
    max_seasons::Int=3,
    recency_half_life::Union{Nothing,Real}=nothing,
    half_life_candidates=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, Inf),
    current_season::Union{Nothing,Integer}=nothing,
)
    data, edges = build_exposure_data(historical_drives; time_edges=time_edges)
    byseason = _season_stats(data)
    seasons = _historical_seasons(byseason, max_seasons)
    isempty(seasons) && return _default_hazard_prior(edges)

    calibration_seasons = sort!(collect(filter(!=(0), keys(byseason))))
    half_life = if recency_half_life === nothing
        _calibrate_recency_half_life(
            byseason,
            calibration_seasons,
            edges;
            candidates=half_life_candidates,
        )
    else
        Float64(recency_half_life)
    end
    observed_seasons = collect(filter(!=(0), seasons))
    reference = if current_season === nothing
        isempty(observed_seasons) ? 0 : maximum(observed_seasons) + 1
    else
        Int(current_season)
    end
    if !isempty(observed_seasons) && reference <= maximum(observed_seasons)
        throw(ArgumentError("current_season must follow historical seasons"))
    end
    historical_weights = Dict(
        Int(season) => _season_weight(Int(season), reference, half_life)
        for season in seasons
    )
    td_hyper, defensive_hyper, td_home_multiplier, defensive_home_multiplier =
        _fit_hyperparameter_vectors(
        byseason, seasons, edges, reference, half_life,
    )
    td_priors, defensive_priors = _build_team_priors(
        byseason,
        seasons,
        edges,
        reference,
        half_life,
        td_hyper,
        defensive_hyper,
        td_home_multiplier,
        defensive_home_multiplier,
    )

    return HazardPrior(
        edges,
        td_hyper,
        defensive_hyper,
        td_priors,
        defensive_priors,
        td_home_multiplier,
        defensive_home_multiplier,
        half_life,
        reference == 0 ? nothing : reference,
        Int.(seasons),
        historical_weights,
    )
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

function _prior_parameters(prior::HazardPrior, kind::Symbol, team::String, time_bin::Int)
    1 <= time_bin <= length(prior.time_edges) - 1 ||
        throw(BoundsError(prior.time_edges, time_bin))
    if kind === :td
        return get(prior.td_team_priors, team, prior.td_hyperparameters)[time_bin]
    elseif kind === :defensive
        return get(
            prior.defensive_team_priors,
            team,
            prior.defensive_hyperparameters,
        )[time_bin]
    end
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end

"""
    fit_hazard_model(drives; historical_drives=nothing, prior=nothing, kwargs...) -> HazardModel

Initialize a current-season two-outcome hazard model. Supply either a
precomputed `prior` or `historical_drives` from which to fit one. If neither
is supplied, a weak default Gamma prior is used. When historical drives are
provided, `current_season` is inferred from the current drives when possible
and otherwise defaults to one season after the latest historical season.
Historical priors include fitted home multipliers when home/away indicators
are available; those multipliers are not re-estimated by current-season
updates.
"""
function fit_hazard_model(
    drives::AbstractDataFrame;
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    prior::Union{Nothing,HazardPrior}=nothing,
    time_edges=DEFAULT_TIME_EDGES,
    max_seasons::Int=3,
    recency_half_life::Union{Nothing,Real}=nothing,
    current_season::Union{Nothing,Integer}=nothing,
)
    edges = _validate_time_edges(time_edges)
    fitted_prior = if prior !== nothing
        prior.time_edges == edges ||
            throw(ArgumentError("prior and model time_edges must match"))
        prior
    elseif historical_drives !== nothing
        reference_season = current_season === nothing ?
            _latest_observed_season(drives) : current_season
        fit_empirical_bayes_prior(
            historical_drives;
            time_edges=edges,
            max_seasons=max_seasons,
            recency_half_life=recency_half_life,
            current_season=reference_season,
        )
    else
        _default_hazard_prior(edges)
    end

    model = HazardModel(edges, fitted_prior, HazardSufficientStats())
    return update_hazard_model!(model, drives)
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
    hazard_posterior(model, kind, team, time_bin; home=false) -> GammaParams

Return the current Gamma posterior parameters for one team's hazard in one
elapsed-time bin. When `home=true`, return the distribution of the
home-adjusted hazard rather than the baseline (away) hazard.
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
    prior = _prior_parameters(model.prior, kind, team_name, Int(time_bin))
    outcome = _outcome_stats(model.stats, kind)
    key = (team_name, Int(time_bin))
    multiplier = home_multiplier(model.prior, kind)
    posterior = GammaParams(
        prior.shape + get(outcome.home_counts, key, 0.0) +
            get(outcome.away_counts, key, 0.0),
        prior.rate + get(outcome.away_exposure, key, 0.0) +
            multiplier * get(outcome.home_exposure, key, 0.0),
    )
    return home ? GammaParams(posterior.shape, posterior.rate / multiplier) : posterior
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
    return posterior.shape / posterior.rate
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
    posteriors = GammaParams[]
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
The Gamma posteriors are transformed exactly to log-hazard moments:
`E[log(lambda)] = digamma(shape) - log(rate)` and
`Var(log(lambda)) = trigamma(shape)`.
"""
function hazard_theta(model::HazardModel, home_team, away_team)
    posteriors, posterior_keys, labels =
        _matchup_theta_posteriors(model, home_team, away_team)
    log_mean = [
        SpecialFunctions.digamma(posterior.shape) - log(posterior.rate)
        for posterior in posteriors
    ]
    covariance = zeros(Float64, length(posteriors), length(posteriors))
    for i in eachindex(posteriors), j in eachindex(posteriors)
        posterior_keys[i] == posterior_keys[j] || continue
        covariance[i, j] = SpecialFunctions.trigamma(posteriors[i].shape)
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

# ----------------------------------------------------------------------
# 6. Renewal-reward game-level aggregation
# ----------------------------------------------------------------------

"""
    GAME_CLOCK_SECONDS

Total regulation game clock, in seconds.
"""
const GAME_CLOCK_SECONDS = 3600.0

function _game_metrics_from_moments(
    home_moments::DriveMoments,
    away_moments::DriveMoments;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    mean_Tc = home_moments.mean_T + away_moments.mean_T
    var_Tc = home_moments.var_T + away_moments.var_T
    mean_Rc = home_moments.mean_S - away_moments.mean_S
    var_Rc = home_moments.var_S + away_moments.var_S
    cov_TcRc = home_moments.cov_TS - away_moments.cov_TS

    rate = mean_Rc / mean_Tc
    mean_spread = horizon * rate
    var_rate = (var_Rc - 2 * rate * cov_TcRc + rate^2 * var_Tc) / mean_Tc
    spread_variance = horizon * var_rate
    win_probability = (
        1 + SpecialFunctions.erf(mean_spread / sqrt(2 * spread_variance))
    ) / 2

    return (;
        mean_spread,
        spread_variance,
        win_probability,
    )
end

function _game_metrics_from_theta(
    theta::AbstractVector{<:Real},
    edges::AbstractVector{<:Real},
    marks::ScoreMarks;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    n_bins = length(edges) - 1
    expected_length = 4 * n_bins
    length(theta) == expected_length ||
        throw(ArgumentError("theta must contain four hazard blocks per time bin"))

    home_td = exp.(theta[1:n_bins])
    away_defensive = exp.(theta[(n_bins + 1):(2 * n_bins)])
    away_td = exp.(theta[(2 * n_bins + 1):(3 * n_bins)])
    home_defensive = exp.(theta[(3 * n_bins + 1):(4 * n_bins)])

    home_moments = _drive_moments_from_hazards(
        edges,
        marks,
        home_td,
        away_defensive,
    )
    away_moments = _drive_moments_from_hazards(
        edges,
        marks,
        away_td,
        home_defensive,
    )
    return _game_metrics_from_moments(home_moments, away_moments; horizon=horizon)
end

"""
    game_spread_distribution(home_moments, away_moments; horizon=GAME_CLOCK_SECONDS)
        -> Distributions.Normal

Approximate the final home-minus-away score spread using the renewal-reward
central limit theorem.
"""
function game_spread_distribution(
    home_moments::DriveMoments,
    away_moments::DriveMoments;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    metrics = _game_metrics_from_moments(
        home_moments,
        away_moments;
        horizon=horizon,
    )
    return Normal(metrics.mean_spread, sqrt(metrics.spread_variance))
end

"""
    ExpectedGameMetrics

Posterior expected game metrics after propagating matchup hazard uncertainty.
`predictive_spread_variance` includes both conditional game variance and
between-hazard-posterior variance in the conditional spread mean.
"""
struct ExpectedGameMetrics
    expected_spread::Float64
    expected_win_probability::Float64
    predictive_spread_variance::Float64
end

function _trace_product(
    left::AbstractMatrix{<:Real},
    right::AbstractMatrix{<:Real},
)
    size(left) == size(right) ||
        throw(ArgumentError("matrix dimensions must match"))
    return sum(
        left[i, j] * right[j, i]
        for i in axes(left, 1), j in axes(left, 2)
    )
end

function _quadratic_form(
    gradient::AbstractVector{<:Real},
    covariance::AbstractMatrix{<:Real},
)
    size(covariance) == (length(gradient), length(gradient)) ||
        throw(ArgumentError("covariance dimensions must match gradient length"))
    return sum(
        gradient[i] * covariance[i, j] * gradient[j]
        for i in eachindex(gradient), j in eachindex(gradient)
    )
end

function _second_order_expectation(
    function_value,
    theta_mean::Vector{Float64},
    covariance::Matrix{Float64},
)
    hessian = ForwardDiff.hessian(function_value, theta_mean)
    return function_value(theta_mean) + 0.5 * _trace_product(hessian, covariance)
end

"""
    expected_game_metrics(
        model,
        marks,
        home_team,
        away_team;
        horizon=GAME_CLOCK_SECONDS,
    ) -> ExpectedGameMetrics

Approximate posterior expected spread and home win probability using a
second-order delta method over the matchup's log-hazard posterior. The
posterior-predictive spread variance uses the law of total variance, with a
second-order approximation for expected conditional variance and a
first-order approximation for the variance of the conditional spread mean.
Score marks, empirical-Bayes hyperparameters, and fitted home multipliers are
treated as fixed.
"""
function expected_game_metrics(
    model::HazardModel,
    marks::ScoreMarks,
    home_team,
    away_team;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    theta = hazard_theta(model, home_team, away_team)
    spread_function = theta_vector ->
        _game_metrics_from_theta(
            theta_vector,
            model.time_edges,
            marks;
            horizon=horizon,
        ).mean_spread
    variance_function = theta_vector ->
        _game_metrics_from_theta(
            theta_vector,
            model.time_edges,
            marks;
            horizon=horizon,
        ).spread_variance
    win_function = theta_vector ->
        _game_metrics_from_theta(
            theta_vector,
            model.time_edges,
            marks;
            horizon=horizon,
        ).win_probability

    expected_spread = _second_order_expectation(
        spread_function,
        theta.log_mean,
        theta.covariance,
    )
    expected_win_probability = _second_order_expectation(
        win_function,
        theta.log_mean,
        theta.covariance,
    )
    expected_conditional_variance = _second_order_expectation(
        variance_function,
        theta.log_mean,
        theta.covariance,
    )
    spread_gradient = ForwardDiff.gradient(spread_function, theta.log_mean)
    parameter_spread_variance = _quadratic_form(
        spread_gradient,
        theta.covariance,
    )
    predictive_spread_variance =
        expected_conditional_variance + parameter_spread_variance

    return ExpectedGameMetrics(
        expected_spread,
        expected_win_probability,
        predictive_spread_variance,
    )
end
