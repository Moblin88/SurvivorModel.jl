const RESET_GAMMA_SHAPE_LOWER = 0.05
const RESET_GAMMA_SHAPE_UPPER = 1.0e6

_reset_probability_from_logit(raw::Real) = inv(1.0 + exp(-Float64(raw)))

function _reset_logit_probability(probability::Float64)
    probability <= 0.0 && return -12.0
    probability >= 1.0 && return 12.0
    return log(probability / (1.0 - probability))
end

function _logsumexp(values)
    isempty(values) && throw(ArgumentError("logsumexp requires at least one value"))
    maximum_value = maximum(values)
    maximum_value == -Inf && return -Inf
    isfinite(maximum_value) || return maximum_value
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function _log_gamma_increment(shape::Float64, count::Int)
    count == 0 && return 0.0
    count <= 64 &&
        return sum(log(shape + offset) for offset in 0:(count - 1))
    return SpecialFunctions.loggamma(shape + count) -
        SpecialFunctions.loggamma(shape)
end

function _validated_gamma_rate_cell(count::Real, exposure::Real)
    count_value = Float64(count)
    exposure_value = Float64(exposure)
    isfinite(count_value) && count_value >= 0.0 ||
        throw(ArgumentError("event counts must be finite and nonnegative"))
    isfinite(exposure_value) && exposure_value >= 0.0 ||
        throw(ArgumentError("transformed exposures must be finite and nonnegative"))
    count_integer = round(Int, count_value)
    isapprox(count_value, count_integer; atol=1e-10) ||
        throw(ArgumentError("event counts must be integer-valued"))
    exposure_value == 0.0 && count_integer > 0 &&
        throw(ArgumentError("positive event counts require positive exposure"))
    return count_integer, exposure_value
end

function _log_gamma_process_marginal(
    component::GammaParams,
    count::Real,
    exposure::Real,
)
    count_integer, exposure_value =
        _validated_gamma_rate_cell(count, exposure)
    count_integer == 0 && exposure_value == 0.0 && return 0.0
    return _log_gamma_increment(component.shape, count_integer) +
        component.shape * log(component.rate) -
        (component.shape + count_integer) * log(component.rate + exposure_value)
end

function _transition_gamma_mixture(
    mixture::GammaMixture,
    persistence::Real,
    fresh_component::GammaParams,
    source_season::Integer,
)
    persistence_value = Float64(persistence)
    isfinite(persistence_value) &&
        0.0 <= persistence_value <= 1.0 ||
        throw(ArgumentError("persistence must be finite and in [0, 1]"))
    return GammaMixture(
        vcat(persistence_value .* mixture.weights, 1.0 - persistence_value),
        vcat(mixture.components, [fresh_component]);
        source_seasons=vcat(mixture.source_seasons, Int(source_season)),
    )
end

mutable struct _RawTeamSeasonCell
    away_durations::Vector{Float64}
    home_durations::Vector{Float64}
    event_count::Int
    home_event_count::Int
    event_log_duration::Float64
end

_RawTeamSeasonCell() =
    _RawTeamSeasonCell(Float64[], Float64[], 0, 0, 0.0)

function _cause_raw_cells(data::AbstractDataFrame, kind::Symbol)
    _validate_cause(kind)
    cells = Dict{String,Dict{Int,_RawTeamSeasonCell}}()
    event_kind = kind === :td ? :td : :defensive
    for row in eachrow(data)
        team = kind === :td ? row.posteam : row.defteam
        is_home = kind === :td ? row.posteam_home : row.defteam_home
        season = ismissing(row.season) ? 0 : Int(row.season)
        season_cells = get!(cells, team, Dict{Int,_RawTeamSeasonCell}())
        cell = get!(season_cells, season, _RawTeamSeasonCell())
        push!(is_home ? cell.home_durations : cell.away_durations, row.duration)
        if row.event === event_kind
            cell.event_count += 1
            is_home && (cell.home_event_count += 1)
            cell.event_log_duration += log(row.duration)
        end
    end
    return cells
end

function _historical_seasons(
    cells::Dict{String,Dict{Int,_RawTeamSeasonCell}},
    max_seasons::Int,
)
    max_seasons > 0 ||
        throw(ArgumentError("max_seasons must be positive"))
    seasons = sort!(unique(
        season
        for team_cells in values(cells)
        for season in keys(team_cells)
        if season != 0
    ))
    if isempty(seasons)
        return any(haskey(team_cells, 0) for team_cells in values(cells)) ?
            [0] : Int[]
    end
    return seasons[max(1, length(seasons) - max_seasons + 1):end]
end

function _cell_exposures(cell::_RawTeamSeasonCell, shape::Float64)
    away_exposure = sum(
        duration^shape for duration in cell.away_durations;
        init=0.0,
    )
    home_exposure = sum(
        duration^shape for duration in cell.home_durations;
        init=0.0,
    )
    return away_exposure, home_exposure
end

function _cause_season_likelihood(
    cells::Dict{String,Dict{Int,_RawTeamSeasonCell}},
    seasons::Vector{Int},
    hyperparameter::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    weibull_shape_value::Float64,
)
    log_likelihood = 0.0
    for team in sort!(collect(keys(cells)))
        team_cells = cells[team]
        mixture = GammaMixture(
            [1.0],
            [hyperparameter];
            source_seasons=[first(seasons)],
        )
        for (season_index, season) in enumerate(seasons)
            season_index > 1 &&
                (mixture = _transition_gamma_mixture(
                    mixture,
                    persistence,
                    hyperparameter,
                    season,
                ))
            cell = get(team_cells, season, _RawTeamSeasonCell())
            away_exposure, home_exposure = _cell_exposures(
                cell,
                weibull_shape_value,
            )
            exposure = away_exposure + home_multiplier_value * home_exposure
            base_event_log_likelihood =
                cell.event_count * log(weibull_shape_value) +
                (weibull_shape_value - 1.0) * cell.event_log_duration +
                cell.home_event_count * log(home_multiplier_value)
            component_log_weights = [
                log(weight) + _log_gamma_process_marginal(
                    component,
                    cell.event_count,
                    exposure,
                )
                for (weight, component) in zip(
                    mixture.weights,
                    mixture.components,
                )
            ]
            evidence = _logsumexp(component_log_weights)
            isfinite(evidence) ||
                throw(ArgumentError("Weibull reset likelihood has no finite component"))
            log_likelihood += base_event_log_likelihood + evidence
            mixture = _update_gamma_mixture(
                mixture,
                cell.event_count,
                exposure,
            )
        end
    end
    return log_likelihood
end

function _aggregate_cause_exposures(
    cells::Dict{String,Dict{Int,_RawTeamSeasonCell}},
    shape::Float64,
)
    home_events = 0.0
    away_events = 0.0
    home_exposure = 0.0
    away_exposure = 0.0
    total_events = 0
    for team_cells in values(cells), cell in values(team_cells)
        away_value, home_value = _cell_exposures(cell, shape)
        home_events += cell.home_event_count
        away_events += cell.event_count - cell.home_event_count
        home_exposure += home_value
        away_exposure += away_value
        total_events += cell.event_count
    end
    return (
        home_events=home_events,
        away_events=away_events,
        home_exposure=home_exposure,
        away_exposure=away_exposure,
        total_events=total_events,
    )
end

function _fit_cause_parameters(
    cells::Dict{String,Dict{Int,_RawTeamSeasonCell}},
    seasons::Vector{Int},
    kind::Symbol,
)
    isempty(seasons) &&
        throw(ArgumentError("historical likelihood requires at least one season"))
    aggregate = _aggregate_cause_exposures(
        cells,
        kind === :td ? 2.0 : 1.5,
    )
    aggregate.total_events > 0 ||
        throw(ArgumentError("$kind historical data contain no observed events"))
    home_multiplier_initial = if aggregate.home_events > 0.0 &&
        aggregate.away_events > 0.0 &&
        aggregate.home_exposure > 0.0 &&
        aggregate.away_exposure > 0.0
        clamp(
            (aggregate.home_events / aggregate.home_exposure) /
                (aggregate.away_events / aggregate.away_exposure),
            0.25,
            4.0,
        )
    else
        1.0
    end

    function initial_for_shape(shape)
        totals = _aggregate_cause_exposures(cells, shape)
        exposure = totals.away_exposure +
            home_multiplier_initial * totals.home_exposure
        mean_rate = max(totals.total_events / max(exposure, eps(Float64)), 1e-10)
        return [
            log(mean_rate),
            log(4.0),
            0.0,
            log(home_multiplier_initial),
            log(shape),
        ]
    end

    lower = [
        log(1e-10),
        log(RESET_GAMMA_SHAPE_LOWER),
        -12.0,
        log(0.05),
        log(0.2),
    ]
    upper = [
        log(100.0),
        log(RESET_GAMMA_SHAPE_UPPER),
        12.0,
        log(20.0),
        log(5.0),
    ]
    function unpack(values)
        mean_rate = exp(values[1])
        gamma_shape = exp(values[2])
        persistence = _reset_probability_from_logit(values[3])
        home_multiplier_value = exp(values[4])
        shape = exp(values[5])
        return (
            hyperparameter=GammaParams(gamma_shape, gamma_shape / mean_rate),
            persistence=persistence,
            home_multiplier=home_multiplier_value,
            shape=shape,
        )
    end
    objective = values -> begin
        parameters = unpack(values)
        likelihood = _cause_season_likelihood(
            cells,
            seasons,
            parameters.hyperparameter,
            parameters.home_multiplier,
            parameters.persistence,
            parameters.shape,
        )
        return isfinite(likelihood) ? -likelihood : floatmax(Float64)
    end

    initial_shapes = kind === :td ? (2.2, 1.0) : (1.4, 1.0)
    options = Optim.Options(
        iterations=1200,
        f_reltol=1e-9,
        x_reltol=1e-7,
        show_trace=false,
        show_warnings=false,
    )
    results = [
        Optim.optimize(
            objective,
            lower,
            upper,
            initial_for_shape(shape),
            Optim.Fminbox(Optim.NelderMead()),
            options,
        )
        for shape in initial_shapes
    ]
    converged_results = filter(Optim.converged, results)
    isempty(converged_results) &&
        throw(ArgumentError("$kind Weibull empirical-Bayes fit did not converge"))
    result = converged_results[argmin(Optim.minimum.(converged_results))]
    parameters = unpack(Optim.minimizer(result))
    boundary_parameters = Symbol[]
    fitted_values = Optim.minimizer(result)
    fitted_values[1] <= lower[1] + 1e-5 &&
        push!(boundary_parameters, :mean_rate_lower)
    fitted_values[1] >= upper[1] - 1e-5 &&
        push!(boundary_parameters, :mean_rate_upper)
    fitted_values[2] <= lower[2] + 1e-5 &&
        push!(boundary_parameters, :gamma_shape_lower)
    fitted_values[2] >= upper[2] - 1e-5 &&
        push!(boundary_parameters, :gamma_shape_upper)
    fitted_values[3] <= lower[3] + 1e-5 &&
        push!(boundary_parameters, :persistence_lower)
    fitted_values[3] >= upper[3] - 1e-5 &&
        push!(boundary_parameters, :persistence_upper)
    fitted_values[4] <= lower[4] + 1e-5 &&
        push!(boundary_parameters, :home_multiplier_lower)
    fitted_values[4] >= upper[4] - 1e-5 &&
        push!(boundary_parameters, :home_multiplier_upper)
    fitted_values[5] <= lower[5] + 1e-5 &&
        push!(boundary_parameters, :weibull_shape_lower)
    fitted_values[5] >= upper[5] - 1e-5 &&
        push!(boundary_parameters, :weibull_shape_upper)
    diagnostics = LikelihoodFitDiagnostics(
        -Optim.minimum(result),
        true,
        Optim.iterations(result),
        Optim.f_calls(result),
        :converged,
        boundary_parameters,
    )
    return (
        shape=parameters.shape,
        hyperparameter=parameters.hyperparameter,
        home_multiplier=parameters.home_multiplier,
        persistence=parameters.persistence,
        diagnostics=diagnostics,
    )
end

function _build_reset_team_mixtures(
    cells::Dict{String,Dict{Int,_RawTeamSeasonCell}},
    seasons::Vector{Int},
    reference::Int,
    hyperparameter::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    shape::Float64,
)
    mixtures = Dict{String,GammaMixture}()
    for team in sort!(collect(keys(cells)))
        mixture = GammaMixture(
            [1.0],
            [hyperparameter];
            source_seasons=[first(seasons)],
        )
        for (season_index, season) in enumerate(seasons)
            season_index > 1 &&
                (mixture = _transition_gamma_mixture(
                    mixture,
                    persistence,
                    hyperparameter,
                    season,
                ))
            cell = get(cells[team], season, _RawTeamSeasonCell())
            away_exposure, home_exposure = _cell_exposures(cell, shape)
            mixture = _update_gamma_mixture(
                mixture,
                cell.event_count,
                away_exposure + home_multiplier_value * home_exposure,
            )
        end
        mixtures[team] = _transition_gamma_mixture(
            mixture,
            persistence,
            hyperparameter,
            reference,
        )
    end
    return mixtures
end

function _log_historical_prior_diagnostics(
    prior::HazardPrior;
    source::Symbol,
    cache_path::Union{Nothing,AbstractString}=nothing,
)
    @debug(
        "historical empirical-Bayes prior summary",
        source=source,
        cache_path=cache_path,
        historical_seasons=copy(prior.historical_seasons),
        team_counts=(
            touchdown=length(prior.td_team_mixtures),
            defensive=length(prior.defensive_team_mixtures),
        ),
        duration_unit=:minutes,
    )

    for cause in (:td, :defensive)
        hyperparameter = cause === :td ?
            prior.td_hyperparameters :
            prior.defensive_hyperparameters
        diagnostics = likelihood_fit_diagnostics(prior, cause)
        persistence_probability = hazard_persistence(prior, cause)
        @debug(
            "historical empirical-Bayes prior cause diagnostics",
            cause=cause,
            source=source,
            diagnostics_source=source === :cache ? :stored : :fresh_fit,
            cache_path=cache_path,
            weibull_shape=weibull_shape(prior, cause),
            gamma_shape=hyperparameter.shape,
            gamma_rate=hyperparameter.rate,
            gamma_mean_cumulative_hazard=hyperparameter.shape / hyperparameter.rate,
            home_multiplier=home_multiplier(prior, cause),
            persistence_probability=persistence_probability,
            reset_probability=1.0 - persistence_probability,
            diagnostics_available=diagnostics !== nothing,
            log_likelihood=diagnostics === nothing ?
                nothing : diagnostics.log_likelihood,
            converged=diagnostics === nothing ?
                nothing : diagnostics.converged,
            status=diagnostics === nothing ? nothing : diagnostics.status,
            iterations=diagnostics === nothing ?
                nothing : diagnostics.iterations,
            function_evaluations=diagnostics === nothing ?
                nothing : diagnostics.function_evaluations,
            boundary_parameters=diagnostics === nothing ?
                nothing : copy(diagnostics.boundary_parameters),
        )
    end
    return nothing
end

"""
    fit_empirical_bayes_prior(historical_drives; kwargs...) -> HazardPrior

Fit separate Weibull shapes, Gamma cumulative-hazard rate priors,
home multipliers, and season-reset probabilities for touchdown and
defensive-event processes. Team-season rates are integrated with the exact
finite-mixture reset filter.
"""
function fit_empirical_bayes_prior(
    historical_drives::AbstractDataFrame;
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    current_season::Union{Nothing,Integer}=nothing,
    method::WeibullEmpiricalBayesFit=DEFAULT_PRIOR_FIT_METHOD,
)
    max_seasons > 0 ||
        throw(ArgumentError("max_seasons must be positive"))
    data = build_drive_data(historical_drives)
    isempty(data) &&
        throw(ArgumentError("historical data contain no usable drives"))

    td_cells = _cause_raw_cells(data, :td)
    defensive_cells = _cause_raw_cells(data, :defensive)
    seasons = _historical_seasons(td_cells, max_seasons)
    isempty(seasons) &&
        throw(ArgumentError("historical data contain no usable seasons"))
    reference = if current_season === nothing
        first(seasons) == 0 ? 0 : maximum(seasons) + 1
    else
        Int(current_season)
    end
    observed_seasons = filter(!=(0), seasons)
    !isempty(observed_seasons) && reference <= maximum(observed_seasons) &&
        throw(ArgumentError("current_season must follow historical seasons"))
    method isa WeibullEmpiricalBayesFit ||
        throw(ArgumentError("unsupported empirical-Bayes method"))

    td_fit = _fit_cause_parameters(td_cells, seasons, :td)
    defensive_fit = _fit_cause_parameters(defensive_cells, seasons, :defensive)
    td_mixtures = _build_reset_team_mixtures(
        td_cells,
        seasons,
        reference,
        td_fit.hyperparameter,
        td_fit.home_multiplier,
        td_fit.persistence,
        td_fit.shape,
    )
    defensive_mixtures = _build_reset_team_mixtures(
        defensive_cells,
        seasons,
        reference,
        defensive_fit.hyperparameter,
        defensive_fit.home_multiplier,
        defensive_fit.persistence,
        defensive_fit.shape,
    )
    prior = HazardPrior(
        td_fit.shape,
        defensive_fit.shape,
        td_fit.hyperparameter,
        defensive_fit.hyperparameter,
        td_mixtures,
        defensive_mixtures,
        td_fit.home_multiplier,
        defensive_fit.home_multiplier,
        td_fit.persistence,
        defensive_fit.persistence,
        seasons,
        td_fit.diagnostics,
        defensive_fit.diagnostics,
    )
    _log_historical_prior_diagnostics(prior; source=:fit)
    return prior
end
