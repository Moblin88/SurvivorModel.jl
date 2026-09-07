"""
Piecewise-constant two-outcome drive model.

Each drive is modeled as a race between an offensive touchdown hazard and a
defensive-event hazard. The hazards are independent conditional on their
team-specific rates, piecewise constant in elapsed time since drive start,
and do not depend on field position.

Team-specific hazards use Gamma posteriors. A historical empirical-Bayes prior
can be fitted from the previous three seasons and updated with current-season
exposure and event counts as data arrive. Historical hyperparameters are fit
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
const MAX_HISTORICAL_SEASONS = 3
const RESET_EM_MAX_ITERATIONS = 100
const RESET_EM_ABSOLUTE_TOLERANCE = 1.0e-8
const RESET_EM_RELATIVE_TOLERANCE = 1.0e-8
const RESET_GAMMA_SHAPE_LOWER = 0.05
const RESET_GAMMA_SHAPE_UPPER = 1.0e8

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

function _logsumexp(values)
    isempty(values) && throw(ArgumentError("logsumexp requires at least one value"))
    maximum_value = maximum(values)
    maximum_value == -Inf && return -Inf
    isfinite(maximum_value) || return maximum_value
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function _log_gamma_event_marginal(
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

    shape = component.shape
    rate = component.rate
    return shape * log(rate) -
        SpecialFunctions.loggamma(shape) +
        SpecialFunctions.loggamma(shape + count_integer) -
        (shape + count_integer) * log(rate + exposure_value)
end

function _unpack_reset_parameters(
    values::AbstractVector{<:Real},
    n_bins::Int,
)
    length(values) == 2 * n_bins + 2 ||
        throw(ArgumentError("invalid reset parameter vector length"))
    means = exp.(Float64.(values[1:n_bins]))
    shapes = exp.(Float64.(values[(n_bins + 1):(2 * n_bins)]))
    hyperparameters = [
        begin
            shape = shapes[index]
            GammaParams(shape, shape / means[index])
        end
        for index in 1:n_bins
    ]
    persistence_raw = Float64(values[2 * n_bins + 1])
    persistence = persistence_raw >= 0.0 ?
        1.0 / (1.0 + exp(-persistence_raw)) :
        exp(persistence_raw) / (1.0 + exp(persistence_raw))
    home_multiplier_value = exp(Float64(values[2 * n_bins + 2]))
    return hyperparameters, home_multiplier_value, persistence
end

function _reset_likelihood_cells(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
)
    teams = Set{String}()
    for season in seasons
        outcome = _outcome_stats(byseason[Int(season)], kind)
        for key in keys(outcome.exposure)
            key[2] <= n_bins || continue
            push!(teams, key[1])
        end
    end

    cells = NamedTuple[]
    for team in sort!(collect(teams))
        for time_bin in 1:n_bins
            counts = Float64[]
            home_counts = Float64[]
            away_exposures = Float64[]
            home_exposures = Float64[]
            for season in seasons
                cell = _team_season_cell(
                    byseason,
                    season,
                    kind,
                    team,
                    time_bin,
                )
                push!(counts, cell.count)
                push!(home_counts, cell.home_count)
                push!(away_exposures, cell.exposure)
                push!(home_exposures, cell.home_exposure)
            end
            push!(
                cells,
                (
                    time_bin=time_bin,
                    counts=Tuple(counts),
                    home_counts=Tuple(home_counts),
                    away_exposures=Tuple(away_exposures),
                    home_exposures=Tuple(home_exposures),
                ),
            )
        end
    end
    return cells
end

function _reset_partition_paths(n_seasons::Int)
    n_seasons == 1 && return (
        (groups=((1, 1),), persistent_links=0),
    )
    n_seasons == 2 && return (
        (groups=((1, 1), (2, 2)), persistent_links=0),
        (groups=((1, 2),), persistent_links=1),
    )
    n_seasons == 3 && return (
        (
            groups=((1, 1), (2, 2), (3, 3)),
            persistent_links=0,
        ),
        (
            groups=((1, 2), (3, 3)),
            persistent_links=1,
        ),
        (
            groups=((1, 1), (2, 3)),
            persistent_links=1,
        ),
        (
            groups=((1, 3),),
            persistent_links=2,
        ),
    )
    throw(ArgumentError("reset likelihood supports one to three seasons"))
end

function _reset_path_log_probability(
    path,
    n_seasons::Int,
    persistence::Float64,
)
    reset_links = n_seasons - 1 - path.persistent_links
    log_probability = 0.0
    if path.persistent_links > 0
        persistence > 0.0 || return -Inf
        log_probability += path.persistent_links * log(persistence)
    end
    if reset_links > 0
        persistence < 1.0 || return -Inf
        log_probability += reset_links * log1p(-persistence)
    end
    return log_probability
end

function _log_reset_group_marginal(
    component::GammaParams,
    counts::Tuple,
    away_exposures::Tuple,
    home_exposures::Tuple,
    home_multiplier_value::Float64,
    first_season::Int,
    last_season::Int,
)
    count = 0.0
    exposure = 0.0
    for season in first_season:last_season
        count += counts[season]
        exposure += away_exposures[season] +
            home_multiplier_value * home_exposures[season]
    end
    return _log_gamma_event_marginal(component, count, exposure)
end

function _log_reset_partition_marginal(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    n_seasons = length(cell.counts)
    1 <= n_seasons <= 3 ||
        throw(ArgumentError("reset likelihood supports one to three seasons"))

    log_home_events = sum(cell.home_counts) * log(home_multiplier_value)
    log_terms = Float64[]
    for path in _reset_partition_paths(n_seasons)
        log_term = _reset_path_log_probability(
            path,
            n_seasons,
            persistence,
        )
        for (first_season, last_season) in path.groups
            log_term += _log_reset_group_marginal(
                component,
                cell.counts,
                cell.away_exposures,
                cell.home_exposures,
                home_multiplier_value,
                first_season,
                last_season,
            )
        end
        push!(log_terms, log_term)
    end
    return log_home_events + _logsumexp(log_terms)
end

function _reset_event_log_likelihood(
    cells::AbstractVector,
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Real,
    persistence::Real,
)
    isempty(cells) && return 0.0
    home_multiplier_float = Float64(home_multiplier_value)
    persistence_float = Float64(persistence)
    isfinite(home_multiplier_float) && home_multiplier_float > 0.0 ||
        return -Inf
    isfinite(persistence_float) && 0.0 <= persistence_float <= 1.0 ||
        return -Inf

    log_likelihood = 0.0
    for cell in cells
        log_likelihood += _log_reset_partition_marginal(
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_float,
            persistence_float,
        )
    end
    return log_likelihood
end

function _reset_group_posterior_moments(
    component::GammaParams,
    counts::Tuple,
    away_exposures::Tuple,
    home_exposures::Tuple,
    home_multiplier_value::Float64,
    first_season::Int,
    last_season::Int,
)
    count = 0.0
    exposure = 0.0
    home_exposure = 0.0
    for season in first_season:last_season
        count += counts[season]
        exposure += away_exposures[season] +
            home_multiplier_value * home_exposures[season]
        home_exposure += home_exposures[season]
    end
    posterior_shape = component.shape + count
    posterior_rate = component.rate + exposure
    return (
        mean=posterior_shape / posterior_rate,
        log_mean=SpecialFunctions.digamma(posterior_shape) -
            log(posterior_rate),
        home_exposure=home_exposure,
    )
end

function _reset_em_expectations(
    cells::AbstractVector,
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Float64,
    persistence::Float64,
)
    n_bins = length(hyperparameters)
    expected_group_counts = zeros(Float64, n_bins)
    expected_log_lambdas = zeros(Float64, n_bins)
    expected_lambdas = zeros(Float64, n_bins)
    expected_home_lambda_exposures = zeros(Float64, n_bins)
    expected_persistent_links = 0.0
    total_transitions = 0.0
    total_home_events = 0.0
    function_evaluations = 0

    for cell in cells
        n_seasons = length(cell.counts)
        paths = _reset_partition_paths(n_seasons)
        log_terms = Float64[]
        for path in paths
            log_term = _reset_path_log_probability(
                path,
                n_seasons,
                persistence,
            )
            for (first_season, last_season) in path.groups
                log_term += _log_reset_group_marginal(
                    hyperparameters[cell.time_bin],
                    cell.counts,
                    cell.away_exposures,
                    cell.home_exposures,
                    home_multiplier_value,
                    first_season,
                    last_season,
                )
            end
            push!(log_terms, log_term)
        end
        log_normalizer = _logsumexp(log_terms)
        total_home_events += sum(cell.home_counts)
        total_transitions += n_seasons - 1
        function_evaluations += length(paths)

        for (path, log_term) in zip(paths, log_terms)
            posterior_weight = exp(log_term - log_normalizer)
            posterior_weight > 0.0 || continue
            expected_persistent_links +=
                posterior_weight * path.persistent_links
            expected_group_counts[cell.time_bin] +=
                posterior_weight * length(path.groups)
            for (first_season, last_season) in path.groups
                moments = _reset_group_posterior_moments(
                    hyperparameters[cell.time_bin],
                    cell.counts,
                    cell.away_exposures,
                    cell.home_exposures,
                    home_multiplier_value,
                    first_season,
                    last_season,
                )
                expected_lambdas[cell.time_bin] +=
                    posterior_weight * moments.mean
                expected_log_lambdas[cell.time_bin] +=
                    posterior_weight * moments.log_mean
                expected_home_lambda_exposures[cell.time_bin] +=
                    posterior_weight * moments.mean * moments.home_exposure
            end
        end
    end

    return (
        expected_group_counts=expected_group_counts,
        expected_log_lambdas=expected_log_lambdas,
        expected_lambdas=expected_lambdas,
        expected_home_lambda_exposures=expected_home_lambda_exposures,
        expected_persistent_links=expected_persistent_links,
        total_transitions=total_transitions,
        total_home_events=total_home_events,
        function_evaluations=function_evaluations,
    )
end

function _reset_gamma_shape_from_moments(
    mean_lambda::Float64,
    mean_log_lambda::Float64,
)
    isfinite(mean_lambda) && mean_lambda > 0.0 ||
        throw(ArgumentError("EM Gamma mean must be finite and positive"))
    isfinite(mean_log_lambda) ||
        throw(ArgumentError("EM Gamma log mean must be finite"))

    discrepancy = max(
        log(mean_lambda) - mean_log_lambda,
        0.0,
    )
    lower_log_shape = log(RESET_GAMMA_SHAPE_LOWER)
    upper_log_shape = log(RESET_GAMMA_SHAPE_UPPER)
    residual(log_shape) = begin
        shape = exp(log_shape)
        log(shape) - SpecialFunctions.digamma(shape) - discrepancy
    end

    lower_residual = residual(lower_log_shape)
    upper_residual = residual(upper_log_shape)
    lower_residual <= 0.0 && return RESET_GAMMA_SHAPE_LOWER
    upper_residual >= 0.0 && return RESET_GAMMA_SHAPE_UPPER

    for _ in 1:100
        midpoint = (lower_log_shape + upper_log_shape) / 2.0
        residual_value = residual(midpoint)
        if residual_value > 0.0
            lower_log_shape = midpoint
        else
            upper_log_shape = midpoint
        end
    end
    return exp((lower_log_shape + upper_log_shape) / 2.0)
end

function _reset_em_maximize(
    expectations,
    old_hyperparameters::AbstractVector{<:GammaParams},
    old_home_multiplier::Float64,
    old_persistence::Float64,
)
    n_bins = length(old_hyperparameters)
    hyperparameters = GammaParams[]
    for time_bin in 1:n_bins
        group_count = expectations.expected_group_counts[time_bin]
        group_count > 0.0 || begin
            push!(hyperparameters, old_hyperparameters[time_bin])
            continue
        end
        mean_lambda =
            expectations.expected_lambdas[time_bin] / group_count
        mean_log_lambda =
            expectations.expected_log_lambdas[time_bin] / group_count
        shape = _reset_gamma_shape_from_moments(
            mean_lambda,
            mean_log_lambda,
        )
        push!(
            hyperparameters,
            GammaParams(shape, shape / mean_lambda),
        )
    end

    home_denominator = sum(expectations.expected_home_lambda_exposures)
    home_multiplier_value = if home_denominator > 0.0
        clamp(
            expectations.total_home_events / home_denominator,
            0.05,
            20.0,
        )
    else
        old_home_multiplier
    end
    persistence = expectations.total_transitions > 0.0 ?
        clamp(
            expectations.expected_persistent_links /
                expectations.total_transitions,
            0.0,
            1.0,
        ) :
        old_persistence
    return hyperparameters, home_multiplier_value, persistence
end

function _reset_bin_event_log_likelihood(
    cells::AbstractVector,
    time_bin::Int,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    log_likelihood = 0.0
    for cell in cells
        cell.time_bin == time_bin || continue
        log_likelihood += _log_reset_partition_marginal(
            cell,
            component,
            home_multiplier_value,
            persistence,
        )
    end
    return log_likelihood
end

function _reset_logit_probability(probability::Float64)
    probability <= 0.0 && return -30.0
    probability >= 1.0 && return 30.0
    return log(probability / (1.0 - probability))
end

function _reset_optimize_bin_parameters(
    cells::AbstractVector,
    time_bin::Int,
    initial::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    objective = values -> begin
        mean_value = exp(values[1])
        shape_value = exp(values[2])
        parameter = GammaParams(shape_value, shape_value / mean_value)
        return -_reset_bin_event_log_likelihood(
            cells,
            time_bin,
            parameter,
            home_multiplier_value,
            persistence,
        )
    end
    lower = [log(1.0e-10), log(RESET_GAMMA_SHAPE_LOWER)]
    upper = [log(10.0), log(RESET_GAMMA_SHAPE_UPPER)]
    initial_values = [
        clamp(
            log(initial.shape / initial.rate),
            lower[1] + 1.0e-8,
            upper[1] - 1.0e-8,
        ),
        clamp(
            log(initial.shape),
            lower[2] + 1.0e-8,
            upper[2] - 1.0e-8,
        ),
    ]
    result = Optim.optimize(
        objective,
        lower,
        upper,
        initial_values,
        Optim.Fminbox(Optim.NelderMead()),
        Optim.Options(
            iterations=300,
            f_reltol=1.0e-8,
            x_reltol=1.0e-7,
            show_trace=false,
            show_warnings=false,
        ),
    )
    Optim.converged(result) ||
        throw(ArgumentError(
            "conditional Gamma likelihood optimization did not converge",
        ))
    fitted_values = Optim.minimizer(result)
    shape = exp(fitted_values[2])
    mean_value = exp(fitted_values[1])
    return (
        parameter=GammaParams(shape, shape / mean_value),
        function_evaluations=Optim.f_calls(result),
    )
end

function _reset_optimize_shared_parameters(
    cells::AbstractVector,
    hyperparameters::AbstractVector{<:GammaParams},
    initial_home_multiplier::Float64,
    initial_persistence::Float64,
)
    objective = values -> begin
        raw_persistence = values[1]
        persistence = raw_persistence >= 0.0 ?
            1.0 / (1.0 + exp(-raw_persistence)) :
            exp(raw_persistence) / (1.0 + exp(raw_persistence))
        home_multiplier_value = exp(values[2])
        return -_reset_event_log_likelihood(
            cells,
            hyperparameters,
            home_multiplier_value,
            persistence,
        )
    end
    lower = [-30.0, log(0.05)]
    upper = [30.0, log(20.0)]
    initial_values = [
        clamp(
            _reset_logit_probability(initial_persistence),
            lower[1] + 1.0e-8,
            upper[1] - 1.0e-8,
        ),
        clamp(
            log(initial_home_multiplier),
            lower[2] + 1.0e-8,
            upper[2] - 1.0e-8,
        ),
    ]
    result = Optim.optimize(
        objective,
        lower,
        upper,
        initial_values,
        Optim.Fminbox(Optim.NelderMead()),
        Optim.Options(
            iterations=300,
            f_reltol=1.0e-8,
            x_reltol=1.0e-7,
            show_trace=false,
            show_warnings=false,
        ),
    )
    Optim.converged(result) ||
        throw(ArgumentError(
            "conditional shared likelihood optimization did not converge",
        ))
    fitted_values = Optim.minimizer(result)
    raw_persistence = fitted_values[1]
    persistence = raw_persistence >= 0.0 ?
        1.0 / (1.0 + exp(-raw_persistence)) :
        exp(raw_persistence) / (1.0 + exp(raw_persistence))
    return (
        home_multiplier=exp(fitted_values[2]),
        persistence=persistence,
        function_evaluations=Optim.f_calls(result),
    )
end

# Apply observed-likelihood conditional maximization after the EM M-step.
# Each bin remains a two-parameter problem conditional on the shared values.
function _reset_ecme_update(
    cells::AbstractVector,
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Float64,
    persistence::Float64,
)
    updated_hyperparameters = GammaParams[]
    function_evaluations = 0
    for time_bin in eachindex(hyperparameters)
        result = _reset_optimize_bin_parameters(
            cells,
            time_bin,
            hyperparameters[time_bin],
            home_multiplier_value,
            persistence,
        )
        push!(updated_hyperparameters, result.parameter)
        function_evaluations += result.function_evaluations
    end
    shared = _reset_optimize_shared_parameters(
        cells,
        updated_hyperparameters,
        home_multiplier_value,
        persistence,
    )
    function_evaluations += shared.function_evaluations
    return (
        hyperparameters=updated_hyperparameters,
        home_multiplier=shared.home_multiplier,
        persistence=shared.persistence,
        function_evaluations=function_evaluations,
    )
end

function _reset_boundary_parameters(
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Float64,
    persistence::Float64,
)
    boundary_parameters = Symbol[]
    for (time_bin, parameter) in enumerate(hyperparameters)
        mean_value = parameter.shape / parameter.rate
        mean_value <= 1.0e-10 * 1.000001 &&
            push!(boundary_parameters, Symbol("mean_lower_$time_bin"))
        mean_value >= 10.0 / 1.000001 &&
            push!(boundary_parameters, Symbol("mean_upper_$time_bin"))
        parameter.shape <= RESET_GAMMA_SHAPE_LOWER * 1.000001 &&
            push!(boundary_parameters, Symbol("shape_lower_$time_bin"))
        parameter.shape >= RESET_GAMMA_SHAPE_UPPER / 1.000001 &&
            push!(boundary_parameters, Symbol("shape_upper_$time_bin"))
    end
    persistence <= 1.0e-6 &&
        push!(boundary_parameters, :persistence_lower)
    persistence >= 1.0 - 1.0e-6 &&
        push!(boundary_parameters, :persistence_upper)
    home_multiplier_value <= 0.050001 &&
        push!(boundary_parameters, :home_multiplier_lower)
    home_multiplier_value >= 19.999 &&
        push!(boundary_parameters, :home_multiplier_upper)
    return boundary_parameters
end

function _initial_home_multiplier(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
)
    home_count = 0.0
    away_count = 0.0
    home_exposure = 0.0
    away_exposure = 0.0
    for team in _teams(byseason, seasons)
        for season in seasons
            for time_bin in 1:n_bins
                cell = _team_season_cell(
                    byseason,
                    season,
                    kind,
                    team,
                    time_bin,
                )
                home_count += cell.home_count
                away_count += cell.away_count
                home_exposure += cell.home_exposure
                away_exposure += cell.exposure
            end
        end
    end
    home_rate = home_exposure > 0.0 ? home_count / home_exposure : 0.0
    away_rate = away_exposure > 0.0 ? away_count / away_exposure : 0.0
    if home_rate > 0.0 && away_rate > 0.0
        return clamp(home_rate / away_rate, 0.25, 4.0)
    end
    return 1.0
end

function _initial_reset_parameter_vector(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
)
    home_multiplier_value = _initial_home_multiplier(
        byseason,
        seasons,
        kind,
        n_bins,
    )
    means = Float64[]
    for time_bin in 1:n_bins
        total_count = 0.0
        total_exposure = 0.0
        for team in _teams(byseason, seasons)
            for season in seasons
                cell = _team_season_cell(
                    byseason,
                    season,
                    kind,
                    team,
                    time_bin,
                )
                total_count += cell.count
                total_exposure += cell.exposure +
                    home_multiplier_value * cell.home_exposure
            end
        end
        push!(
            means,
            max(total_count / max(total_exposure, eps(Float64)), 1.0e-8),
        )
    end
    return vcat(
        log.(means),
        fill(log(4.0), n_bins),
        [0.0, log(home_multiplier_value)],
    )
end

function _fit_reset_outcome_parameters_with_diagnostics(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    kind::Symbol,
)
    isempty(seasons) && throw(ArgumentError("historical likelihood requires seasons"))
    length(seasons) <= MAX_HISTORICAL_SEASONS ||
        throw(ArgumentError("historical likelihood supports at most three seasons"))
    n_bins = length(time_edges) - 1
    n_bins > 0 || throw(ArgumentError("historical likelihood requires time bins"))
    initial = _initial_reset_parameter_vector(
        byseason,
        seasons,
        kind,
        n_bins,
    )
    likelihood_cells = _reset_likelihood_cells(
        byseason,
        seasons,
        kind,
        n_bins,
    )
    isempty(likelihood_cells) &&
        throw(ArgumentError("$kind historical data contain no exposure cells"))
    hyperparameters, home_multiplier_value, persistence =
        _unpack_reset_parameters(initial, n_bins)
    log_likelihood = _reset_event_log_likelihood(
        likelihood_cells,
        hyperparameters,
        home_multiplier_value,
        persistence,
    )
    converged = false
    iterations = 0
    function_evaluations = 0

    for iteration in 1:RESET_EM_MAX_ITERATIONS
        expectations = _reset_em_expectations(
            likelihood_cells,
            hyperparameters,
            home_multiplier_value,
            persistence,
        )
        function_evaluations += expectations.function_evaluations
        em_hyperparameters, em_home_multiplier, em_persistence =
            _reset_em_maximize(
                expectations,
                hyperparameters,
                home_multiplier_value,
                persistence,
            )
        ecme = _reset_ecme_update(
            likelihood_cells,
            em_hyperparameters,
            em_home_multiplier,
            em_persistence,
        )
        function_evaluations += ecme.function_evaluations
        next_hyperparameters = ecme.hyperparameters
        next_home_multiplier = ecme.home_multiplier
        next_persistence = ecme.persistence
        next_log_likelihood = _reset_event_log_likelihood(
            likelihood_cells,
            next_hyperparameters,
            next_home_multiplier,
            next_persistence,
        )
        isfinite(next_log_likelihood) ||
            throw(ArgumentError(
                "$kind historical EM produced a non-finite likelihood",
            ))
        iterations = iteration
        if abs(next_log_likelihood - log_likelihood) <=
            RESET_EM_ABSOLUTE_TOLERANCE +
            RESET_EM_RELATIVE_TOLERANCE * max(1.0, abs(log_likelihood))
            hyperparameters = next_hyperparameters
            home_multiplier_value = next_home_multiplier
            persistence = next_persistence
            log_likelihood = next_log_likelihood
            converged = true
            break
        end
        hyperparameters = next_hyperparameters
        home_multiplier_value = next_home_multiplier
        persistence = next_persistence
        log_likelihood = next_log_likelihood
    end

    converged ||
        throw(ArgumentError(
            "$kind historical EM did not converge after " *
            "$RESET_EM_MAX_ITERATIONS iterations",
        ))
    boundary_parameters = _reset_boundary_parameters(
        hyperparameters,
        home_multiplier_value,
        persistence,
    )
    diagnostics = LikelihoodFitDiagnostics(
        log_likelihood,
        true,
        iterations,
        function_evaluations,
        :converged,
        boundary_parameters,
    )
    return hyperparameters, home_multiplier_value, persistence, diagnostics
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

function _season_cells(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    time_bin::Int,
)
    cells = NamedTuple[]
    for season in seasons
        stats = byseason[Int(season)]
        outcome = _outcome_stats(stats, kind)
        teams = Set{String}()
        for key in keys(outcome.home_exposure)
            key[2] == time_bin || continue
            push!(teams, key[1])
        end
        for key in keys(outcome.away_exposure)
            key[2] == time_bin || continue
            push!(teams, key[1])
        end
        for team in teams
            key = (team, time_bin)
            home_exposure = get(outcome.home_exposure, key, 0.0)
            away_exposure = get(outcome.away_exposure, key, 0.0)
            home_exposure + away_exposure > 0.0 || continue
            push!(
                cells,
                (
                    season=Int(season),
                    team=team,
                    home_count=get(outcome.home_counts, key, 0.0),
                    away_count=get(outcome.away_counts, key, 0.0),
                    home_exposure=home_exposure,
                    away_exposure=away_exposure,
                ),
            )
        end
    end
    return cells
end

function _sequential_cells(
    cells::AbstractVector,
    seasons::AbstractVector{<:Integer},
)
    by_key = Dict{Tuple{String,Int},Any}()
    for cell in cells
        by_key[(cell.team, cell.season)] = cell
    end

    pairs = NamedTuple[]
    for index in 1:(length(seasons) - 1)
        previous_season = Int(seasons[index])
        current_season = Int(seasons[index + 1])
        teams = intersect(
            Set(cell.team for cell in cells if cell.season == previous_season),
            Set(cell.team for cell in cells if cell.season == current_season),
        )
        for team in teams
            previous = get(by_key, (team, previous_season), nothing)
            current = get(by_key, (team, current_season), nothing)
            if isnothing(previous) || isnothing(current)
                continue
            end
            push!(pairs, (previous=previous, current=current))
        end
    end
    return pairs
end

function _mean_or_nothing(values::AbstractVector{<:Real})
    isempty(values) && return nothing
    result = mean(values)
    return isfinite(result) ? Float64(result) : nothing
end

function _moment_summary(cells::AbstractVector, pairs::AbstractVector)
    away_rates = Float64[]
    home_rates = Float64[]
    away_factorials = Float64[]
    home_factorials = Float64[]
    same_season_products = Float64[]
    for cell in cells
        if cell.away_exposure > 0.0
            push!(away_rates, cell.away_count / cell.away_exposure)
            push!(
                away_factorials,
                cell.away_count * (cell.away_count - 1.0) /
                    cell.away_exposure^2,
            )
        end
        if cell.home_exposure > 0.0
            push!(home_rates, cell.home_count / cell.home_exposure)
            push!(
                home_factorials,
                cell.home_count * (cell.home_count - 1.0) /
                    cell.home_exposure^2,
            )
        end
        if cell.home_exposure > 0.0 && cell.away_exposure > 0.0
            push!(
                same_season_products,
                cell.home_count * cell.away_count /
                    (cell.home_exposure * cell.away_exposure),
            )
        end
    end

    sequential_away_away = Float64[]
    sequential_away_home = Float64[]
    sequential_home_away = Float64[]
    sequential_home_home = Float64[]
    for pair in pairs
        previous, current = pair.previous, pair.current
        if previous.away_exposure > 0.0 && current.away_exposure > 0.0
            push!(
                sequential_away_away,
                (previous.away_count / previous.away_exposure) *
                (current.away_count / current.away_exposure),
            )
        end
        if previous.away_exposure > 0.0 && current.home_exposure > 0.0
            push!(
                sequential_away_home,
                (previous.away_count / previous.away_exposure) *
                (current.home_count / current.home_exposure),
            )
        end
        if previous.home_exposure > 0.0 && current.away_exposure > 0.0
            push!(
                sequential_home_away,
                (previous.home_count / previous.home_exposure) *
                (current.away_count / current.away_exposure),
            )
        end
        if previous.home_exposure > 0.0 && current.home_exposure > 0.0
            push!(
                sequential_home_home,
                (previous.home_count / previous.home_exposure) *
                (current.home_count / current.home_exposure),
            )
        end
    end

    return (
        away_mean=_mean_or_nothing(away_rates),
        home_mean=_mean_or_nothing(home_rates),
        away_factorial=_mean_or_nothing(away_factorials),
        home_factorial=_mean_or_nothing(home_factorials),
        same_season_product=_mean_or_nothing(same_season_products),
        sequential_away_away=_mean_or_nothing(sequential_away_away),
        sequential_away_home=_mean_or_nothing(sequential_away_home),
        sequential_home_away=_mean_or_nothing(sequential_home_away),
        sequential_home_home=_mean_or_nothing(sequential_home_home),
        n_away_mean=length(away_rates),
        n_home_mean=length(home_rates),
        n_away_factorial=length(away_factorials),
        n_home_factorial=length(home_factorials),
        n_same_season_product=length(same_season_products),
        n_sequential_away_away=length(sequential_away_away),
        n_sequential_away_home=length(sequential_away_home),
        n_sequential_home_away=length(sequential_home_away),
        n_sequential_home_home=length(sequential_home_home),
    )
end

function _weighted_moment_projection(values, scales, counts)
    numerator = 0.0
    denominator = 0.0
    for index in eachindex(values)
        value = values[index]
        count = Float64(counts[index])
        scale = Float64(scales[index])
        isnothing(value) && continue
        isfinite(value) && count > 0.0 && isfinite(scale) ||
            continue
        numerator += count * scale * Float64(value)
        denominator += count * scale^2
    end
    denominator > 0.0 || return nothing
    return numerator / denominator
end

function _shared_moment_count(counts)
    valid_counts = [
        Float64(count) for count in counts if Float64(count) > 0.0
    ]
    isempty(valid_counts) && return 0.0
    return minimum(valid_counts)
end

function _season_moment_summaries(
    cells::AbstractVector,
    seasons::AbstractVector{<:Integer},
)
    summaries = NamedTuple[]
    for season in seasons
        season_cells = [
            cell for cell in cells if cell.season == Int(season)
        ]
        push!(
            summaries,
            (
                season=Int(season),
                moments=_moment_summary(season_cells, NamedTuple[]),
            ),
        )
    end
    return summaries
end

function _transition_moment_summaries(
    cells::AbstractVector,
    pairs::AbstractVector,
)
    pairs_by_transition =
        Dict{Tuple{Int,Int},Vector{NamedTuple}}()
    for pair in pairs
        transition = (pair.previous.season, pair.current.season)
        push!(
            get!(pairs_by_transition, transition, NamedTuple[]),
            pair,
        )
    end

    cells_by_season = Dict{Int,Vector{NamedTuple}}()
    for cell in cells
        push!(get!(cells_by_season, cell.season, NamedTuple[]), cell)
    end

    summaries = NamedTuple[]
    for transition in sort!(collect(keys(pairs_by_transition)))
        previous_season, current_season = transition
        transition_cells = vcat(
            get(cells_by_season, previous_season, NamedTuple[]),
            get(cells_by_season, current_season, NamedTuple[]),
        )
        push!(
            summaries,
            (
                previous_season=previous_season,
                current_season=current_season,
                moments=_moment_summary(
                    transition_cells,
                    pairs_by_transition[transition],
                ),
            ),
        )
    end
    return summaries
end

function _estimate_home_multiplier(blocks::AbstractVector)
    log_ratios = Float64[]
    weights = Float64[]
    for block in blocks
        moments = block.moments
        isnothing(moments.away_mean) && continue
        isnothing(moments.home_mean) && continue
        moments.away_mean > 0.0 || continue
        moments.home_mean > 0.0 || continue
        weight = min(moments.n_away_mean, moments.n_home_mean)
        weight > 0 || continue
        push!(
            log_ratios,
            log(moments.home_mean / moments.away_mean),
        )
        push!(weights, Float64(weight))
    end
    isempty(log_ratios) && return 1.0
    multiplier = exp(
        sum(weight * value for (weight, value) in zip(weights, log_ratios)) /
        sum(weights),
    )
    return clamp(multiplier, 0.25, 4.0)
end

function _reset_moment_parameters(
    blocks::AbstractVector,
)
    home_multiplier = _estimate_home_multiplier(blocks)
    means = Float64[]
    variances = Float64[]
    pooled_cross_covariance = 0.0
    pooled_variance = 0.0

    for block in blocks
        moments = block.moments
        mean_rate = _weighted_moment_projection(
            (moments.away_mean, moments.home_mean),
            (1.0, home_multiplier),
            (moments.n_away_mean, moments.n_home_mean),
        )
        mean_rate = isnothing(mean_rate) ?
            1.0e-3 : max(Float64(mean_rate), 1.0e-8)

        second_moment = _weighted_moment_projection(
            (
                moments.away_factorial,
                moments.home_factorial,
                moments.same_season_product,
            ),
            (1.0, home_multiplier^2, home_multiplier),
            (
                moments.n_away_factorial,
                moments.n_home_factorial,
                moments.n_same_season_product,
            ),
        )
        variance = isnothing(second_moment) ?
            mean_rate^2 * 1.0e-4 :
            Float64(second_moment) - mean_rate^2
        latent_variance = max(variance, mean_rate^2 * 1.0e-4)

        cross_difference = nothing
        cross_weight = 0
        if hasproperty(block, :transition_moments) &&
            hasproperty(block, :season_moments)
            season_means = Dict{Int,Float64}()
            for season_summary in block.season_moments
                season_moments = season_summary.moments
                season_mean = _weighted_moment_projection(
                    (
                        season_moments.away_mean,
                        season_moments.home_mean,
                    ),
                    (1.0, home_multiplier),
                    (
                        season_moments.n_away_mean,
                        season_moments.n_home_mean,
                    ),
                )
                isnothing(season_mean) && continue
                season_means[season_summary.season] = Float64(season_mean)
            end

            centered_cross_sum = 0.0
            centered_cross_weight = 0
            for transition_summary in block.transition_moments
                previous_mean = get(
                    season_means,
                    transition_summary.previous_season,
                    nothing,
                )
                current_mean = get(
                    season_means,
                    transition_summary.current_season,
                    nothing,
                )
                isnothing(previous_mean) && continue
                isnothing(current_mean) && continue

                transition_moments = transition_summary.moments
                transition_cross = _weighted_moment_projection(
                    (
                        transition_moments.sequential_away_away,
                        transition_moments.sequential_away_home,
                        transition_moments.sequential_home_away,
                        transition_moments.sequential_home_home,
                    ),
                    (
                        1.0,
                        home_multiplier,
                        home_multiplier,
                        home_multiplier^2,
                    ),
                    (
                        transition_moments.n_sequential_away_away,
                        transition_moments.n_sequential_away_home,
                        transition_moments.n_sequential_home_away,
                        transition_moments.n_sequential_home_home,
                    ),
                )
                isnothing(transition_cross) && continue
                transition_weight = _shared_moment_count(
                    (
                        transition_moments.n_sequential_away_away,
                        transition_moments.n_sequential_away_home,
                        transition_moments.n_sequential_home_away,
                        transition_moments.n_sequential_home_home,
                    ),
                )
                transition_weight > 0 || continue
                finite_sample_correction =
                    transition_weight > 1.0 ?
                    transition_weight / (transition_weight - 1.0) : 1.0
                centered_cross_sum += transition_weight * (
                    finite_sample_correction * (
                        Float64(transition_cross) -
                        previous_mean * current_mean
                    )
                )
                centered_cross_weight += transition_weight
            end

            if centered_cross_weight > 0
                cross_difference =
                    centered_cross_sum / centered_cross_weight
                cross_weight = centered_cross_weight
            end
        end

        if isnothing(cross_difference)
            sequential_moment = _weighted_moment_projection(
                (
                    moments.sequential_away_away,
                    moments.sequential_away_home,
                    moments.sequential_home_away,
                    moments.sequential_home_home,
                ),
                (1.0, home_multiplier, home_multiplier, home_multiplier^2),
                (
                    moments.n_sequential_away_away,
                    moments.n_sequential_away_home,
                    moments.n_sequential_home_away,
                    moments.n_sequential_home_home,
                ),
            )
            if !isnothing(sequential_moment)
                cross_difference =
                    Float64(sequential_moment) - mean_rate^2
                cross_weight = _shared_moment_count(
                    (
                        moments.n_sequential_away_away,
                        moments.n_sequential_away_home,
                        moments.n_sequential_home_away,
                        moments.n_sequential_home_home,
                    ),
                )
            end
        end

        if !isnothing(cross_difference) && variance > 0.0
            second_weight = _shared_moment_count(
                (
                    moments.n_away_factorial,
                    moments.n_home_factorial,
                    moments.n_same_season_product,
                ),
            )
            moment_weight = min(second_weight, cross_weight)
            if moment_weight > 0
                pooled_cross_covariance +=
                    moment_weight * Float64(cross_difference)
                pooled_variance += moment_weight * variance
            end
        end

        push!(means, mean_rate)
        push!(variances, latent_variance)
    end

    persistence = pooled_variance > 0.0 ?
        clamp(pooled_cross_covariance / pooled_variance, 0.0, 1.0) : 0.5
    return means, variances, home_multiplier, persistence
end

function _fit_reset_outcome_parameters(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    kind::Symbol,
)
    parameters, home_multiplier_value, persistence, _ =
        _fit_reset_outcome_parameters_with_diagnostics(
            byseason,
            seasons,
            time_edges,
            kind,
        )
    return parameters, home_multiplier_value, persistence
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

function _team_season_cell(
    byseason::Dict{Int,HazardSufficientStats},
    season::Integer,
    kind::Symbol,
    team::String,
    time_bin::Int,
)
    outcome = _outcome_stats(byseason[Int(season)], kind)
    key = (team, time_bin)
    return (
        count=get(outcome.home_counts, key, 0.0) +
            get(outcome.away_counts, key, 0.0),
        home_count=get(outcome.home_counts, key, 0.0),
        away_count=get(outcome.away_counts, key, 0.0),
        exposure=get(outcome.away_exposure, key, 0.0),
        home_exposure=get(outcome.home_exposure, key, 0.0),
    )
end

function _build_reset_team_mixtures(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    reference::Int,
    kind::Symbol,
    hyperparameters::AbstractVector{GammaParams},
    home_multiplier_value::Real,
    persistence::Real,
)
    mixtures = Dict{String,Vector{GammaMixture}}()
    n_bins = length(time_edges) - 1
    for team in _teams(byseason, seasons)
        team_mixtures = GammaMixture[]
        for time_bin in 1:n_bins
            mixture = nothing
            for season in seasons
                if isnothing(mixture)
                    mixture = GammaMixture(
                        [1.0],
                        [hyperparameters[time_bin]];
                        source_seasons=[Int(season)],
                    )
                else
                    mixture = _transition_gamma_mixture(
                        mixture,
                        persistence,
                        hyperparameters[time_bin],
                        Int(season),
                    )
                end
                cell = _team_season_cell(
                    byseason,
                    season,
                    kind,
                    team,
                    time_bin,
                )
                effective_exposure = cell.exposure +
                    home_multiplier_value * cell.home_exposure
                mixture = _update_gamma_mixture(
                    mixture,
                    cell.count,
                    effective_exposure,
                )
            end
            mixture = _transition_gamma_mixture(
                mixture,
                persistence,
                hyperparameters[time_bin],
                reference,
            )
            push!(team_mixtures, mixture)
        end
        mixtures[team] = team_mixtures
    end
    return mixtures
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

"""
    fit_empirical_bayes_prior(historical_drives; kwargs...) -> HazardPrior

Fit stationary league Gamma parameters and season-to-season persistence
probabilities with the event-process marginal likelihood. The likelihood
retains the competing-risk exposure term, uses the home multiplier in both
event and integrated-hazard contributions, and integrates the latent
team-season hazards through the probabilistic reset filter. Team-specific
season-opening priors are finite mixtures of at most four Gamma components.

The historical fit is performed separately for the touchdown and defensive
processes because the joint competing-risk likelihood factorizes conditional
on the observed risk intervals. The fitted home multiplier is shared across
time bins within each outcome, and one persistence probability is shared
across that outcome's hazard curve. EM or conditional optimization failure is
reported rather than silently replaced with a default prior.
"""
function fit_empirical_bayes_prior(
    historical_drives::AbstractDataFrame;
    time_edges=DEFAULT_TIME_EDGES,
    max_seasons::Int=3,
    current_season::Union{Nothing,Integer}=nothing,
)
    data, edges = build_exposure_data(historical_drives; time_edges=time_edges)
    byseason = _season_stats(data)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    effective_max_seasons = min(max_seasons, MAX_HISTORICAL_SEASONS)
    seasons = _historical_seasons(byseason, effective_max_seasons)
    isempty(seasons) &&
        throw(ArgumentError("historical data contain no usable seasons"))

    observed_seasons = collect(filter(!=(0), seasons))
    reference = if current_season === nothing
        isempty(observed_seasons) ? 0 : maximum(observed_seasons) + 1
    else
        Int(current_season)
    end
    if !isempty(observed_seasons) && reference <= maximum(observed_seasons)
        throw(ArgumentError("current_season must follow historical seasons"))
    end
    td_hyper, td_home_multiplier, td_persistence, td_diagnostics =
        _fit_reset_outcome_parameters_with_diagnostics(
            byseason,
            seasons,
            edges,
            :td,
        )
    defensive_hyper, defensive_home_multiplier, defensive_persistence,
    defensive_diagnostics =
        _fit_reset_outcome_parameters_with_diagnostics(
            byseason,
            seasons,
            edges,
            :defensive,
        )
    td_mixtures = _build_reset_team_mixtures(
        byseason,
        seasons,
        edges,
        reference,
        :td,
        td_hyper,
        td_home_multiplier,
        td_persistence,
    )
    defensive_mixtures = _build_reset_team_mixtures(
        byseason,
        seasons,
        edges,
        reference,
        :defensive,
        defensive_hyper,
        defensive_home_multiplier,
        defensive_persistence,
    )

    return HazardPrior(
        edges,
        td_hyper,
        defensive_hyper,
        td_mixtures,
        defensive_mixtures,
        td_home_multiplier,
        defensive_home_multiplier,
        td_persistence,
        defensive_persistence,
        Int.(seasons),
        td_diagnostics,
        defensive_diagnostics,
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

"""
    ExpectedGameSpreadMetrics

Posterior expected spread and predictive spread variance for a matchup.
"""
struct ExpectedGameSpreadMetrics
    expected_spread::Float64
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

function _expected_game_win_probability(
    theta::HazardTheta,
    edges::AbstractVector{<:Real},
    marks::ScoreMarks;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    win_function = theta_vector ->
        _game_metrics_from_theta(
            theta_vector,
            edges,
            marks;
            horizon=horizon,
        ).win_probability
    approximation = _second_order_expectation(
        win_function,
        theta.log_mean,
        theta.covariance,
    )
    isfinite(approximation) ||
        throw(ArgumentError("posterior win probability is not finite"))
    # The Hessian approximation can leave [0, 1] under high posterior
    # uncertainty even though the underlying probability is bounded.
    return clamp(approximation, 0.0, 1.0)
end

function _expected_game_spread_metrics(
    theta::HazardTheta,
    edges::AbstractVector{<:Real},
    marks::ScoreMarks;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    spread_function = theta_vector ->
        _game_metrics_from_theta(
            theta_vector,
            edges,
            marks;
            horizon=horizon,
        ).mean_spread
    variance_function = theta_vector ->
        _game_metrics_from_theta(
            theta_vector,
            edges,
            marks;
            horizon=horizon,
        ).spread_variance

    expected_spread = _second_order_expectation(
        spread_function,
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

    return ExpectedGameSpreadMetrics(
        expected_spread,
        predictive_spread_variance,
    )
end

"""
    expected_game_win_probability(
        model,
        marks,
        home_team,
        away_team;
        horizon=GAME_CLOCK_SECONDS,
    ) -> Float64

Approximate the posterior expected home win probability without evaluating
spread or predictive-variance metrics.
"""
function expected_game_win_probability(
    model::HazardModel,
    marks::ScoreMarks,
    home_team,
    away_team;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    theta = hazard_theta(model, home_team, away_team)
    return _expected_game_win_probability(
        theta,
        model.time_edges,
        marks;
        horizon=horizon,
    )
end

"""
    expected_game_spread_metrics(
        model,
        marks,
        home_team,
        away_team;
        horizon=GAME_CLOCK_SECONDS,
    ) -> ExpectedGameSpreadMetrics

Compute posterior expected spread and predictive spread variance without
evaluating the posterior expected win probability.
"""
function expected_game_spread_metrics(
    model::HazardModel,
    marks::ScoreMarks,
    home_team,
    away_team;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    theta = hazard_theta(model, home_team, away_team)
    return _expected_game_spread_metrics(
        theta,
        model.time_edges,
        marks;
        horizon=horizon,
    )
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
    away_team    ;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    theta = hazard_theta(model, home_team, away_team)
    expected_win_probability = _expected_game_win_probability(
        theta,
        model.time_edges,
        marks;
        horizon=horizon,
    )
    spread_metrics = _expected_game_spread_metrics(
        theta,
        model.time_edges,
        marks;
        horizon=horizon,
    )

    return ExpectedGameMetrics(
        spread_metrics.expected_spread,
        expected_win_probability,
        spread_metrics.predictive_spread_variance,
    )
end
