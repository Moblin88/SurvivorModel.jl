struct EMLBFGSFit end

const DEFAULT_HISTORICAL_SEASONS = 5
const DEFAULT_PRIOR_FIT_METHOD = EMLBFGSFit()

function _logsumexp(values)
    isempty(values) && throw(ArgumentError("logsumexp requires at least one value"))
    maximum_value = maximum(values)
    maximum_value == -Inf && return -Inf
    isfinite(maximum_value) || return maximum_value
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

const _RESET_GAMMA_RECURRENCE_MAX_COUNT = 32
const _GammaSpecialValues = NTuple{2,Float64}

function _validated_gamma_event_count_exposure(
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
    return count_integer, exposure_value
end

@inline function _gamma_shape_special_values(
    shape::Float64,
)
    return (
        SpecialFunctions.loggamma(shape),
        SpecialFunctions.digamma(shape),
    )
end

@inline function _gamma_shape_shifted_special_values(
    shape::Float64,
    count::Int,
    base_values::_GammaSpecialValues,
)
    count == 0 && return base_values
    if count <= _RESET_GAMMA_RECURRENCE_MAX_COUNT
        loggamma_value = base_values[1]
        digamma_value = base_values[2]
        for offset in 0:(count - 1)
            shifted_shape = shape + offset
            inverse_shape = 1.0 / shifted_shape
            loggamma_value += log(shifted_shape)
            digamma_value += inverse_shape
        end
        return loggamma_value, digamma_value
    end
    shifted_shape = shape + count
    return _gamma_shape_special_values(shifted_shape)
end

# Keep the recurrence as differences to avoid subtracting large special values.
@inline function _gamma_shape_difference_values(
    shape::Float64,
    count::Int,
    base_values::_GammaSpecialValues,
)
    count == 0 && return 0.0, 0.0
    if count <= _RESET_GAMMA_RECURRENCE_MAX_COUNT
        loggamma_difference = 0.0
        digamma_difference = 0.0
        for offset in 0:(count - 1)
            shifted_shape = shape + offset
            inverse_shape = 1.0 / shifted_shape
            loggamma_difference += log(shifted_shape)
            digamma_difference += inverse_shape
        end
        return loggamma_difference, digamma_difference
    end
    shifted_values = _gamma_shape_special_values(shape + count)
    return (
        shifted_values[1] - base_values[1],
        shifted_values[2] - base_values[2],
    )
end

function _log_gamma_event_marginal(
    component::GammaParams,
    count::Real,
    exposure::Real,
)
    return _log_gamma_event_marginal(
        component,
        count,
        exposure,
        _gamma_shape_special_values(component.shape),
    )
end

function _log_gamma_event_marginal(
    component::GammaParams,
    count::Real,
    exposure::Real,
    base_values::_GammaSpecialValues,
)
    count_integer, exposure_value =
        _validated_gamma_event_count_exposure(count, exposure)
    exposure_value == 0.0 &&
        return count_integer == 0 ? 0.0 : -Inf

    shape = component.shape
    rate = component.rate
    differences = _gamma_shape_difference_values(
        shape,
        count_integer,
        base_values,
    )
    return shape * log(rate) +
        differences[1] -
        (shape + count_integer) * log(rate + exposure_value)
end

function _log_gamma_event_marginal_derivative_values(
    component::GammaParams,
    count::Real,
    exposure::Real,
)
    return _log_gamma_event_marginal_derivative_values(
        component,
        count,
        exposure,
        _gamma_shape_special_values(component.shape),
    )
end

function _log_gamma_event_marginal_derivative_values(
    component::GammaParams,
    count::Real,
    exposure::Real,
    base_values::_GammaSpecialValues,
)
    count_integer, exposure_value =
        _validated_gamma_event_count_exposure(count, exposure)
    shape = component.shape
    rate = component.rate
    differences = _gamma_shape_difference_values(
        shape,
        count_integer,
        base_values,
    )
    value = exposure_value == 0.0 ?
        (count_integer == 0 ? 0.0 : -Inf) :
        shape * log(rate) +
            differences[1] -
            (shape + count_integer) * log(rate + exposure_value)
    denominator = rate + exposure_value
    d_shape = log(rate) +
        differences[2] -
        log(denominator)
    d_rate = shape / rate - (shape + count_integer) / denominator
    d_exposure = -(shape + count_integer) / denominator

    return (
        value=value,
        d_log_mean=-rate * d_rate,
        d_log_shape=shape * d_shape + rate * d_rate,
        d_exposure=d_exposure,
    )
end

function _log_gamma_event_marginal_with_derivatives(
    component::GammaParams,
    count::Real,
    exposure::Real,
)
    return _log_gamma_event_marginal_with_derivatives(
        component,
        count,
        exposure,
        _gamma_shape_special_values(component.shape),
    )
end

function _log_gamma_event_marginal_with_derivatives(
    component::GammaParams,
    count::Real,
    exposure::Real,
    base_values::_GammaSpecialValues,
)
    values = _log_gamma_event_marginal_derivative_values(
        component,
        count,
        exposure,
        base_values,
    )
    return (
        value=values.value,
        d_log_mean=values.d_log_mean,
        d_log_shape=values.d_log_shape,
        d_exposure=values.d_exposure,
    )
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
    persistence = _reset_probability_from_logit(values[2 * n_bins + 1])
    home_multiplier_value = exp(Float64(values[2 * n_bins + 2]))
    return hyperparameters, home_multiplier_value, persistence
end

function _reset_likelihood_cells(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
)
    isempty(seasons) &&
        throw(ArgumentError("reset likelihood requires at least one season"))
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
            season_cells = [
                _team_season_cell(
                    byseason,
                    season,
                    kind,
                    team,
                    time_bin,
                )
                for season in seasons
            ]
            push!(
                cells,
                (
                    time_bin=time_bin,
                    counts=[season_cell.count for season_cell in season_cells],
                    home_counts=[
                        season_cell.home_count for season_cell in season_cells
                    ],
                    away_exposures=[
                        season_cell.exposure for season_cell in season_cells
                    ],
                    home_exposures=[
                        season_cell.home_exposure
                        for season_cell in season_cells
                    ],
                ),
            )
        end
    end
    return cells
end

function _log_reset_group_marginal(
    component::GammaParams,
    counts,
    away_exposures,
    home_exposures,
    home_multiplier_value::Float64,
    first_season::Int,
    last_season::Int,
)
    return _log_reset_group_marginal(
        component,
        counts,
        away_exposures,
        home_exposures,
        home_multiplier_value,
        first_season,
        last_season,
        _gamma_shape_special_values(component.shape),
    )
end

function _log_reset_group_marginal(
    component::GammaParams,
    counts,
    away_exposures,
    home_exposures,
    home_multiplier_value::Float64,
    first_season::Int,
    last_season::Int,
    base_values::_GammaSpecialValues,
)
    count = 0.0
    exposure = 0.0
    for season in first_season:last_season
        count += counts[season]
        exposure += away_exposures[season] +
            home_multiplier_value * home_exposures[season]
    end
    return _log_gamma_event_marginal(
        component,
        count,
        exposure,
        base_values,
    )
end

function _reset_segment_transition(
    first_season::Int,
    last_season::Int,
    persistence::Float64,
    log_persistence::Float64,
    log_reset::Float64,
    include_reset::Bool=true,
)
    persistent_links = last_season - first_season
    reset_links = include_reset && first_season > 1 ? 1 : 0
    log_value = 0.0
    if persistent_links > 0
        isfinite(log_persistence) || return -Inf, 0.0
        log_value += persistent_links * log_persistence
    end
    if reset_links > 0
        isfinite(log_reset) || return -Inf, 0.0
        log_value += log_reset
    end
    gradient = persistent_links * (1.0 - persistence) -
        reset_links * persistence
    return log_value, gradient
end

function _reset_segment_tables(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    base_values::_GammaSpecialValues;
    with_gradient::Bool=false,
)
    n_seasons = length(cell.counts)
    n_seasons > 0 || throw(ArgumentError("reset likelihood requires seasons"))
    log_segments = fill(-Inf, n_seasons, n_seasons)
    segment_gradients = with_gradient ?
        zeros(Float64, n_seasons, n_seasons, 4) : nothing
    for first_season in 1:n_seasons
        count = 0.0
        exposure = 0.0
        home_exposure = 0.0
        for last_season in first_season:n_seasons
            count += cell.counts[last_season]
            home_exposure += cell.home_exposures[last_season]
            exposure += cell.away_exposures[last_season] +
                home_multiplier_value * cell.home_exposures[last_season]
            if with_gradient
                marginal = _log_gamma_event_marginal_with_derivatives(
                    component,
                    count,
                    exposure,
                    base_values,
                )
                log_segments[first_season, last_season] = marginal.value
                segment_gradients[first_season, last_season, :] .= (
                    marginal.d_log_mean,
                    marginal.d_log_shape,
                    0.0,
                    marginal.d_exposure * home_multiplier_value * home_exposure,
                )
            else
                log_segments[first_season, last_season] =
                    _log_gamma_event_marginal(
                        component,
                        count,
                        exposure,
                        base_values,
                    )
            end
        end
    end
    return (log_segments=log_segments, segment_gradients=segment_gradients)
end

function _reset_segment_dynamic_program(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    base_values::_GammaSpecialValues;
    with_gradient::Bool=false,
)
    isfinite(home_multiplier_value) && home_multiplier_value > 0.0 ||
        throw(ArgumentError("home multiplier must be finite and positive"))
    isfinite(persistence) && 0.0 <= persistence <= 1.0 ||
        throw(ArgumentError("persistence must be finite and in [0, 1]"))
    n_seasons = length(cell.counts)
    n_seasons > 0 || throw(ArgumentError("reset likelihood requires seasons"))

    tables = _reset_segment_tables(
        cell,
        component,
        home_multiplier_value,
        persistence,
        base_values;
        with_gradient=with_gradient,
    )
    log_segments = tables.log_segments
    log_persistence = persistence > 0.0 ? log(persistence) : -Inf
    log_reset = persistence < 1.0 ? log1p(-persistence) : -Inf

    forward = fill(-Inf, n_seasons)
    backward = fill(-Inf, n_seasons + 1)
    forward_gradients = with_gradient ? zeros(Float64, n_seasons, 4) : nothing

    for last_season in 1:n_seasons
        candidate_logs = fill(-Inf, last_season)
        candidate_gradients = with_gradient ?
            zeros(Float64, last_season, 4) : nothing
        for first_season in 1:last_season
            transition_log, transition_gradient = _reset_segment_transition(
                first_season,
                last_season,
                persistence,
                log_persistence,
                log_reset,
            )
            candidate_logs[first_season] =
                (first_season == 1 ? 0.0 : forward[first_season - 1]) +
                transition_log +
                log_segments[first_season, last_season]
            with_gradient || continue

            local_gradient = copy(
                tables.segment_gradients[first_season, last_season, :],
            )
            local_gradient[3] += transition_gradient
            if first_season == 1
                candidate_gradients[first_season, :] = local_gradient
            else
                previous = first_season - 1
                candidate_gradients[first_season, :] =
                    forward_gradients[previous, :] + local_gradient
            end
        end
        normalizer = _logsumexp(candidate_logs)
        forward[last_season] = normalizer
        if with_gradient
            weights = exp.(candidate_logs .- normalizer)
            forward_gradients[last_season, :] =
                vec(sum(weights .* candidate_gradients; dims=1))
        end
    end

    backward[n_seasons + 1] = 0.0
    for first_season in n_seasons:-1:1
        candidate_logs = Float64[]
        for last_season in first_season:n_seasons
            transition_log, _ = _reset_segment_transition(
                first_season,
                last_season,
                persistence,
                log_persistence,
                log_reset,
                false,
            )
            suffix_log = last_season < n_seasons ?
                log_reset + backward[last_season + 1] : 0.0
            push!(
                candidate_logs,
                log_segments[first_season, last_season] +
                transition_log +
                suffix_log,
            )
        end
        backward[first_season] = _logsumexp(candidate_logs)
    end

    log_normalizer = backward[1]
    segment_posteriors = zeros(Float64, n_seasons, n_seasons)
    for first_season in 1:n_seasons
        prefix_log = first_season == 1 ? 0.0 : forward[first_season - 1]
        for last_season in first_season:n_seasons
            transition_log, _ = _reset_segment_transition(
                first_season,
                last_season,
                persistence,
                log_persistence,
                log_reset,
            )
            suffix_log = last_season < n_seasons ?
                log_reset + backward[last_season + 1] : 0.0
            term = prefix_log + transition_log +
                log_segments[first_season, last_season] + suffix_log
            segment_posteriors[first_season, last_season] =
                isfinite(term) ? exp(term - log_normalizer) : 0.0
        end
    end

    result = (
        log_normalizer=log_normalizer,
        segment_posteriors=segment_posteriors,
    )
    with_gradient || return result
    return merge(result, (gradient=forward_gradients[n_seasons, :],))
end

function _log_reset_partition_marginal!(
    log_terms::AbstractVector{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    return _log_reset_partition_marginal!(
        log_terms,
        cell,
        component,
        home_multiplier_value,
        persistence,
        _gamma_shape_special_values(component.shape),
    )
end

function _log_reset_partition_marginal!(
    log_terms::AbstractVector{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    base_values::_GammaSpecialValues,
)
    result = _reset_segment_dynamic_program(
        cell,
        component,
        home_multiplier_value,
        persistence,
        base_values,
    )
    log_home_events = sum(cell.home_counts) * log(home_multiplier_value)
    return log_home_events + result.log_normalizer
end

function _log_reset_partition_marginal(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    return _log_reset_partition_marginal!(
        Float64[],
        cell,
        component,
        home_multiplier_value,
        persistence,
    )
end

function _log_reset_partition_marginal(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    base_values::_GammaSpecialValues,
)
    return _log_reset_partition_marginal!(
        Float64[],
        cell,
        component,
        home_multiplier_value,
        persistence,
        base_values,
    )
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
    shape_special_values = [
        _gamma_shape_special_values(component.shape)
        for component in hyperparameters
    ]
    for cell in cells
        log_likelihood += _log_reset_partition_marginal!(
            Float64[],
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_float,
            persistence_float,
            shape_special_values[cell.time_bin],
        )
    end
    return log_likelihood
end

function _reset_event_log_likelihood_with_gradient(
    cells::AbstractVector,
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Float64,
    persistence::Float64,
)
    n_bins = length(hyperparameters)
    gradient = zeros(Float64, 2 * n_bins + 2)
    log_likelihood = 0.0
    shape_special_values = [
        _gamma_shape_special_values(component.shape)
        for component in hyperparameters
    ]
    for cell in cells
        result = _reset_segment_dynamic_program(
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_value,
            persistence,
            shape_special_values[cell.time_bin];
            with_gradient=true,
        )
        log_likelihood += result.log_normalizer +
            sum(cell.home_counts) * log(home_multiplier_value)
        gradient[cell.time_bin] += result.gradient[1]
        gradient[n_bins + cell.time_bin] += result.gradient[2]
        gradient[2 * n_bins + 1] += result.gradient[3]
        gradient[2 * n_bins + 2] += result.gradient[4] +
            sum(cell.home_counts)
    end
    return (log_likelihood=log_likelihood, gradient=gradient)
end

function _reset_group_posterior_moments(
    component::GammaParams,
    counts,
    away_exposures,
    home_exposures,
    home_multiplier_value::Float64,
    first_season::Int,
    last_season::Int,
)
    return _reset_group_posterior_moments(
        component,
        counts,
        away_exposures,
        home_exposures,
        home_multiplier_value,
        first_season,
        last_season,
        _gamma_shape_special_values(component.shape),
    )
end

function _reset_group_posterior_moments(
    component::GammaParams,
    counts,
    away_exposures,
    home_exposures,
    home_multiplier_value::Float64,
    first_season::Int,
    last_season::Int,
    base_values::_GammaSpecialValues,
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
    posterior_special_values = _gamma_shape_shifted_special_values(
        component.shape,
        round(Int, count),
        base_values,
    )
    return (
        mean=posterior_shape / posterior_rate,
        log_mean=posterior_special_values[2] - log(posterior_rate),
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
    shape_special_values = [
        _gamma_shape_special_values(component.shape)
        for component in hyperparameters
    ]

    for cell in cells
        result = _reset_segment_dynamic_program(
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_value,
            persistence,
            shape_special_values[cell.time_bin],
        )
        n_seasons = length(cell.counts)
        total_home_events += sum(cell.home_counts)
        total_transitions += n_seasons - 1
        function_evaluations += n_seasons * (n_seasons + 1) ÷ 2
        for first_season in 1:n_seasons
            for last_season in first_season:n_seasons
                posterior_weight =
                    result.segment_posteriors[first_season, last_season]
                posterior_weight > 0.0 || continue
                expected_group_counts[cell.time_bin] += posterior_weight
                expected_persistent_links += posterior_weight *
                    (last_season - first_season)
                moments = _reset_group_posterior_moments(
                    hyperparameters[cell.time_bin],
                    cell.counts,
                    cell.away_exposures,
                    cell.home_exposures,
                    home_multiplier_value,
                    first_season,
                    last_season,
                    shape_special_values[cell.time_bin],
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
    base_values = _gamma_shape_special_values(component.shape)
    for cell in cells
        cell.time_bin == time_bin || continue
        log_likelihood += _log_reset_partition_marginal!(
            Float64[],
            cell,
            component,
            home_multiplier_value,
            persistence,
            base_values,
        )
    end
    return log_likelihood
end

function _reset_bin_event_log_likelihood_with_gradient(
    cells::AbstractVector,
    time_bin::Int,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    log_likelihood = 0.0
    gradient = zeros(Float64, 2)
    base_values = _gamma_shape_special_values(component.shape)
    for cell in cells
        cell.time_bin == time_bin || continue
        result = _reset_segment_dynamic_program(
            cell,
            component,
            home_multiplier_value,
            persistence,
            base_values,
            with_gradient=true,
        )
        log_likelihood += result.log_normalizer +
            sum(cell.home_counts) * log(home_multiplier_value)
        gradient[1] += result.gradient[1]
        gradient[2] += result.gradient[2]
    end
    return (log_likelihood=log_likelihood, gradient=gradient)
end

function _reset_probability_from_logit(raw::Real)
    raw_value = Float64(raw)
    return raw_value >= 0.0 ?
        1.0 / (1.0 + exp(-raw_value)) :
        exp(raw_value) / (1.0 + exp(raw_value))
end

function _reset_logit_probability(probability::Float64)
    probability <= 0.0 && return -30.0
    probability >= 1.0 && return 30.0
    return log(probability / (1.0 - probability))
end

function _reset_optimize_bounded(
    objective,
    gradient!,
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
    initial_values::AbstractVector{<:Real},
    options,
    failure_message::AbstractString,
)
    result = Optim.optimize(
        objective,
        gradient!,
        lower,
        upper,
        initial_values,
        Optim.LBFGSB(),
        options,
    )
    if !Optim.converged(result)
        result = Optim.optimize(
            objective,
            lower,
            upper,
            initial_values,
            Optim.Fminbox(Optim.NelderMead()),
            options,
        )
    end
    Optim.converged(result) ||
        throw(ArgumentError(failure_message))
    return result
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
    gradient! = (storage, values) -> begin
        mean_value = exp(values[1])
        shape_value = exp(values[2])
        component = GammaParams(shape_value, shape_value / mean_value)
        result = _reset_bin_event_log_likelihood_with_gradient(
            cells,
            time_bin,
            component,
            home_multiplier_value,
            persistence,
        )
        storage[1] = -result.gradient[1]
        storage[2] = -result.gradient[2]
        return storage
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
    options = Optim.Options(
        iterations=300,
        f_reltol=1.0e-8,
        x_reltol=1.0e-7,
        show_trace=false,
        show_warnings=false,
    )
    result = _reset_optimize_bounded(
        objective,
        gradient!,
        lower,
        upper,
        initial_values,
        options,
        "conditional Gamma likelihood optimization did not converge",
    )
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
    has_transitions = any(length(cell.counts) > 1 for cell in cells)
    if !has_transitions
        return (
            home_multiplier=clamp(initial_home_multiplier, 0.05, 20.0),
            persistence=clamp(initial_persistence, 0.0, 1.0),
            function_evaluations=0,
        )
    end
    objective = values -> begin
        raw_persistence = values[1]
        persistence = _reset_probability_from_logit(raw_persistence)
        home_multiplier_value = exp(values[2])
        return -_reset_event_log_likelihood(
            cells,
            hyperparameters,
            home_multiplier_value,
            persistence,
        )
    end
    gradient! = (storage, values) -> begin
        persistence = _reset_probability_from_logit(values[1])
        home_multiplier_value = exp(values[2])
        result = _reset_event_log_likelihood_with_gradient(
            cells,
            hyperparameters,
            home_multiplier_value,
            persistence,
        )
        storage[1] = -result.gradient[2 * length(hyperparameters) + 1]
        storage[2] = -result.gradient[2 * length(hyperparameters) + 2]
        return storage
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
    options = Optim.Options(
        iterations=300,
        f_reltol=1.0e-8,
        x_reltol=1.0e-7,
        show_trace=false,
        show_warnings=false,
    )
    result = _reset_optimize_bounded(
        objective,
        gradient!,
        lower,
        upper,
        initial_values,
        options,
        "conditional shared likelihood optimization did not converge",
    )
    fitted_values = Optim.minimizer(result)
    raw_persistence = fitted_values[1]
    persistence = _reset_probability_from_logit(raw_persistence)
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
            "$kind historical EM-LBFGS optimization did not converge",
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
    fit_empirical_bayes_prior(historical_drives; kwargs...) -> HazardPrior

Fit stationary league Gamma parameters and season-to-season persistence
probabilities with the event-process marginal likelihood. The likelihood
retains the competing-risk exposure term, uses the home multiplier in both
event and integrated-hazard contributions, and integrates the latent
team-season hazards through the probabilistic reset filter. Team-specific
season-opening priors are finite Gamma mixtures whose component count grows
with the supplied history.

The historical fit is performed separately for the touchdown and defensive
processes because the joint competing-risk likelihood factorizes conditional
on the observed risk intervals. The fitted home multiplier is shared across
time bins within each outcome, and one persistence probability is shared
across that outcome's hazard curve. Solver failure is reported rather than
silently replaced with a default prior. The typed `method` keyword selects
`EMLBFGSFit()`, the only supported fitter.
"""
function fit_empirical_bayes_prior(
    historical_drives::AbstractDataFrame;
    time_edges=DEFAULT_TIME_EDGES,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    current_season::Union{Nothing,Integer}=nothing,
    method::EMLBFGSFit=DEFAULT_PRIOR_FIT_METHOD,
)
    data, edges = build_exposure_data(historical_drives; time_edges=time_edges)
    byseason = _season_stats(data)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    seasons = _historical_seasons(byseason, max_seasons)
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

    prior = HazardPrior(
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
    return prior
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
updates. When `historical_drives` is supplied, the typed `method` is forwarded
to `fit_empirical_bayes_prior`.
"""
function fit_hazard_model(
    drives::AbstractDataFrame;
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    prior::Union{Nothing,HazardPrior}=nothing,
    time_edges=DEFAULT_TIME_EDGES,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    current_season::Union{Nothing,Integer}=nothing,
    method::EMLBFGSFit=DEFAULT_PRIOR_FIT_METHOD,
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
            method=method,
        )
    else
        _default_hazard_prior(edges)
    end

    model = HazardModel(edges, fitted_prior, HazardSufficientStats())
    return update_hazard_model!(model, drives)
end
