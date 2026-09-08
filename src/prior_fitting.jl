"""
    PriorFitMethod

Typed selector for the empirical-Bayes prior-fitting algorithm.
"""
abstract type PriorFitMethod end

struct EMECMEFit <: PriorFitMethod end
struct EMLBFGSFit <: PriorFitMethod end
struct DirectLBFGSFit <: PriorFitMethod end
struct DirectBFGSFit <: PriorFitMethod end
struct MomentFit <: PriorFitMethod end
struct MomentLBFGSFit <: PriorFitMethod end
struct HybridFit <: PriorFitMethod end
struct BlockNewtonFit <: PriorFitMethod end
struct SchurNewtonFit <: PriorFitMethod end

prior_fit_method_name(::EMECMEFit) = :em_ecme
prior_fit_method_name(::EMLBFGSFit) = :em_lbfgs
prior_fit_method_name(::DirectLBFGSFit) = :direct_lbfgs
prior_fit_method_name(::DirectBFGSFit) = :direct_bfgs
prior_fit_method_name(::MomentFit) = :moment
prior_fit_method_name(::MomentLBFGSFit) = :moment_lbfgs
prior_fit_method_name(::HybridFit) = :hybrid
prior_fit_method_name(::BlockNewtonFit) = :block_newton
prior_fit_method_name(::SchurNewtonFit) = :schur_newton

function Base.show(io::IO, method::PriorFitMethod)
    print(io, prior_fit_method_name(method))
end

@inline _fit_method_symbol(method::PriorFitMethod) =
    prior_fit_method_name(method)

function _logsumexp(values)
    isempty(values) && throw(ArgumentError("logsumexp requires at least one value"))
    maximum_value = maximum(values)
    maximum_value == -Inf && return -Inf
    isfinite(maximum_value) || return maximum_value
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function _logsumexp_prefix(values::AbstractVector{<:Real}, count::Int)
    count > 0 || throw(ArgumentError("logsumexp requires at least one value"))
    maximum_value = values[1]
    for index in 2:count
        maximum_value = max(maximum_value, values[index])
    end
    maximum_value == -Inf && return -Inf
    isfinite(maximum_value) || return maximum_value
    total = 0.0
    for index in 1:count
        total += exp(values[index] - maximum_value)
    end
    return maximum_value + log(total)
end

const _RESET_GAMMA_RECURRENCE_MAX_COUNT = 32
const _GammaSpecialValues = NTuple{3,Float64}

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
        SpecialFunctions.trigamma(shape),
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
        trigamma_value = base_values[3]
        for offset in 0:(count - 1)
            shifted_shape = shape + offset
            inverse_shape = 1.0 / shifted_shape
            loggamma_value += log(shifted_shape)
            digamma_value += inverse_shape
            trigamma_value -= inverse_shape * inverse_shape
        end
        return loggamma_value, digamma_value, trigamma_value
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
    count == 0 && return 0.0, 0.0, 0.0
    if count <= _RESET_GAMMA_RECURRENCE_MAX_COUNT
        loggamma_difference = 0.0
        digamma_difference = 0.0
        trigamma_difference = 0.0
        for offset in 0:(count - 1)
            shifted_shape = shape + offset
            inverse_shape = 1.0 / shifted_shape
            loggamma_difference += log(shifted_shape)
            digamma_difference += inverse_shape
            trigamma_difference -= inverse_shape * inverse_shape
        end
        return loggamma_difference, digamma_difference, trigamma_difference
    end
    shifted_values = _gamma_shape_special_values(shape + count)
    return (
        shifted_values[1] - base_values[1],
        shifted_values[2] - base_values[2],
        shifted_values[3] - base_values[3],
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

    h_shape_shape =
        differences[3]
    h_shape_rate = 1.0 / rate - 1.0 / denominator
    h_shape_exposure = -1.0 / denominator
    h_rate_rate =
        -shape / rate^2 +
        (shape + count_integer) / denominator^2
    h_rate_exposure = (shape + count_integer) / denominator^2
    h_exposure_exposure = (shape + count_integer) / denominator^2

    return (
        value=value,
        d_log_mean=-rate * d_rate,
        d_log_shape=shape * d_shape + rate * d_rate,
        d_exposure=d_exposure,
        h11=h_rate_rate * rate^2 + d_rate * rate,
        h12=-rate * (h_shape_rate * shape + h_rate_rate * rate) -
            d_rate * rate,
        h13=-rate * h_rate_exposure,
        h22=h_shape_shape * shape^2 +
            2.0 * h_shape_rate * shape * rate +
            h_rate_rate * rate^2 +
            d_shape * shape +
            d_rate * rate,
        h23=h_shape_exposure * shape + h_rate_exposure * rate,
        h33=h_exposure_exposure,
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

function _log_gamma_event_marginal_with_hessian(
    component::GammaParams,
    count::Real,
    exposure::Real,
)
    values = _log_gamma_event_marginal_derivative_values(
        component,
        count,
        exposure,
    )
    gradient = [
        values.d_log_mean,
        values.d_log_shape,
        values.d_exposure,
    ]
    hessian = zeros(Float64, 3, 3)
    hessian[1, 1] = values.h11
    hessian[1, 2] = values.h12
    hessian[1, 3] = values.h13
    hessian[2, 1] = values.h12
    hessian[2, 2] = values.h22
    hessian[2, 3] = values.h23
    hessian[3, 1] = hessian[1, 3]
    hessian[3, 2] = hessian[2, 3]
    hessian[3, 3] = values.h33
    return (value=values.value, gradient=gradient, hessian=hessian)
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

function _pack_reset_parameters(
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Float64,
    persistence::Float64,
)
    return vcat(
        [
            log(parameter.shape / parameter.rate)
            for parameter in hyperparameters
        ],
        [log(parameter.shape) for parameter in hyperparameters],
        [
            _reset_logit_probability(persistence),
            log(home_multiplier_value),
        ],
    )
end

function _reset_likelihood_cells(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
)
    n_seasons = length(seasons)
    n_seasons == 1 &&
        return _reset_likelihood_cells_typed(
            byseason,
            seasons,
            kind,
            n_bins,
            Val(1),
        )
    n_seasons == 2 &&
        return _reset_likelihood_cells_typed(
            byseason,
            seasons,
            kind,
            n_bins,
            Val(2),
        )
    n_seasons == 3 &&
        return _reset_likelihood_cells_typed(
            byseason,
            seasons,
            kind,
            n_bins,
            Val(3),
        )
    throw(ArgumentError("reset likelihood supports one to three seasons"))
end

function _reset_likelihood_cells_typed(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
    ::Val{N},
) where {N}
    teams = Set{String}()
    for season in seasons
        outcome = _outcome_stats(byseason[Int(season)], kind)
        for key in keys(outcome.exposure)
            key[2] <= n_bins || continue
            push!(teams, key[1])
        end
    end

    cell_type = NamedTuple{
        (
            :time_bin,
            :counts,
            :home_counts,
            :away_exposures,
            :home_exposures,
        ),
        Tuple{
            Int,
            NTuple{N,Float64},
            NTuple{N,Float64},
            NTuple{N,Float64},
            NTuple{N,Float64},
        },
    }
    cells = cell_type[]
    for team in sort!(collect(teams))
        for time_bin in 1:n_bins
            season_cells = ntuple(
                season_index -> _team_season_cell(
                    byseason,
                    seasons[season_index],
                    kind,
                    team,
                    time_bin,
                ),
                Val(N),
            )
            push!(
                cells,
                cell_type((
                    time_bin=time_bin,
                    counts=ntuple(
                        season_index -> season_cells[season_index].count,
                        Val(N),
                    ),
                    home_counts=ntuple(
                        season_index ->
                            season_cells[season_index].home_count,
                        Val(N),
                    ),
                    away_exposures=ntuple(
                        season_index ->
                            season_cells[season_index].exposure,
                        Val(N),
                    ),
                    home_exposures=ntuple(
                        season_index ->
                            season_cells[season_index].home_exposure,
                        Val(N),
                    ),
                )),
            )
        end
    end
    return cells
end

function _reset_partition_paths(n_seasons::Int)
    n_seasons == 1 && return RESET_PARTITION_PATHS_1
    n_seasons == 2 && return RESET_PARTITION_PATHS_2
    n_seasons == 3 && return RESET_PARTITION_PATHS_3
    throw(ArgumentError("reset likelihood supports one to three seasons"))
end

function _reset_path_log_probability(
    path,
    n_seasons::Int,
    persistence::Float64,
)
    return _reset_path_log_probability_from_logs(
        path,
        n_seasons,
        persistence > 0.0 ? log(persistence) : -Inf,
        persistence < 1.0 ? log1p(-persistence) : -Inf,
    )
end

function _reset_path_log_probability_from_logs(
    path,
    n_seasons::Int,
    log_persistence::Float64,
    log_reset::Float64,
)
    reset_links = n_seasons - 1 - path.persistent_links
    log_probability = 0.0
    if path.persistent_links > 0
        isfinite(log_persistence) || return -Inf
        log_probability += path.persistent_links * log_persistence
    end
    if reset_links > 0
        isfinite(log_reset) || return -Inf
        log_probability += reset_links * log_reset
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
    counts::Tuple,
    away_exposures::Tuple,
    home_exposures::Tuple,
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

function _reset_partition_log_terms!(
    log_terms::AbstractVector{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    return _reset_partition_log_terms!(
        log_terms,
        cell,
        component,
        home_multiplier_value,
        persistence,
        _gamma_shape_special_values(component.shape),
    )
end

function _reset_partition_log_terms!(
    log_terms::AbstractVector{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    base_values::_GammaSpecialValues,
)
    n_seasons = length(cell.counts)
    paths = _reset_partition_paths(n_seasons)
    n_paths = length(paths)
    log_persistence = persistence > 0.0 ? log(persistence) : -Inf
    log_reset = persistence < 1.0 ? log1p(-persistence) : -Inf
    fill!(log_terms, 0.0)
    for (path_index, path) in enumerate(paths)
        log_term = _reset_path_log_probability_from_logs(
            path,
            n_seasons,
            log_persistence,
            log_reset,
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
                base_values,
            )
        end
        log_terms[path_index] = log_term
    end
    return n_paths
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
    n_seasons = length(cell.counts)
    1 <= n_seasons <= 3 ||
        throw(ArgumentError("reset likelihood supports one to three seasons"))

    log_home_events = sum(cell.home_counts) * log(home_multiplier_value)
    n_paths = _reset_partition_log_terms!(
        log_terms,
        cell,
        component,
        home_multiplier_value,
        persistence,
        base_values,
    )
    return log_home_events + _logsumexp_prefix(log_terms, n_paths)
end

function _log_reset_partition_marginal(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    log_terms = zeros(Float64, 4)
    return _log_reset_partition_marginal!(
        log_terms,
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
    log_terms = zeros(Float64, 4)
    return _log_reset_partition_marginal!(
        log_terms,
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
    log_terms = zeros(Float64, 4)
    shape_special_values = [
        _gamma_shape_special_values(component.shape)
        for component in hyperparameters
    ]
    for cell in cells
        log_likelihood += _log_reset_partition_marginal!(
            log_terms,
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_float,
            persistence_float,
            shape_special_values[cell.time_bin],
        )
    end
    return log_likelihood
end

function _reset_path_log_probability_derivative(
    path,
    n_seasons::Int,
    persistence::Float64,
)
    isfinite(persistence) && 0.0 <= persistence <= 1.0 ||
        throw(ArgumentError(
            "persistence must be finite and in [0, 1]",
        ))
    return _reset_path_log_probability_derivative_unchecked(
        path,
        n_seasons,
        persistence,
    )
end

@inline function _reset_path_log_probability_derivative_unchecked(
    path,
    n_seasons::Int,
    persistence::Float64,
)
    reset_links = n_seasons - 1 - path.persistent_links
    return path.persistent_links * (1.0 - persistence) -
        reset_links * persistence
end

function _reset_cell_log_likelihood_gradient!(
    log_terms::AbstractVector{Float64},
    path_gradients::AbstractMatrix{Float64},
    gradient::AbstractVector{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    return _reset_cell_log_likelihood_gradient!(
        log_terms,
        path_gradients,
        gradient,
        cell,
        component,
        home_multiplier_value,
        persistence,
        _gamma_shape_special_values(component.shape),
    )
end

function _reset_cell_log_likelihood_gradient!(
    log_terms::AbstractVector{Float64},
    path_gradients::AbstractMatrix{Float64},
    gradient::AbstractVector{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    base_values::_GammaSpecialValues,
)
    isfinite(home_multiplier_value) && home_multiplier_value > 0.0 ||
        throw(ArgumentError("home multiplier must be finite and positive"))
    isfinite(persistence) && 0.0 <= persistence <= 1.0 ||
        throw(ArgumentError("persistence must be finite and in [0, 1]"))
    paths = _reset_partition_paths(length(cell.counts))
    n_paths = length(paths)
    n_seasons = length(cell.counts)
    log_persistence = persistence > 0.0 ? log(persistence) : -Inf
    log_reset = persistence < 1.0 ? log1p(-persistence) : -Inf
    fill!(log_terms, 0.0)
    fill!(path_gradients, 0.0)
    fill!(gradient, 0.0)
    home_event_count = sum(cell.home_counts)

    for (path_index, path) in enumerate(paths)
        log_term = _reset_path_log_probability_from_logs(
            path,
            n_seasons,
            log_persistence,
            log_reset,
        )
        for (first_season, last_season) in path.groups
            count = 0.0
            exposure = 0.0
            home_exposure = 0.0
            for season in first_season:last_season
                count += cell.counts[season]
                exposure += cell.away_exposures[season] +
                    home_multiplier_value * cell.home_exposures[season]
                home_exposure += cell.home_exposures[season]
            end
            marginal = _log_gamma_event_marginal_with_derivatives(
                component,
                count,
                exposure,
                base_values,
            )
            log_term += marginal.value
            path_gradients[path_index, 1] += marginal.d_log_mean
            path_gradients[path_index, 2] += marginal.d_log_shape
            path_gradients[path_index, 4] +=
                home_multiplier_value * home_exposure * marginal.d_exposure
        end
        path_gradients[path_index, 3] =
            _reset_path_log_probability_derivative_unchecked(
                path,
                n_seasons,
                persistence,
            )
        log_terms[path_index] = log_term
    end

    log_normalizer = _logsumexp_prefix(log_terms, n_paths)
    for path_index in 1:n_paths
        posterior_weight = exp(log_terms[path_index] - log_normalizer)
        for coordinate in 1:4
            gradient[coordinate] +=
                posterior_weight * path_gradients[path_index, coordinate]
        end
    end
    gradient[4] += home_event_count
    return home_event_count * log(home_multiplier_value) + log_normalizer
end

function _reset_cell_log_likelihood_gradient(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    log_terms = zeros(Float64, 4)
    path_gradients = zeros(Float64, 4, 4)
    gradient = zeros(Float64, 4)
    log_likelihood = _reset_cell_log_likelihood_gradient!(
        log_terms,
        path_gradients,
        gradient,
        cell,
        component,
        home_multiplier_value,
        persistence,
    )
    return (log_likelihood=log_likelihood, gradient=gradient)
end

function _reset_cell_log_likelihood_with_hessian!(
    log_terms::AbstractVector{Float64},
    path_gradients::AbstractMatrix{Float64},
    path_hessians::Array{Float64,3},
    posterior_weights::AbstractVector{Float64},
    gradient::AbstractVector{Float64},
    hessian::AbstractMatrix{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    return _reset_cell_log_likelihood_with_hessian!(
        log_terms,
        path_gradients,
        path_hessians,
        posterior_weights,
        gradient,
        hessian,
        cell,
        component,
        home_multiplier_value,
        persistence,
        _gamma_shape_special_values(component.shape),
    )
end

function _reset_cell_log_likelihood_with_hessian!(
    log_terms::AbstractVector{Float64},
    path_gradients::AbstractMatrix{Float64},
    path_hessians::Array{Float64,3},
    posterior_weights::AbstractVector{Float64},
    gradient::AbstractVector{Float64},
    hessian::AbstractMatrix{Float64},
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
    base_values::_GammaSpecialValues,
)
    isfinite(home_multiplier_value) && home_multiplier_value > 0.0 ||
        throw(ArgumentError("home multiplier must be finite and positive"))
    isfinite(persistence) && 0.0 <= persistence <= 1.0 ||
        throw(ArgumentError("persistence must be finite and in [0, 1]"))
    paths = _reset_partition_paths(length(cell.counts))
    n_paths = length(paths)
    home_event_count = sum(cell.home_counts)
    n_seasons = length(cell.counts)
    log_persistence = persistence > 0.0 ? log(persistence) : -Inf
    log_reset = persistence < 1.0 ? log1p(-persistence) : -Inf
    fill!(log_terms, 0.0)
    fill!(path_gradients, 0.0)
    fill!(path_hessians, 0.0)
    fill!(posterior_weights, 0.0)
    fill!(gradient, 0.0)
    fill!(hessian, 0.0)

    for (path_index, path) in enumerate(paths)
        log_term = _reset_path_log_probability_from_logs(
            path,
            n_seasons,
            log_persistence,
            log_reset,
        )
        for (first_season, last_season) in path.groups
            count = 0.0
            exposure = 0.0
            home_exposure = 0.0
            for season in first_season:last_season
                count += cell.counts[season]
                exposure += cell.away_exposures[season] +
                    home_multiplier_value * cell.home_exposures[season]
                home_exposure += cell.home_exposures[season]
            end
            marginal = _log_gamma_event_marginal_derivative_values(
                component,
                count,
                exposure,
                base_values,
            )
            home_exposure_derivative =
                home_multiplier_value * home_exposure
            log_term += marginal.value
            path_gradients[path_index, 1] += marginal.d_log_mean
            path_gradients[path_index, 2] += marginal.d_log_shape
            path_gradients[path_index, 4] +=
                marginal.d_exposure * home_exposure_derivative

            path_hessians[path_index, 1, 1] += marginal.h11
            path_hessians[path_index, 1, 2] += marginal.h12
            path_hessians[path_index, 2, 1] += marginal.h12
            path_hessians[path_index, 2, 2] += marginal.h22
            path_hessians[path_index, 1, 4] +=
                marginal.h13 * home_exposure_derivative
            path_hessians[path_index, 4, 1] =
                path_hessians[path_index, 1, 4]
            path_hessians[path_index, 2, 4] +=
                marginal.h23 * home_exposure_derivative
            path_hessians[path_index, 4, 2] =
                path_hessians[path_index, 2, 4]
            path_hessians[path_index, 4, 4] +=
                marginal.h33 * home_exposure_derivative^2 +
                marginal.d_exposure * home_exposure_derivative
        end
        path_gradients[path_index, 3] =
            _reset_path_log_probability_derivative_unchecked(
                path,
                n_seasons,
                persistence,
            )
        path_hessians[path_index, 3, 3] =
            -(n_seasons - 1) * persistence * (1.0 - persistence)
        log_terms[path_index] = log_term
    end

    log_normalizer = _logsumexp_prefix(log_terms, n_paths)
    for path_index in 1:n_paths
        posterior_weight = exp(log_terms[path_index] - log_normalizer)
        posterior_weights[path_index] = posterior_weight
        for coordinate in 1:4
            gradient[coordinate] +=
                posterior_weight * path_gradients[path_index, coordinate]
            for second_coordinate in 1:4
                hessian[coordinate, second_coordinate] +=
                    posterior_weight *
                    path_hessians[
                        path_index,
                        coordinate,
                        second_coordinate,
                    ]
            end
        end
    end
    for first_index in 1:4
        for second_index in 1:4
            covariance = 0.0
            for path_index in 1:n_paths
                posterior_weight = posterior_weights[path_index]
                covariance += posterior_weight *
                    path_gradients[path_index, first_index] *
                    path_gradients[path_index, second_index]
            end
            hessian[first_index, second_index] += covariance -
                gradient[first_index] * gradient[second_index]
        end
    end
    gradient[4] += home_event_count
    return home_event_count * log(home_multiplier_value) + log_normalizer
end

function _reset_cell_log_likelihood_with_hessian(
    cell,
    component::GammaParams,
    home_multiplier_value::Float64,
    persistence::Float64,
)
    log_terms = zeros(Float64, 4)
    path_gradients = zeros(Float64, 4, 4)
    path_hessians = zeros(Float64, 4, 4, 4)
    posterior_weights = zeros(Float64, 4)
    gradient = zeros(Float64, 4)
    hessian = zeros(Float64, 4, 4)
    log_likelihood = _reset_cell_log_likelihood_with_hessian!(
        log_terms,
        path_gradients,
        path_hessians,
        posterior_weights,
        gradient,
        hessian,
        cell,
        component,
        home_multiplier_value,
        persistence,
    )
    return (
        log_likelihood=log_likelihood,
        gradient=gradient,
        hessian=hessian,
    )
end

function _reset_event_log_likelihood_with_gradient(
    cells::AbstractVector,
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Float64,
    persistence::Float64,
)
    n_bins = length(hyperparameters)
    gradient = zeros(Float64, 2 * n_bins + 2)
    log_terms = zeros(Float64, 4)
    path_gradients = zeros(Float64, 4, 4)
    cell_gradient = zeros(Float64, 4)
    log_likelihood = 0.0
    shape_special_values = Vector{_GammaSpecialValues}(undef, n_bins)
    for time_bin in 1:n_bins
        shape_special_values[time_bin] =
            _gamma_shape_special_values(hyperparameters[time_bin].shape)
    end
    for cell in cells
        log_likelihood += _reset_cell_log_likelihood_gradient!(
            log_terms,
            path_gradients,
            cell_gradient,
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_value,
            persistence,
            shape_special_values[cell.time_bin],
        )
        gradient[cell.time_bin] += cell_gradient[1]
        gradient[n_bins + cell.time_bin] += cell_gradient[2]
        gradient[2 * n_bins + 1] += cell_gradient[3]
        gradient[2 * n_bins + 2] += cell_gradient[4]
    end
    return (log_likelihood=log_likelihood, gradient=gradient)
end

function _reset_event_log_likelihood_with_hessian(
    cells::AbstractVector,
    hyperparameters::AbstractVector{<:GammaParams},
    home_multiplier_value::Float64,
    persistence::Float64,
)
    n_bins = length(hyperparameters)
    gradient = zeros(Float64, 2 * n_bins + 2)
    hessian = zeros(Float64, 2 * n_bins + 2, 2 * n_bins + 2)
    log_terms = zeros(Float64, 4)
    path_gradients = zeros(Float64, 4, 4)
    path_hessians = zeros(Float64, 4, 4, 4)
    posterior_weights = zeros(Float64, 4)
    cell_gradient = zeros(Float64, 4)
    cell_hessian = zeros(Float64, 4, 4)
    log_likelihood = 0.0
    shape_special_values = Vector{_GammaSpecialValues}(undef, n_bins)
    for time_bin in 1:n_bins
        shape_special_values[time_bin] =
            _gamma_shape_special_values(hyperparameters[time_bin].shape)
    end
    for cell in cells
        log_likelihood += _reset_cell_log_likelihood_with_hessian!(
            log_terms,
            path_gradients,
            path_hessians,
            posterior_weights,
            cell_gradient,
            cell_hessian,
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_value,
            persistence,
            shape_special_values[cell.time_bin],
        )
        local_indices = (
            cell.time_bin,
            n_bins + cell.time_bin,
            2 * n_bins + 1,
            2 * n_bins + 2,
        )
        for local_index in 1:4
            global_index = local_indices[local_index]
            gradient[global_index] += cell_gradient[local_index]
            for second_local_index in 1:4
                hessian[
                    global_index,
                    local_indices[second_local_index],
                ] += cell_hessian[local_index, second_local_index]
            end
        end
    end
    return (
        log_likelihood=log_likelihood,
        gradient=gradient,
        hessian=hessian,
    )
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
    counts::Tuple,
    away_exposures::Tuple,
    home_exposures::Tuple,
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
    log_terms = zeros(Float64, 4)
    shape_special_values = Vector{_GammaSpecialValues}(undef, n_bins)
    for time_bin in 1:n_bins
        shape_special_values[time_bin] =
            _gamma_shape_special_values(hyperparameters[time_bin].shape)
    end

    for cell in cells
        n_seasons = length(cell.counts)
        paths = _reset_partition_paths(n_seasons)
        n_paths = _reset_partition_log_terms!(
            log_terms,
            cell,
            hyperparameters[cell.time_bin],
            home_multiplier_value,
            persistence,
            shape_special_values[cell.time_bin],
        )
        log_normalizer = _logsumexp_prefix(log_terms, n_paths)
        total_home_events += sum(cell.home_counts)
        total_transitions += n_seasons - 1
        function_evaluations += n_paths

        for path_index in 1:n_paths
            path = paths[path_index]
            log_term = log_terms[path_index]
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
    log_terms = zeros(Float64, 4)
    base_values = _gamma_shape_special_values(component.shape)
    for cell in cells
        cell.time_bin == time_bin || continue
        log_likelihood += _log_reset_partition_marginal!(
            log_terms,
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
    log_terms = zeros(Float64, 4)
    path_gradients = zeros(Float64, 4, 4)
    cell_gradient = zeros(Float64, 4)
    base_values = _gamma_shape_special_values(component.shape)
    for cell in cells
        cell.time_bin == time_bin || continue
        log_likelihood += _reset_cell_log_likelihood_gradient!(
            log_terms,
            path_gradients,
            cell_gradient,
            cell,
            component,
            home_multiplier_value,
            persistence,
            base_values,
        )
        gradient[1] += cell_gradient[1]
        gradient[2] += cell_gradient[2]
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
    method::Symbol,
    options,
    failure_message::AbstractString,
)
    result = if method === :nelder_mead
        Optim.optimize(
            objective,
            lower,
            upper,
            initial_values,
            Optim.Fminbox(Optim.NelderMead()),
            options,
        )
    elseif method === :lbfgs
        Optim.optimize(
            objective,
            gradient!,
            lower,
            upper,
            initial_values,
            Optim.LBFGSB(),
            options,
        )
    elseif method === :bfgs
        Optim.optimize(
            objective,
            gradient!,
            lower,
            upper,
            initial_values,
            Optim.Fminbox(Optim.BFGS()),
            options,
        )
    else
        throw(ArgumentError("unsupported bounded reset optimizer: $method"))
    end
    if !Optim.converged(result) && method !== :nelder_mead
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
    ;
    method::Symbol=:nelder_mead,
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
        method,
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
    ;
    method::Symbol=:nelder_mead,
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
        method,
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
    ;
    method::Symbol=:nelder_mead,
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
            method=method,
        )
        push!(updated_hyperparameters, result.parameter)
        function_evaluations += result.function_evaluations
    end
    shared = _reset_optimize_shared_parameters(
        cells,
        updated_hyperparameters,
        home_multiplier_value,
        persistence,
        method=method,
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

function _reset_moment_initial_parameter_vector(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
    ;
    iterated::Bool=true,
)
    blocks = _reset_moment_blocks(
        byseason,
        seasons,
        kind,
        n_bins,
    )
    means, variances, home_multiplier_value, persistence = if iterated
        moment_fit = _reset_iterated_moment_parameters(blocks)
        (
            moment_fit.means,
            moment_fit.variances,
            moment_fit.home_multiplier,
            moment_fit.persistence,
        )
    else
        _reset_moment_parameters(blocks)
    end
    bounded_means = [
        clamp(Float64(mean_value), 1.0e-10, 10.0)
        for mean_value in means
    ]
    bounded_shapes = [
        clamp(
            mean_value^2 / max(Float64(variance), mean_value^2 * 1.0e-8),
            RESET_GAMMA_SHAPE_LOWER,
            RESET_GAMMA_SHAPE_UPPER,
        )
        for (mean_value, variance) in zip(bounded_means, variances)
    ]
    return vcat(
        log.(bounded_means),
        log.(bounded_shapes),
        [
            _reset_logit_probability(clamp(persistence, 0.0, 1.0)),
            log(clamp(home_multiplier_value, 0.05, 20.0)),
        ],
    )
end

function _reset_validate_solver(solver::Symbol)
    solver in (
        :em_ecme,
        :em_lbfgs,
        :direct_lbfgs,
        :direct_bfgs,
        :moment,
        :moment_lbfgs,
        :hybrid,
        :block_newton,
        :schur_newton,
    ) || throw(ArgumentError(
        "solver must be one of :em_ecme, :em_lbfgs, :direct_lbfgs, " *
        ":direct_bfgs, :moment, :moment_lbfgs, :hybrid, :block_newton, or " *
        ":schur_newton; got $solver",
    ))
    return solver
end

function _reset_optimize_joint_parameters(
    cells::AbstractVector,
    initial_values::AbstractVector{<:Real},
    n_bins::Int,
    ;
    method::Symbol=:lbfgs,
)
    objective = values -> begin
        hyperparameters, home_multiplier_value, persistence =
            _unpack_reset_parameters(values, n_bins)
        return -_reset_event_log_likelihood(
            cells,
            hyperparameters,
            home_multiplier_value,
            persistence,
        )
    end
    gradient! = (storage, values) -> begin
        hyperparameters, home_multiplier_value, persistence =
            _unpack_reset_parameters(values, n_bins)
        result = _reset_event_log_likelihood_with_gradient(
            cells,
            hyperparameters,
            home_multiplier_value,
            persistence,
        )
        storage .= -result.gradient
        return storage
    end
    lower, upper = _reset_joint_parameter_bounds(n_bins)
    initial = [
        clamp(
            Float64(value),
            lower[index] + 1.0e-8,
            upper[index] - 1.0e-8,
        )
        for (index, value) in enumerate(initial_values)
    ]
    options = Optim.Options(
        iterations=600,
        f_reltol=1.0e-9,
        x_reltol=1.0e-8,
        show_trace=false,
        show_warnings=false,
    )
    result = if method === :lbfgs
        Optim.optimize(
            objective,
            gradient!,
            lower,
            upper,
            initial,
            Optim.LBFGSB(),
            options,
        )
    elseif method === :bfgs
        Optim.optimize(
            objective,
            gradient!,
            lower,
            upper,
            initial,
            Optim.Fminbox(Optim.BFGS()),
            options,
        )
    else
        throw(ArgumentError("unsupported joint reset optimizer: $method"))
    end
    if !Optim.converged(result) && method === :bfgs
        result = Optim.optimize(
            objective,
            gradient!,
            lower,
            upper,
            initial,
            Optim.LBFGSB(),
            options,
        )
    end
    Optim.converged(result) ||
        throw(ArgumentError("joint reset likelihood optimization did not converge"))
    fitted_values = Optim.minimizer(result)
    hyperparameters, home_multiplier_value, persistence =
        _unpack_reset_parameters(fitted_values, n_bins)
    return (
        hyperparameters=hyperparameters,
        home_multiplier=home_multiplier_value,
        persistence=persistence,
        log_likelihood=-Optim.minimum(result),
        iterations=Optim.iterations(result),
        function_evaluations=Optim.f_calls(result),
    )
end

function _reset_joint_parameter_bounds(n_bins::Int)
    lower = vcat(
        fill(log(1.0e-10), n_bins),
        fill(log(RESET_GAMMA_SHAPE_LOWER), n_bins),
        [-30.0, log(0.05)],
    )
    upper = vcat(
        fill(log(10.0), n_bins),
        fill(log(RESET_GAMMA_SHAPE_UPPER), n_bins),
        [30.0, log(20.0)],
    )
    return lower, upper
end

function _reset_positive_definite(matrix::AbstractMatrix{<:Real})
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension || return false
    all(isfinite, matrix) || return false
    dimension == 0 && return true
    if dimension == 1
        return matrix[1, 1] > 0.0
    elseif dimension == 2
        return matrix[1, 1] > 0.0 &&
            matrix[1, 1] * matrix[2, 2] -
            matrix[1, 2] * matrix[2, 1] > 0.0
    end
    return false
end

function _reset_positive_definite_solve(
    matrix::AbstractMatrix{<:Real},
    right_hand_side::AbstractVecOrMat{<:Real},
)
    _reset_positive_definite(matrix) || return nothing
    factor = try
        cholesky(Symmetric(matrix); check=false)
    catch error
        error isa LinearAlgebra.PosDefException || rethrow()
        return nothing
    end
    issuccess(factor) || return nothing
    solution = factor \ right_hand_side
    all(isfinite, solution) || return nothing
    return solution
end

function _reset_positive_definite_factor(
    matrix::AbstractMatrix{<:Real},
)
    all(isfinite, matrix) || return nothing
    factor = try
        cholesky(Symmetric(matrix); check=false)
    catch error
        error isa LinearAlgebra.PosDefException || rethrow()
        return nothing
    end
    issuccess(factor) || return nothing
    return factor
end

function _reset_lowrank_downdate(
    factor::Cholesky,
    vectors::AbstractMatrix{<:Real},
)
    updated = factor
    for column in axes(vectors, 2)
        updated = try
            lowrankdowndate(updated, @view vectors[:, column])
        catch error
            error isa LinearAlgebra.PosDefException || rethrow()
            return nothing
        end
        issuccess(updated) || return nothing
    end
    return updated
end

function _reset_newton_active_coordinates(
    gradient::AbstractVector{<:Real},
    values::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
    n_bins::Int,
    has_transitions::Bool,
)
    n_parameters = length(values)
    persistence_index = 2 * n_bins + 1
    active = falses(n_parameters)
    for index in 1:n_parameters
        at_lower = values[index] <= lower[index] + 1.0e-10
        at_upper = values[index] >= upper[index] - 1.0e-10
        blocked_persistence =
            index == persistence_index && !has_transitions
        blocked_lower = at_lower && gradient[index] <= 0.0
        blocked_upper = at_upper && gradient[index] >= 0.0
        active[index] =
            blocked_persistence || blocked_lower || blocked_upper
    end
    return active
end

function _reset_projected_gradient_norm(
    gradient::AbstractVector{<:Real},
    active::AbstractVector{Bool},
)
    norm = 0.0
    for index in eachindex(gradient, active)
        active[index] || (norm = max(norm, abs(gradient[index])))
    end
    return norm
end

function _reset_schur_newton_direction(
    hessian::AbstractMatrix{<:Real},
    gradient::AbstractVector{<:Real},
    values::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
    n_bins::Int,
    has_transitions::Bool,
    regularization::Float64,
)
    n_parameters = length(values)
    persistence_index = 2 * n_bins + 1
    active = _reset_newton_active_coordinates(
        gradient,
        values,
        lower,
        upper,
        n_bins,
        has_transitions,
    )

    bin_groups = Vector{Vector{Int}}()
    for time_bin in 1:n_bins
        group = Int[]
        for index in (time_bin, n_bins + time_bin)
            active[index] || push!(group, index)
        end
        isempty(group) || push!(bin_groups, group)
    end
    shared_indices = Int[]
    for index in (persistence_index, persistence_index + 1)
        active[index] || push!(shared_indices, index)
    end
    isempty(bin_groups) && isempty(shared_indices) && return nothing

    shared_dimension = length(shared_indices)
    schur = zeros(Float64, shared_dimension, shared_dimension)
    right_hand_side = Float64[
        gradient[index]
        for index in shared_indices
    ]
    for first_index in 1:shared_dimension
        for second_index in 1:shared_dimension
            schur[first_index, second_index] =
                -hessian[
                    shared_indices[first_index],
                    shared_indices[second_index],
                ] +
                (first_index == second_index ? regularization : 0.0)
        end
    end
    schur_factor = if shared_dimension == 0
        nothing
    else
        _reset_positive_definite_factor(schur)
    end
    shared_dimension > 0 && schur_factor === nothing && return nothing

    block_data = NamedTuple[]
    for group in bin_groups
        dimension = length(group)
        block = zeros(Float64, dimension, dimension)
        cross_block = zeros(Float64, dimension, shared_dimension)
        for first_index in 1:dimension
            for second_index in 1:dimension
                block[first_index, second_index] =
                    -hessian[group[first_index], group[second_index]] +
                    (first_index == second_index ? regularization : 0.0)
            end
            for second_index in 1:shared_dimension
                cross_block[first_index, second_index] =
                    -hessian[group[first_index], shared_indices[second_index]]
            end
        end
        block_gradient = Float64[
            gradient[index]
            for index in group
        ]
        block_factor = _reset_positive_definite_factor(block)
        block_factor === nothing && return nothing
        solved_gradient = block_factor \ block_gradient
        all(isfinite, solved_gradient) || return nothing
        solved_shared = if shared_dimension == 0
            zeros(Float64, dimension, 0)
        else
            block_factor \ cross_block
        end
        solved_shared === nothing && return nothing
        all(isfinite, solved_shared) || return nothing
        if shared_dimension > 0
            right_hand_side .-=
                transpose(cross_block) * solved_gradient
            whitened_cross = block_factor.L \ cross_block
            schur_factor = _reset_lowrank_downdate(
                schur_factor,
                transpose(whitened_cross),
            )
            schur_factor === nothing && return nothing
        end
        push!(
            block_data,
            (
                indices=group,
                solved_gradient=solved_gradient,
                solved_shared=solved_shared,
            ),
        )
    end

    shared_direction = if shared_dimension == 0
        Float64[]
    else
        schur_factor \ right_hand_side
    end
    shared_direction === nothing && return nothing
    all(isfinite, shared_direction) || return nothing

    direction = zeros(Float64, n_parameters)
    for (index, shared_index) in enumerate(shared_indices)
        direction[shared_index] = shared_direction[index]
    end
    for block in block_data
        block_direction =
            block.solved_gradient -
            block.solved_shared * shared_direction
        direction[block.indices] .= block_direction
    end
    directional_derivative = dot(gradient, direction)
    isfinite(directional_derivative) && directional_derivative > 0.0 ||
        return nothing
    return (
        direction=direction,
        active_coordinates=count(active),
        directional_derivative=directional_derivative,
    )
end

function _reset_block_newton_direction(
    hessian::AbstractMatrix{<:Real},
    gradient::AbstractVector{<:Real},
    values::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
    n_bins::Int,
    has_transitions::Bool,
    regularization::Float64,
    block_kind::Symbol,
)
    n_parameters = length(values)
    persistence_index = 2 * n_bins + 1
    active = _reset_newton_active_coordinates(
        gradient,
        values,
        lower,
        upper,
        n_bins,
        has_transitions,
    )
    direction = zeros(Float64, n_parameters)

    groups = Vector{Vector{Int}}()
    if block_kind === :shared
        group = [
            index for index in
            (persistence_index, persistence_index + 1) if !active[index]
        ]
        isempty(group) || push!(groups, group)
    elseif block_kind === :bins
        for time_bin in 1:n_bins
            group = [
                index for index in (time_bin, n_bins + time_bin) if
                !active[index]
            ]
            isempty(group) || push!(groups, group)
        end
    else
        throw(ArgumentError("unsupported Newton block: $block_kind"))
    end
    isempty(groups) && return nothing

    for group in groups
        dimension = length(group)
        block = zeros(Float64, dimension, dimension)
        block_gradient = zeros(Float64, dimension)
        for first_index in 1:dimension
            global_first = group[first_index]
            block_gradient[first_index] = gradient[global_first]
            for second_index in 1:dimension
                block[first_index, second_index] =
                    -hessian[global_first, group[second_index]] +
                    (first_index == second_index ? regularization : 0.0)
            end
        end
        factor = _reset_positive_definite_factor(block)
        factor === nothing && return nothing
        block_direction = factor \ block_gradient
        all(isfinite, block_direction) || return nothing
        direction[group] .= block_direction
    end

    directional_derivative = dot(gradient, direction)
    isfinite(directional_derivative) && directional_derivative > 0.0 ||
        return nothing
    return (
        direction=direction,
        active_coordinates=count(active),
        directional_derivative=directional_derivative,
    )
end

function _reset_newton_solver_metrics(
    ;
    solver::Symbol=:schur_newton,
    gradient_evaluations::Int=0,
    hessian_evaluations::Int=0,
    newton_iterations::Int=0,
    damping_steps::Int=0,
    backtracking_steps::Int=0,
    schur_corrections::Int=0,
    fallback_count::Int=0,
    active_coordinates::Int=0,
    gradient_norm::Float64=NaN,
    parameter_step_norm::Float64=NaN,
    status::Symbol=:not_run,
)
    return (
        solver=solver,
        gradient_evaluations=gradient_evaluations,
        hessian_evaluations=hessian_evaluations,
        newton_iterations=newton_iterations,
        damping_steps=damping_steps,
        backtracking_steps=backtracking_steps,
        schur_corrections=schur_corrections,
        fallback_count=fallback_count,
        active_coordinates=active_coordinates,
        gradient_norm=gradient_norm,
        parameter_step_norm=parameter_step_norm,
        status=status,
    )
end

function _reset_optimize_schur_newton(
    cells::AbstractVector,
    initial_values::AbstractVector{<:Real},
    n_bins::Int,
)
    lower, upper = _reset_joint_parameter_bounds(n_bins)
    values = [
        clamp(
            Float64(value),
            lower[index],
            upper[index],
        )
        for (index, value) in enumerate(initial_values)
    ]
    has_transitions = any(length(cell.counts) > 1 for cell in cells)
    hyperparameters, home_multiplier_value, persistence =
        _unpack_reset_parameters(values, n_bins)
    current = _reset_event_log_likelihood_with_hessian(
        cells,
        hyperparameters,
        home_multiplier_value,
        persistence,
    )
    isfinite(current.log_likelihood) ||
        throw(ArgumentError("Schur-Newton initialization was non-finite"))
    function_evaluations = 1
    gradient_evaluations = 1
    hessian_evaluations = 1
    iterations = 0
    newton_iterations = 0
    damping_steps = 0
    backtracking_steps = 0
    fallback_count = 0
    active_coordinates = 0
    parameter_step_norm = 0.0
    converged = false
    status = :max_iterations

    for iteration in 1:RESET_NEWTON_MAX_ITERATIONS
        iterations = iteration
        newton_iterations = iteration
        active = _reset_newton_active_coordinates(
            current.gradient,
            values,
            lower,
            upper,
            n_bins,
            has_transitions,
        )
        gradient_norm = _reset_projected_gradient_norm(
            current.gradient,
            active,
        )
        if gradient_norm <= RESET_NEWTON_SCORE_TOLERANCE
            converged = true
            status = :converged
            break
        end

        accepted = false
        regularizations = (
            0.0,
            1.0e-10,
            1.0e-8,
            1.0e-6,
            1.0e-4,
            1.0e-2,
            1.0,
            1.0e2,
            1.0e4,
        )
        for regularization in regularizations
            regularization > 0.0 && (damping_steps += 1)
            direction_result = _reset_schur_newton_direction(
                current.hessian,
                current.gradient,
                values,
                lower,
                upper,
                n_bins,
                has_transitions,
                regularization,
            )
            direction_result === nothing && continue
            active_coordinates = direction_result.active_coordinates
            direction = direction_result.direction
            directional_derivative =
                direction_result.directional_derivative
            step_scale = 1.0
            for backtrack in 0:RESET_NEWTON_MAX_BACKTRACKS
                candidate = clamp.(
                    values .+ step_scale .* direction,
                    lower,
                    upper,
                )
                actual_step = candidate - values
                parameter_step_norm = maximum(abs, actual_step)
                parameter_step_norm <= RESET_NEWTON_STEP_TOLERANCE && break
                actual_directional_derivative =
                    dot(current.gradient, actual_step)
                actual_directional_derivative > 0.0 || break
                candidate_hyperparameters, candidate_home_multiplier,
                candidate_persistence =
                    _unpack_reset_parameters(candidate, n_bins)
                candidate_likelihood = _reset_event_log_likelihood(
                    cells,
                    candidate_hyperparameters,
                    candidate_home_multiplier,
                    candidate_persistence,
                )
                function_evaluations += 1
                if isfinite(candidate_likelihood) &&
                    candidate_likelihood >= current.log_likelihood +
                    1.0e-4 * actual_directional_derivative
                    values = candidate
                    current = _reset_event_log_likelihood_with_hessian(
                        cells,
                        candidate_hyperparameters,
                        candidate_home_multiplier,
                        candidate_persistence,
                    )
                    function_evaluations += 1
                    gradient_evaluations += 1
                    hessian_evaluations += 1
                    accepted = true
                    backtracking_steps += backtrack
                    break
                end
                backtracking_steps += 1
                step_scale *= 0.5
            end
            accepted && break
        end

        accepted || break
        active = _reset_newton_active_coordinates(
            current.gradient,
            values,
            lower,
            upper,
            n_bins,
            has_transitions,
        )
        gradient_norm = _reset_projected_gradient_norm(
            current.gradient,
            active,
        )
        if parameter_step_norm <= RESET_NEWTON_STEP_TOLERANCE &&
            gradient_norm <=
            10.0 * RESET_NEWTON_SCORE_TOLERANCE
            converged = true
            status = :converged
            break
        end
    end

    active = _reset_newton_active_coordinates(
        current.gradient,
        values,
        lower,
        upper,
        n_bins,
        has_transitions,
    )
    gradient_norm = _reset_projected_gradient_norm(
        current.gradient,
        active,
    )
    if !converged && gradient_norm <= RESET_NEWTON_SCORE_TOLERANCE
        converged = true
        status = :converged
    end
    if !converged
        fallback_count = 1
        fallback = try
            _reset_optimize_joint_parameters(
                cells,
                values,
                n_bins;
                method=:lbfgs,
            )
        catch error
            error isa ArgumentError || rethrow()
            _reset_optimize_joint_parameters(
                cells,
                initial_values,
                n_bins;
                method=:lbfgs,
            )
        end
        if fallback.log_likelihood + 1.0e-8 >= current.log_likelihood
            hyperparameters = fallback.hyperparameters
            home_multiplier_value = fallback.home_multiplier
            persistence = fallback.persistence
            values = _pack_reset_parameters(
                hyperparameters,
                home_multiplier_value,
                persistence,
            )
            current = _reset_event_log_likelihood_with_hessian(
                cells,
                hyperparameters,
                home_multiplier_value,
                persistence,
            )
            function_evaluations += fallback.function_evaluations + 1
            gradient_evaluations += 1
            hessian_evaluations += 1
            iterations += fallback.iterations
            converged = true
            status = :fallback
            active = _reset_newton_active_coordinates(
                current.gradient,
                values,
                lower,
                upper,
                n_bins,
                has_transitions,
            )
            gradient_norm = _reset_projected_gradient_norm(
                current.gradient,
                active,
            )
        else
            status = :failed
        end
    end

    hyperparameters, home_multiplier_value, persistence =
        _unpack_reset_parameters(values, n_bins)
    active_coordinates = count(active)
    return (
        hyperparameters=hyperparameters,
        home_multiplier=home_multiplier_value,
        persistence=persistence,
        log_likelihood=current.log_likelihood,
        iterations=iterations,
        function_evaluations=function_evaluations,
        converged=converged,
        solver_metrics=_reset_newton_solver_metrics(
            gradient_evaluations=gradient_evaluations,
            hessian_evaluations=hessian_evaluations,
            newton_iterations=newton_iterations,
            damping_steps=damping_steps,
            backtracking_steps=backtracking_steps,
            fallback_count=fallback_count,
            active_coordinates=active_coordinates,
            gradient_norm=gradient_norm,
            parameter_step_norm=parameter_step_norm,
            status=status,
        ),
    )
end

function _reset_try_newton_step(
    cells::AbstractVector,
    values::AbstractVector{<:Real},
    current,
    direction::AbstractVector{<:Real},
    n_bins::Int,
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
)
    step_scale = 1.0
    for backtrack in 0:RESET_NEWTON_MAX_BACKTRACKS
        candidate = clamp.(
            values .+ step_scale .* direction,
            lower,
            upper,
        )
        actual_step = candidate - values
        parameter_step_norm = maximum(abs, actual_step)
        parameter_step_norm <= RESET_NEWTON_STEP_TOLERANCE && break
        actual_directional_derivative =
            dot(current.gradient, actual_step)
        actual_directional_derivative > 0.0 || break
        candidate_hyperparameters, candidate_home_multiplier,
        candidate_persistence =
            _unpack_reset_parameters(candidate, n_bins)
        candidate_likelihood = _reset_event_log_likelihood(
            cells,
            candidate_hyperparameters,
            candidate_home_multiplier,
            candidate_persistence,
        )
        if isfinite(candidate_likelihood) &&
            candidate_likelihood >= current.log_likelihood +
            1.0e-4 * actual_directional_derivative
            candidate_current = _reset_event_log_likelihood_with_hessian(
                cells,
                candidate_hyperparameters,
                candidate_home_multiplier,
                candidate_persistence,
            )
            return (
                accepted=true,
                values=candidate,
                current=candidate_current,
                function_evaluations=2,
                gradient_evaluations=1,
                hessian_evaluations=1,
                parameter_step_norm=parameter_step_norm,
                backtracking_steps=backtrack,
            )
        end
        step_scale *= 0.5
    end
    return (
        accepted=false,
        values=values,
        current=current,
        function_evaluations=0,
        gradient_evaluations=0,
        hessian_evaluations=0,
        parameter_step_norm=0.0,
        backtracking_steps=RESET_NEWTON_MAX_BACKTRACKS,
    )
end

function _reset_optimize_block_newton(
    cells::AbstractVector,
    initial_values::AbstractVector{<:Real},
    n_bins::Int,
)
    lower, upper = _reset_joint_parameter_bounds(n_bins)
    values = [
        clamp(
            Float64(value),
            lower[index],
            upper[index],
        )
        for (index, value) in enumerate(initial_values)
    ]
    has_transitions = any(length(cell.counts) > 1 for cell in cells)
    hyperparameters, home_multiplier_value, persistence =
        _unpack_reset_parameters(values, n_bins)
    current = _reset_event_log_likelihood_with_hessian(
        cells,
        hyperparameters,
        home_multiplier_value,
        persistence,
    )
    isfinite(current.log_likelihood) ||
        throw(ArgumentError("block-Newton initialization was non-finite"))

    function_evaluations = 1
    gradient_evaluations = 1
    hessian_evaluations = 1
    iterations = 0
    newton_iterations = 0
    damping_steps = 0
    backtracking_steps = 0
    schur_corrections = 0
    fallback_count = 0
    active_coordinates = 0
    parameter_step_norm = 0.0
    converged = false
    status = :max_iterations
    regularizations = (
        0.0,
        1.0e-10,
        1.0e-8,
        1.0e-6,
        1.0e-4,
        1.0e-2,
        1.0,
        1.0e2,
        1.0e4,
    )

    for sweep in 1:RESET_BLOCK_NEWTON_MAX_SWEEPS
        iterations = sweep
        newton_iterations = sweep
        active = _reset_newton_active_coordinates(
            current.gradient,
            values,
            lower,
            upper,
            n_bins,
            has_transitions,
        )
        gradient_norm = _reset_projected_gradient_norm(
            current.gradient,
            active,
        )
        if gradient_norm <= RESET_NEWTON_SCORE_TOLERANCE
            converged = true
            status = :converged
            break
        end

        sweep_accepted = false
        sweep_step_norm = 0.0
        for block_kind in (:shared, :bins)
            active = _reset_newton_active_coordinates(
                current.gradient,
                values,
                lower,
                upper,
                n_bins,
                has_transitions,
            )
            block_has_free_coordinate = if block_kind === :shared
                any(
                    !active[index]
                    for index in (2 * n_bins + 1, 2 * n_bins + 2)
                )
            else
                any(
                    !active[index]
                    for time_bin in 1:n_bins
                    for index in (time_bin, n_bins + time_bin)
                )
            end
            block_has_free_coordinate || continue

            accepted = false
            for regularization in regularizations
                regularization > 0.0 && (damping_steps += 1)
                direction_result = _reset_block_newton_direction(
                    current.hessian,
                    current.gradient,
                    values,
                    lower,
                    upper,
                    n_bins,
                    has_transitions,
                    regularization,
                    block_kind,
                )
                direction_result === nothing && continue
                active_coordinates = direction_result.active_coordinates
                trial = _reset_try_newton_step(
                    cells,
                    values,
                    current,
                    direction_result.direction,
                    n_bins,
                    lower,
                    upper,
                )
                function_evaluations += trial.function_evaluations
                gradient_evaluations += trial.gradient_evaluations
                hessian_evaluations += trial.hessian_evaluations
                backtracking_steps += trial.backtracking_steps
                if trial.accepted
                    values = trial.values
                    current = trial.current
                    sweep_step_norm = max(
                        sweep_step_norm,
                        trial.parameter_step_norm,
                    )
                    sweep_accepted = true
                    accepted = true
                    break
                end
            end
            accepted || break
        end

        correction_accepted = false
        for regularization in regularizations
            regularization > 0.0 && (damping_steps += 1)
            direction_result = _reset_schur_newton_direction(
                current.hessian,
                current.gradient,
                values,
                lower,
                upper,
                n_bins,
                has_transitions,
                regularization,
            )
            direction_result === nothing && continue
            active_coordinates = direction_result.active_coordinates
            trial = _reset_try_newton_step(
                cells,
                values,
                current,
                direction_result.direction,
                n_bins,
                lower,
                upper,
            )
            function_evaluations += trial.function_evaluations
            gradient_evaluations += trial.gradient_evaluations
            hessian_evaluations += trial.hessian_evaluations
            backtracking_steps += trial.backtracking_steps
            if trial.accepted
                values = trial.values
                current = trial.current
                sweep_step_norm = max(
                    sweep_step_norm,
                    trial.parameter_step_norm,
                )
                correction_accepted = true
                schur_corrections += 1
                break
            end
        end
        sweep_accepted |= correction_accepted
        sweep_accepted || break
        parameter_step_norm = sweep_step_norm

        active = _reset_newton_active_coordinates(
            current.gradient,
            values,
            lower,
            upper,
            n_bins,
            has_transitions,
        )
        gradient_norm = _reset_projected_gradient_norm(
            current.gradient,
            active,
        )
        if parameter_step_norm <= RESET_NEWTON_STEP_TOLERANCE &&
            gradient_norm <= 10.0 * RESET_NEWTON_SCORE_TOLERANCE
            converged = true
            status = :converged
            break
        end
    end

    active = _reset_newton_active_coordinates(
        current.gradient,
        values,
        lower,
        upper,
        n_bins,
        has_transitions,
    )
    gradient_norm = _reset_projected_gradient_norm(
        current.gradient,
        active,
    )
    if !converged
        fallback_count = 1
        fallback = try
            _reset_optimize_joint_parameters(
                cells,
                values,
                n_bins;
                method=:lbfgs,
            )
        catch error
            error isa ArgumentError || rethrow()
            _reset_optimize_joint_parameters(
                cells,
                initial_values,
                n_bins;
                method=:lbfgs,
            )
        end
        if fallback.log_likelihood + 1.0e-8 >= current.log_likelihood
            hyperparameters = fallback.hyperparameters
            home_multiplier_value = fallback.home_multiplier
            persistence = fallback.persistence
            values = _pack_reset_parameters(
                hyperparameters,
                home_multiplier_value,
                persistence,
            )
            current = _reset_event_log_likelihood_with_hessian(
                cells,
                hyperparameters,
                home_multiplier_value,
                persistence,
            )
            function_evaluations += fallback.function_evaluations + 1
            gradient_evaluations += 1
            hessian_evaluations += 1
            iterations += fallback.iterations
            converged = true
            status = :fallback
            active = _reset_newton_active_coordinates(
                current.gradient,
                values,
                lower,
                upper,
                n_bins,
                has_transitions,
            )
            gradient_norm = _reset_projected_gradient_norm(
                current.gradient,
                active,
            )
        else
            status = :failed
        end
    end

    hyperparameters, home_multiplier_value, persistence =
        _unpack_reset_parameters(values, n_bins)
    return (
        hyperparameters=hyperparameters,
        home_multiplier=home_multiplier_value,
        persistence=persistence,
        log_likelihood=current.log_likelihood,
        iterations=iterations,
        function_evaluations=function_evaluations,
        converged=converged,
        solver_metrics=_reset_newton_solver_metrics(
            solver=:block_newton,
            gradient_evaluations=gradient_evaluations,
            hessian_evaluations=hessian_evaluations,
            newton_iterations=newton_iterations,
            damping_steps=damping_steps,
            backtracking_steps=backtracking_steps,
            schur_corrections=schur_corrections,
            fallback_count=fallback_count,
            active_coordinates=count(active),
            gradient_norm=gradient_norm,
            parameter_step_norm=parameter_step_norm,
            status=status,
        ),
    )
end

function _fit_reset_outcome_parameters_with_diagnostics(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    kind::Symbol,
    ;
    method::PriorFitMethod=HybridFit(),
)
    solver = _fit_method_symbol(method)
    _reset_validate_solver(solver)
    isempty(seasons) && throw(ArgumentError("historical likelihood requires seasons"))
    length(seasons) <= MAX_HISTORICAL_SEASONS ||
        throw(ArgumentError("historical likelihood supports at most three seasons"))
    n_bins = length(time_edges) - 1
    n_bins > 0 || throw(ArgumentError("historical likelihood requires time bins"))
    moment_fit = solver === :moment ?
        _reset_iterated_moment_fit(
            byseason,
            seasons,
            kind,
            n_bins,
        ) :
        nothing
    initial = solver === :moment ?
        _reset_moment_parameter_vector(moment_fit) :
        solver in (:moment_lbfgs, :hybrid) ?
        _reset_moment_initial_parameter_vector(
            byseason,
            seasons,
            kind,
            n_bins,
        ) :
        solver in (:block_newton, :schur_newton) ?
        _reset_moment_initial_parameter_vector(
            byseason,
            seasons,
            kind,
            n_bins,
        ) :
        _initial_reset_parameter_vector(byseason, seasons, kind, n_bins)
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
    solver_metrics = _reset_newton_solver_metrics(
        solver=solver,
        status=:not_run,
    )

    solver === :moment && begin
        hyperparameters, home_multiplier_value, persistence =
            _unpack_reset_parameters(initial, n_bins)
        iterations = moment_fit.iterations
        function_evaluations = 1
        converged = moment_fit.converged
        solver_metrics = _reset_newton_solver_metrics(
            solver=:moment,
            newton_iterations=moment_fit.iterations,
            status=moment_fit.converged ? :moment : :max_iterations,
        )
    end

    if solver in (:em_ecme, :em_lbfgs, :hybrid)
        conditional_method = solver === :em_ecme ? :nelder_mead : :lbfgs
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
                em_persistence;
                method=conditional_method,
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
    end

    solver === :block_newton && begin
        block_newton = _reset_optimize_block_newton(
            likelihood_cells,
            initial,
            n_bins,
        )
        if block_newton.log_likelihood + 1.0e-8 < log_likelihood
            throw(ArgumentError(
                "$kind block-Newton optimization decreased the observed likelihood",
            ))
        end
        hyperparameters = block_newton.hyperparameters
        home_multiplier_value = block_newton.home_multiplier
        persistence = block_newton.persistence
        log_likelihood = block_newton.log_likelihood
        iterations = block_newton.iterations
        function_evaluations = block_newton.function_evaluations
        converged = block_newton.converged
        solver_metrics = block_newton.solver_metrics
    end

    solver === :schur_newton && begin
        newton = _reset_optimize_schur_newton(
            likelihood_cells,
            initial,
            n_bins,
        )
        if newton.log_likelihood + 1.0e-8 < log_likelihood
            throw(ArgumentError(
                "$kind Schur-Newton optimization decreased the observed likelihood",
            ))
        end
        hyperparameters = newton.hyperparameters
        home_multiplier_value = newton.home_multiplier
        persistence = newton.persistence
        log_likelihood = newton.log_likelihood
        iterations = newton.iterations
        function_evaluations = newton.function_evaluations
        converged = newton.converged
        solver_metrics = newton.solver_metrics
    end

    solver in (:direct_lbfgs, :direct_bfgs, :moment_lbfgs, :hybrid) && begin
        method = solver === :direct_bfgs ? :bfgs : :lbfgs
        direct = _reset_optimize_joint_parameters(
            likelihood_cells,
            solver === :hybrid ? begin
                _pack_reset_parameters(
                    hyperparameters,
                    home_multiplier_value,
                    persistence,
                )
            end : initial,
            n_bins;
            method=method,
        )
        if direct.log_likelihood + 1.0e-8 < log_likelihood
            solver === :hybrid || throw(ArgumentError(
                "$kind joint likelihood polishing decreased the observed likelihood",
            ))
        else
            hyperparameters = direct.hyperparameters
            home_multiplier_value = direct.home_multiplier
            persistence = direct.persistence
            log_likelihood = direct.log_likelihood
        end
        function_evaluations += direct.function_evaluations
        iterations += direct.iterations
        converged = true
    end

    converged ||
        throw(ArgumentError(
            "$kind historical $solver optimization did not converge",
        ))
    solver_metrics = solver_metrics.status === :not_run ?
        _reset_newton_solver_metrics(
            solver=solver,
            status=:converged,
        ) :
        solver_metrics
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
        solver === :moment ? :moment : :converged,
        boundary_parameters,
    )
    return (
        hyperparameters,
        home_multiplier_value,
        persistence,
        diagnostics,
        solver_metrics,
    )
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

function _reset_moment_blocks(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
)
    blocks = NamedTuple[]
    for time_bin in 1:n_bins
        cells = _season_cells(byseason, seasons, kind, time_bin)
        pairs = _sequential_cells(cells, seasons)
        push!(
            blocks,
            (
                time_bin=time_bin,
                cells=cells,
                pairs=pairs,
                moments=_moment_summary(cells, pairs),
                season_moments=_season_moment_summaries(cells, seasons),
                transition_moments=_transition_moment_summaries(cells, pairs),
            ),
        )
    end
    return blocks
end

function _reset_moment_rate_mean(
    cells::AbstractVector,
    mean_rate::Float64,
    latent_variance::Float64,
    persistence::Float64,
    home_multiplier::Float64,
)
    groups = Dict{String,Vector{Any}}()
    for cell in cells
        push!(get!(groups, cell.team, Any[]), cell)
    end

    numerator = 0.0
    denominator = 0.0
    for group in values(groups)
        count = length(group)
        rates = zeros(Float64, count)
        covariance = zeros(Float64, count, count)
        for first_index in 1:count
            first_cell = group[first_index]
            first_exposure = first_cell.away_exposure +
                home_multiplier * first_cell.home_exposure
            first_exposure > 0.0 || continue
            rates[first_index] = (
                first_cell.away_count + first_cell.home_count
            ) / first_exposure
            for second_index in 1:count
                second_cell = group[second_index]
                second_exposure = second_cell.away_exposure +
                    home_multiplier * second_cell.home_exposure
                second_exposure > 0.0 || continue
                if first_index == second_index
                    covariance[first_index, second_index] =
                        mean_rate / first_exposure + latent_variance
                else
                    distance = abs(
                        first_cell.season - second_cell.season,
                    )
                    covariance[first_index, second_index] =
                        latent_variance * persistence^distance
                end
            end
        end

        scale = 1.0
        for index in 1:count
            scale = max(scale, covariance[index, index])
        end
        for index in 1:count
            covariance[index, index] += 1.0e-10 * scale
        end
        factor = _reset_positive_definite_factor(covariance)
        if factor === nothing
            for index in 1:count
                variance = covariance[index, index]
                variance > 0.0 || continue
                numerator += rates[index] / variance
                denominator += 1.0 / variance
            end
        else
            unit_vector = ones(Float64, count)
            solved_ones = factor \ unit_vector
            solved_rates = factor \ rates
            numerator += dot(unit_vector, solved_rates)
            denominator += dot(unit_vector, solved_ones)
        end
    end

    denominator > 0.0 || return mean_rate
    estimate = numerator / denominator
    return isfinite(estimate) && estimate > 0.0 ?
        estimate : mean_rate
end

function _reset_moment_home_multiplier(
    blocks::AbstractVector,
    means::AbstractVector{<:Real},
    home_multiplier::Float64,
)
    numerator = 0.0
    denominator = 0.0
    for (block_index, block) in enumerate(blocks)
        mean_rate = max(Float64(means[block_index]), 1.0e-12)
        for cell in block.cells
            away_exposure = Float64(cell.away_exposure)
            home_exposure = Float64(cell.home_exposure)
            away_exposure > 0.0 && home_exposure > 0.0 || continue
            variance =
                mean_rate * home_multiplier / home_exposure +
                mean_rate * home_multiplier^2 / away_exposure
            variance = max(variance, 1.0e-12)
            weight = 1.0 / variance
            away_rate = cell.away_count / away_exposure
            home_rate = cell.home_count / home_exposure
            numerator += weight * home_rate
            denominator += weight * away_rate
        end
    end
    denominator > 0.0 || return home_multiplier
    estimate = numerator / denominator
    return isfinite(estimate) && estimate > 0.0 ?
        clamp(estimate, 0.05, 20.0) : home_multiplier
end

function _reset_moment_factorial_variance(
    mean_rate::Float64,
    latent_variance::Float64,
    exposure::Float64,
)
    exposure > 0.0 || return 1.0
    mean_rate > 0.0 || return 1.0
    latent_variance = max(
        latent_variance,
        mean_rate^2 / RESET_GAMMA_SHAPE_UPPER,
    )
    shape = clamp(
        mean_rate^2 / latent_variance,
        RESET_GAMMA_SHAPE_LOWER,
        RESET_GAMMA_SHAPE_UPPER,
    )
    second_rate_moment = mean_rate^2 + latent_variance
    third_rate_moment =
        mean_rate^3 * (1.0 + 1.0 / shape) * (1.0 + 2.0 / shape)
    fourth_rate_moment =
        mean_rate^4 *
        (1.0 + 1.0 / shape) *
        (1.0 + 2.0 / shape) *
        (1.0 + 3.0 / shape)
    variance =
        fourth_rate_moment - second_rate_moment^2 +
        4.0 * third_rate_moment / exposure +
        2.0 * second_rate_moment / exposure^2
    return max(variance, 1.0e-12)
end

function _reset_moment_latent_variance(
    block,
    mean_rate::Float64,
    latent_variance::Float64,
    home_multiplier::Float64,
)
    numerator = 0.0
    denominator = 0.0
    for cell in block.cells
        exposure = cell.away_exposure +
            home_multiplier * cell.home_exposure
        exposure > 0.0 || continue
        count = cell.away_count + cell.home_count
        factorial_moment = count * (count - 1.0) / exposure^2
        weight = 1.0 / _reset_moment_factorial_variance(
            mean_rate,
            latent_variance,
            exposure,
        )
        numerator += weight * factorial_moment
        denominator += weight
    end
    denominator > 0.0 || return latent_variance
    second_moment = numerator / denominator
    estimate = second_moment - mean_rate^2
    lower = mean_rate^2 / RESET_GAMMA_SHAPE_UPPER
    upper = mean_rate^2 / RESET_GAMMA_SHAPE_LOWER
    return isfinite(estimate) ?
        clamp(estimate, lower, upper) :
        clamp(latent_variance, lower, upper)
end

function _reset_moment_pair_variance(
    mean_rate::Float64,
    latent_variance::Float64,
    persistence::Float64,
    previous_exposure::Float64,
    current_exposure::Float64,
)
    previous_exposure > 0.0 && current_exposure > 0.0 ||
        return 1.0
    mean_rate > 0.0 || return 1.0
    latent_variance = max(
        latent_variance,
        mean_rate^2 / RESET_GAMMA_SHAPE_UPPER,
    )
    shape = clamp(
        mean_rate^2 / latent_variance,
        RESET_GAMMA_SHAPE_LOWER,
        RESET_GAMMA_SHAPE_UPPER,
    )
    second_rate_moment = mean_rate^2 + latent_variance
    third_rate_moment =
        mean_rate^3 * (1.0 + 1.0 / shape) * (1.0 + 2.0 / shape)
    fourth_rate_moment =
        mean_rate^4 *
        (1.0 + 1.0 / shape) *
        (1.0 + 2.0 / shape) *
        (1.0 + 3.0 / shape)
    persistence_value = clamp(persistence, 0.0, 1.0)
    mean_product =
        mean_rate^2 + persistence_value * latent_variance
    mixed_second =
        (1.0 - persistence_value) * second_rate_moment^2 +
        persistence_value * fourth_rate_moment
    mixed_second_one =
        (1.0 - persistence_value) * second_rate_moment * mean_rate +
        persistence_value * third_rate_moment
    second_product =
        mixed_second +
        mixed_second_one / previous_exposure +
        mixed_second_one / current_exposure +
        mean_product / (previous_exposure * current_exposure)
    return max(second_product - mean_product^2, 1.0e-12)
end

function _reset_moment_persistence(
    blocks::AbstractVector,
    means::AbstractVector{<:Real},
    variances::AbstractVector{<:Real},
    home_multiplier::Float64,
    persistence::Float64,
)
    numerator = 0.0
    denominator = 0.0
    for (block_index, block) in enumerate(blocks)
        mean_rate = max(Float64(means[block_index]), 1.0e-12)
        latent_variance = max(Float64(variances[block_index]), 0.0)
        latent_variance > 0.0 || continue
        for pair in block.pairs
            previous = pair.previous
            current = pair.current
            previous_exposure = previous.away_exposure +
                home_multiplier * previous.home_exposure
            current_exposure = current.away_exposure +
                home_multiplier * current.home_exposure
            previous_exposure > 0.0 && current_exposure > 0.0 || continue
            previous_count = previous.away_count + previous.home_count
            current_count = current.away_count + current.home_count
            product = previous_count * current_count /
                (previous_exposure * current_exposure)
            weight = 1.0 / _reset_moment_pair_variance(
                mean_rate,
                latent_variance,
                persistence,
                previous_exposure,
                current_exposure,
            )
            numerator += weight * latent_variance * (
                product - mean_rate^2
            )
            denominator += weight * latent_variance^2
        end
    end
    denominator > 0.0 || return persistence
    estimate = numerator / denominator
    return isfinite(estimate) ?
        clamp(estimate, 0.0, 1.0) :
        persistence
end

function _reset_iterated_moment_parameters(
    blocks::AbstractVector;
    max_iterations::Int=RESET_MOMENT_MAX_ITERATIONS,
    tolerance::Float64=RESET_MOMENT_TOLERANCE,
    damping::Float64=RESET_MOMENT_DAMPING,
)
    max_iterations > 0 ||
        throw(ArgumentError("moment max_iterations must be positive"))
    tolerance > 0.0 ||
        throw(ArgumentError("moment tolerance must be positive"))
    0.0 < damping <= 1.0 ||
        throw(ArgumentError("moment damping must be in (0, 1]"))
    isempty(blocks) &&
        throw(ArgumentError("moment estimation requires at least one bin"))

    initial_means, initial_variances, initial_home, initial_persistence =
        _reset_moment_parameters(blocks)
    means = [
        clamp(Float64(value), 1.0e-10, 10.0)
        for value in initial_means
    ]
    variances = [
        max(Float64(value), mean^2 / RESET_GAMMA_SHAPE_UPPER)
        for (value, mean) in zip(initial_variances, means)
    ]
    home_multiplier = clamp(initial_home, 0.05, 20.0)
    persistence = clamp(initial_persistence, 0.0, 1.0)

    has_raw_cells = all(hasproperty(block, :cells) for block in blocks)
    has_raw_cells &= all(hasproperty(block, :pairs) for block in blocks)
    if !has_raw_cells
        return (
            means=means,
            variances=variances,
            home_multiplier=home_multiplier,
            persistence=persistence,
            iterations=0,
            converged=true,
        )
    end

    converged = false
    iterations = 0
    for iteration in 1:max_iterations
        iterations = iteration
        candidate_home = _reset_moment_home_multiplier(
            blocks,
            means,
            home_multiplier,
        )
        candidate_means = [
            _reset_moment_rate_mean(
                block.cells,
                means[index],
                variances[index],
                persistence,
                candidate_home,
            )
            for (index, block) in enumerate(blocks)
        ]
        candidate_variances = [
            _reset_moment_latent_variance(
                block,
                candidate_means[index],
                variances[index],
                candidate_home,
            )
            for (index, block) in enumerate(blocks)
        ]
        candidate_persistence = _reset_moment_persistence(
            blocks,
            candidate_means,
            candidate_variances,
            candidate_home,
            persistence,
        )

        next_means = [
            clamp(
                exp(
                    (1.0 - damping) * log(means[index]) +
                    damping * log(max(candidate_means[index], 1.0e-10)),
                ),
                1.0e-10,
                10.0,
            )
            for index in eachindex(means)
        ]
        next_variances = [
            begin
                lower = next_means[index]^2 / RESET_GAMMA_SHAPE_UPPER
                upper = next_means[index]^2 / RESET_GAMMA_SHAPE_LOWER
                clamp(
                    exp(
                        (1.0 - damping) * log(max(variances[index], lower)) +
                        damping * log(max(candidate_variances[index], lower)),
                    ),
                    lower,
                    upper,
                )
            end
            for index in eachindex(variances)
        ]
        next_home = clamp(
            exp(
                (1.0 - damping) * log(home_multiplier) +
                damping * log(candidate_home),
            ),
            0.05,
            20.0,
        )
        next_persistence = clamp(
            (1.0 - damping) * persistence +
            damping * candidate_persistence,
            0.0,
            1.0,
        )

        change = maximum(
            vcat(
                [
                    abs(log(next_means[index] / means[index]))
                    for index in eachindex(means)
                ],
                [
                    abs(log(next_variances[index] / variances[index]))
                    for index in eachindex(variances)
                ],
                abs(log(next_home / home_multiplier)),
                abs(next_persistence - persistence),
            ),
        )
        means = next_means
        variances = next_variances
        home_multiplier = next_home
        persistence = next_persistence
        if change <= tolerance
            converged = true
            break
        end
    end

    return (
        means=means,
        variances=variances,
        home_multiplier=home_multiplier,
        persistence=persistence,
        iterations=iterations,
        converged=converged,
    )
end

function _reset_iterated_moment_fit(
    byseason::AbstractDict,
    seasons::AbstractVector{<:Integer},
    kind::Symbol,
    n_bins::Int,
)
    blocks = _reset_moment_blocks(
        byseason,
        seasons,
        kind,
        n_bins,
    )
    return _reset_iterated_moment_parameters(blocks)
end

function _reset_moment_parameter_vector(moment_fit)
    bounded_means = [
        clamp(Float64(value), 1.0e-10, 10.0)
        for value in moment_fit.means
    ]
    bounded_shapes = [
        clamp(
            mean_value^2 / max(Float64(variance), mean_value^2 * 1.0e-8),
            RESET_GAMMA_SHAPE_LOWER,
            RESET_GAMMA_SHAPE_UPPER,
        )
        for (mean_value, variance) in
            zip(bounded_means, moment_fit.variances)
    ]
    return vcat(
        log.(bounded_means),
        log.(bounded_shapes),
        [
            _reset_logit_probability(
                clamp(moment_fit.persistence, 0.0, 1.0),
            ),
            log(clamp(moment_fit.home_multiplier, 0.05, 20.0)),
        ],
    )
end

function _fit_reset_outcome_parameters(
    byseason::Dict{Int,HazardSufficientStats},
    seasons::AbstractVector{<:Integer},
    time_edges::AbstractVector{<:Real},
    kind::Symbol,
    ;
    method::PriorFitMethod=HybridFit(),
)
    parameters, home_multiplier_value, persistence, _, _ =
        _fit_reset_outcome_parameters_with_diagnostics(
            byseason,
            seasons,
            time_edges,
            kind,
            method=method,
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
across that outcome's hazard curve. Solver failure is reported rather than
silently replaced with a default prior. The The typed `method` keyword selects `HybridFit()` by default, or one of
`MomentFit()`, `EMECMEFit()`, `EMLBFGSFit()`, `DirectLBFGSFit()`,
`DirectBFGSFit()`, `MomentLBFGSFit()`, `BlockNewtonFit()`, or
`SchurNewtonFit()`.
The standalone moment estimate is not an MLE; its diagnostics report
`status=:moment` and its likelihood is the exact likelihood evaluated at the
moment estimate.
"""
function fit_empirical_bayes_prior(
    historical_drives::AbstractDataFrame;
    time_edges=DEFAULT_TIME_EDGES,
    max_seasons::Int=3,
    current_season::Union{Nothing,Integer}=nothing,
    method::PriorFitMethod=HybridFit(),
    _return_solver_metrics::Bool=false,
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
    td_hyper, td_home_multiplier, td_persistence, td_diagnostics,
    td_solver_metrics =
        _fit_reset_outcome_parameters_with_diagnostics(
            byseason,
            seasons,
            edges,
            :td,
            method=method,
        )
    defensive_hyper, defensive_home_multiplier, defensive_persistence,
        defensive_diagnostics, defensive_solver_metrics =
        _fit_reset_outcome_parameters_with_diagnostics(
            byseason,
            seasons,
            edges,
            :defensive,
            method=method,
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
    return _return_solver_metrics ?
        (
            prior=prior,
            solver_metrics=(
                td=td_solver_metrics,
                defensive=defensive_solver_metrics,
            ),
        ) :
        prior
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
    max_seasons::Int=3,
    current_season::Union{Nothing,Integer}=nothing,
    method::PriorFitMethod=HybridFit(),
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
