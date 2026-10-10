const SurvivorParameterKey = Tuple{Symbol,String}

struct SurvivorParameterSystem
    keys::Vector{SurvivorParameterKey}
    indices::Dict{SurvivorParameterKey,Int}
    log_mean::Vector{Float64}
    variance::Vector{Float64}
end

struct SurvivorCandidateDerivatives
    base_probability::Float64
    gradient::Vector{Float64}
    hessian_covariance::Float64
end

struct SurvivorObjectiveInputs
    parameters::SurvivorParameterSystem
    derivatives::Vector{SurvivorCandidateDerivatives}
    covariance_gradient_gram::Matrix{Float64}
end

function SurvivorObjectiveInputs(
    parameters::SurvivorParameterSystem,
    derivatives::AbstractVector{<:SurvivorCandidateDerivatives},
)
    derivative_values = collect(derivatives)
    n_candidates = length(derivative_values)
    n_parameters = length(parameters.keys)
    all(
        length(derivative.gradient) == n_parameters
        for derivative in derivative_values
    ) || throw(ArgumentError(
        "survivor derivative gradients must match parameter coordinates",
    ))
    covariance_gradient_gram = zeros(Float64, n_candidates, n_candidates)
    for first in 1:n_candidates, second in first:n_candidates
        covariance_gradient_gram[first, second] = sum(
            derivative_values[first].gradient[index] *
            parameters.variance[index] *
            derivative_values[second].gradient[index]
            for index in 1:n_parameters
        )
        covariance_gradient_gram[second, first] =
            covariance_gradient_gram[first, second]
    end
    all(isfinite, covariance_gradient_gram) ||
        throw(ArgumentError(
            "survivor covariance gradient contractions must be finite",
        ))
    return SurvivorObjectiveInputs(
        parameters,
        derivative_values,
        covariance_gradient_gram,
    )
end

function _survivor_matchup_parameter_requests(
    model::HazardModel,
    home_team,
    away_team,
)
    return (
        (:td, String(home_team), true),
        (:defensive, String(away_team), false),
        (:td, String(away_team), false),
        (:defensive, String(home_team), true),
    )
end

function _survivor_candidate_matchup(row)
    team = String(row.team)
    opponent = String(row.opponent)
    return Bool(row.is_home) ? (team, opponent) : (opponent, team)
end

function _survivor_parameter_system(
    model::HazardModel,
    data::AbstractDataFrame,
    cache::_HazardLogMomentCache,
)
    keys = SurvivorParameterKey[]
    seen = Set{SurvivorParameterKey}()
    for row in eachrow(data)
        home_team, away_team = _survivor_candidate_matchup(row)
        requests = _survivor_matchup_parameter_requests(
            model,
            home_team,
            away_team,
        )
        for (kind, team, _) in requests
            key = (kind, team)
            key in seen && continue
            push!(seen, key)
            push!(keys, key)
        end
    end

    indices = Dict(key => index for (index, key) in enumerate(keys))
    log_mean = Float64[]
    variance = Float64[]
    for (kind, team) in keys
        moments = _hazard_log_moments(
            model,
            kind,
            team,
            cache=cache,
        )
        isfinite(moments[1]) && isfinite(moments[2]) ||
            throw(ArgumentError("survivor hazard parameter moments must be finite"))
        moments[2] >= 0.0 ||
            throw(ArgumentError("survivor hazard parameter variances must be nonnegative"))
        push!(log_mean, moments[1])
        push!(variance, moments[2])
    end
    return SurvivorParameterSystem(keys, indices, log_mean, variance)
end

function _survivor_candidate_local_parameters(
    model::HazardModel,
    row,
    parameters::SurvivorParameterSystem,
)
    home_team, away_team = _survivor_candidate_matchup(row)
    requests = _survivor_matchup_parameter_requests(
        model,
        home_team,
        away_team,
    )
    local_mean = Float64[]
    global_indices = Int[]
    for (kind, team, home) in requests
        key = (kind, team)
        index = parameters.indices[key]
        multiplier = home_multiplier(model.prior, kind)
        push!(
            local_mean,
            parameters.log_mean[index] + (home ? log(multiplier) : 0.0),
        )
        push!(global_indices, index)
    end
    return local_mean, global_indices
end

function _survivor_matchup_derivatives(
    model::HazardModel,
    marks::ScoreMarks,
    row,
    parameters::SurvivorParameterSystem;
    horizon::Real,
)
    local_mean, global_indices = _survivor_candidate_local_parameters(
        model,
        row,
        parameters,
    )
    derivatives = _game_probability_derivatives(
        local_mean,
        model.prior,
        marks;
        horizon=horizon,
    )
    base_probability = derivatives.probability
    isfinite(base_probability) &&
        0.0 <= base_probability <= 1.0 ||
        throw(ArgumentError("survivor base win probabilities must be in [0, 1]"))

    n_parameters = length(parameters.keys)
    gradient = zeros(Float64, n_parameters)
    local_gradient = derivatives.gradient
    local_hessian = derivatives.hessian
    for (local_index, global_index) in enumerate(global_indices)
        gradient[global_index] += local_gradient[local_index]
    end
    all(isfinite, gradient) ||
        throw(ArgumentError("survivor candidate gradients must be finite"))
    hessian_covariance = sum(
        local_hessian[first_local, second_local] *
            parameters.variance[global_indices[first_local]]
        for first_local in eachindex(global_indices),
            second_local in eachindex(global_indices)
        if global_indices[first_local] == global_indices[second_local]
    )
    isfinite(hessian_covariance) ||
        throw(ArgumentError("survivor Hessian covariance contraction is not finite"))

    return SurvivorCandidateDerivatives(
        Float64(base_probability),
        gradient,
        Float64(hessian_covariance),
    )
end

function _survivor_oriented_derivatives(
    home_derivatives::SurvivorCandidateDerivatives,
    is_home::Bool,
)
    is_home && return home_derivatives
    return SurvivorCandidateDerivatives(
        1.0 - home_derivatives.base_probability,
        -home_derivatives.gradient,
        -home_derivatives.hessian_covariance,
    )
end

function _survivor_candidate_derivatives(
    model::HazardModel,
    marks::ScoreMarks,
    row,
    parameters::SurvivorParameterSystem;
    horizon::Real,
)
    home_derivatives = _survivor_matchup_derivatives(
        model,
        marks,
        row,
        parameters;
        horizon=horizon,
    )
    return _survivor_oriented_derivatives(
        home_derivatives,
        Bool(row.is_home),
    )
end

function _survivor_objective_inputs(
    model::HazardModel,
    marks::ScoreMarks,
    data::AbstractDataFrame;
    horizon::Real,
)
    cache = _HazardLogMomentCache(model)
    parameters = _survivor_parameter_system(model, data, cache)
    matchup_derivatives = Dict{
        Tuple{String,String},
        SurvivorCandidateDerivatives,
    }()
    derivatives = [
        let
            home_team, away_team = _survivor_candidate_matchup(row)
            matchup_key = (String(home_team), String(away_team))
            home_derivatives = get!(
                matchup_derivatives,
                matchup_key,
            ) do
                _survivor_matchup_derivatives(
                    model,
                    marks,
                    row,
                    parameters;
                    horizon=horizon,
                )
            end
            _survivor_oriented_derivatives(
                home_derivatives,
                Bool(row.is_home),
            )
        end
        for row in eachrow(data)
    ]
    return SurvivorObjectiveInputs(parameters, derivatives)
end
