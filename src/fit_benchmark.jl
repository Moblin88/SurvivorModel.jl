const FIT_BENCHMARK_TIME_EDGES = (0.0, 120.0, 240.0, Inf)
const FIT_BENCHMARK_SYNTHETIC_PERSISTENCE = 0.7
const FIT_BENCHMARK_SYNTHETIC_HOME_MULTIPLIER = 1.35
const FIT_BENCHMARK_SYNTHETIC_TD_MEAN = 0.0008
const FIT_BENCHMARK_SYNTHETIC_DEFENSIVE_MEAN = 0.0012
const FIT_BENCHMARK_SYNTHETIC_GAMMA_SHAPE = 4.0

const FIT_BENCHMARK_METHODS = (
    EMECMEFit(),
    EMLBFGSFit(),
    DirectLBFGSFit(),
    DirectBFGSFit(),
    MomentFit(),
    MomentLBFGSFit(),
    HybridFit(),
    BlockNewtonFit(),
    SchurNewtonFit(),
)

function _synthetic_unit_interval(seed::Integer)
    value = sin(Float64(seed) * 12.9898 + 78.233) * 43758.5453123
    return value - floor(value)
end

function _synthetic_gamma_rate(
    seed::Integer,
    mean_rate::Float64,
    shape::Int,
)
    product = 1.0
    for offset in 1:shape
        uniform = max(
            _synthetic_unit_interval(seed + offset),
            eps(Float64),
        )
        product *= uniform
    end
    return mean_rate * (-log(product) / shape)
end

const FIT_RECOVERY_TIME_EDGES = (0.0, 120.0, 240.0, Inf)
const FIT_RECOVERY_SEASONS = (2021, 2022, 2023)
const FIT_RECOVERY_TEAM_COUNT = 32
const FIT_RECOVERY_GAMES_PER_TEAM = 17
const FIT_RECOVERY_APPROX_DRIVES_PER_TEAM_GAME = 11
const FIT_RECOVERY_GAME_HORIZON_SECONDS = Int(GAME_CLOCK_SECONDS)
const FIT_RECOVERY_TD_PERSISTENCE = 0.70
const FIT_RECOVERY_DEFENSIVE_PERSISTENCE = 0.55
const FIT_RECOVERY_TD_HOME_MULTIPLIER = 1.40
const FIT_RECOVERY_DEFENSIVE_HOME_MULTIPLIER = 1.20

function _default_fit_recovery_td_hyperparameters()
    return [
        GammaParams(5.0, 5.0 / 0.00090),
        GammaParams(5.0, 5.0 / 0.00110),
        GammaParams(5.0, 5.0 / 0.00130),
    ]
end

function _default_fit_recovery_defensive_hyperparameters()
    return [
        GammaParams(4.0, 4.0 / 0.00400),
        GammaParams(4.0, 4.0 / 0.00480),
        GammaParams(4.0, 4.0 / 0.00570),
    ]
end

function _normalize_fit_recovery_hyperparameters(
    values,
    n_bins::Int,
    label::AbstractString,
)
    parameters = GammaParams[]
    for value in values
        if value isa GammaParams
            push!(parameters, value)
        elseif value isa Tuple && length(value) == 2
            push!(parameters, GammaParams(value[1], value[2]))
        else
            throw(ArgumentError(
                "$label entries must be GammaParams or (shape, rate) tuples",
            ))
        end
    end
    length(parameters) == n_bins ||
        throw(ArgumentError("$label must contain one distribution per time bin"))
    return parameters
end

function _fit_recovery_schedule(
    seasons::AbstractVector{<:Integer},
    teams::AbstractVector{<:AbstractString},
    games_per_team::Int,
)
    length(teams) % 2 == 0 ||
        throw(ArgumentError("recovery schedule requires an even team count"))
    games_per_team > 0 ||
        throw(ArgumentError("games_per_team must be positive"))

    rotating = collect(eachindex(teams))
    rounds = Vector{Vector{Tuple{Int,Int}}}()
    for _ in 1:(length(teams) - 1)
        push!(
            rounds,
            [
                (rotating[index], rotating[end - index + 1])
                for index in 1:(length(teams) ÷ 2)
            ],
        )
        rotating = vcat(
            rotating[1:1],
            rotating[end:end],
            rotating[2:(end - 1)],
        )
    end

    game_ids = String[]
    season_values = Int[]
    game_types = String[]
    weeks = Int[]
    away_teams = String[]
    home_teams = String[]
    results = Union{Missing,Int}[]
    for season in seasons
        for round_index in 1:games_per_team
            round = rounds[mod1(round_index, length(rounds))]
            cycle = (round_index - 1) ÷ length(rounds)
            for (game_index, (first_index, second_index)) in enumerate(round)
                home_index, away_index = isodd(round_index + cycle) ?
                    (first_index, second_index) :
                    (second_index, first_index)
                push!(
                    game_ids,
                    "$(season)_$(round_index)_$(game_index)",
                )
                push!(season_values, Int(season))
                push!(game_types, "REG")
                push!(weeks, round_index)
                push!(away_teams, String(teams[away_index]))
                push!(home_teams, String(teams[home_index]))
                push!(results, missing)
            end
        end
    end
    return DataFrame(
        game_id=game_ids,
        season=season_values,
        game_type=game_types,
        week=weeks,
        away_team=away_teams,
        home_team=home_teams,
        result=results,
    )
end

function _fit_recovery_empty_drives()
    return DataFrame(
        game_id=String[],
        fixed_drive=Int[],
        posteam=String[],
        defteam=String[],
        posteam_home=Bool[],
        defteam_home=Bool[],
        drive_result=String[],
        time_of_possession=Second[],
    )
end

"""
    synthetic_fit_recovery_drives(; kwargs...) -> NamedTuple

Simulate scheduled seasons for 32 teams from a truth-configured piecewise
competing-risk model. The schedule simulator samples one latent
Gamma rate per team, outcome, and time bin for each season, reuses it across
that season's games, and generates the number and durations of drives
endogenously. `games_per_team` controls the hypothetical schedule volume.
The legacy `drives_per_team` keyword remains as an approximate schedule-volume
alias, using approximately eleven drives per team-game. The default hazard
means are calibrated to produce roughly 160-second drives and a 20% touchdown
share in the two-outcome event process; score rewards are not simulated.
"""
function synthetic_fit_recovery_drives(
    ;
    seasons=FIT_RECOVERY_SEASONS,
    teams=["T$(index)" for index in 1:FIT_RECOVERY_TEAM_COUNT],
    games_per_team::Int=FIT_RECOVERY_GAMES_PER_TEAM,
    drives_per_team::Union{Nothing,Int}=nothing,
    time_edges=FIT_RECOVERY_TIME_EDGES,
    td_hyperparameters=_default_fit_recovery_td_hyperparameters(),
    defensive_hyperparameters=_default_fit_recovery_defensive_hyperparameters(),
    td_persistence::Real=FIT_RECOVERY_TD_PERSISTENCE,
    defensive_persistence::Real=FIT_RECOVERY_DEFENSIVE_PERSISTENCE,
    td_home_multiplier::Real=FIT_RECOVERY_TD_HOME_MULTIPLIER,
    defensive_home_multiplier::Real=FIT_RECOVERY_DEFENSIVE_HOME_MULTIPLIER,
    horizon_seconds::Int=FIT_RECOVERY_GAME_HORIZON_SECONDS,
    max_duration_seconds::Union{Nothing,Int}=nothing,
    seed::Integer=1,
)
    season_values = Int.(collect(seasons))
    isempty(season_values) &&
        throw(ArgumentError("recovery simulation requires at least one season"))
    issorted(season_values) ||
        throw(ArgumentError("recovery seasons must be sorted"))
    length(unique(season_values)) == length(season_values) ||
        throw(ArgumentError("recovery seasons must be unique"))

    team_names = String.(collect(teams))
    length(team_names) == FIT_RECOVERY_TEAM_COUNT ||
        throw(ArgumentError(
            "recovery simulation requires exactly $(FIT_RECOVERY_TEAM_COUNT) teams",
        ))
    length(unique(team_names)) == length(team_names) ||
        throw(ArgumentError("recovery teams must be unique"))
    games_per_team > 0 ||
        throw(ArgumentError("games_per_team must be positive"))
    if drives_per_team !== nothing
        drives_per_team > 0 ||
            throw(ArgumentError("drives_per_team must be positive"))
        games_per_team = max(
            1,
            cld(drives_per_team, FIT_RECOVERY_APPROX_DRIVES_PER_TEAM_GAME),
        )
    end
    if max_duration_seconds !== nothing
        max_duration_seconds > 0 ||
            throw(ArgumentError("max_duration_seconds must be positive"))
        horizon_seconds = max_duration_seconds
    end
    horizon_seconds > 0 ||
        throw(ArgumentError("horizon_seconds must be positive"))

    edges = _validate_time_edges(time_edges)
    n_bins = length(edges) - 1
    isinf(edges[end]) ||
        throw(ArgumentError("recovery time_edges must end at Inf"))
    edges[end - 1] < horizon_seconds ||
        throw(ArgumentError(
            "horizon_seconds must exceed the last finite time edge",
        ))
    td_parameters = _normalize_fit_recovery_hyperparameters(
        td_hyperparameters,
        n_bins,
        "td_hyperparameters",
    )
    defensive_parameters = _normalize_fit_recovery_hyperparameters(
        defensive_hyperparameters,
        n_bins,
        "defensive_hyperparameters",
    )

    0.0 <= td_persistence <= 1.0 ||
        throw(ArgumentError("td_persistence must be in [0, 1]"))
    0.0 <= defensive_persistence <= 1.0 ||
        throw(ArgumentError("defensive_persistence must be in [0, 1]"))
    td_home_multiplier > 0.0 ||
        throw(ArgumentError("td_home_multiplier must be positive"))
    defensive_home_multiplier > 0.0 ||
        throw(ArgumentError("defensive_home_multiplier must be positive"))

    schedule = _fit_recovery_schedule(
        season_values,
        team_names,
        games_per_team,
    )
    prior = HazardPrior(
        edges,
        td_parameters,
        defensive_parameters,
        Dict{String,Vector{GammaMixture}}(),
        Dict{String,Vector{GammaMixture}}(),
        Float64(td_home_multiplier),
        Float64(defensive_home_multiplier),
        Float64(td_persistence),
        Float64(defensive_persistence),
        Int[],
        nothing,
        nothing,
    )
    model = fit_hazard_model(
        _fit_recovery_empty_drives();
        prior=prior,
        time_edges=edges,
    )
    rng = MersenneTwister(seed)
    simulation = simulate_renewal_schedule(
        model,
        schedule;
        rng=rng,
        horizon=horizon_seconds,
        opening_possession=:random,
    )

    truth = (
        seasons=season_values,
        teams=team_names,
        time_edges=edges,
        td_hyperparameters=td_parameters,
        defensive_hyperparameters=defensive_parameters,
        td_persistence=Float64(td_persistence),
        defensive_persistence=Float64(defensive_persistence),
        td_home_multiplier=Float64(td_home_multiplier),
        defensive_home_multiplier=Float64(defensive_home_multiplier),
        games_per_team=games_per_team,
        horizon_seconds=horizon_seconds,
        max_duration_seconds=horizon_seconds,
        seed=Int(seed),
    )
    return (
        drives=simulation.drives,
        truth=truth,
        schedule=schedule,
        games=simulation.games,
        latent_rates=simulation.latent_rates,
    )
end

"""
    synthetic_fit_benchmark_drives(; kwargs...) -> DataFrame

Return a deterministic drive-level data set with persistent latent team rates
for fitting benchmarks.
"""
function synthetic_fit_benchmark_drives(
    ;
    seasons=2021:2023,
    teams=["T$(index)" for index in 1:60],
    drives_per_team::Int=300,
    persistence::Real=FIT_BENCHMARK_SYNTHETIC_PERSISTENCE,
    touchdown_mean::Real=FIT_BENCHMARK_SYNTHETIC_TD_MEAN,
    defensive_mean::Real=FIT_BENCHMARK_SYNTHETIC_DEFENSIVE_MEAN,
    home_multiplier::Real=FIT_BENCHMARK_SYNTHETIC_HOME_MULTIPLIER,
)
    drives_per_team > 0 ||
        throw(ArgumentError("drives_per_team must be positive"))
    0.0 <= persistence <= 1.0 ||
        throw(ArgumentError("persistence must be in [0, 1]"))
    touchdown_mean > 0.0 ||
        throw(ArgumentError("touchdown_mean must be positive"))
    defensive_mean > 0.0 ||
        throw(ArgumentError("defensive_mean must be positive"))
    home_multiplier > 0.0 ||
        throw(ArgumentError("home_multiplier must be positive"))
    isempty(teams) && throw(ArgumentError("teams must not be empty"))
    length(unique(teams)) == length(teams) ||
        throw(ArgumentError("teams must be unique"))
    shape = round(Int, FIT_BENCHMARK_SYNTHETIC_GAMMA_SHAPE)

    rows = DataFrame(
        game_id=String[],
        fixed_drive=Int[],
        posteam=String[],
        defteam=String[],
        posteam_home=Bool[],
        defteam_home=Bool[],
        drive_result=String[],
        time_of_possession=Second[],
        home_spread_change=Float64[],
    )
    td_rates = Dict{Tuple{String,Int},Float64}()
    defensive_rates = Dict{Tuple{String,Int},Float64}()
    season_values = collect(seasons)
    for (season_index, season) in enumerate(season_values)
        for (team_index, team_value) in enumerate(teams)
            team = string(team_value)
            for time_bin in 1:3
                td_key = (team, time_bin)
                td_rates[td_key] = if season_index == 1 ||
                    _synthetic_unit_interval(
                        100_000 * season_index +
                        10_000 * time_bin +
                        1_000 * team_index +
                        11,
                    ) >= persistence
                    _synthetic_gamma_rate(
                        10_000 * season_index +
                        1_000 * time_bin +
                        100 * team_index +
                        17,
                        Float64(touchdown_mean),
                        shape,
                    )
                else
                    td_rates[td_key]
                end
                defensive_key = (team, time_bin)
                defensive_rates[defensive_key] = if season_index == 1 ||
                    _synthetic_unit_interval(
                        100_000 * season_index +
                        10_000 * time_bin +
                        1_000 * team_index +
                        23,
                    ) >= persistence
                    _synthetic_gamma_rate(
                        10_000 * season_index +
                        1_000 * time_bin +
                        100 * team_index +
                        29,
                        Float64(defensive_mean),
                        shape,
                    )
                else
                    defensive_rates[defensive_key]
                end
            end
        end

        for (team_index, team_value) in enumerate(teams)
            team = string(team_value)
            for index in 1:drives_per_team
                defensive_team_index =
                    mod(team_index + index + season_index - 2, length(teams)) + 1
                defensive_team = string(teams[defensive_team_index])
                posteam_home = iseven(index + season + team_index)
                defteam_home = !posteam_home
                planned_duration = 60 + 60 * mod(index + season_index, 5)
                seed = 1_000_000 * season_index +
                    10_000 * team_index +
                    100 * index
                event_observed = false
                touchdown = false
                duration = planned_duration
                elapsed = 0
                for time_bin in 1:3
                    bin_end = time_bin == 1 ? 120 :
                        time_bin == 2 ? 240 :
                        planned_duration
                    interval_end = min(planned_duration, bin_end)
                    exposure = interval_end - elapsed
                    exposure <= 0 && continue
                    td_rate = td_rates[(team, time_bin)] *
                        (posteam_home ? home_multiplier : 1.0)
                    defensive_rate =
                        defensive_rates[(defensive_team, time_bin)] *
                        (defteam_home ? home_multiplier : 1.0)
                    total_rate = td_rate + defensive_rate
                    event_probability =
                        1.0 - exp(-total_rate * exposure)
                    if _synthetic_unit_interval(seed + 10 * time_bin) <
                        event_probability
                        event_observed = true
                        touchdown = _synthetic_unit_interval(
                            seed + 100 + time_bin,
                        ) < td_rate / total_rate
                        event_offset = 1 + floor(Int,
                            _synthetic_unit_interval(
                                seed + 1_000 + time_bin,
                            ) * exposure,
                        )
                        duration = elapsed + min(event_offset, Int(exposure))
                        break
                    end
                    elapsed = interval_end
                end
                drive_result = event_observed ?
                    (touchdown ? "Touchdown" : "Punt") :
                    "End of half"
                push!(
                    rows,
                    (
                        "$(season)_$(team)_$(defensive_team)_$(index)",
                        1,
                        team,
                        defensive_team,
                        posteam_home,
                        defteam_home,
                        drive_result,
                        Second(duration),
                        touchdown && posteam_home ? 7.0 : 0.0,
                    ),
                )
            end
        end
    end
    return rows
end

function _fit_benchmark_once(
    drives::AbstractDataFrame,
    method::PriorFitMethod;
    current_season::Integer,
    max_seasons::Int,
    time_edges,
)
    return fit_empirical_bayes_prior(
        drives;
        time_edges=time_edges,
        current_season=current_season,
        max_seasons=max_seasons,
        method=method,
        _return_solver_metrics=true,
    )
end

function _fit_benchmark_failure_row(method::PriorFitMethod, error)
    return (
        method=string(prior_fit_method_name(method)),
        median_ms=NaN,
        touchdown_log_likelihood=NaN,
        defensive_log_likelihood=NaN,
        total_log_likelihood=NaN,
        likelihood_gap=NaN,
        converged=false,
        status="failed",
        iterations=0,
        function_evaluations=0,
        boundary_parameters=0,
        error=sprint(showerror, error),
    )
end

function _append_fit_benchmark_parameter_rows!(
    rows::Vector{<:NamedTuple},
    method::PriorFitMethod,
    prior::HazardPrior,
)
    for kind in (:td, :defensive)
        hyperparameters = kind === :td ?
            prior.td_hyperparameters :
            prior.defensive_hyperparameters
        fitted_home_multiplier = home_multiplier(prior, kind)
        fitted_persistence = hazard_persistence(prior, kind)
        for (time_bin, parameter) in enumerate(hyperparameters)
            push!(
                rows,
                (
                    method=string(prior_fit_method_name(method)),
                    kind=string(kind),
                    time_bin=time_bin,
                    home_advantage=fitted_home_multiplier,
                    persistence=fitted_persistence,
                    hazard_mean=parameter.shape / parameter.rate,
                    hazard_variance=parameter.shape / parameter.rate^2,
                ),
            )
        end
    end
    return nothing
end

"""
    _fit_benchmark_with_parameters(drives; kwargs...)

Benchmark the available empirical-Bayes fitting methods on one data set.
`median_ms` excludes one warm-up fit, and `likelihood_gap` is measured against
the best successful total log likelihood in the returned summary table.
"""
function _fit_benchmark_with_parameters(
    drives::AbstractDataFrame;
    current_season::Integer=2024,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=FIT_BENCHMARK_TIME_EDGES,
    repeats::Int=3,
    methods=FIT_BENCHMARK_METHODS,
)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    repeats > 0 || throw(ArgumentError("repeats must be positive"))
    method_list = collect(methods)
    isempty(method_list) &&
        throw(ArgumentError("fit benchmark requires at least one method"))
    all(method -> method isa PriorFitMethod, method_list) ||
        throw(ArgumentError("fit benchmark methods must be PriorFitMethod values"))

    rows = NamedTuple[]
    parameter_rows = NamedTuple[]
    for method in method_list
        try
            _fit_benchmark_once(
                drives,
                method;
                current_season=current_season,
                max_seasons=max_seasons,
                time_edges=time_edges,
            )
            elapsed_ms = Float64[]
            fitted = nothing
            for _ in 1:repeats
                started = time_ns()
                fitted = _fit_benchmark_once(
                    drives,
                    method;
                    current_season=current_season,
                    max_seasons=max_seasons,
                    time_edges=time_edges,
                )
                push!(elapsed_ms, (time_ns() - started) / 1.0e6)
            end

            td = likelihood_fit_diagnostics(fitted.prior, :td)
            defensive = likelihood_fit_diagnostics(fitted.prior, :defensive)
            total_log_likelihood =
                td.log_likelihood + defensive.log_likelihood
            statuses = unique([td.status, defensive.status])
            boundary_parameters = unique(vcat(
                td.boundary_parameters,
                defensive.boundary_parameters,
            ))
            push!(
                rows,
                (
                    method=string(prior_fit_method_name(method)),
                    median_ms=median(elapsed_ms),
                    touchdown_log_likelihood=td.log_likelihood,
                    defensive_log_likelihood=defensive.log_likelihood,
                    total_log_likelihood=total_log_likelihood,
                    likelihood_gap=NaN,
                    converged=td.converged && defensive.converged,
                    status=join(string.(statuses), "/"),
                    iterations=td.iterations + defensive.iterations,
                    function_evaluations=
                        td.function_evaluations + defensive.function_evaluations,
                    boundary_parameters=length(boundary_parameters),
                    error="",
                ),
            )
            _append_fit_benchmark_parameter_rows!(
                parameter_rows,
                method,
                fitted.prior,
            )
        catch error
            error isa ArgumentError || rethrow()
            push!(rows, _fit_benchmark_failure_row(method, error))
        end
    end

    result = DataFrame(rows)
    successful_likelihoods = [
        row.total_log_likelihood
        for row in eachrow(result)
        if isfinite(row.total_log_likelihood)
    ]
    best_log_likelihood = isempty(successful_likelihoods) ?
        NaN :
        maximum(successful_likelihoods)
    result.likelihood_gap = [
        isfinite(row.total_log_likelihood) && isfinite(best_log_likelihood) ?
            best_log_likelihood - row.total_log_likelihood :
            NaN
        for row in eachrow(result)
    ]
    return (
        summary=result,
        parameters=DataFrame(parameter_rows),
    )
end

function _fit_recovery_tables(
    parameters::AbstractDataFrame,
    truth,
)
    if isempty(parameters)
        return (
            shared=DataFrame(
                method=String[],
                kind=String[],
                fitted_home_advantage=Float64[],
                target_home_advantage=Float64[],
                home_advantage_error=Float64[],
                fitted_persistence=Float64[],
                target_persistence=Float64[],
                persistence_error=Float64[],
            ),
            bins=DataFrame(
                method=String[],
                kind=String[],
                time_bin=Int[],
                fitted_mean=Float64[],
                target_mean=Float64[],
                mean_relative_error=Float64[],
                fitted_variance=Float64[],
                target_variance=Float64[],
                variance_relative_error=Float64[],
            ),
            quality=DataFrame(
                method=String[],
                kind=String[],
                mean_abs_relative_error=Float64[],
                variance_abs_relative_error=Float64[],
                home_abs_error=Float64[],
                persistence_abs_error=Float64[],
                max_abs_relative_error=Float64[],
            ),
        )
    end
    shared_rows = NamedTuple[]
    bin_rows = NamedTuple[]
    quality_rows = NamedTuple[]
    truth_by_kind = (
        td=(
            parameters=truth.td_hyperparameters,
            persistence=truth.td_persistence,
            home_multiplier=truth.td_home_multiplier,
        ),
        defensive=(
            parameters=truth.defensive_hyperparameters,
            persistence=truth.defensive_persistence,
            home_multiplier=truth.defensive_home_multiplier,
        ),
    )

    for method in unique(parameters.method)
        for kind in (:td, :defensive)
            kind_name = string(kind)
            fitted = parameters[
                (parameters.method .== method) .&
                (parameters.kind .== kind_name),
                :,
            ]
            isempty(fitted) && continue
            target = getproperty(truth_by_kind, kind)
            fitted_home = first(fitted.home_advantage)
            fitted_persistence = first(fitted.persistence)
            home_error = fitted_home - target.home_multiplier
            persistence_error =
                fitted_persistence - target.persistence
            push!(
                shared_rows,
                (
                    method=method,
                    kind=kind_name,
                    fitted_home_advantage=fitted_home,
                    target_home_advantage=target.home_multiplier,
                    home_advantage_error=home_error,
                    fitted_persistence=fitted_persistence,
                    target_persistence=target.persistence,
                    persistence_error=persistence_error,
                ),
            )

            mean_relative_errors = Float64[]
            variance_relative_errors = Float64[]
            for row in eachrow(fitted)
                target_parameter = target.parameters[row.time_bin]
                target_mean = target_parameter.shape / target_parameter.rate
                target_variance =
                    target_parameter.shape / target_parameter.rate^2
                mean_relative_error =
                    (row.hazard_mean - target_mean) / target_mean
                variance_relative_error =
                    (row.hazard_variance - target_variance) /
                    target_variance
                push!(mean_relative_errors, mean_relative_error)
                push!(variance_relative_errors, variance_relative_error)
                push!(
                    bin_rows,
                    (
                        method=method,
                        kind=kind_name,
                        time_bin=row.time_bin,
                        fitted_mean=row.hazard_mean,
                        target_mean=target_mean,
                        mean_relative_error=mean_relative_error,
                        fitted_variance=row.hazard_variance,
                        target_variance=target_variance,
                        variance_relative_error=variance_relative_error,
                    ),
                )
            end
            relative_shared_errors = [
                home_error / target.home_multiplier,
                persistence_error / max(target.persistence, eps()),
            ]
            all_relative_errors = vcat(
                abs.(mean_relative_errors),
                abs.(variance_relative_errors),
                abs.(relative_shared_errors),
            )
            push!(
                quality_rows,
                (
                    method=method,
                    kind=kind_name,
                    mean_abs_relative_error=mean(abs.(mean_relative_errors)),
                    variance_abs_relative_error=
                        mean(abs.(variance_relative_errors)),
                    home_abs_error=abs(home_error),
                    persistence_abs_error=abs(persistence_error),
                    max_abs_relative_error=maximum(all_relative_errors),
                ),
            )
        end
    end
    return (
        shared=DataFrame(shared_rows),
        bins=DataFrame(bin_rows),
        quality=DataFrame(quality_rows),
    )
end

"""
    fit_recovery_benchmark(drives, truth; kwargs...) -> NamedTuple

Fit every requested method to a correctly specified synthetic data set and
return runtime, likelihood, and parameter-recovery tables. `truth` is the
named tuple returned by [`synthetic_fit_recovery_drives`](@ref).
"""
function fit_recovery_benchmark(
    drives::AbstractDataFrame,
    truth;
    current_season::Union{Nothing,Integer}=nothing,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=truth.time_edges,
    repeats::Int=3,
    methods=FIT_BENCHMARK_METHODS,
)
    season_values = Int.(collect(truth.seasons))
    isempty(season_values) &&
        throw(ArgumentError("recovery truth must contain at least one season"))
    issorted(season_values) ||
        throw(ArgumentError("recovery truth seasons must be sorted"))
    length(unique(season_values)) == length(season_values) ||
        throw(ArgumentError("recovery truth seasons must be unique"))
    reference_season = current_season === nothing ?
        maximum(season_values) + 1 :
        Int(current_season)
    result = _fit_benchmark_with_parameters(
        drives;
        current_season=reference_season,
        max_seasons=max_seasons,
        time_edges=time_edges,
        repeats=repeats,
        methods=methods,
    )
    recovery = _fit_recovery_tables(result.parameters, truth)
    return (
        summary=result.summary,
        parameters=result.parameters,
        recovery_shared=recovery.shared,
        recovery_bins=recovery.bins,
        recovery_quality=recovery.quality,
        truth=truth,
    )
end

"""
    fit_benchmark(drives; kwargs...) -> DataFrame

Benchmark the available empirical-Bayes fitting methods on one data set.
`median_ms` excludes one warm-up fit, and `likelihood_gap` is measured against
the best successful total log likelihood in the returned table.
"""
function fit_benchmark(
    drives::AbstractDataFrame;
    current_season::Integer=2024,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=FIT_BENCHMARK_TIME_EDGES,
    repeats::Int=3,
    methods=FIT_BENCHMARK_METHODS,
)
    return _fit_benchmark_with_parameters(
        drives;
        current_season=current_season,
        max_seasons=max_seasons,
        time_edges=time_edges,
        repeats=repeats,
        methods=methods,
    ).summary
end
