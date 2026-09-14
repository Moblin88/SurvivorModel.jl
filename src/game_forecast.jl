"""
    GAME_CLOCK_SECONDS

Total regulation game clock, in seconds.
"""
const GAME_CLOCK_SECONDS = 3600.0

const SCHEDULE_REQUIRED_COLUMNS = (
    :game_id,
    :season,
    :game_type,
    :week,
    :away_team,
    :home_team,
    :result,
)

function _require_columns(data::AbstractDataFrame, required, label::AbstractString)
    missing_columns = setdiff(collect(required), propertynames(data))
    isempty(missing_columns) ||
        throw(ArgumentError("$label is missing required columns: $missing_columns"))
    return nothing
end

function _schedule_integer(value, column::Symbol)
    ismissing(value) &&
        throw(ArgumentError("schedule column $column cannot contain missing values"))
    parsed = value isa Integer ? Int(value) : tryparse(Int, string(value))
    parsed === nothing &&
        throw(ArgumentError("schedule column $column must contain integers"))
    return parsed
end

function _schedule_string(value, column::Symbol)
    ismissing(value) &&
        throw(ArgumentError("schedule column $column cannot contain missing values"))
    return string(value)
end

function _normalize_schedule(schedule::AbstractDataFrame)
    _require_columns(schedule, SCHEDULE_REQUIRED_COLUMNS, "schedule")
    data = DataFrame(schedule)

    data.game_id = [_schedule_string(value, :game_id) for value in data.game_id]
    data.season = [_schedule_integer(value, :season) for value in data.season]
    data.game_type = [_schedule_string(value, :game_type) for value in data.game_type]
    data.week = [_schedule_integer(value, :week) for value in data.week]
    data.away_team = [_schedule_string(value, :away_team) for value in data.away_team]
    data.home_team = [_schedule_string(value, :home_team) for value in data.home_team]

    length(unique(data.game_id)) == nrow(data) ||
        throw(ArgumentError("schedule game_id values must be unique"))
    return data
end

"""
    load_schedule() -> DataFrame
    load_schedule(schedule::AbstractDataFrame) -> DataFrame

Load and normalize the NFL schedule through `NFLData`, or normalize an
injected schedule DataFrame. The returned table includes regular-season and
postseason rows; callers can filter by `game_type`.
"""
load_schedule() = _normalize_schedule(NFLData.load_schedules())
load_schedule(schedule::AbstractDataFrame) = _normalize_schedule(schedule)

function _regular_season_schedule(schedule::AbstractDataFrame, season::Integer)
    regular = schedule[
        (schedule.game_type .== "REG") .& (schedule.season .== Int(season)),
        :,
    ]
    sort!(regular, [:week, :game_id])
    return regular
end

function _schedule_game_metadata(schedule::AbstractDataFrame)
    metadata = Dict{String,NamedTuple{(:season, :week, :game_type),Tuple{Int,Int,String}}}()
    for row in eachrow(schedule)
        haskey(metadata, row.game_id) &&
            throw(ArgumentError("schedule game_id values must be unique"))
        metadata[row.game_id] = (
            season=Int(row.season),
            week=Int(row.week),
            game_type=String(row.game_type),
        )
    end
    return metadata
end

function _regular_season_drives(
    drives::AbstractDataFrame,
    schedule::AbstractDataFrame,
)
    _require_columns(drives, (:game_id,), "drives")
    metadata = _schedule_game_metadata(schedule)
    keep = falses(nrow(drives))
    seasons = Int[]
    weeks = Int[]
    game_types = String[]

    for (index, game_id) in enumerate(drives.game_id)
        ismissing(game_id) && continue
        game_metadata = get(metadata, string(game_id), nothing)
        isnothing(game_metadata) && continue
        game_metadata.game_type == "REG" || continue
        keep[index] = true
        push!(seasons, game_metadata.season)
        push!(weeks, game_metadata.week)
        push!(game_types, game_metadata.game_type)
    end

    filtered = DataFrame(drives[keep, :])
    filtered.season = seasons
    filtered.week = weeks
    filtered.game_type = game_types
    return filtered
end

function _empty_drive_data(drives::AbstractDataFrame)
    return drives[1:0, :]
end

function _load_forecast_drives(
    season::Integer,
    max_seasons::Int,
    historical_drives,
    current_drives,
    ;
    allow_missing_current::Bool=false,
)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))

    historical = if historical_drives !== nothing
        historical_drives
    elseif season <= 1999
        nothing
    else
        first_season = max(1999, Int(season) - max_seasons)
        load_drive_pbp(first_season:(Int(season) - 1))
    end

    current = if current_drives !== nothing
        current_drives
    elseif allow_missing_current && season > NFLData.most_recent_season()
        historical === nothing &&
            throw(ArgumentError(
                "historical drive data is required when current-season PBP " *
                "is unavailable",
            ))
        _empty_drive_data(historical)
    elseif season > NFLData.most_recent_season()
        throw(ArgumentError(
            "NFL PBP data for season $season is not available yet; " *
            "provide current_drives or wait for the season data release",
        ))
    else
        load_drive_pbp(season)
    end

    if historical_drives !== nothing
        return historical, current
    end
    if season <= 1999
        return _empty_drive_data(current), current
    end
    return historical, current
end

function _forecast_training_data(
    schedule::AbstractDataFrame,
    season::Integer,
    as_of_week::Integer,
    historical_drives::AbstractDataFrame,
    current_drives::AbstractDataFrame,
    ;
    _schedule_indexed_drives::Bool=false,
)
    historical = _schedule_indexed_drives ?
        historical_drives :
        _regular_season_drives(historical_drives, schedule)
    current = _schedule_indexed_drives ?
        current_drives :
        _regular_season_drives(current_drives, schedule)

    historical = historical[historical.season .< Int(season), :]
    cutoff = current[
        (current.season .== Int(season)) .&
        (current.week .< Int(as_of_week)),
        :,
    ]
    training = vcat(historical, cutoff; cols=:union)
    nrow(training) > 0 ||
        throw(ArgumentError("no regular-season drive data is available before week $as_of_week"))
    return historical, cutoff, training
end

"""
    RegularSeasonForecastContext

Reusable fitted state for regular-season forecasts. The context contains the
frozen hazard model, score marks, normalized schedule, and target-season games
at or after the cutoff week.
"""
struct RegularSeasonForecastContext
    season::Int
    as_of_week::Int
    schedule::DataFrame
    games::DataFrame
    model::HazardModel
    marks::ScoreMarks
end

"""
    fit_regular_season_forecast(
        season;
        as_of_week,
        schedule=nothing,
        historical_drives=nothing,
        current_drives=nothing,
        max_seasons=DEFAULT_HISTORICAL_SEASONS,
        time_edges=DEFAULT_TIME_EDGES,
        method=DEFAULT_PRIOR_FIT_METHOD,
        prior=nothing,
    ) -> RegularSeasonForecastContext

Fit a frozen model using regular-season data available before `as_of_week`.
Historical seasons provide the empirical-Bayes prior; target-season drives
from weeks before the cutoff provide the current-season update. The returned
context can be reused to request probabilities, spreads, or the full metric
table without reloading data or refitting the model. A prior fitted for the
same target season can be supplied when evaluating multiple weekly snapshots
to avoid repeating the historical empirical-Bayes fit.
"""
function fit_regular_season_forecast(
    season::Integer;
    as_of_week::Integer,
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    current_drives::Union{Nothing,AbstractDataFrame}=nothing,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=DEFAULT_TIME_EDGES,
    method::PriorFitMethod=DEFAULT_PRIOR_FIT_METHOD,
    prior::Union{Nothing,HazardPrior}=nothing,
    _normalized_schedule::Bool=false,
    _schedule_indexed_drives::Bool=false,
)
    1 <= as_of_week <= 18 ||
        throw(ArgumentError("as_of_week must be between 1 and 18"))
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))

    normalized_schedule = if schedule === nothing
        load_schedule()
    elseif _normalized_schedule && schedule isa DataFrame
        schedule
    else
        load_schedule(schedule)
    end
    target_schedule = _regular_season_schedule(normalized_schedule, season)
    isempty(target_schedule) &&
        throw(ArgumentError("schedule has no regular-season games for season $season"))

    historical, current = _load_forecast_drives(
        season,
        max_seasons,
        historical_drives,
        current_drives,
        ;
        allow_missing_current=as_of_week == 1,
    )
    historical, cutoff, training = _forecast_training_data(
        normalized_schedule,
        season,
        as_of_week,
        historical,
        current,
        ;
        _schedule_indexed_drives=_schedule_indexed_drives,
    )

    fitted_prior = prior === nothing ? fit_empirical_bayes_prior(
        historical;
        time_edges=time_edges,
        max_seasons=max_seasons,
        current_season=season,
        method=method,
    ) : prior
    model = fit_hazard_model(cutoff; prior=fitted_prior, time_edges=time_edges)
    marks = fit_score_marks(training)
    games = target_schedule[target_schedule.week .>= as_of_week, :]

    return RegularSeasonForecastContext(
        Int(season),
        Int(as_of_week),
        normalized_schedule,
        games,
        model,
        marks,
    )
end

function _selected_forecast_games(
    context::RegularSeasonForecastContext;
    include_completed::Bool,
)
    games = copy(context.games)
    if !include_completed
        games = games[ismissing.(games.result), :]
    end
    return games
end

const FORECAST_OUTPUT_COLUMNS = (
    :game_id,
    :season,
    :game_type,
    :week,
    :gameday,
    :away_team,
    :home_team,
    :away_score,
    :home_score,
    :result,
)

function _forecast_output(
    games::AbstractDataFrame;
    full_schedule::Bool,
)
    full_schedule && return DataFrame(games)
    columns = [
        column for column in FORECAST_OUTPUT_COLUMNS
        if column in propertynames(games)
    ]
    return select(games, columns)
end

function _validate_win_probability(probability::Real)
    value = Float64(probability)
    isfinite(value) && 0.0 <= value <= 1.0 ||
        throw(ArgumentError("model returned an invalid win probability"))
    return value
end

"""
    forecast_win_probabilities(
        context::RegularSeasonForecastContext;
        include_completed=true,
        horizon=GAME_CLOCK_SECONDS,
        full_schedule=false,
    ) -> DataFrame

Return only schedule identifiers/results and home/away win probabilities.
This path does not evaluate spread or predictive-variance metrics.
"""
function forecast_win_probabilities(
    context::RegularSeasonForecastContext;
    include_completed::Bool=true,
    horizon::Real=GAME_CLOCK_SECONDS,
    full_schedule::Bool=false,
)
    games = _selected_forecast_games(context; include_completed=include_completed)
    forecast = _forecast_output(games; full_schedule=full_schedule)
    home_probabilities = Float64[]
    away_probabilities = Float64[]
    cache = _HazardLogMomentCache(context.model)

    for row in eachrow(games)
        home_probability = _validate_win_probability(
            _expected_game_win_probability_with_cache(
                context.model,
                context.marks,
                row.home_team,
                row.away_team,
                cache;
                horizon=horizon,
            ),
        )
        push!(home_probabilities, home_probability)
        push!(away_probabilities, 1.0 - home_probability)
    end

    forecast.home_win_probability = home_probabilities
    forecast.away_win_probability = away_probabilities
    forecast.game_completed = .!ismissing.(games.result)
    return forecast
end

"""
    forecast_win_probabilities(
        season;
        as_of_week,
        ...
    ) -> DataFrame

Fit a forecast context and return the probability-only output. Reuse
`fit_regular_season_forecast` directly when requesting multiple output views.
"""
function forecast_win_probabilities(
    season::Integer;
    as_of_week::Integer,
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    current_drives::Union{Nothing,AbstractDataFrame}=nothing,
    include_completed::Bool=true,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=DEFAULT_TIME_EDGES,
    method::PriorFitMethod=DEFAULT_PRIOR_FIT_METHOD,
    horizon::Real=GAME_CLOCK_SECONDS,
    full_schedule::Bool=false,
)
    context = fit_regular_season_forecast(
        season;
        as_of_week=as_of_week,
        schedule=schedule,
        historical_drives=historical_drives,
        current_drives=current_drives,
        max_seasons=max_seasons,
        time_edges=time_edges,
        method=method,
    )
    return forecast_win_probabilities(
        context;
        include_completed=include_completed,
        horizon=horizon,
        full_schedule=full_schedule,
    )
end

"""
    forecast_spreads(
        context::RegularSeasonForecastContext;
        include_completed=true,
        horizon=GAME_CLOCK_SECONDS,
        full_schedule=false,
    ) -> DataFrame

Return expected spread and predictive spread variance for the forecast games
without evaluating the posterior expected win probability.
"""
function forecast_spreads(
    context::RegularSeasonForecastContext;
    include_completed::Bool=true,
    horizon::Real=GAME_CLOCK_SECONDS,
    full_schedule::Bool=false,
)
    games = _selected_forecast_games(context; include_completed=include_completed)
    forecast = _forecast_output(games; full_schedule=full_schedule)
    expected_spreads = Float64[]
    predictive_variances = Float64[]
    cache = _HazardLogMomentCache(context.model)

    for row in eachrow(games)
        metrics = _expected_game_spread_metrics_with_cache(
            context.model,
            context.marks,
            row.home_team,
            row.away_team,
            cache;
            horizon=horizon,
        )
        push!(expected_spreads, metrics.expected_spread)
        push!(predictive_variances, metrics.predictive_spread_variance)
    end

    forecast.expected_spread = expected_spreads
    forecast.predictive_spread_variance = predictive_variances
    forecast.game_completed = .!ismissing.(games.result)
    return forecast
end

"""
    forecast_spreads(
        season;
        as_of_week,
        ...
    ) -> DataFrame

Fit a forecast context and return expected spread metrics. Reuse
`fit_regular_season_forecast` directly when requesting multiple output views.
"""
function forecast_spreads(
    season::Integer;
    as_of_week::Integer,
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    current_drives::Union{Nothing,AbstractDataFrame}=nothing,
    include_completed::Bool=true,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=DEFAULT_TIME_EDGES,
    method::PriorFitMethod=DEFAULT_PRIOR_FIT_METHOD,
    horizon::Real=GAME_CLOCK_SECONDS,
    full_schedule::Bool=false,
)
    context = fit_regular_season_forecast(
        season;
        as_of_week=as_of_week,
        schedule=schedule,
        historical_drives=historical_drives,
        current_drives=current_drives,
        max_seasons=max_seasons,
        time_edges=time_edges,
        method=method,
    )
    return forecast_spreads(
        context;
        include_completed=include_completed,
        horizon=horizon,
        full_schedule=full_schedule,
    )
end

"""
    regular_season_results(
        season;
        schedule=nothing,
        from_week=1,
        through_week=18,
        include_unplayed=true,
        full_schedule=false,
    ) -> DataFrame

Return regular-season schedule results without loading PBP or fitting a model.
Unplayed games are retained by default with missing scores/results.
"""
function regular_season_results(
    season::Integer;
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    from_week::Integer=1,
    through_week::Integer=18,
    include_unplayed::Bool=true,
    full_schedule::Bool=false,
)
    1 <= from_week <= through_week <= 18 ||
        throw(ArgumentError("week range must be within 1:18"))
    normalized_schedule = schedule === nothing ? load_schedule() : load_schedule(schedule)
    games = _regular_season_schedule(normalized_schedule, season)
    isempty(games) &&
        throw(ArgumentError("schedule has no regular-season games for season $season"))
    games = games[
        (games.week .>= from_week) .& (games.week .<= through_week),
        :,
    ]
    if !include_unplayed
        games = games[.!ismissing.(games.result), :]
    end
    results = _forecast_output(games; full_schedule=full_schedule)
    results.game_completed = .!ismissing.(games.result)
    return results
end

"""
    forecast_regular_season(
        season;
        as_of_week,
        schedule=nothing,
        historical_drives=nothing,
        current_drives=nothing,
        include_completed=true,
        max_seasons=DEFAULT_HISTORICAL_SEASONS,
        time_edges=DEFAULT_TIME_EDGES,
        horizon=GAME_CLOCK_SECONDS,
    ) -> DataFrame

Fit a frozen model and return the full backward-compatible schedule and metric
table for every regular-season game from `as_of_week` through week 18.
"""
function forecast_regular_season(
    season::Integer;
    as_of_week::Integer,
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    current_drives::Union{Nothing,AbstractDataFrame}=nothing,
    include_completed::Bool=true,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=DEFAULT_TIME_EDGES,
    method::PriorFitMethod=DEFAULT_PRIOR_FIT_METHOD,
    horizon::Real=GAME_CLOCK_SECONDS,
)
    context = fit_regular_season_forecast(
        season;
        as_of_week=as_of_week,
        schedule=schedule,
        historical_drives=historical_drives,
        current_drives=current_drives,
        max_seasons=max_seasons,
        time_edges=time_edges,
        method=method,
    )
    return forecast_regular_season(
        context;
        include_completed=include_completed,
        horizon=horizon,
    )
end

"""
    forecast_regular_season(
        context::RegularSeasonForecastContext;
        include_completed=true,
        horizon=GAME_CLOCK_SECONDS,
    ) -> DataFrame

Return the full schedule and metric table from a fitted forecast context.
"""
function forecast_regular_season(
    context::RegularSeasonForecastContext;
    include_completed::Bool=true,
    horizon::Real=GAME_CLOCK_SECONDS,
)
    games = _selected_forecast_games(context; include_completed=include_completed)
    forecast = _forecast_output(games; full_schedule=true)

    home_probabilities = Float64[]
    away_probabilities = Float64[]
    expected_spreads = Float64[]
    predictive_variances = Float64[]
    cache = _HazardLogMomentCache(context.model)
    for row in eachrow(games)
        metrics = _expected_game_metrics_with_cache(
            context.model,
            context.marks,
            row.home_team,
            row.away_team,
            cache;
            horizon=horizon,
        )
        home_probability = _validate_win_probability(metrics.expected_win_probability)
        push!(home_probabilities, home_probability)
        push!(away_probabilities, 1.0 - home_probability)
        push!(expected_spreads, metrics.expected_spread)
        push!(predictive_variances, metrics.predictive_spread_variance)
    end

    forecast.home_win_probability = home_probabilities
    forecast.away_win_probability = away_probabilities
    forecast.expected_spread = expected_spreads
    forecast.predictive_spread_variance = predictive_variances
    forecast.game_completed = .!ismissing.(games.result)
    return forecast
end

# ----------------------------------------------------------------------
# Renewal-reward game-level aggregation
# ----------------------------------------------------------------------

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

function _expected_game_win_probability_with_cache(
    model::HazardModel,
    marks::ScoreMarks,
    home_team,
    away_team,
    cache::_HazardLogMomentCache;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    theta = _hazard_theta_with_cache(
        model,
        home_team,
        away_team,
        cache,
    )
    return _expected_game_win_probability(
        theta,
        model.time_edges,
        marks;
        horizon=horizon,
    )
end

function _expected_game_spread_metrics_with_cache(
    model::HazardModel,
    marks::ScoreMarks,
    home_team,
    away_team,
    cache::_HazardLogMomentCache;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    theta = _hazard_theta_with_cache(
        model,
        home_team,
        away_team,
        cache,
    )
    return _expected_game_spread_metrics(
        theta,
        model.time_edges,
        marks;
        horizon=horizon,
    )
end

function _expected_game_metrics_with_cache(
    model::HazardModel,
    marks::ScoreMarks,
    home_team,
    away_team,
    cache::_HazardLogMomentCache;
    horizon::Real=GAME_CLOCK_SECONDS,
)
    theta = _hazard_theta_with_cache(
        model,
        home_team,
        away_team,
        cache,
    )
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
