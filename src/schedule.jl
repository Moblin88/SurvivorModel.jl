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
)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    effective_max_seasons = min(max_seasons, MAX_HISTORICAL_SEASONS)
    current = current_drives === nothing ? load_drive_pbp(season) : current_drives

    if historical_drives !== nothing
        return historical_drives, current
    end

    if season <= 1999
        return _empty_drive_data(current), current
    end

    first_season = max(1999, Int(season) - effective_max_seasons)
    historical = load_drive_pbp(first_season:(Int(season) - 1))
    return historical, current
end

function _forecast_training_data(
    schedule::AbstractDataFrame,
    season::Integer,
    as_of_week::Integer,
    historical_drives::AbstractDataFrame,
    current_drives::AbstractDataFrame,
)
    historical = _regular_season_drives(historical_drives, schedule)
    current = _regular_season_drives(current_drives, schedule)

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
        max_seasons=3,
        time_edges=DEFAULT_TIME_EDGES,
        recency_half_life=DEFAULT_RECENCY_HALF_LIFE,
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
    max_seasons::Int=3,
    time_edges=DEFAULT_TIME_EDGES,
    recency_half_life::Union{Nothing,Real}=DEFAULT_RECENCY_HALF_LIFE,
    prior::Union{Nothing,HazardPrior}=nothing,
)
    1 <= as_of_week <= 18 ||
        throw(ArgumentError("as_of_week must be between 1 and 18"))
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))

    normalized_schedule = schedule === nothing ? load_schedule() : load_schedule(schedule)
    target_schedule = _regular_season_schedule(normalized_schedule, season)
    isempty(target_schedule) &&
        throw(ArgumentError("schedule has no regular-season games for season $season"))

    historical, current = _load_forecast_drives(
        season,
        max_seasons,
        historical_drives,
        current_drives,
    )
    historical, cutoff, training = _forecast_training_data(
        normalized_schedule,
        season,
        as_of_week,
        historical,
        current,
    )

    fitted_prior = prior === nothing ? fit_empirical_bayes_prior(
        historical;
        time_edges=time_edges,
        max_seasons=max_seasons,
        recency_half_life=recency_half_life,
        current_season=season,
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

    for row in eachrow(games)
        home_probability = _validate_win_probability(
            expected_game_win_probability(
                context.model,
                context.marks,
                row.home_team,
                row.away_team;
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
    max_seasons::Int=3,
    time_edges=DEFAULT_TIME_EDGES,
    recency_half_life::Union{Nothing,Real}=DEFAULT_RECENCY_HALF_LIFE,
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
        recency_half_life=recency_half_life,
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

    for row in eachrow(games)
        metrics = expected_game_spread_metrics(
            context.model,
            context.marks,
            row.home_team,
            row.away_team;
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
    max_seasons::Int=3,
    time_edges=DEFAULT_TIME_EDGES,
    recency_half_life::Union{Nothing,Real}=DEFAULT_RECENCY_HALF_LIFE,
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
        recency_half_life=recency_half_life,
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
        max_seasons=3,
        time_edges=DEFAULT_TIME_EDGES,
        recency_half_life=DEFAULT_RECENCY_HALF_LIFE,
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
    max_seasons::Int=3,
    time_edges=DEFAULT_TIME_EDGES,
    recency_half_life::Union{Nothing,Real}=DEFAULT_RECENCY_HALF_LIFE,
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
        recency_half_life=recency_half_life,
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
    for row in eachrow(games)
        metrics = expected_game_metrics(
            context.model,
            context.marks,
            row.home_team,
            row.away_team;
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
