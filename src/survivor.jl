const DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY = 0.65

"""
    SurvivorPoolState

State required to optimize a survivor-pool plan. `picks_made` maps completed
weeks to the teams selected in those weeks. `strikes_remaining` is the number
of losses that the personal reach-discount model still allows.
"""
struct SurvivorPoolState
    season::Int
    current_week::Int
    picks_made::Dict{Int,String}
    strikes_remaining::Int
end

function _normalize_survivor_picks(picks_made)
    picks_made === nothing && return Dict{Int,String}()
    picks_made isa AbstractDict ||
        throw(ArgumentError("picks_made must be an associative collection"))

    normalized = Dict{Int,String}()
    for (week, team) in pairs(picks_made)
        week_value = _schedule_integer(week, :week)
        1 <= week_value <= 18 ||
            throw(ArgumentError("picked weeks must be between 1 and 18"))
        team_value = _schedule_string(team, :team)
        isempty(team_value) && throw(ArgumentError("picked teams cannot be empty"))
        haskey(normalized, week_value) &&
            throw(ArgumentError("picks_made cannot contain duplicate weeks"))
        normalized[week_value] = team_value
    end

    length(unique(values(normalized))) == length(normalized) ||
        throw(ArgumentError("picks_made cannot reuse a team"))
    return normalized
end

function SurvivorPoolState(
    season::Integer,
    current_week::Integer;
    picks_made=Dict{Int,String}(),
    strikes_remaining::Integer=1,
)
    season > 0 || throw(ArgumentError("season must be positive"))
    1 <= current_week <= 18 ||
        throw(ArgumentError("current_week must be between 1 and 18"))
    strikes_remaining >= 0 ||
        throw(ArgumentError("strikes_remaining must be nonnegative"))

    normalized_picks = _normalize_survivor_picks(picks_made)
    all(week < current_week for week in keys(normalized_picks)) ||
        throw(ArgumentError("picks_made weeks must precede current_week"))

    return SurvivorPoolState(
        Int(season),
        Int(current_week),
        normalized_picks,
        Int(strikes_remaining),
    )
end

function SurvivorPoolState(
    season::Integer,
    current_week::Integer,
    picks_made::AbstractDict;
    strikes_remaining::Integer=1,
)
    return SurvivorPoolState(
        season,
        current_week;
        picks_made=picks_made,
        strikes_remaining=strikes_remaining,
    )
end

"""
    SurvivorPoolPlan

The selected forward plan returned by `optimize_survivor_pool`. `selections`
contains one row per planned week, `current_pick` contains the current week's
single selected row, and `discounts` records the fixed personal reach
discount used for each week.
"""
struct SurvivorPoolPlan
    state::SurvivorPoolState
    selections::DataFrame
    current_pick::DataFrame
    discounts::DataFrame
    objective_value::Float64
end

function _validate_survivor_probability(probability::Real)
    value = Float64(probability)
    isfinite(value) && 0.0 <= value <= 1.0 ||
        throw(ArgumentError("weekly_survival_probability must be finite and in [0, 1]"))
    return value
end

"""
    survivor_reach_discounts(
        number_of_weeks;
        weekly_survival_probability=0.65,
        strikes_remaining=0,
    ) -> Vector{Float64}

Return the probability of reaching each week in a future horizon under a
fixed weekly survival probability. The first entry is always `1.0`. With `k`
prior future weeks and `s` remaining strikes, the discount is the probability
of at most `s` losses in those `k` weeks.
"""
function survivor_reach_discounts(
    number_of_weeks::Integer;
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    strikes_remaining::Integer=0,
)
    number_of_weeks >= 0 ||
        throw(ArgumentError("number_of_weeks must be nonnegative"))
    strikes_remaining >= 0 ||
        throw(ArgumentError("strikes_remaining must be nonnegative"))
    probability = _validate_survivor_probability(weekly_survival_probability)
    number_of_weeks == 0 && return Float64[]

    discounts = Float64[]
    for prior_weeks in 0:(number_of_weeks - 1)
        reach_probability = 0.0
        for losses in 0:min(Int(strikes_remaining), prior_weeks)
            reach_probability +=
                binomial(prior_weeks, losses) *
                (1.0 - probability)^losses *
                probability^(prior_weeks - losses)
        end
        push!(discounts, reach_probability)
    end
    return discounts
end

"""
    survivor_reach_discounts(
        current_week,
        through_week;
        weekly_survival_probability=0.65,
        strikes_remaining=0,
    ) -> Vector{Float64}

Return discounts indexed by the weeks from `current_week` through
`through_week`, inclusive.
"""
function survivor_reach_discounts(
    current_week::Integer,
    through_week::Integer;
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    strikes_remaining::Integer=0,
)
    1 <= current_week <= through_week <= 18 ||
        throw(ArgumentError("week range must be within 1:18"))
    return survivor_reach_discounts(
        through_week - current_week + 1;
        weekly_survival_probability=weekly_survival_probability,
        strikes_remaining=strikes_remaining,
    )
end

const SURVIVOR_FORECAST_COLUMNS = (
    :game_id,
    :week,
    :away_team,
    :home_team,
    :away_win_probability,
    :home_win_probability,
)

function _forecast_game_completed(row, columns)
    if :game_completed in columns
        return Bool(row.game_completed)
    elseif :result in columns
        return !ismissing(row.result)
    end
    return false
end

"""
    build_survivor_candidates(forecast; ...)

Expand a forecast table into one candidate row for each team in each game.
The input must contain game identifiers, weeks, home/away teams, and the
corresponding home/away win probabilities.
"""
function build_survivor_candidates(
    forecast::AbstractDataFrame;
    from_week::Integer=1,
    through_week::Integer=18,
    include_completed::Bool=false,
    picks_made=nothing,
)
    1 <= from_week <= through_week <= 18 ||
        throw(ArgumentError("week range must be within 1:18"))
    _require_columns(forecast, SURVIVOR_FORECAST_COLUMNS, "forecast")
    columns = propertynames(forecast)
    normalized_picks = _normalize_survivor_picks(picks_made)
    used_teams = Set(values(normalized_picks))
    candidates = DataFrame(
        game_id=String[],
        week=Int[],
        team=String[],
        opponent=String[],
        is_home=Bool[],
        win_probability=Float64[],
    )
    seen_games = Set{String}()
    seen_team_weeks = Set{Tuple{Int,String}}()

    for row in eachrow(forecast)
        game_id = _schedule_string(row.game_id, :game_id)
        game_id in seen_games &&
            throw(ArgumentError("forecast game_id values must be unique"))
        push!(seen_games, game_id)

        week = _schedule_integer(row.week, :week)
        1 <= week <= 18 ||
            throw(ArgumentError("forecast weeks must be between 1 and 18"))
        away_team = _schedule_string(row.away_team, :away_team)
        home_team = _schedule_string(row.home_team, :home_team)
        away_team != home_team ||
            throw(ArgumentError("a game cannot have the same home and away team"))

        for team in (away_team, home_team)
            key = (week, team)
            key in seen_team_weeks &&
                throw(ArgumentError("a team cannot have multiple games in one week"))
            push!(seen_team_weeks, key)
        end

        completed = _forecast_game_completed(row, columns)
        include_completed || !completed || continue
        from_week <= week <= through_week || continue

        away_probability = _validate_win_probability(row.away_win_probability)
        home_probability = _validate_win_probability(row.home_win_probability)
        away_team in used_teams || push!(
            candidates,
            (
                game_id=game_id,
                week=week,
                team=away_team,
                opponent=home_team,
                is_home=false,
                win_probability=away_probability,
            ),
        )
        home_team in used_teams || push!(
            candidates,
            (
                game_id=game_id,
                week=week,
                team=home_team,
                opponent=away_team,
                is_home=true,
                win_probability=home_probability,
            ),
        )
    end

    return candidates
end

function build_survivor_candidates(
    context::RegularSeasonForecastContext;
    through_week::Integer=18,
    include_completed::Bool=false,
    picks_made=nothing,
    horizon::Real=GAME_CLOCK_SECONDS,
)
    context.as_of_week <= through_week <= 18 ||
        throw(ArgumentError("through_week must be at least the context as_of_week and at most 18"))
    forecast = forecast_win_probabilities(
        context;
        include_completed=include_completed,
        horizon=horizon,
        full_schedule=true,
    )
    return build_survivor_candidates(
        forecast;
        from_week=context.as_of_week,
        through_week=through_week,
        include_completed=true,
        picks_made=picks_made,
    )
end

function _normalize_survivor_candidates(
    candidates::AbstractDataFrame,
    state::SurvivorPoolState,
    through_week::Integer,
)
    _require_columns(
        candidates,
        (:game_id, :week, :team, :opponent, :is_home, :win_probability),
        "survivor candidates",
    )
    1 <= state.current_week <= through_week <= 18 ||
        throw(ArgumentError("week range must be within 1:18"))

    data = DataFrame(candidates)
    isempty(data) &&
        throw(ArgumentError("survivor candidates cannot be empty"))
    data.game_id = [_schedule_string(value, :game_id) for value in data.game_id]
    data.week = [_schedule_integer(value, :week) for value in data.week]
    data.team = [_schedule_string(value, :team) for value in data.team]
    data.opponent = [_schedule_string(value, :opponent) for value in data.opponent]
    data.is_home = [Bool(value) for value in data.is_home]
    data.win_probability = [
        _validate_win_probability(value) for value in data.win_probability
    ]

    used_teams = Set(values(state.picks_made))
    keep = [
        state.current_week <= week <= through_week && !(team in used_teams)
        for (week, team) in zip(data.week, data.team)
    ]
    data = data[keep, :]
    isempty(data) &&
        throw(ArgumentError("no eligible survivor candidates remain"))

    seen_team_weeks = Set{Tuple{Int,String}}()
    for row in eachrow(data)
        key = (row.week, row.team)
        row.team != row.opponent ||
            throw(ArgumentError("a survivor candidate cannot select its opponent"))
        key in seen_team_weeks &&
            throw(ArgumentError("a team cannot have multiple survivor candidates in one week"))
        push!(seen_team_weeks, key)
    end
    missing_weeks = setdiff(
        collect(state.current_week:through_week),
        sort(unique(data.week)),
    )
    isempty(missing_weeks) ||
        throw(ArgumentError("no eligible survivor candidates for week(s) $missing_weeks"))

    return data
end

function _survivor_discount_table(
    state::SurvivorPoolState,
    through_week::Integer,
    weekly_survival_probability::Real,
)
    discounts = survivor_reach_discounts(
        state.current_week,
        through_week;
        weekly_survival_probability=weekly_survival_probability,
        strikes_remaining=state.strikes_remaining,
    )
    return DataFrame(
        week=collect(state.current_week:through_week),
        discount=discounts,
    )
end

"""
    optimize_survivor_pool(candidates, state; ...)

Solve the survivor assignment problem from an injected team-level candidate
table. The objective maximizes expected future wins using fixed personal reach
discounts; it does not model the probability that the entire pool survives.
"""
function optimize_survivor_pool(
    candidates::AbstractDataFrame,
    state::SurvivorPoolState;
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    through_week::Integer=18,
    optimizer=HiGHS.Optimizer,
)
    probability = _validate_survivor_probability(weekly_survival_probability)
    data = _normalize_survivor_candidates(candidates, state, through_week)
    discount_table = _survivor_discount_table(state, through_week, probability)
    discount_by_week = Dict(
        row.week => row.discount for row in eachrow(discount_table)
    )
    data.discount = [discount_by_week[week] for week in data.week]
    data.objective_contribution = data.discount .* data.win_probability

    model = Model(optimizer)
    set_silent(model)
    candidate_indices = 1:nrow(data)
    @variable(model, selected[candidate_indices], Bin)

    for week in state.current_week:through_week
        indices = findall(==(week), data.week)
        @constraint(model, sum(selected[index] for index in indices) == 1)
    end
    for team in unique(data.team)
        indices = findall(==(team), data.team)
        @constraint(model, sum(selected[index] for index in indices) <= 1)
    end
    @objective(
        model,
        Max,
        sum(data.objective_contribution[index] * selected[index] for index in candidate_indices),
    )
    optimize!(model)

    JuMP.is_solved_and_feasible(model) ||
        throw(ArgumentError(
            "survivor optimization failed with termination status $(termination_status(model))",
        ))

    selected_indices = [
        index for index in candidate_indices if value(selected[index]) > 0.5
    ]
    selections = sort(data[selected_indices, :], [:week, :team])
    current_pick = selections[selections.week .== state.current_week, :]
    nrow(current_pick) == 1 ||
        throw(ArgumentError("survivor optimization did not select one current-week pick"))

    return SurvivorPoolPlan(
        state,
        selections,
        current_pick,
        discount_table,
        Float64(objective_value(model)),
    )
end

"""
    optimize_survivor_pool(context, state; ...)

Forecast unplayed games once from a fitted context, then solve the survivor
assignment model using those probabilities.
"""
function optimize_survivor_pool(
    context::RegularSeasonForecastContext,
    state::SurvivorPoolState;
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    through_week::Integer=18,
    include_completed::Bool=false,
    horizon::Real=GAME_CLOCK_SECONDS,
    optimizer=HiGHS.Optimizer,
)
    state.season == context.season ||
        throw(ArgumentError("survivor state season must match forecast context season"))
    state.current_week == context.as_of_week ||
        throw(ArgumentError("survivor state current_week must match context as_of_week"))
    candidates = build_survivor_candidates(
        context;
        through_week=through_week,
        include_completed=include_completed,
        picks_made=state.picks_made,
        horizon=horizon,
    )
    return optimize_survivor_pool(
        candidates,
        state;
        weekly_survival_probability=weekly_survival_probability,
        through_week=through_week,
        optimizer=optimizer,
    )
end

"""
    optimize_survivor_pool(season; as_of_week, ...)

Fit one frozen regular-season forecast context and solve a forward
survivor-pool plan. Re-run this once after refreshing the context for a new
week and passing the updated `picks_made` and `strikes_remaining`.
"""
function optimize_survivor_pool(
    season::Integer;
    as_of_week::Integer,
    picks_made=Dict{Int,String}(),
    strikes_remaining::Integer=1,
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    through_week::Integer=18,
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    current_drives::Union{Nothing,AbstractDataFrame}=nothing,
    max_seasons::Int=3,
    time_edges=DEFAULT_TIME_EDGES,
    recency_half_life::Union{Nothing,Real}=nothing,
    include_completed::Bool=false,
    horizon::Real=GAME_CLOCK_SECONDS,
    optimizer=HiGHS.Optimizer,
)
    state = SurvivorPoolState(
        season,
        as_of_week;
        picks_made=picks_made,
        strikes_remaining=strikes_remaining,
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
    return optimize_survivor_pool(
        context,
        state;
        weekly_survival_probability=weekly_survival_probability,
        through_week=through_week,
        include_completed=include_completed,
        horizon=horizon,
        optimizer=optimizer,
    )
end

function optimize_survivor_pool(
    context::RegularSeasonForecastContext;
    picks_made=Dict{Int,String}(),
    strikes_remaining::Integer=1,
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    through_week::Integer=18,
    include_completed::Bool=false,
    horizon::Real=GAME_CLOCK_SECONDS,
    optimizer=HiGHS.Optimizer,
)
    state = SurvivorPoolState(
        context.season,
        context.as_of_week;
        picks_made=picks_made,
        strikes_remaining=strikes_remaining,
    )
    return optimize_survivor_pool(
        context,
        state;
        weekly_survival_probability=weekly_survival_probability,
        through_week=through_week,
        include_completed=include_completed,
        horizon=horizon,
        optimizer=optimizer,
    )
end
