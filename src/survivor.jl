const DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY = 0.65
const DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD = 2.0
const DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY = 0.5
const DEFAULT_SURVIVOR_OBJECTIVE = :exact_milp
const DEFAULT_SURVIVOR_REACH_DISCOUNT_POLICY = :binomial
const DEFAULT_SURVIVOR_MARKET_GUARD_WEEKS = 2
const DEFAULT_SURVIVOR_MISSING_MARKET_POLICY = :allow

const SURVIVOR_OBJECTIVES = (
    :milp,
    :exact_milp,
)
const SURVIVOR_REACH_DISCOUNT_POLICIES = (:binomial,)
const SURVIVOR_MISSING_MARKET_POLICIES = (:allow, :exclude)

function _canonical_survivor_objective(objective::Symbol)
    objective in SURVIVOR_OBJECTIVES && return objective
    throw(ArgumentError(
        "objective must be one of $(collect(SURVIVOR_OBJECTIVES)); got $objective",
    ))
end

"""
    SurvivorPoolState

State required to optimize a survivor-pool plan. `picks_made` maps completed
weeks to the teams selected in those weeks. `strikes_remaining` is the number of strikes remaining before elimination. A
positive value `s` means the `s`-th future loss is terminal; zero means the
next loss is terminal.
"""
struct SurvivorPoolState
    season::Int
    current_week::Int
    picks_made::Dict{Int,String}
    strikes_remaining::Int
end

"""
    SurvivorSelectionConfig(; kwargs...)

Configuration for survivor selection. `:exact_milp` is the default exact
expected-weeks formulation; `:milp` uses the tractable approximation.
"""
struct SurvivorSelectionConfig
    objective::Symbol
    weekly_survival_probability::Float64
    reach_discount_policy::Symbol
    minimum_favorite_spread::Union{Nothing,Float64}
    missing_market_policy::Symbol
    market_guard_weeks::Int
    through_week::Int
end

function SurvivorSelectionConfig(
    ;
    objective::Symbol=DEFAULT_SURVIVOR_OBJECTIVE,
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    reach_discount_policy::Symbol=DEFAULT_SURVIVOR_REACH_DISCOUNT_POLICY,
    minimum_favorite_spread=DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD,
    missing_market_policy::Symbol=DEFAULT_SURVIVOR_MISSING_MARKET_POLICY,
    market_guard_weeks::Integer=DEFAULT_SURVIVOR_MARKET_GUARD_WEEKS,
    through_week::Integer=18,
)
    objective = _canonical_survivor_objective(objective)
    probability = _validate_survivor_probability(weekly_survival_probability)
    reach_discount_policy in SURVIVOR_REACH_DISCOUNT_POLICIES ||
        throw(ArgumentError(
            "reach_discount_policy must be one of " *
            "$(collect(SURVIVOR_REACH_DISCOUNT_POLICIES)); got " *
            "$reach_discount_policy",
        ))
    normalized_spread = if minimum_favorite_spread === nothing
        nothing
    else
        value = Float64(minimum_favorite_spread)
        isfinite(value) && value >= 0.0 ||
            throw(ArgumentError(
                "minimum_favorite_spread must be nothing or a finite nonnegative value",
            ))
        value
    end
    missing_market_policy in SURVIVOR_MISSING_MARKET_POLICIES ||
        throw(ArgumentError(
            "missing_market_policy must be one of " *
            "$(collect(SURVIVOR_MISSING_MARKET_POLICIES)); got " *
            "$missing_market_policy",
        ))
    market_guard_weeks >= 0 ||
        throw(ArgumentError("market_guard_weeks must be nonnegative"))
    1 <= through_week <= 18 ||
        throw(ArgumentError("through_week must be between 1 and 18"))
    return SurvivorSelectionConfig(
        objective,
        probability,
        reach_discount_policy,
        normalized_spread,
        missing_market_policy,
        Int(market_guard_weeks),
        Int(through_week),
    )
end

function _survivor_optimizer(
    config::SurvivorSelectionConfig,
    optimizer,
)
    optimizer !== nothing && return optimizer
    return HiGHS.Optimizer
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
    strikes_remaining::Integer=2,
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
    strikes_remaining::Integer=2,
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
discount used for each week. For `:exact_milp`, `selections` also contains
plan-specific survival and elimination probabilities. `selection_config`
records the effective objective and eligibility policies.
"""
struct SurvivorPoolPlan
    state::SurvivorPoolState
    selections::DataFrame
    current_pick::DataFrame
    discounts::DataFrame
    objective_value::Float64
    selection_config::SurvivorSelectionConfig
end

function _validate_survivor_probability(probability::Real)
    value = Float64(probability)
    isfinite(value) && 0.0 <= value <= 1.0 ||
        throw(ArgumentError("weekly_survival_probability must be finite and in [0, 1]"))
    return value
end

function _survivor_market_spread(value)
    ismissing(value) && return missing
    parsed = value isa Real ? Float64(value) : tryparse(Float64, string(value))
    parsed === nothing ||
        (isfinite(parsed) && return parsed)
    throw(ArgumentError("market spread must be finite or missing"))
end

function _survivor_market_eligible(
    week::Integer,
    market_spread,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
)
    config.minimum_favorite_spread === nothing && return true
    protected_week = state.current_week <= week <
        state.current_week + config.market_guard_weeks
    !protected_week && return true
    ismissing(market_spread) &&
        return config.missing_market_policy === :allow
    return market_spread >= config.minimum_favorite_spread
end

function _survivor_market_guard_mask(
    candidates::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
)
    return [
        _survivor_market_eligible(week, market_spread, state, config)
        for (week, market_spread) in zip(
            candidates.week,
            candidates.market_spread,
        )
    ]
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
of fewer than `s` losses in those `k` weeks; zero strikes allows no losses.
"""
function survivor_reach_discounts(
    number_of_weeks::Integer;
    weekly_survival_probability::Real=DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY,
    strikes_remaining::Integer=0,
    reach_discount_policy::Symbol=DEFAULT_SURVIVOR_REACH_DISCOUNT_POLICY,
)
    number_of_weeks >= 0 ||
        throw(ArgumentError("number_of_weeks must be nonnegative"))
    strikes_remaining >= 0 ||
        throw(ArgumentError("strikes_remaining must be nonnegative"))
    probability = _validate_survivor_probability(weekly_survival_probability)
    reach_discount_policy in SURVIVOR_REACH_DISCOUNT_POLICIES ||
        throw(ArgumentError(
            "reach_discount_policy must be one of " *
            "$(collect(SURVIVOR_REACH_DISCOUNT_POLICIES)); got " *
            "$reach_discount_policy",
        ))
    number_of_weeks == 0 && return Float64[]

    discounts = Float64[]
    for prior_weeks in 0:(number_of_weeks - 1)
        reach_probability = 0.0
        for losses in 0:min(max(Int(strikes_remaining) - 1, 0), prior_weeks)
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
    reach_discount_policy::Symbol=DEFAULT_SURVIVOR_REACH_DISCOUNT_POLICY,
)
    1 <= current_week <= through_week <= 18 ||
        throw(ArgumentError("week range must be within 1:18"))
    return survivor_reach_discounts(
        through_week - current_week + 1;
        weekly_survival_probability=weekly_survival_probability,
        strikes_remaining=strikes_remaining,
        reach_discount_policy=reach_discount_policy,
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
Only teams with at least a 0.5 model win probability are included. The input
must contain game identifiers, weeks, home/away teams, and the corresponding
home/away win probabilities.
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
    has_spread_line = :spread_line in columns
    candidates = DataFrame(
        game_id=String[],
        week=Int[],
        team=String[],
        opponent=String[],
        is_home=Bool[],
        win_probability=Float64[],
        market_spread=Union{Missing,Float64}[],
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
        spread_line = has_spread_line ?
            _survivor_market_spread(row.spread_line) :
            missing

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
        away_team in used_teams ||
            away_probability < DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY ||
            push!(
            candidates,
            (
                game_id=game_id,
                week=week,
                team=away_team,
                opponent=home_team,
                is_home=false,
                win_probability=away_probability,
                market_spread=ismissing(spread_line) ? missing : -spread_line,
            ),
        )
        home_team in used_teams ||
            home_probability < DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY ||
            push!(
            candidates,
            (
                game_id=game_id,
                week=week,
                team=home_team,
                opponent=away_team,
                is_home=true,
                win_probability=home_probability,
                market_spread=spread_line,
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
    data = data[
        data.win_probability .>= DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY,
        :,
    ]
    isempty(data) &&
        throw(ArgumentError(
            "no survivor candidates meet the model-favorite threshold of " *
            "$DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY",
        ))
    if :market_spread in propertynames(data)
        data.market_spread = [
            _survivor_market_spread(value) for value in data.market_spread
        ]
    else
        data.market_spread = Union{Missing,Float64}[missing for _ in 1:nrow(data)]
    end

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
    config::SurvivorSelectionConfig,
)
    discounts = survivor_reach_discounts(
        state.current_week,
        config.through_week;
        weekly_survival_probability=config.weekly_survival_probability,
        strikes_remaining=state.strikes_remaining,
        reach_discount_policy=config.reach_discount_policy,
    )
    return DataFrame(
        week=collect(state.current_week:config.through_week),
        discount=discounts,
    )
end

function _add_survivor_assignment_constraints!(
    model,
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
    selected,
)
    candidate_indices = 1:nrow(data)
    for week in state.current_week:config.through_week
        indices = findall(==(week), data.week)
        @constraint(model, sum(selected[index] for index in indices) == 1)
    end
    for team in unique(data.team)
        indices = findall(==(team), data.team)
        @constraint(model, sum(selected[index] for index in indices) <= 1)
    end
    market_eligible = _survivor_market_guard_mask(data, state, config)
    for index in candidate_indices
        market_eligible[index] || @constraint(model, selected[index] == 0)
    end
    return candidate_indices
end

function _survivor_loss_threshold(state::SurvivorPoolState)
    return max(1, state.strikes_remaining)
end

function _survivor_week_indices(
    data::AbstractDataFrame,
)
    week_indices = Dict{Int,Vector{Int}}()
    for index in 1:nrow(data)
        push!(get!(week_indices, Int(data.week[index]), Int[]), index)
    end
    return week_indices
end

function _survivor_selected_indices(model, selected, candidate_indices)
    return [
        index for index in candidate_indices if value(selected[index]) > 0.5
    ]
end

function _survivor_objective_contribution(
    probability::Real,
    discount::Real,
    objective::Symbol,
)
    objective === :milp &&
        return Float64(discount) * Float64(probability)
    throw(ArgumentError("unsupported survivor objective: $objective"))
end

function _survivor_constant_plan(
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
    discount_table::AbstractDataFrame;
    optimizer=nothing,
)
    model = Model(isnothing(optimizer) ? HiGHS.Optimizer : optimizer)
    set_silent(model)
    candidate_indices = 1:nrow(data)
    @variable(model, selected[candidate_indices], Bin)
    _add_survivor_assignment_constraints!(
        model,
        data,
        state,
        config,
        selected,
    )
    @objective(model, Max, 0.0)
    optimize!(model)
    JuMP.is_solved_and_feasible(model) ||
        throw(ArgumentError(
            "survivor optimization failed with termination status " *
            "$(termination_status(model))",
        ))

    selected_indices = _survivor_selected_indices(
        model,
        selected,
        candidate_indices,
    )
    selections = sort(data[selected_indices, :], [:week, :team])
    selections.survival_probability = ones(Float64, nrow(selections))
    selections.elimination_probability = zeros(Float64, nrow(selections))
    selections.objective_contribution = selections.survival_probability
    current_pick = selections[selections.week .== state.current_week, :]
    nrow(current_pick) == 1 ||
        throw(ArgumentError("survivor optimization did not select one current-week pick"))

    return SurvivorPoolPlan(
        state,
        selections,
        current_pick,
        discount_table,
        Float64(config.through_week - state.current_week + 1),
        config,
    )
end

function _survivor_state_statistics(
    probabilities::AbstractVector{<:Real},
    losses_to_elimination::Integer,
)
    losses_to_elimination >= 1 ||
        throw(ArgumentError("losses_to_elimination must be positive"))
    alive = zeros(Float64, losses_to_elimination)
    alive[1] = 1.0
    survival_probability = zeros(Float64, length(probabilities))
    for (position, probability_value) in enumerate(probabilities)
        probability = Float64(probability_value)
        0.0 <= probability <= 1.0 ||
            throw(ArgumentError("win probabilities must be between 0 and 1"))
        next_alive = zeros(Float64, losses_to_elimination)
        next_alive[1] = probability * alive[1]
        for loss_state in 2:losses_to_elimination
            next_alive[loss_state] =
                probability * alive[loss_state] +
                (1.0 - probability) * alive[loss_state - 1]
        end
        alive = next_alive
        survival_probability[position] = sum(alive)
    end
    return (
        survival_probability=survival_probability,
        expected_weeks=sum(survival_probability),
    )
end

function _optimize_survivor_expected_weeks_exact_milp(
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
    discount_table::AbstractDataFrame;
    optimizer=nothing,
)
    number_of_weeks = config.through_week - state.current_week + 1
    losses_to_elimination = _survivor_loss_threshold(state)
    losses_to_elimination > number_of_weeks &&
        return _survivor_constant_plan(
            data,
            state,
            config,
            discount_table;
            optimizer=optimizer,
        )

    model = Model(_survivor_optimizer(config, optimizer))
    set_silent(model)
    candidate_indices = 1:nrow(data)
    @variable(model, selected[candidate_indices], Bin)
    _add_survivor_assignment_constraints!(
        model,
        data,
        state,
        config,
        selected,
    )

    state_indices = 1:losses_to_elimination
    @variable(
        model,
        0 <= alive[1:(number_of_weeks + 1), state_indices] <= 1,
    )
    @variable(model, 0 <= transition[candidate_indices, state_indices] <= 1)
    @constraint(model, alive[1, 1] == 1.0)
    for loss_state in 2:losses_to_elimination
        @constraint(model, alive[1, loss_state] == 0.0)
    end

    week_indices = _survivor_week_indices(data)
    for index in candidate_indices
        position = Int(data.week[index]) - state.current_week + 1
        for loss_state in state_indices
            @constraint(model, transition[index, loss_state] <= selected[index])
        end
    end

    for position in 1:number_of_weeks
        week = state.current_week + position - 1
        indices = week_indices[week]
        for loss_state in state_indices
            @constraint(
                model,
                sum(transition[index, loss_state] for index in indices) ==
                    alive[position, loss_state],
            )
        end
        @constraint(
            model,
            alive[position + 1, 1] ==
                sum(
                    data.win_probability[index] * transition[index, 1]
                    for index in indices
                ),
        )
        for loss_state in 2:losses_to_elimination
            @constraint(
                model,
                alive[position + 1, loss_state] ==
                    sum(
                        data.win_probability[index] * transition[index, loss_state] +
                        (1.0 - data.win_probability[index]) *
                        transition[index, loss_state - 1]
                        for index in indices
                    ),
            )
        end
    end

    @objective(
        model,
        Max,
        sum(
            alive[position + 1, loss_state]
            for position in 1:number_of_weeks,
            loss_state in state_indices
        ),
    )
    optimize!(model)

    JuMP.is_solved_and_feasible(model) ||
        throw(ArgumentError(
            "survivor optimization failed with termination status " *
            "$(termination_status(model))",
        ))
    selected_indices = _survivor_selected_indices(
        model,
        selected,
        candidate_indices,
    )
    selections = sort(data[selected_indices, :], [:week, :team])
    nrow(selections) == number_of_weeks ||
        throw(ArgumentError("survivor optimization did not select one team per week"))

    probabilities = Float64.(selections.win_probability)
    statistics = _survivor_state_statistics(
        probabilities,
        losses_to_elimination,
    )
    model_expected_weeks = Float64(objective_value(model))
    isapprox(
        model_expected_weeks,
        statistics.expected_weeks;
        rtol=1e-6,
        atol=2e-6,
    ) || throw(ArgumentError(
        "survivor state-transition objective and selected-plan evaluation disagree: " *
        "$model_expected_weeks vs $(statistics.expected_weeks)",
    ))

    survival_by_week = Dict(
        state.current_week + position - 1 => probability
        for (position, probability) in enumerate(
            statistics.survival_probability,
        )
    )
    selections.survival_probability = [
        survival_by_week[Int(week)] for week in selections.week
    ]
    selections.elimination_probability = 1.0 .- selections.survival_probability
    selections.objective_contribution = selections.survival_probability
    current_pick = selections[selections.week .== state.current_week, :]
    nrow(current_pick) == 1 ||
        throw(ArgumentError("survivor optimization did not select one current-week pick"))

    return SurvivorPoolPlan(
        state,
        selections,
        current_pick,
        discount_table,
        Float64(statistics.expected_weeks),
        config,
    )
end

"""
    optimize_survivor_pool(candidates, state; ...)

Solve the survivor assignment problem from an injected team-level candidate
table. `selection_config` controls the objective, reach discounts, market
guard, missing-line policy, and planning horizon. The default objective
`:exact_milp` uses a state-transition MILP and reports expected completed
weeks survived. The `:milp` objective uses fixed personal reach discounts as a
tractable approximation.
"""
function optimize_survivor_pool(
    candidates::AbstractDataFrame,
    state::SurvivorPoolState;
    selection_config::SurvivorSelectionConfig=SurvivorSelectionConfig(),
    optimizer=nothing,
)
    config = selection_config
    state.current_week <= config.through_week ||
        throw(ArgumentError(
            "selection through_week must be at least the current week",
        ))
    data = _normalize_survivor_candidates(
        candidates,
        state,
        config.through_week,
    )
    discount_table = _survivor_discount_table(state, config)
    discount_by_week = Dict(
        row.week => row.discount for row in eachrow(discount_table)
    )
    data.discount = [discount_by_week[week] for week in data.week]
    if config.objective === :exact_milp
        return _optimize_survivor_expected_weeks_exact_milp(
            data,
            state,
            config,
            discount_table;
            optimizer=optimizer,
        )
    end
    data.objective_contribution = [
        _survivor_objective_contribution(
            probability,
            discount,
            config.objective,
        )
        for (probability, discount) in zip(
            data.win_probability,
            data.discount,
        )
    ]

    model = Model(_survivor_optimizer(config, optimizer))
    set_silent(model)
    candidate_indices = 1:nrow(data)
    @variable(model, selected[candidate_indices], Bin)
    _add_survivor_assignment_constraints!(
        model,
        data,
        state,
        config,
        selected,
    )
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

    selected_indices = _survivor_selected_indices(
        model,
        selected,
        candidate_indices,
    )
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
        config,
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
    selection_config::SurvivorSelectionConfig=SurvivorSelectionConfig(),
    include_completed::Bool=false,
    horizon::Real=GAME_CLOCK_SECONDS,
    optimizer=nothing,
)
    state.season == context.season ||
        throw(ArgumentError("survivor state season must match forecast context season"))
    state.current_week == context.as_of_week ||
        throw(ArgumentError("survivor state current_week must match context as_of_week"))
    candidates = build_survivor_candidates(
        context;
        through_week=selection_config.through_week,
        include_completed=include_completed,
        picks_made=state.picks_made,
        horizon=horizon,
    )
    return optimize_survivor_pool(
        candidates,
        state;
        selection_config=selection_config,
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
    strikes_remaining::Integer=2,
    selection_config::SurvivorSelectionConfig=SurvivorSelectionConfig(),
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    historical_drives::Union{Nothing,AbstractDataFrame}=nothing,
    current_drives::Union{Nothing,AbstractDataFrame}=nothing,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    time_edges=DEFAULT_TIME_EDGES,
    method::PriorFitMethod=DEFAULT_PRIOR_FIT_METHOD,
    include_completed::Bool=false,
    horizon::Real=GAME_CLOCK_SECONDS,
    optimizer=nothing,
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
        method=method,
    )
    return optimize_survivor_pool(
        context,
        state;
        selection_config=selection_config,
        include_completed=include_completed,
        horizon=horizon,
        optimizer=optimizer,
    )
end

function optimize_survivor_pool(
    context::RegularSeasonForecastContext;
    picks_made=Dict{Int,String}(),
    strikes_remaining::Integer=2,
    selection_config::SurvivorSelectionConfig=SurvivorSelectionConfig(),
    include_completed::Bool=false,
    horizon::Real=GAME_CLOCK_SECONDS,
    optimizer=nothing,
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
        selection_config=selection_config,
        include_completed=include_completed,
        horizon=horizon,
        optimizer=optimizer,
    )
end
