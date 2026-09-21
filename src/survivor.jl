const DEFAULT_SURVIVOR_WEEKLY_SURVIVAL_PROBABILITY = 0.65
const DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD = 2.0
const DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY = 0.5
const DEFAULT_SURVIVOR_OBJECTIVE = :exact_milp
const DEFAULT_SURVIVOR_REACH_DISCOUNT_POLICY = :binomial
const DEFAULT_SURVIVOR_MARKET_GUARD_WEEKS = 2
const DEFAULT_SURVIVOR_MISSING_MARKET_POLICY = :allow

const SURVIVOR_OBJECTIVES = (
    :milp,
    :fixed_exact_milp,
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

Configuration for survivor selection. `:exact_milp` is the default
covariance-aware expected-weeks formulation for fitted contexts;
`:fixed_exact_milp` is the fixed-probability compatibility formulation; and
`:milp` uses the tractable reach-discount approximation. `timeout_seconds`
limits the default HiGHS solve and returns its best feasible incumbent when
the limit is reached.
"""
struct SurvivorSelectionConfig
    objective::Symbol
    weekly_survival_probability::Float64
    reach_discount_policy::Symbol
    minimum_favorite_spread::Union{Nothing,Float64}
    missing_market_policy::Symbol
    market_guard_weeks::Int
    through_week::Int
    timeout_seconds::Union{Nothing,Float64}
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
    timeout_seconds=nothing,
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
    normalized_timeout = if timeout_seconds === nothing
        nothing
    else
        value = Float64(timeout_seconds)
        isfinite(value) && value > 0.0 ||
            throw(ArgumentError(
                "timeout_seconds must be nothing or a finite positive value",
            ))
        value
    end
    return SurvivorSelectionConfig(
        objective,
        probability,
        reach_discount_policy,
        normalized_spread,
        missing_market_policy,
        Int(market_guard_weeks),
        Int(through_week),
        normalized_timeout,
    )
end

function _survivor_optimizer(
    config::SurvivorSelectionConfig,
    optimizer,
)
    optimizer !== nothing && return optimizer
    attributes = Pair{String,Any}[
        "threads" => BLAS.get_num_threads(),
        "parallel" => "on",
    ]
    config.timeout_seconds === nothing ||
        push!(attributes, "time_limit" => config.timeout_seconds)
    return optimizer_with_attributes(HiGHS.Optimizer, attributes...)
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
discount used for each week. For exact objectives, `selections` also contains plan-specific survival and
elimination probabilities. The covariance-aware objective additionally reports
the posterior-mean survival probability, its parameter-variance adjustment,
and the adjusted survival estimate. `selection_config` records the effective
objective and eligibility policies.
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

function _survivor_has_feasible_incumbent(model)
    JuMP.is_solved_and_feasible(model) && return true
    return termination_status(model) == JuMP.MOI.TIME_LIMIT &&
        primal_status(model) == JuMP.MOI.FEASIBLE_POINT &&
        JuMP.has_values(model)
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
    @objective(model, Max, 0.0)
    optimize!(model)
    _survivor_has_feasible_incumbent(model) ||
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
    _set_survivor_plan_diagnostics!(
        selections,
        ones(Float64, nrow(selections)),
        zeros(Float64, nrow(selections)),
    )
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

function _set_survivor_plan_diagnostics!(
    selections::AbstractDataFrame,
    survival_probability::AbstractVector{<:Real},
    parameter_variance_adjustment::AbstractVector{<:Real},
)
    length(survival_probability) == nrow(selections) ||
        throw(ArgumentError("survival probabilities must match selections"))
    length(parameter_variance_adjustment) == nrow(selections) ||
        throw(ArgumentError("variance adjustments must match selections"))
    base = Float64.(survival_probability)
    adjustment = Float64.(parameter_variance_adjustment)
    adjusted = base .+ adjustment
    all(isfinite, base) && all(isfinite, adjustment) && all(isfinite, adjusted) ||
        throw(ArgumentError("survivor plan diagnostics must be finite"))
    selections.survival_probability = base
    selections.elimination_probability = 1.0 .- base
    selections.parameter_variance_adjustment = adjustment
    selections.variance_adjusted_survival_probability = adjusted
    selections.objective_contribution = adjusted
    return selections
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

function _optimize_survivor_expected_weeks_fixed_milp(
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
                    data.win_probability[index] *
                    transition[index, 1]
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

    _survivor_has_feasible_incumbent(model) ||
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
    survival_probability = [
        survival_by_week[Int(week)] for week in selections.week
    ]
    _set_survivor_plan_diagnostics!(
        selections,
        survival_probability,
        zeros(Float64, nrow(selections)),
    )
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

function _survivor_interval_product(
    lower::Float64,
    upper::Float64,
    coefficient::Float64,
)
    lower <= upper ||
        throw(ArgumentError("survivor interval bounds must be ordered"))
    coefficient >= 0.0 ?
        (coefficient * lower, coefficient * upper) :
        (coefficient * upper, coefficient * lower)
end

function _survivor_widen_interval(
    lower::Float64,
    upper::Float64,
)
    lower <= upper ||
        throw(ArgumentError("survivor interval bounds must be ordered"))
    scale = max(abs(lower), abs(upper), 1.0e-12)
    margin = 1.0e-9 * scale
    return lower - margin, upper + margin
end

function _survivor_fixed_selected_indices(
    data::AbstractDataFrame,
    selections::AbstractDataFrame,
    state::SurvivorPoolState,
    number_of_weeks::Integer,
)
    selected_by_position = zeros(Int, number_of_weeks)
    for row in eachrow(selections)
        position = Int(row.week) - state.current_week + 1
        1 <= position <= number_of_weeks ||
            throw(ArgumentError(
                "survivor warm-start selection is outside the optimization horizon",
            ))
        matches = findall(
            index ->
                Int(data.week[index]) == Int(row.week) &&
                String(data.team[index]) == String(row.team),
            1:nrow(data),
        )
        length(matches) == 1 ||
            throw(ArgumentError(
                "survivor warm-start selection does not identify one candidate",
            ))
        selected_by_position[position] = only(matches)
    end
    all(>(0), selected_by_position) ||
        throw(ArgumentError(
            "survivor warm-start plan did not select one candidate per week",
        ))
    return selected_by_position
end


function _survivor_interval_difference(
    first_lower::Float64,
    first_upper::Float64,
    second_lower::Float64,
    second_upper::Float64,
)
    first_lower <= first_upper ||
        throw(ArgumentError("survivor interval bounds must be ordered"))
    second_lower <= second_upper ||
        throw(ArgumentError("survivor interval bounds must be ordered"))
    return first_lower - second_upper, first_upper - second_lower
end

function _survivor_other_interval(
    lower::AbstractArray{<:Real},
    upper::AbstractArray{<:Real},
    indices,
    excluded::Integer,
    coordinates::Vararg{Int},
)
    other_indices = [index for index in indices if index != excluded]
    isempty(other_indices) && return nothing
    lower_value = minimum(
        lower[index, coordinates...]
        for index in other_indices
    )
    upper_value = maximum(
        upper[index, coordinates...]
        for index in other_indices
    )
    return Float64(lower_value), Float64(upper_value)
end

function _survivor_add_gated_recurrence!(
    model,
    aggregate,
    recurrence,
    selected,
    candidate_lower::Float64,
    candidate_upper::Float64,
    other_interval,
)
    candidate_lower <= candidate_upper ||
        throw(ArgumentError("survivor candidate recurrence bounds must be ordered"))
    if isnothing(other_interval)
        @constraint(model, aggregate == recurrence)
    else
        other_lower, other_upper = other_interval
        @constraint(
            model,
            aggregate - recurrence >=
                (other_lower - candidate_upper) * (1.0 - selected),
        )
        @constraint(
            model,
            aggregate - recurrence <=
                (other_upper - candidate_lower) * (1.0 - selected),
        )
    end
    return nothing
end

function _survivor_scalar_bounds(
    inputs::SurvivorObjectiveInputs,
    candidate_positions::AbstractVector{<:Integer},
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
)
    n_candidates = length(inputs.derivatives)
    length(candidate_positions) == n_candidates ||
        throw(ArgumentError(
            "survivor candidate positions must match derivative inputs",
        ))
    all(
        position -> 1 <= position <= number_of_weeks,
        candidate_positions,
    ) ||
        throw(ArgumentError("survivor candidate positions must be in the horizon"))
    week_indices = [
        findall(==(position), candidate_positions)
        for position in 1:number_of_weeks
    ]
    all(!isempty, week_indices) ||
        throw(ArgumentError(
            "survivor derivative bounds require candidates in every week",
        ))

    probability_lower =
        zeros(Float64, number_of_weeks + 1, losses_to_elimination)
    probability_upper =
        zeros(Float64, number_of_weeks + 1, losses_to_elimination)
    gradient_lower = zeros(
        Float64,
        number_of_weeks + 1,
        losses_to_elimination,
        n_candidates,
    )
    gradient_upper = zeros(
        Float64,
        number_of_weeks + 1,
        losses_to_elimination,
        n_candidates,
    )
    hessian_lower =
        zeros(Float64, number_of_weeks + 1, losses_to_elimination)
    hessian_upper =
        zeros(Float64, number_of_weeks + 1, losses_to_elimination)

    candidate_probability_lower =
        zeros(Float64, n_candidates, losses_to_elimination)
    candidate_probability_upper =
        zeros(Float64, n_candidates, losses_to_elimination)
    candidate_gradient_lower = zeros(
        Float64,
        n_candidates,
        losses_to_elimination,
        n_candidates,
    )
    candidate_gradient_upper = zeros(
        Float64,
        n_candidates,
        losses_to_elimination,
        n_candidates,
    )
    candidate_hessian_lower =
        zeros(Float64, n_candidates, losses_to_elimination)
    candidate_hessian_upper =
        zeros(Float64, n_candidates, losses_to_elimination)

    probability_lower[1, 1] = 1.0
    probability_upper[1, 1] = 1.0
    for position in 1:number_of_weeks
        for index in week_indices[position]
            derivative = inputs.derivatives[index]
            probability = derivative.base_probability
            for loss_state in 1:losses_to_elimination
                previous_lower, previous_upper = loss_state == 1 ?
                    (0.0, 0.0) :
                    (
                        probability_lower[position, loss_state - 1],
                        probability_upper[position, loss_state - 1],
                    )
                current_lower = probability_lower[position, loss_state]
                current_upper = probability_upper[position, loss_state]
                lower, upper = _survivor_interval_product(
                    current_lower,
                    current_upper,
                    probability,
                )
                previous_term_lower, previous_term_upper =
                    _survivor_interval_product(
                        previous_lower,
                        previous_upper,
                        1.0 - probability,
                    )
                lower += previous_term_lower
                upper += previous_term_upper
                candidate_probability_lower[index, loss_state],
                    candidate_probability_upper[index, loss_state] =
                    _survivor_widen_interval(lower, upper)

                probability_difference_lower, probability_difference_upper =
                    _survivor_interval_difference(
                        current_lower,
                        current_upper,
                        previous_lower,
                        previous_upper,
                    )
                for reference in 1:n_candidates
                    current_gradient_lower =
                        gradient_lower[position, loss_state, reference]
                    current_gradient_upper =
                        gradient_upper[position, loss_state, reference]
                    lower, upper = _survivor_interval_product(
                        current_gradient_lower,
                        current_gradient_upper,
                        probability,
                    )
                    previous_gradient_lower, previous_gradient_upper =
                        loss_state == 1 ?
                        (0.0, 0.0) :
                        (
                            gradient_lower[
                                position,
                                loss_state - 1,
                                reference,
                            ],
                            gradient_upper[
                                position,
                                loss_state - 1,
                                reference,
                            ],
                        )
                    previous_term_lower, previous_term_upper =
                        _survivor_interval_product(
                            previous_gradient_lower,
                            previous_gradient_upper,
                            1.0 - probability,
                        )
                    lower += previous_term_lower
                    upper += previous_term_upper
                    source_lower, source_upper = _survivor_interval_product(
                        probability_difference_lower,
                        probability_difference_upper,
                        inputs.covariance_gradient_gram[index, reference],
                    )
                    lower += source_lower
                    upper += source_upper
                    candidate_gradient_lower[index, loss_state, reference],
                        candidate_gradient_upper[index, loss_state, reference] =
                        _survivor_widen_interval(lower, upper)
                end

                current_hessian_lower = hessian_lower[position, loss_state]
                current_hessian_upper = hessian_upper[position, loss_state]
                lower, upper = _survivor_interval_product(
                    current_hessian_lower,
                    current_hessian_upper,
                    probability,
                )
                previous_hessian_lower, previous_hessian_upper =
                    loss_state == 1 ?
                    (0.0, 0.0) :
                    (
                        hessian_lower[position, loss_state - 1],
                        hessian_upper[position, loss_state - 1],
                    )
                previous_term_lower, previous_term_upper =
                    _survivor_interval_product(
                        previous_hessian_lower,
                        previous_hessian_upper,
                        1.0 - probability,
                    )
                lower += previous_term_lower
                upper += previous_term_upper
                source_lower, source_upper = _survivor_interval_product(
                    probability_difference_lower,
                    probability_difference_upper,
                    derivative.hessian_covariance,
                )
                lower += source_lower
                upper += source_upper
                current_gradient_lower =
                    gradient_lower[position, loss_state, index]
                current_gradient_upper =
                    gradient_upper[position, loss_state, index]
                previous_gradient_lower, previous_gradient_upper =
                    loss_state == 1 ?
                    (0.0, 0.0) :
                    (
                        gradient_lower[position, loss_state - 1, index],
                        gradient_upper[position, loss_state - 1, index],
                    )
                gradient_difference_lower, gradient_difference_upper =
                    _survivor_interval_difference(
                        current_gradient_lower,
                        current_gradient_upper,
                        previous_gradient_lower,
                        previous_gradient_upper,
                    )
                source_lower, source_upper = _survivor_interval_product(
                    gradient_difference_lower,
                    gradient_difference_upper,
                    2.0,
                )
                lower += source_lower
                upper += source_upper
                candidate_hessian_lower[index, loss_state],
                    candidate_hessian_upper[index, loss_state] =
                    _survivor_widen_interval(lower, upper)
            end
        end

        for loss_state in 1:losses_to_elimination
            probability_lower[position + 1, loss_state],
                probability_upper[position + 1, loss_state] =
                _survivor_widen_interval(
                    minimum(
                        candidate_probability_lower[index, loss_state]
                        for index in week_indices[position]
                    ),
                    maximum(
                        candidate_probability_upper[index, loss_state]
                        for index in week_indices[position]
                    ),
                )
            for reference in 1:n_candidates
                gradient_lower[position + 1, loss_state, reference],
                    gradient_upper[position + 1, loss_state, reference] =
                    _survivor_widen_interval(
                        minimum(
                            candidate_gradient_lower[
                                index,
                                loss_state,
                                reference,
                            ]
                            for index in week_indices[position]
                        ),
                        maximum(
                            candidate_gradient_upper[
                                index,
                                loss_state,
                                reference,
                            ]
                            for index in week_indices[position]
                        ),
                    )
            end
            hessian_lower[position + 1, loss_state],
                hessian_upper[position + 1, loss_state] =
                _survivor_widen_interval(
                    minimum(
                        candidate_hessian_lower[index, loss_state]
                        for index in week_indices[position]
                    ),
                    maximum(
                        candidate_hessian_upper[index, loss_state]
                        for index in week_indices[position]
                    ),
                )
        end
    end

    all(isfinite, probability_lower) &&
        all(isfinite, probability_upper) &&
        all(isfinite, gradient_lower) &&
        all(isfinite, gradient_upper) &&
        all(isfinite, hessian_lower) &&
        all(isfinite, hessian_upper) ||
        throw(ArgumentError("survivor scalar recurrence bounds are not finite"))
    return (
        probability=(lower=probability_lower, upper=probability_upper),
        gradient=(lower=gradient_lower, upper=gradient_upper),
        hessian=(lower=hessian_lower, upper=hessian_upper),
        candidate_probability=(
            lower=candidate_probability_lower,
            upper=candidate_probability_upper,
        ),
        candidate_gradient=(
            lower=candidate_gradient_lower,
            upper=candidate_gradient_upper,
        ),
        candidate_hessian=(
            lower=candidate_hessian_lower,
            upper=candidate_hessian_upper,
        ),
    )
end

function _survivor_scalar_forward_values(
    selected_by_position::AbstractVector{<:Integer},
    inputs::SurvivorObjectiveInputs,
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
)
    n_candidates = length(inputs.derivatives)
    length(selected_by_position) == number_of_weeks ||
        throw(ArgumentError(
            "survivor warm-start selections must cover the horizon",
        ))
    all(
        index -> 1 <= index <= n_candidates,
        selected_by_position,
    ) || throw(ArgumentError(
        "survivor warm-start selections must identify candidates",
    ))
    probability = zeros(
        Float64,
        number_of_weeks + 1,
        losses_to_elimination,
    )
    gradient = zeros(
        Float64,
        number_of_weeks + 1,
        losses_to_elimination,
        n_candidates,
    )
    hessian = zeros(
        Float64,
        number_of_weeks + 1,
        losses_to_elimination,
    )
    probability[1, 1] = 1.0
    for position in 1:number_of_weeks
        index = Int(selected_by_position[position])
        derivative = inputs.derivatives[index]
        win_probability = derivative.base_probability
        for loss_state in 1:losses_to_elimination
            previous_probability = loss_state == 1 ?
                0.0 :
                probability[position, loss_state - 1]
            probability[position + 1, loss_state] =
                win_probability * probability[position, loss_state] +
                (1.0 - win_probability) * previous_probability
            for reference in 1:n_candidates
                previous_gradient = loss_state == 1 ?
                    0.0 :
                    gradient[position, loss_state - 1, reference]
                gradient[position + 1, loss_state, reference] =
                    win_probability *
                    gradient[position, loss_state, reference] +
                    (1.0 - win_probability) * previous_gradient +
                    inputs.covariance_gradient_gram[index, reference] *
                    (
                        probability[position, loss_state] -
                        previous_probability
                    )
            end
            previous_hessian = loss_state == 1 ?
                0.0 :
                hessian[position, loss_state - 1]
            previous_gradient_for_selected = loss_state == 1 ?
                0.0 :
                gradient[position, loss_state - 1, index]
            hessian[position + 1, loss_state] =
                win_probability * hessian[position, loss_state] +
                (1.0 - win_probability) * previous_hessian +
                derivative.hessian_covariance *
                (
                    probability[position, loss_state] -
                    previous_probability
                ) +
                2.0 *
                (
                    gradient[position, loss_state, index] -
                    previous_gradient_for_selected
                )
        end
    end
    return (
        probability=probability,
        gradient=gradient,
        hessian=hessian,
    )
end

function _survivor_scalar_warm_start(
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    fixed_plan::SurvivorPoolPlan,
    inputs::SurvivorObjectiveInputs,
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
)
    selected_by_position = _survivor_fixed_selected_indices(
        data,
        fixed_plan.selections,
        state,
        number_of_weeks,
    )
    values = _survivor_scalar_forward_values(
        selected_by_position,
        inputs,
        number_of_weeks,
        losses_to_elimination,
    )
    return merge(values, (selected=selected_by_position,))
end

function _set_survivor_scalar_warm_start!(
    selected,
    probability,
    gradient,
    hessian,
    warm_start,
    candidate_indices,
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
)
    for index in candidate_indices
        set_start_value(
            selected[index],
            any(warm_start.selected .== index) ? 1.0 : 0.0,
        )
    end
    for position in 1:(number_of_weeks + 1), loss_state in 1:losses_to_elimination
        set_start_value(
            probability[position, loss_state],
            warm_start.probability[position, loss_state],
        )
        set_start_value(
            hessian[position, loss_state],
            warm_start.hessian[position, loss_state],
        )
        for reference in candidate_indices
            set_start_value(
                gradient[position, loss_state, reference],
                warm_start.gradient[position, loss_state, reference],
            )
        end
    end
    return nothing
end

function _optimize_survivor_expected_weeks_scalar_milp(
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
    discount_table::AbstractDataFrame,
    inputs::SurvivorObjectiveInputs;
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
    length(inputs.derivatives) == nrow(data) ||
        throw(ArgumentError("survivor derivative inputs must match candidates"))

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
    candidate_positions = [
        Int(data.week[index]) - state.current_week + 1
        for index in candidate_indices
    ]
    bounds = _survivor_scalar_bounds(
        inputs,
        candidate_positions,
        number_of_weeks,
        losses_to_elimination,
    )
    @variable(
        model,
        probability[1:(number_of_weeks + 1), state_indices],
    )
    @variable(
        model,
        gradient[
            1:(number_of_weeks + 1),
            state_indices,
            candidate_indices,
        ],
    )
    @variable(
        model,
        hessian[1:(number_of_weeks + 1), state_indices],
    )

    for position in 1:(number_of_weeks + 1), loss_state in state_indices
        set_lower_bound(
            probability[position, loss_state],
            bounds.probability.lower[position, loss_state],
        )
        set_upper_bound(
            probability[position, loss_state],
            bounds.probability.upper[position, loss_state],
        )
        set_lower_bound(
            hessian[position, loss_state],
            bounds.hessian.lower[position, loss_state],
        )
        set_upper_bound(
            hessian[position, loss_state],
            bounds.hessian.upper[position, loss_state],
        )
        for reference in candidate_indices
            set_lower_bound(
                gradient[position, loss_state, reference],
                bounds.gradient.lower[position, loss_state, reference],
            )
            set_upper_bound(
                gradient[position, loss_state, reference],
                bounds.gradient.upper[position, loss_state, reference],
            )
        end
    end

    @constraint(model, probability[1, 1] == 1.0)
    for loss_state in 2:losses_to_elimination
        @constraint(model, probability[1, loss_state] == 0.0)
    end
    for loss_state in state_indices
        @constraint(model, hessian[1, loss_state] == 0.0)
        for reference in candidate_indices
            @constraint(model, gradient[1, loss_state, reference] == 0.0)
        end
    end

    week_indices = [
        findall(==(position), candidate_positions)
        for position in 1:number_of_weeks
    ]
    for position in 1:number_of_weeks
        indices = week_indices[position]
        for loss_state in state_indices
            previous_probability = loss_state == 1 ?
                0.0 :
                probability[position, loss_state - 1]
            for index in indices
                derivative = inputs.derivatives[index]
                recurrence =
                    derivative.base_probability *
                    probability[position, loss_state] +
                    (1.0 - derivative.base_probability) *
                    previous_probability
                other_interval = _survivor_other_interval(
                    bounds.candidate_probability.lower,
                    bounds.candidate_probability.upper,
                    indices,
                    index,
                    loss_state,
                )
                _survivor_add_gated_recurrence!(
                    model,
                    probability[position + 1, loss_state],
                    recurrence,
                    selected[index],
                    bounds.candidate_probability.lower[index, loss_state],
                    bounds.candidate_probability.upper[index, loss_state],
                    other_interval,
                )

                probability_difference =
                    probability[position, loss_state] -
                    previous_probability
                for reference in candidate_indices
                    previous_gradient = loss_state == 1 ?
                        0.0 :
                        gradient[position, loss_state - 1, reference]
                    recurrence =
                        derivative.base_probability *
                        gradient[position, loss_state, reference] +
                        (1.0 - derivative.base_probability) *
                        previous_gradient +
                        inputs.covariance_gradient_gram[index, reference] *
                        probability_difference
                    other_interval = _survivor_other_interval(
                        bounds.candidate_gradient.lower,
                        bounds.candidate_gradient.upper,
                        indices,
                        index,
                        loss_state,
                        reference,
                    )
                    _survivor_add_gated_recurrence!(
                        model,
                        gradient[
                            position + 1,
                            loss_state,
                            reference,
                        ],
                        recurrence,
                        selected[index],
                        bounds.candidate_gradient.lower[
                            index,
                            loss_state,
                            reference,
                        ],
                        bounds.candidate_gradient.upper[
                            index,
                            loss_state,
                            reference,
                        ],
                        other_interval,
                    )
                end

                previous_hessian = loss_state == 1 ?
                    0.0 :
                    hessian[position, loss_state - 1]
                previous_gradient_for_selected = loss_state == 1 ?
                    0.0 :
                    gradient[position, loss_state - 1, index]
                recurrence =
                    derivative.base_probability *
                    hessian[position, loss_state] +
                    (1.0 - derivative.base_probability) *
                    previous_hessian +
                    derivative.hessian_covariance *
                    probability_difference +
                    2.0 *
                    (
                        gradient[position, loss_state, index] -
                        previous_gradient_for_selected
                    )
                other_interval = _survivor_other_interval(
                    bounds.candidate_hessian.lower,
                    bounds.candidate_hessian.upper,
                    indices,
                    index,
                    loss_state,
                )
                _survivor_add_gated_recurrence!(
                    model,
                    hessian[position + 1, loss_state],
                    recurrence,
                    selected[index],
                    bounds.candidate_hessian.lower[index, loss_state],
                    bounds.candidate_hessian.upper[index, loss_state],
                    other_interval,
                )
            end
        end
    end

    fixed_plan = _optimize_survivor_expected_weeks_fixed_milp(
        data,
        state,
        config,
        discount_table;
        optimizer=optimizer,
    )
    warm_start = _survivor_scalar_warm_start(
        data,
        state,
        fixed_plan,
        inputs,
        number_of_weeks,
        losses_to_elimination,
    )
    _set_survivor_scalar_warm_start!(
        selected,
        probability,
        gradient,
        hessian,
        warm_start,
        candidate_indices,
        number_of_weeks,
        losses_to_elimination,
    )

    @objective(
        model,
        Max,
        sum(
            probability[position + 1, loss_state] +
            0.5 * hessian[position + 1, loss_state]
            for position in 1:number_of_weeks,
            loss_state in state_indices
        ),
    )
    optimize!(model)

    _survivor_has_feasible_incumbent(model) ||
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

    selected_by_position = _survivor_fixed_selected_indices(
        data,
        selections,
        state,
        number_of_weeks,
    )
    selected_values = _survivor_scalar_forward_values(
        selected_by_position,
        inputs,
        number_of_weeks,
        losses_to_elimination,
    )
    base_by_week = Dict{Int,Float64}()
    adjustment_by_week = Dict{Int,Float64}()
    for position in 1:number_of_weeks
        week = state.current_week + position - 1
        base_by_week[week] = sum(
            selected_values.probability[position + 1, loss_state]
            for loss_state in state_indices
        )
        adjustment_by_week[week] = 0.5 * sum(
            selected_values.hessian[position + 1, loss_state]
            for loss_state in state_indices
        )
    end
    base_survival = [base_by_week[Int(week)] for week in selections.week]
    parameter_variance_adjustment = [
        adjustment_by_week[Int(week)] for week in selections.week
    ]
    _set_survivor_plan_diagnostics!(
        selections,
        base_survival,
        parameter_variance_adjustment,
    )
    model_expected_weeks = Float64(objective_value(model))
    selected_expected_weeks = sum(selections.objective_contribution)
    isapprox(
        model_expected_weeks,
        selected_expected_weeks;
        rtol=1e-6,
        atol=2e-6,
    ) || throw(ArgumentError(
        "survivor covariance objective and selected-plan evaluation disagree: " *
        "$model_expected_weeks vs $selected_expected_weeks",
    ))

    current_pick = selections[selections.week .== state.current_week, :]
    nrow(current_pick) == 1 ||
        throw(ArgumentError("survivor optimization did not select one current-week pick"))

    return SurvivorPoolPlan(
        state,
        selections,
        current_pick,
        discount_table,
        selected_expected_weeks,
        config,
    )
end

"""
    optimize_survivor_pool(candidates, state; ...)

Solve the survivor assignment problem from an injected team-level candidate
table. `selection_config` controls the objective, reach discounts, market
guard, missing-line policy, and planning horizon. The default objective
`:exact_milp` requires a fitted context for its covariance-aware objective.
Use `:fixed_exact_milp` for the fixed-probability exact state-transition MILP
with injected candidates. The `:milp` objective uses fixed personal reach
discounts as a tractable approximation.
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
        throw(ArgumentError(
            "covariance-aware :exact_milp requires a " *
            "RegularSeasonForecastContext; use objective=:fixed_exact_milp " *
            "for injected candidate tables",
        ))
    elseif config.objective === :fixed_exact_milp
        return _optimize_survivor_expected_weeks_fixed_milp(
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

    _survivor_has_feasible_incumbent(model) ||
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
    selection_config.objective !== :exact_milp &&
        return optimize_survivor_pool(
            candidates,
            state;
            selection_config=selection_config,
            optimizer=optimizer,
        )

    data = _normalize_survivor_candidates(
        candidates,
        state,
        selection_config.through_week,
    )
    discount_table = _survivor_discount_table(state, selection_config)
    discount_by_week = Dict(
        row.week => row.discount for row in eachrow(discount_table)
    )
    data.discount = [discount_by_week[week] for week in data.week]
    inputs = _survivor_objective_inputs(
        context.model,
        context.marks,
        data;
        horizon=horizon,
    )
    return _optimize_survivor_expected_weeks_scalar_milp(
        data,
        state,
        selection_config,
        discount_table,
        inputs;
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
