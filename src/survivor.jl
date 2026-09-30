const DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD = 2.0
const DEFAULT_SURVIVOR_MIN_MODEL_WIN_PROBABILITY = 0.5
const DEFAULT_SURVIVOR_MARKET_GUARD_WEEKS = 2
const DEFAULT_SURVIVOR_MISSING_MARKET_POLICY = :allow

const SURVIVOR_MISSING_MARKET_POLICIES = (:allow, :exclude)

function _normalize_survivor_team_abbreviations(
    teams::AbstractVector{<:AbstractString},
    option::AbstractString,
)
    normalized = String[]
    for raw_team in teams
        team = uppercase(strip(String(raw_team)))
        isempty(team) &&
            throw(ArgumentError("$option must contain nonempty team abbreviations"))
        team in normalized || push!(normalized, team)
    end
    return normalized
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
covariance-aware expected-weeks formulation. `timeout_seconds` limits the
HiGHS solve and returns its best feasible incumbent when the limit is reached.
`hessian_weeks` controls how many future weeks receive the covariance-aware
Hessian adjustment; later weeks retain their posterior-mean probability terms
only. `banned_first_pick_teams` excludes the listed teams from the current-week
pick only; they may still be used in later weeks.
"""
struct SurvivorSelectionConfig
    minimum_favorite_spread::Union{Nothing,Float64}
    missing_market_policy::Symbol
    market_guard_weeks::Int
    through_week::Int
    banned_first_pick_teams::Vector{String}
    hessian_weeks::Int
    timeout_seconds::Union{Nothing,Float64}
end

function SurvivorSelectionConfig(
    ;
    minimum_favorite_spread=DEFAULT_SURVIVOR_MIN_FAVORITE_SPREAD,
    missing_market_policy::Symbol=DEFAULT_SURVIVOR_MISSING_MARKET_POLICY,
    market_guard_weeks::Integer=DEFAULT_SURVIVOR_MARKET_GUARD_WEEKS,
    through_week::Integer=18,
    banned_first_pick_teams::AbstractVector{<:AbstractString}=String[],
    hessian_weeks::Integer=3,
    timeout_seconds=nothing,
)
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
    hessian_weeks >= 0 ||
        throw(ArgumentError("hessian_weeks must be nonnegative"))
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
    normalized_banned_first_pick_teams =
        _normalize_survivor_team_abbreviations(
            banned_first_pick_teams,
            "banned_first_pick_teams",
        )
    return SurvivorSelectionConfig(
        normalized_spread,
        missing_market_policy,
        Int(market_guard_weeks),
        Int(through_week),
        normalized_banned_first_pick_teams,
        Int(hessian_weeks),
        normalized_timeout,
    )
end

function _survivor_debug_logging_enabled()
    # Mirror @debug's module lookup so JULIA_DEBUG=SurvivorModel is honored.
    logger = Base.CoreLogging.current_logger_for_env(
        Logging.Debug,
        :none,
        @__MODULE__,
    )
    return logger !== nothing && Logging.shouldlog(
        logger,
        Logging.Debug,
        @__MODULE__,
        :none,
        :survivor_milp_progress,
    )
end

function _survivor_optimizer(
    config::SurvivorSelectionConfig,
    optimizer;
    debug_logging::Bool=false,
)
    optimizer !== nothing && return optimizer
    attributes = Pair{String,Any}[
        "parallel" => "on",
    ]
    config.timeout_seconds === nothing ||
        push!(attributes, "time_limit" => config.timeout_seconds)
    if debug_logging
        # Keep HiGHS' native console output away from stdout, which holds the pick.
        append!(
            attributes,
            Pair{String,Any}[
                "output_flag" => true,
                "log_to_console" => false,
                "log_file" => Sys.iswindows() ? "CON" : "/dev/stderr",
                "mip_report_level" => 2,
            ],
        )
    end
    return optimizer_with_attributes(HiGHS.Optimizer, attributes...)
end

function _survivor_milp_model(
    config::SurvivorSelectionConfig,
    optimizer,
)
    debug_logging = optimizer === nothing && _survivor_debug_logging_enabled()
    model = Model(
        _survivor_optimizer(
            config,
            optimizer;
            debug_logging=debug_logging,
        ),
    )
    debug_logging || set_silent(model)
    return model
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
contains one row per planned week and `current_pick` contains the current
week's single selected row. The selections include plan-specific survival and
elimination probabilities, the posterior-mean survival probability, its
parameter-variance adjustment, and the adjusted survival estimate.
`selection_config` records the eligibility policies.
"""
struct SurvivorPoolPlan
    state::SurvivorPoolState
    selections::DataFrame
    current_pick::DataFrame
    objective_value::Float64
    selection_config::SurvivorSelectionConfig
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
    ;
    banned_first_pick_teams::AbstractVector{<:AbstractString}=String[],
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
    normalized_banned_first_pick_teams =
        _normalize_survivor_team_abbreviations(
            banned_first_pick_teams,
            "banned_first_pick_teams",
        )
    if !isempty(normalized_banned_first_pick_teams)
        current_week_has_candidates = any(
            ==(state.current_week),
            data.week,
        )
        banned_teams = Set(normalized_banned_first_pick_teams)
        data = data[
            [
                week != state.current_week || !(team in banned_teams)
                for (week, team) in zip(data.week, data.team)
            ],
            :,
        ]
        current_week_has_candidates &&
                !any(==(state.current_week), data.week) &&
                throw(ArgumentError(
                    "no eligible survivor candidates remain for week " *
                    "$(state.current_week) after applying first-pick bans",
                ))
    end

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

function _survivor_gradient_reference_indices(
    candidate_positions::AbstractVector{<:Integer},
    number_of_weeks::Integer,
    covariance_gradient_gram::Union{Nothing,AbstractMatrix}=nothing,
    ;
    maximum_reference_position::Integer=number_of_weeks,
)
    all(
        position -> 1 <= position <= number_of_weeks,
        candidate_positions,
    ) || throw(ArgumentError(
        "survivor candidate positions must be in the horizon",
    ))
    0 <= maximum_reference_position <= number_of_weeks ||
        throw(ArgumentError(
            "survivor maximum gradient reference position must be in the horizon",
        ))
    n_candidates = length(candidate_positions)
    if covariance_gradient_gram !== nothing
        size(covariance_gradient_gram) == (n_candidates, n_candidates) ||
            throw(ArgumentError(
                "survivor covariance gradient Gram dimensions must match candidates",
            ))
        all(isfinite, covariance_gradient_gram) ||
            throw(ArgumentError(
                "survivor covariance gradient Gram entries must be finite",
            ))
    end
    references = Vector{Vector{Int}}(undef, maximum_reference_position)
    for position in 1:maximum_reference_position
        suffix = findall(
            index ->
                position <= candidate_positions[index] <= maximum_reference_position,
            eachindex(candidate_positions),
        )
        if covariance_gradient_gram === nothing
            references[position] = suffix
            continue
        end
        prior_candidates = findall(<(position), candidate_positions)
        references[position] = [
            reference for reference in suffix
            if any(
                !iszero(covariance_gradient_gram[index, reference])
                for index in prior_candidates
            )
        ]
    end
    return references
end

function _survivor_gradient_switch_position(
    gradient_reference_indices::AbstractVector,
    number_of_parameters::Integer,
)
    number_of_parameters >= 0 ||
        throw(ArgumentError(
            "survivor parameter count must be nonnegative",
        ))
    for position in eachindex(gradient_reference_indices)
        all(
            length(gradient_reference_indices[suffix]) <= number_of_parameters
            for suffix in position:lastindex(gradient_reference_indices)
        ) && return position
    end
    return length(gradient_reference_indices) + 1
end

function _survivor_parameter_gradient_contraction(
    gradient_state::AbstractMatrix,
    loss_state::Integer,
    reference::Integer,
    inputs::SurvivorObjectiveInputs,
)
    return sum(
        gradient_state[loss_state, parameter] *
        inputs.parameters.variance[parameter] *
        inputs.derivatives[reference].gradient[parameter]
        for parameter in eachindex(inputs.parameters.variance);
        init=0.0,
    )
end

function _survivor_greedy_selected_indices(
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
    inputs::SurvivorObjectiveInputs,
)
    length(inputs.derivatives) == nrow(data) ||
        throw(ArgumentError("survivor derivative inputs must match candidates"))
    candidate_positions = [
        Int(data.week[index]) - state.current_week + 1
        for index in 1:nrow(data)
    ]
    number_of_weeks = config.through_week - state.current_week + 1
    week_indices = [
        findall(==(position), candidate_positions)
        for position in 1:number_of_weeks
    ]
    market_eligible = _survivor_market_guard_mask(data, state, config)
    used_teams = Set(values(state.picks_made))
    selected_by_position = zeros(Int, number_of_weeks)
    for position in 1:number_of_weeks
        eligible_indices = [
            index for index in week_indices[position]
            if market_eligible[index] && !(String(data.team[index]) in used_teams)
        ]
        isempty(eligible_indices) &&
            throw(ArgumentError(
                "survivor greedy warm start has no eligible candidate for " *
                "week $(state.current_week + position - 1)",
            ))
        best_index = eligible_indices[1]
        best_probability = inputs.derivatives[best_index].base_probability
        for index in eligible_indices[2:end]
            probability = inputs.derivatives[index].base_probability
            probability > best_probability || continue
            best_index = index
            best_probability = probability
        end
        selected_by_position[position] = best_index
        push!(used_teams, String(data.team[best_index]))
    end
    return selected_by_position
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

function _survivor_optional_result_attribute(getter::Function)
    try
        return getter()
    catch error
        error isa JuMP.MOI.UnsupportedAttribute && return nothing
        rethrow()
    end
end

function _survivor_log_milp_result(
    model,
    phase::Symbol,
    elapsed_seconds::Real,
)
    has_values = JuMP.has_values(model)
    incumbent_objective = has_values ?
        Float64(JuMP.objective_value(model)) :
        nothing
    objective_bound = _survivor_optional_result_attribute(
        () -> Float64(JuMP.objective_bound(model)),
    )
    relative_gap = _survivor_optional_result_attribute(
        () -> Float64(JuMP.relative_gap(model)),
    )
    node_count = _survivor_optional_result_attribute(
        () -> Int(JuMP.node_count(model)),
    )
    @debug "survivor MILP solve complete" phase=phase elapsed_seconds=Float64(elapsed_seconds) termination_status=JuMP.termination_status(model) primal_status=JuMP.primal_status(model) result_count=JuMP.result_count(model) has_values=has_values incumbent_objective=incumbent_objective objective_bound=objective_bound relative_gap=relative_gap node_count=node_count
    return (
        phase=phase,
        elapsed_seconds=Float64(elapsed_seconds),
        termination_status=JuMP.termination_status(model),
        primal_status=JuMP.primal_status(model),
        result_count=JuMP.result_count(model),
        has_values,
        incumbent_objective,
        objective_bound,
        relative_gap,
        node_count,
    )
end

function _survivor_constant_plan(
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig;
    optimizer=nothing,
)
    model = _survivor_milp_model(config, optimizer)
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
    solve_started_at = time_ns()
    optimize!(model)
    _survivor_log_milp_result(
        model,
        :constant_plan,
        (time_ns() - solve_started_at) / 1.0e9,
    )
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

function _survivor_add_one_hot_dummies!(
    model,
    aggregate,
    recurrences,
    selected,
    candidate_lower,
    candidate_upper,
    candidate_indices,
    ;
    dummy_start_values=nothing,
    recurrence_lower=candidate_lower,
    recurrence_upper=candidate_upper,
)
    isempty(candidate_indices) &&
        throw(ArgumentError("survivor one-hot dummies require candidates"))
    dummies = Dict{Int,JuMP.AffExpr}()
    for index in candidate_indices
        haskey(recurrences, index) ||
            throw(ArgumentError(
                "survivor one-hot dummy recurrence is missing a candidate",
            ))
        lower = Float64(candidate_lower[index])
        upper = Float64(candidate_upper[index])
        lower <= upper && isfinite(lower) && isfinite(upper) ||
            throw(ArgumentError(
                "survivor selected-branch dummy bounds must be finite and ordered",
            ))
        all_lower = Float64(recurrence_lower[index])
        all_upper = Float64(recurrence_upper[index])
        all_lower <= all_upper &&
                isfinite(all_lower) && isfinite(all_upper) ||
            throw(ArgumentError(
                "survivor all-history recurrence bounds must be finite and ordered",
            ))
        all_lower <= lower <= upper <= all_upper ||
            throw(ArgumentError(
                "survivor selected-branch bounds must lie within all-history bounds",
            ))
        start_value = if dummy_start_values === nothing
            nothing
        else
            haskey(dummy_start_values, index) ||
                throw(ArgumentError(
                    "survivor one-hot dummy start is missing a candidate",
                ))
            value = Float64(dummy_start_values[index])
            isfinite(value) ||
                throw(ArgumentError(
                    "survivor one-hot dummy starts must be finite",
                ))
            value
        end
        if lower == upper
            dummy = iszero(lower) ?
                JuMP.AffExpr(0.0) :
                lower * selected[index]
            if all_lower != lower || all_upper != upper
                recurrence = recurrences[index]
                @constraint(
                    model,
                    dummy >= recurrence - all_upper * (1.0 - selected[index]),
                )
                @constraint(
                    model,
                    dummy <= recurrence - all_lower * (1.0 - selected[index]),
                )
            end
            dummies[index] = dummy
            continue
        end
        dummy = @variable(model)
        set_lower_bound(dummy, min(0.0, lower))
        set_upper_bound(dummy, max(0.0, upper))
        recurrence = recurrences[index]
        if !iszero(lower)
            @constraint(
                model,
                dummy >= lower * selected[index],
            )
        end
        if !iszero(upper)
            @constraint(
                model,
                dummy <= upper * selected[index],
            )
        end
        @constraint(
            model,
            dummy >= recurrence - all_upper * (1.0 - selected[index]),
        )
        @constraint(
            model,
            dummy <= recurrence - all_lower * (1.0 - selected[index]),
        )
        if start_value !== nothing
            set_start_value(dummy, start_value)
        end
        dummies[index] = 1.0 * dummy
    end
    @constraint(
        model,
        aggregate == sum(dummies[index] for index in candidate_indices),
    )
    return dummies
end

function _survivor_scalar_bounds_pass(
    inputs::SurvivorObjectiveInputs,
    candidate_positions::AbstractVector{<:Integer},
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
    ;
    curvature_weeks::Integer=number_of_weeks,
    gradient_reference_indices=nothing,
    candidate_teams=nothing,
    excluded_team=nothing,
)
    n_candidates = length(inputs.derivatives)
    length(candidate_positions) == n_candidates ||
        throw(ArgumentError(
            "survivor candidate positions must match derivative inputs",
        ))
    candidate_teams === nothing || length(candidate_teams) == n_candidates ||
        throw(ArgumentError(
            "survivor candidate teams must match derivative inputs",
        ))
    excluded_team === nothing || candidate_teams !== nothing ||
        throw(ArgumentError(
            "survivor leave-out bounds require candidate teams",
        ))
    all(
        position -> 1 <= position <= number_of_weeks,
        candidate_positions,
    ) ||
        throw(ArgumentError("survivor candidate positions must be in the horizon"))
    0 <= curvature_weeks <= number_of_weeks ||
        throw(ArgumentError(
            "survivor curvature weeks must be within the optimization horizon",
        ))
    week_indices = [
        findall(==(position), candidate_positions)
        for position in 1:number_of_weeks
    ]
    all(!isempty, week_indices) ||
        throw(ArgumentError(
            "survivor derivative bounds require candidates in every week",
        ))
    history_week_indices = if excluded_team === nothing
        week_indices
    else
        [
            filter(
                index -> !isequal(candidate_teams[index], excluded_team),
                week_indices[position],
            )
            for position in 1:number_of_weeks
        ]
    end
    history_reachable = falses(number_of_weeks + 1)
    history_reachable[1] = true
    for position in 1:number_of_weeks
        history_reachable[position + 1] =
            history_reachable[position] &&
            !isempty(history_week_indices[position])
    end
    gradient_reference_indices = isnothing(gradient_reference_indices) ?
        [collect(1:n_candidates) for _ in 1:curvature_weeks] :
        gradient_reference_indices
    length(gradient_reference_indices) == curvature_weeks ||
        throw(ArgumentError(
            "survivor gradient reference positions must match curvature weeks",
        ))

    probability_lower =
        zeros(Float64, number_of_weeks + 1, losses_to_elimination)
    probability_upper =
        zeros(Float64, number_of_weeks + 1, losses_to_elimination)
    gradient_lower = zeros(
        Float64,
        curvature_weeks + 1,
        losses_to_elimination,
        n_candidates,
    )
    gradient_upper = zeros(
        Float64,
        curvature_weeks + 1,
        losses_to_elimination,
        n_candidates,
    )
    n_parameters = length(inputs.parameters.keys)
    parameter_gradient_lower = zeros(
        Float64,
        curvature_weeks + 1,
        losses_to_elimination,
        n_parameters,
    )
    parameter_gradient_upper = zeros(
        Float64,
        curvature_weeks + 1,
        losses_to_elimination,
        n_parameters,
    )
    hessian_lower =
        zeros(Float64, curvature_weeks + 1, losses_to_elimination)
    hessian_upper =
        zeros(Float64, curvature_weeks + 1, losses_to_elimination)

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
    candidate_parameter_gradient_lower = zeros(
        Float64,
        n_candidates,
        losses_to_elimination,
        n_parameters,
    )
    candidate_parameter_gradient_upper = zeros(
        Float64,
        n_candidates,
        losses_to_elimination,
        n_parameters,
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
                    candidate_probability_upper[index, loss_state] = lower, upper

                probability_difference_lower, probability_difference_upper =
                    _survivor_interval_difference(
                        current_lower,
                        current_upper,
                        previous_lower,
                        previous_upper,
                    )
                if position < curvature_weeks
                    for parameter in 1:n_parameters
                        current_gradient_lower =
                            parameter_gradient_lower[
                                position,
                                loss_state,
                                parameter,
                            ]
                        current_gradient_upper =
                            parameter_gradient_upper[
                                position,
                                loss_state,
                                parameter,
                            ]
                        lower, upper = _survivor_interval_product(
                            current_gradient_lower,
                            current_gradient_upper,
                            probability,
                        )
                        previous_gradient_lower, previous_gradient_upper =
                            loss_state == 1 ?
                            (0.0, 0.0) :
                            (
                                parameter_gradient_lower[
                                    position,
                                    loss_state - 1,
                                    parameter,
                                ],
                                parameter_gradient_upper[
                                    position,
                                    loss_state - 1,
                                    parameter,
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
                        source_lower, source_upper =
                            _survivor_interval_product(
                                probability_difference_lower,
                                probability_difference_upper,
                                derivative.gradient[parameter],
                            )
                        lower += source_lower
                        upper += source_upper
                        candidate_parameter_gradient_lower[
                            index,
                            loss_state,
                            parameter,
                        ],
                        candidate_parameter_gradient_upper[
                            index,
                            loss_state,
                            parameter,
                        ] = lower, upper
                    end
                    for reference in gradient_reference_indices[position + 1]
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
                            lower, upper
                    end
                end

                if position <= curvature_weeks
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
                        lower, upper
                end
            end
        end

        for loss_state in 1:losses_to_elimination
            eligible_indices = history_week_indices[position]
            isempty(eligible_indices) && continue
            probability_lower[position + 1, loss_state],
                probability_upper[position + 1, loss_state] =
                minimum(
                    candidate_probability_lower[index, loss_state]
                    for index in eligible_indices
                ),
                maximum(
                    candidate_probability_upper[index, loss_state]
                    for index in eligible_indices
                )
            if position < curvature_weeks
                for reference in gradient_reference_indices[position + 1]
                    gradient_lower[position + 1, loss_state, reference],
                        gradient_upper[position + 1, loss_state, reference] =
                        minimum(
                            candidate_gradient_lower[
                                index,
                                loss_state,
                                reference,
                            ]
                            for index in eligible_indices
                        ),
                        maximum(
                            candidate_gradient_upper[
                                index,
                                loss_state,
                                reference,
                            ]
                            for index in eligible_indices
                        )
                end
                for parameter in 1:n_parameters
                    parameter_gradient_lower[
                        position + 1,
                        loss_state,
                        parameter,
                    ],
                    parameter_gradient_upper[
                        position + 1,
                        loss_state,
                        parameter,
                    ] =
                    minimum(
                        candidate_parameter_gradient_lower[
                            index,
                            loss_state,
                            parameter,
                        ]
                        for index in eligible_indices
                    ),
                    maximum(
                        candidate_parameter_gradient_upper[
                            index,
                            loss_state,
                            parameter,
                        ]
                        for index in eligible_indices
                    )
                end
            end
            if position <= curvature_weeks
                hessian_lower[position + 1, loss_state],
                    hessian_upper[position + 1, loss_state] =
                    minimum(
                        candidate_hessian_lower[index, loss_state]
                        for index in eligible_indices
                    ),
                    maximum(
                        candidate_hessian_upper[index, loss_state]
                        for index in eligible_indices
                    )
            end
        end
    end

    all(isfinite, probability_lower) &&
        all(isfinite, probability_upper) &&
        all(isfinite, gradient_lower) &&
        all(isfinite, gradient_upper) &&
        all(isfinite, parameter_gradient_lower) &&
        all(isfinite, parameter_gradient_upper) &&
        all(isfinite, hessian_lower) &&
        all(isfinite, hessian_upper) ||
        throw(ArgumentError("survivor scalar recurrence bounds are not finite"))
    return (
        probability=(lower=probability_lower, upper=probability_upper),
        gradient=(lower=gradient_lower, upper=gradient_upper),
        parameter_gradient=(
            lower=parameter_gradient_lower,
            upper=parameter_gradient_upper,
        ),
        hessian=(lower=hessian_lower, upper=hessian_upper),
        candidate_probability=(
            lower=candidate_probability_lower,
            upper=candidate_probability_upper,
        ),
        candidate_gradient=(
            lower=candidate_gradient_lower,
            upper=candidate_gradient_upper,
        ),
        candidate_parameter_gradient=(
            lower=candidate_parameter_gradient_lower,
            upper=candidate_parameter_gradient_upper,
        ),
        candidate_hessian=(
            lower=candidate_hessian_lower,
            upper=candidate_hessian_upper,
        ),
        history_reachable=history_reachable,
    )
end

function _survivor_bound_intersection(
    all_history_lower::Float64,
    all_history_upper::Float64,
    conditioned_lowers,
    conditioned_uppers,
)
    lower = max(all_history_lower, minimum(conditioned_lowers))
    upper = min(all_history_upper, maximum(conditioned_uppers))
    lower <= upper ||
        throw(ArgumentError(
            "team-leave-out survivor bounds do not intersect all-history bounds",
        ))
    return lower, upper
end

function _survivor_scalar_bounds(
    inputs::SurvivorObjectiveInputs,
    candidate_positions::AbstractVector{<:Integer},
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
    ;
    curvature_weeks::Integer=number_of_weeks,
    gradient_reference_indices=nothing,
    candidate_teams=nothing,
)
    n_candidates = length(inputs.derivatives)
    candidate_teams === nothing || length(candidate_teams) == n_candidates ||
        throw(ArgumentError(
            "survivor candidate teams must match derivative inputs",
        ))
    candidate_teams === nothing || !any(ismissing, candidate_teams) ||
        throw(ArgumentError("survivor candidate teams cannot be missing"))
    references = isnothing(gradient_reference_indices) ?
        [collect(1:n_candidates) for _ in 1:curvature_weeks] :
        gradient_reference_indices
    all_history = _survivor_scalar_bounds_pass(
        inputs,
        candidate_positions,
        number_of_weeks,
        losses_to_elimination;
        curvature_weeks=curvature_weeks,
        gradient_reference_indices=references,
        candidate_teams=candidate_teams,
    )
    candidate_teams === nothing && return all_history

    selected_candidate_probability_lower =
        copy(all_history.candidate_probability.lower)
    selected_candidate_probability_upper =
        copy(all_history.candidate_probability.upper)
    selected_candidate_gradient_lower =
        copy(all_history.candidate_gradient.lower)
    selected_candidate_gradient_upper =
        copy(all_history.candidate_gradient.upper)
    selected_candidate_parameter_gradient_lower =
        copy(all_history.candidate_parameter_gradient.lower)
    selected_candidate_parameter_gradient_upper =
        copy(all_history.candidate_parameter_gradient.upper)
    selected_candidate_hessian_lower =
        copy(all_history.candidate_hessian.lower)
    selected_candidate_hessian_upper =
        copy(all_history.candidate_hessian.upper)

    for team in unique(candidate_teams)
        leave_out_bounds = _survivor_scalar_bounds_pass(
            inputs,
            candidate_positions,
            number_of_weeks,
            losses_to_elimination;
            curvature_weeks=curvature_weeks,
            gradient_reference_indices=references,
            candidate_teams=candidate_teams,
            excluded_team=team,
        )
        for index in 1:n_candidates
            isequal(candidate_teams[index], team) || continue
            position = Int(candidate_positions[index])
            leave_out_bounds.history_reachable[position] || continue
            selected_candidate_probability_lower[index, :] .=
                leave_out_bounds.candidate_probability.lower[index, :]
            selected_candidate_probability_upper[index, :] .=
                leave_out_bounds.candidate_probability.upper[index, :]
            selected_candidate_gradient_lower[index, :, :] .=
                leave_out_bounds.candidate_gradient.lower[index, :, :]
            selected_candidate_gradient_upper[index, :, :] .=
                leave_out_bounds.candidate_gradient.upper[index, :, :]
            selected_candidate_parameter_gradient_lower[index, :, :] .=
                leave_out_bounds.candidate_parameter_gradient.lower[
                    index,
                    :,
                    :,
                ]
            selected_candidate_parameter_gradient_upper[index, :, :] .=
                leave_out_bounds.candidate_parameter_gradient.upper[
                    index,
                    :,
                    :,
                ]
            selected_candidate_hessian_lower[index, :] .=
                leave_out_bounds.candidate_hessian.lower[index, :]
            selected_candidate_hessian_upper[index, :] .=
                leave_out_bounds.candidate_hessian.upper[index, :]
        end
    end

    week_indices = [
        findall(==(position), candidate_positions)
        for position in 1:number_of_weeks
    ]
    probability_lower = copy(all_history.probability.lower)
    probability_upper = copy(all_history.probability.upper)
    for position in 1:number_of_weeks, loss_state in 1:losses_to_elimination
        probability_lower[position + 1, loss_state],
            probability_upper[position + 1, loss_state] =
            _survivor_bound_intersection(
                all_history.probability.lower[position + 1, loss_state],
                all_history.probability.upper[position + 1, loss_state],
                (
                    selected_candidate_probability_lower[index, loss_state]
                    for index in week_indices[position]
                ),
                (
                    selected_candidate_probability_upper[index, loss_state]
                    for index in week_indices[position]
                ),
            )
    end

    parameter_gradient_lower = copy(all_history.parameter_gradient.lower)
    parameter_gradient_upper = copy(all_history.parameter_gradient.upper)
    for position in 1:max(0, curvature_weeks - 1),
        loss_state in 1:losses_to_elimination,
        parameter in eachindex(inputs.parameters.keys)
        parameter_gradient_lower[position + 1, loss_state, parameter],
            parameter_gradient_upper[position + 1, loss_state, parameter] =
            _survivor_bound_intersection(
                all_history.parameter_gradient.lower[
                    position + 1,
                    loss_state,
                    parameter,
                ],
                all_history.parameter_gradient.upper[
                    position + 1,
                    loss_state,
                    parameter,
                ],
                (
                    selected_candidate_parameter_gradient_lower[
                        index,
                        loss_state,
                        parameter,
                    ]
                    for index in week_indices[position]
                ),
                (
                    selected_candidate_parameter_gradient_upper[
                        index,
                        loss_state,
                        parameter,
                    ]
                    for index in week_indices[position]
                ),
            )
    end

    gradient_lower = copy(all_history.gradient.lower)
    gradient_upper = copy(all_history.gradient.upper)
    for position in 1:max(0, curvature_weeks - 1),
        loss_state in 1:losses_to_elimination,
        reference in references[position + 1]
        gradient_lower[position + 1, loss_state, reference],
            gradient_upper[position + 1, loss_state, reference] =
            _survivor_bound_intersection(
                all_history.gradient.lower[
                    position + 1,
                    loss_state,
                    reference,
                ],
                all_history.gradient.upper[
                    position + 1,
                    loss_state,
                    reference,
                ],
                (
                    selected_candidate_gradient_lower[
                        index,
                        loss_state,
                        reference,
                    ]
                    for index in week_indices[position]
                ),
                (
                    selected_candidate_gradient_upper[
                        index,
                        loss_state,
                        reference,
                    ]
                    for index in week_indices[position]
                ),
            )
    end

    hessian_lower = copy(all_history.hessian.lower)
    hessian_upper = copy(all_history.hessian.upper)
    for position in 1:curvature_weeks,
        loss_state in 1:losses_to_elimination
        hessian_lower[position + 1, loss_state],
            hessian_upper[position + 1, loss_state] =
            _survivor_bound_intersection(
                all_history.hessian.lower[position + 1, loss_state],
                all_history.hessian.upper[position + 1, loss_state],
                (
                    selected_candidate_hessian_lower[index, loss_state]
                    for index in week_indices[position]
                ),
                (
                    selected_candidate_hessian_upper[index, loss_state]
                    for index in week_indices[position]
                ),
            )
    end

    return merge(
        all_history,
        (
            probability=(lower=probability_lower, upper=probability_upper),
            parameter_gradient=(
                lower=parameter_gradient_lower,
                upper=parameter_gradient_upper,
            ),
            gradient=(lower=gradient_lower, upper=gradient_upper),
            hessian=(lower=hessian_lower, upper=hessian_upper),
            team_conditioned=(
                candidate_probability=(
                    lower=selected_candidate_probability_lower,
                    upper=selected_candidate_probability_upper,
                ),
                candidate_parameter_gradient=(
                    lower=selected_candidate_parameter_gradient_lower,
                    upper=selected_candidate_parameter_gradient_upper,
                ),
                candidate_gradient=(
                    lower=selected_candidate_gradient_lower,
                    upper=selected_candidate_gradient_upper,
                ),
                candidate_hessian=(
                    lower=selected_candidate_hessian_lower,
                    upper=selected_candidate_hessian_upper,
                ),
            ),
        ),
    )
end

function _survivor_scalar_forward_values(
    selected_by_position::AbstractVector{<:Integer},
    inputs::SurvivorObjectiveInputs,
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
    ;
    curvature_weeks::Integer=number_of_weeks,
    gradient_reference_indices=nothing,
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
    0 <= curvature_weeks <= number_of_weeks ||
        throw(ArgumentError(
            "survivor curvature weeks must be within the optimization horizon",
        ))
    gradient_reference_indices = isnothing(gradient_reference_indices) ?
        [collect(1:n_candidates) for _ in 1:curvature_weeks] :
        gradient_reference_indices
    length(gradient_reference_indices) == curvature_weeks ||
        throw(ArgumentError(
            "survivor gradient reference positions must match curvature weeks",
        ))
    probability = zeros(
        Float64,
        number_of_weeks + 1,
        losses_to_elimination,
    )
    gradient = zeros(
        Float64,
        curvature_weeks + 1,
        losses_to_elimination,
        n_candidates,
    )
    n_parameters = length(inputs.parameters.keys)
    parameter_gradient = zeros(
        Float64,
        curvature_weeks + 1,
        losses_to_elimination,
        n_parameters,
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
            if position < curvature_weeks
                probability_difference =
                    probability[position, loss_state] - previous_probability
                for parameter in 1:n_parameters
                    previous_parameter_gradient = loss_state == 1 ?
                        0.0 :
                        parameter_gradient[
                            position,
                            loss_state - 1,
                            parameter,
                        ]
                    parameter_gradient[
                        position + 1,
                        loss_state,
                        parameter,
                    ] =
                        win_probability *
                        parameter_gradient[
                            position,
                            loss_state,
                            parameter,
                        ] +
                        (1.0 - win_probability) *
                        previous_parameter_gradient +
                        derivative.gradient[parameter] *
                        probability_difference
                end
                for reference in gradient_reference_indices[position + 1]
                    current_reference = findfirst(
                        ==(reference),
                        gradient_reference_indices[position],
                    )
                    current_gradient = current_reference === nothing ?
                        0.0 :
                        gradient[position, loss_state, reference]
                    previous_gradient = loss_state == 1 ||
                            current_reference === nothing ?
                        0.0 :
                        gradient[position, loss_state - 1, reference]
                    gradient[position + 1, loss_state, reference] =
                        win_probability * current_gradient +
                        (1.0 - win_probability) * previous_gradient +
                        inputs.covariance_gradient_gram[index, reference] *
                        (
                            probability[position, loss_state] -
                            previous_probability
                        )
                end
            end
            if position <= curvature_weeks
                previous_hessian = loss_state == 1 ?
                    0.0 :
                    hessian[position, loss_state - 1]
                current_gradient_reference = findfirst(
                    ==(index),
                    gradient_reference_indices[position],
                )
                current_gradient = current_gradient_reference === nothing ?
                    0.0 :
                    gradient[position, loss_state, index]
                previous_gradient_for_selected =
                    loss_state == 1 || current_gradient_reference === nothing ?
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
                        current_gradient -
                        previous_gradient_for_selected
                    )
            end
        end
    end
    return (
        probability=probability,
        gradient=gradient,
        parameter_gradient=parameter_gradient,
        hessian=hessian,
    )
end

function _survivor_scalar_warm_start(
    data::AbstractDataFrame,
    state::SurvivorPoolState,
    config::SurvivorSelectionConfig,
    inputs::SurvivorObjectiveInputs,
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
    curvature_weeks::Integer,
    gradient_reference_indices,
)
    selected_by_position = _survivor_greedy_selected_indices(
        data,
        state,
        config,
        inputs,
    )
    values = _survivor_scalar_forward_values(
        selected_by_position,
        inputs,
        number_of_weeks,
        losses_to_elimination,
        curvature_weeks=curvature_weeks,
        gradient_reference_indices=gradient_reference_indices,
    )
    return merge(values, (selected=selected_by_position,))
end

function _set_survivor_scalar_warm_start!(
    selected,
    probability,
    parameter_gradient,
    gradient,
    hessian,
    warm_start,
    candidate_indices,
    gradient_reference_indices,
    gradient_switch_position::Integer,
    number_of_weeks::Integer,
    losses_to_elimination::Integer,
    curvature_weeks::Integer,
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
    end
    for position in 1:(curvature_weeks + 1), loss_state in 1:losses_to_elimination
        set_start_value(
            hessian[position, loss_state],
            warm_start.hessian[position, loss_state],
        )
    end
    for position in eachindex(parameter_gradient),
        loss_state in 1:losses_to_elimination,
        parameter in axes(parameter_gradient[position], 2)
        set_start_value(
            parameter_gradient[position][loss_state, parameter],
            warm_start.parameter_gradient[position, loss_state, parameter],
        )
    end
    for position in gradient_switch_position:curvature_weeks,
        loss_state in 1:losses_to_elimination
        gradient_state = gradient[position]
        gradient_state === nothing && continue
        for (slot, reference) in enumerate(gradient_reference_indices[position])
            set_start_value(
                gradient_state[loss_state, slot],
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
    inputs::SurvivorObjectiveInputs;
    optimizer=nothing,
)
    number_of_weeks = config.through_week - state.current_week + 1
    curvature_weeks = min(config.hessian_weeks, number_of_weeks)
    0 <= curvature_weeks <= number_of_weeks ||
        throw(ArgumentError(
            "survivor curvature weeks must be within the optimization horizon",
        ))
    losses_to_elimination = _survivor_loss_threshold(state)
    losses_to_elimination > number_of_weeks &&
        return _survivor_constant_plan(
            data,
            state,
            config;
            optimizer=optimizer,
        )
    length(inputs.derivatives) == nrow(data) ||
        throw(ArgumentError("survivor derivative inputs must match candidates"))

    model = _survivor_milp_model(config, optimizer)
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
    # Track parameter gradients before the suffix-safe switch, then only the
    # candidate contractions needed for reference weeks w′ >= w.
    gradient_reference_indices = _survivor_gradient_reference_indices(
        candidate_positions,
        number_of_weeks,
        inputs.covariance_gradient_gram,
        maximum_reference_position=curvature_weeks,
    )
    gradient_reference_slots = [
        Dict(reference => slot for (slot, reference) in enumerate(references))
        for references in gradient_reference_indices
    ]
    parameter_count = length(inputs.parameters.keys)
    gradient_switch_position = _survivor_gradient_switch_position(
        gradient_reference_indices,
        parameter_count,
    )
    parameter_state_count = min(
        curvature_weeks,
        gradient_switch_position - 1,
    )
    bounds = _survivor_scalar_bounds(
        inputs,
        candidate_positions,
        number_of_weeks,
        losses_to_elimination,
        curvature_weeks=curvature_weeks,
        gradient_reference_indices=gradient_reference_indices,
        candidate_teams=data.team,
    )
    candidate_probability_sum_lower = [
        sum(
            bounds.candidate_probability.lower[index, loss_state]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    candidate_probability_sum_upper = [
        sum(
            bounds.candidate_probability.upper[index, loss_state]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    team_candidate_probability_sum_lower = [
        sum(
            bounds.team_conditioned.candidate_probability.lower[
                index,
                loss_state,
            ]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    team_candidate_probability_sum_upper = [
        sum(
            bounds.team_conditioned.candidate_probability.upper[
                index,
                loss_state,
            ]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    candidate_adjusted_sum_lower = [
        candidate_probability_sum_lower[index] +
        0.5 * sum(
            bounds.candidate_hessian.lower[index, loss_state]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    team_candidate_adjusted_sum_lower = [
        team_candidate_probability_sum_lower[index] +
        0.5 * sum(
            bounds.team_conditioned.candidate_hessian.lower[
                index,
                loss_state,
            ]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    team_candidate_adjusted_sum_upper = [
        team_candidate_probability_sum_upper[index] +
        0.5 * sum(
            bounds.team_conditioned.candidate_hessian.upper[
                index,
                loss_state,
            ]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    candidate_adjusted_sum_upper = [
        candidate_probability_sum_upper[index] +
        0.5 * sum(
            bounds.candidate_hessian.upper[index, loss_state]
            for loss_state in state_indices
        )
        for index in candidate_indices
    ]
    warm_start = _survivor_scalar_warm_start(
        data,
        state,
        config,
        inputs,
        number_of_weeks,
        losses_to_elimination,
        curvature_weeks,
        gradient_reference_indices,
    )
    warm_start_selected = Set(warm_start.selected)
    @variable(
        model,
        probability[1:(number_of_weeks + 1), state_indices],
    )
    parameter_gradient =
        Vector{Matrix{JuMP.VariableRef}}(undef, parameter_state_count)
    for position in 1:parameter_state_count
        parameter_gradient[position] = @variable(
            model,
            [state_indices, 1:parameter_count],
        )
    end
    gradient =
        Vector{Union{Nothing,Matrix{JuMP.VariableRef}}}(undef, curvature_weeks)
    fill!(gradient, nothing)
    for position in gradient_switch_position:curvature_weeks
        isempty(gradient_reference_indices[position]) && continue
        gradient[position] = @variable(
            model,
            [state_indices, 1:length(gradient_reference_indices[position])],
        )
    end
    @variable(
        model,
        hessian[1:(curvature_weeks + 1), state_indices],
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
        if position <= parameter_state_count
            for parameter in 1:parameter_count
                set_lower_bound(
                    parameter_gradient[position][loss_state, parameter],
                    bounds.parameter_gradient.lower[
                        position,
                        loss_state,
                        parameter,
                    ],
                )
                set_upper_bound(
                    parameter_gradient[position][loss_state, parameter],
                    bounds.parameter_gradient.upper[
                        position,
                        loss_state,
                        parameter,
                    ],
                )
            end
        end
        if position <= curvature_weeks + 1
            set_lower_bound(
                hessian[position, loss_state],
                bounds.hessian.lower[position, loss_state],
            )
            set_upper_bound(
                hessian[position, loss_state],
                bounds.hessian.upper[position, loss_state],
            )
        end
        if position <= curvature_weeks && gradient[position] !== nothing
            for (slot, reference) in enumerate(gradient_reference_indices[position])
                gradient_state = gradient[position]::Matrix{JuMP.VariableRef}
                set_lower_bound(
                    gradient_state[loss_state, slot],
                    bounds.gradient.lower[position, loss_state, reference],
                )
                set_upper_bound(
                    gradient_state[loss_state, slot],
                    bounds.gradient.upper[position, loss_state, reference],
                )
            end
        end
    end

    @constraint(model, probability[1, 1] == 1.0)
    for loss_state in 2:losses_to_elimination
        @constraint(model, probability[1, loss_state] == 0.0)
    end
    for loss_state in state_indices
        if curvature_weeks >= 0
            @constraint(model, hessian[1, loss_state] == 0.0)
        end
        if parameter_state_count >= 1
            for parameter in 1:parameter_count
                @constraint(
                    model,
                    parameter_gradient[1][loss_state, parameter] == 0.0,
                )
            end
        elseif curvature_weeks >= 1 && gradient[1] !== nothing
            for slot in axes(gradient[1], 2)
                gradient_state = gradient[1]::Matrix{JuMP.VariableRef}
                @constraint(
                    model,
                    gradient_state[loss_state, slot] == 0.0,
                )
            end
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
            probability_recurrences = Dict{Int,Any}()
            probability_dummy_starts = Dict{Int,Float64}()
            for index in indices
                derivative = inputs.derivatives[index]
                probability_recurrences[index] =
                    derivative.base_probability *
                    probability[position, loss_state] +
                    (1.0 - derivative.base_probability) *
                    previous_probability
                probability_dummy_starts[index] =
                    index in warm_start_selected ?
                    derivative.base_probability *
                    warm_start.probability[position, loss_state] +
                    (1.0 - derivative.base_probability) *
                    (
                        loss_state == 1 ?
                        0.0 :
                        warm_start.probability[position, loss_state - 1]
                    ) :
                    0.0
            end
            _survivor_add_one_hot_dummies!(
                model,
                probability[position + 1, loss_state],
                probability_recurrences,
                selected,
                bounds.team_conditioned.candidate_probability.lower[
                    :,
                    loss_state,
                ],
                bounds.team_conditioned.candidate_probability.upper[
                    :,
                    loss_state,
                ],
                indices,
                dummy_start_values=probability_dummy_starts,
                recurrence_lower=
                    bounds.candidate_probability.lower[:, loss_state],
                recurrence_upper=
                    bounds.candidate_probability.upper[:, loss_state],
            )
            probability_difference =
                probability[position, loss_state] -
                previous_probability
            if position < parameter_state_count
                parameter_gradient_target = parameter_gradient[position + 1]
                for parameter in 1:parameter_count
                    parameter_gradient_recurrences = Dict{Int,Any}()
                    parameter_gradient_dummy_starts = Dict{Int,Float64}()
                    for index in indices
                        derivative = inputs.derivatives[index]
                        previous_parameter_gradient =
                            loss_state == 1 ?
                            0.0 :
                            parameter_gradient[position][
                                loss_state - 1,
                                parameter,
                            ]
                        parameter_gradient_recurrences[index] =
                            derivative.base_probability *
                            parameter_gradient[position][
                                loss_state,
                                parameter,
                            ] +
                            (1.0 - derivative.base_probability) *
                            previous_parameter_gradient +
                            derivative.gradient[parameter] *
                            probability_difference
                        parameter_gradient_dummy_starts[index] =
                            index in warm_start_selected ?
                            derivative.base_probability *
                            warm_start.parameter_gradient[
                                position,
                                loss_state,
                                parameter,
                            ] +
                            (1.0 - derivative.base_probability) *
                            (
                                loss_state == 1 ?
                                0.0 :
                                warm_start.parameter_gradient[
                                    position,
                                    loss_state - 1,
                                    parameter,
                                ]
                            ) +
                            derivative.gradient[parameter] *
                            (
                                warm_start.probability[position, loss_state] -
                                (
                                    loss_state == 1 ?
                                    0.0 :
                                    warm_start.probability[
                                        position,
                                        loss_state - 1,
                                    ]
                                )
                            ) :
                            0.0
                    end
                    _survivor_add_one_hot_dummies!(
                        model,
                        parameter_gradient_target[loss_state, parameter],
                        parameter_gradient_recurrences,
                        selected,
                        bounds.team_conditioned.candidate_parameter_gradient.lower[
                            :, loss_state, parameter
                        ],
                        bounds.team_conditioned.candidate_parameter_gradient.upper[
                            :, loss_state, parameter
                        ],
                        indices,
                        dummy_start_values=parameter_gradient_dummy_starts,
                        recurrence_lower=
                            bounds.candidate_parameter_gradient.lower[
                                :, loss_state, parameter
                            ],
                        recurrence_upper=
                            bounds.candidate_parameter_gradient.upper[
                                :, loss_state, parameter
                            ],
                    )
                end
            end
            if position < curvature_weeks &&
                    position + 1 >= gradient_switch_position
                next_references = gradient_reference_indices[position + 1]
                for reference in next_references
                    next_slot = gradient_reference_slots[position + 1][reference]
                    gradient_recurrences = Dict{Int,Any}()
                    gradient_dummy_starts = Dict{Int,Float64}()
                    current_slot = get(
                        gradient_reference_slots[position],
                        reference,
                        nothing,
                    )
                    if position < gradient_switch_position
                        current_gradient = current_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                parameter_gradient[position],
                                loss_state,
                                reference,
                                inputs,
                            )
                        warm_current_gradient = current_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                @view(warm_start.parameter_gradient[
                                    position,
                                    :,
                                    :
                                ]),
                                loss_state,
                                reference,
                                inputs,
                            )
                        previous_gradient =
                            loss_state == 1 || current_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                parameter_gradient[position],
                                loss_state - 1,
                                reference,
                                inputs,
                            )
                        warm_previous_gradient =
                            loss_state == 1 || current_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                @view(warm_start.parameter_gradient[
                                    position,
                                    :,
                                    :
                                ]),
                                loss_state - 1,
                                reference,
                                inputs,
                            )
                    else
                        current_gradient = current_slot === nothing ?
                            0.0 :
                            (gradient[position]::Matrix{JuMP.VariableRef})[
                                loss_state,
                                current_slot,
                            ]
                        warm_current_gradient = current_slot === nothing ?
                            0.0 :
                            warm_start.gradient[
                                position,
                                loss_state,
                                reference,
                            ]
                        previous_gradient =
                            loss_state == 1 || current_slot === nothing ?
                            0.0 :
                            (gradient[position]::Matrix{JuMP.VariableRef})[
                                loss_state - 1,
                                current_slot,
                            ]
                        warm_previous_gradient =
                            loss_state == 1 || current_slot === nothing ?
                            0.0 :
                            warm_start.gradient[
                                position,
                                loss_state - 1,
                                reference,
                            ]
                    end
                    for index in indices
                        derivative = inputs.derivatives[index]
                        gradient_recurrences[index] =
                            derivative.base_probability *
                            current_gradient +
                            (1.0 - derivative.base_probability) *
                            previous_gradient +
                            inputs.covariance_gradient_gram[index, reference] *
                            probability_difference
                        gradient_dummy_starts[index] =
                            index in warm_start_selected ?
                            derivative.base_probability *
                            warm_current_gradient +
                            (1.0 - derivative.base_probability) *
                            warm_previous_gradient +
                            inputs.covariance_gradient_gram[index, reference] *
                            (
                                warm_start.probability[position, loss_state] -
                                (
                                    loss_state == 1 ?
                                    0.0 :
                                    warm_start.probability[
                                        position,
                                        loss_state - 1,
                                    ]
                                )
                            ) :
                            0.0
                    end
                    target_gradient =
                        gradient[position + 1]::Matrix{JuMP.VariableRef}
                    _survivor_add_one_hot_dummies!(
                        model,
                        target_gradient[loss_state, next_slot],
                        gradient_recurrences,
                        selected,
                        bounds.team_conditioned.candidate_gradient.lower[
                            :, loss_state, reference
                        ],
                        bounds.team_conditioned.candidate_gradient.upper[
                            :, loss_state, reference
                        ],
                        indices,
                        dummy_start_values=gradient_dummy_starts,
                        recurrence_lower=
                            bounds.candidate_gradient.lower[
                                :, loss_state, reference
                            ],
                        recurrence_upper=
                            bounds.candidate_gradient.upper[
                                :, loss_state, reference
                            ],
                    )
                end
            end

            if position <= curvature_weeks
                hessian_recurrences = Dict{Int,Any}()
                hessian_dummy_starts = Dict{Int,Float64}()
                for index in indices
                    derivative = inputs.derivatives[index]
                    previous_hessian = loss_state == 1 ?
                        0.0 :
                        hessian[position, loss_state - 1]
                    current_gradient_slot = get(
                        gradient_reference_slots[position],
                        index,
                        nothing,
                    )
                    if position < gradient_switch_position
                        current_gradient = current_gradient_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                parameter_gradient[position],
                                loss_state,
                                index,
                                inputs,
                            )
                        previous_gradient_for_selected =
                            loss_state == 1 ||
                                    current_gradient_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                parameter_gradient[position],
                                loss_state - 1,
                                index,
                                inputs,
                            )
                        warm_current_gradient =
                            current_gradient_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                @view(warm_start.parameter_gradient[
                                    position,
                                    :,
                                    :
                                ]),
                                loss_state,
                                index,
                                inputs,
                            )
                        warm_previous_gradient =
                            loss_state == 1 ||
                                    current_gradient_slot === nothing ?
                            0.0 :
                            _survivor_parameter_gradient_contraction(
                                @view(warm_start.parameter_gradient[
                                    position,
                                    :,
                                    :
                                ]),
                                loss_state - 1,
                                index,
                                inputs,
                            )
                    else
                        current_gradient =
                            current_gradient_slot === nothing ?
                            0.0 :
                            (gradient[position]::Matrix{JuMP.VariableRef})[
                                loss_state,
                                current_gradient_slot,
                            ]
                        previous_gradient_for_selected =
                            loss_state == 1 ||
                                    current_gradient_slot === nothing ?
                            0.0 :
                            (gradient[position]::Matrix{JuMP.VariableRef})[
                                loss_state - 1,
                                current_gradient_slot,
                            ]
                        warm_current_gradient =
                            current_gradient_slot === nothing ?
                            0.0 :
                            warm_start.gradient[
                                position,
                                loss_state,
                                index,
                            ]
                        warm_previous_gradient =
                            loss_state == 1 ||
                                    current_gradient_slot === nothing ?
                            0.0 :
                            warm_start.gradient[
                                position,
                                loss_state - 1,
                                index,
                            ]
                    end
                    hessian_recurrences[index] =
                        derivative.base_probability *
                        hessian[position, loss_state] +
                        (1.0 - derivative.base_probability) *
                        previous_hessian +
                        derivative.hessian_covariance *
                        probability_difference +
                        2.0 *
                        (
                            current_gradient -
                            previous_gradient_for_selected
                        )
                    hessian_dummy_starts[index] =
                        index in warm_start_selected ?
                        derivative.base_probability *
                        warm_start.hessian[position, loss_state] +
                        (1.0 - derivative.base_probability) *
                        (
                            loss_state == 1 ?
                            0.0 :
                            warm_start.hessian[position, loss_state - 1]
                        ) +
                        derivative.hessian_covariance *
                        (
                            warm_start.probability[position, loss_state] -
                            (
                                loss_state == 1 ?
                                0.0 :
                                warm_start.probability[
                                    position,
                                    loss_state - 1,
                                ]
                            )
                        ) +
                        2.0 *
                        (
                            warm_current_gradient -
                            warm_previous_gradient
                        ) :
                        0.0
                end
                _survivor_add_one_hot_dummies!(
                    model,
                    hessian[position + 1, loss_state],
                    hessian_recurrences,
                    selected,
                    bounds.team_conditioned.candidate_hessian.lower[
                        :,
                        loss_state,
                    ],
                    bounds.team_conditioned.candidate_hessian.upper[
                        :,
                        loss_state,
                    ],
                    indices,
                    dummy_start_values=hessian_dummy_starts,
                    recurrence_lower=
                        bounds.candidate_hessian.lower[:, loss_state],
                    recurrence_upper=
                        bounds.candidate_hessian.upper[:, loss_state],
                )
            end
        end

        current_probability_sum =
            sum(probability[position, loss_state] for loss_state in state_indices)
        next_probability_sum =
            sum(probability[position + 1, loss_state] for loss_state in state_indices)
        @constraint(model, next_probability_sum <= current_probability_sum)
        probability_sum_recurrences = Dict{Int,Any}()
        probability_sum_dummy_starts = Dict{Int,Float64}()
        warm_current_probability_sum = sum(
            warm_start.probability[position, loss_state]
            for loss_state in state_indices
        )
        warm_probability_terminal =
            warm_start.probability[position, losses_to_elimination]
        for index in indices
            derivative = inputs.derivatives[index]
            probability_terminal = probability[position, losses_to_elimination]
            probability_sum_recurrences[index] =
                current_probability_sum -
                (1.0 - derivative.base_probability) * probability_terminal
            probability_sum_dummy_starts[index] =
                index in warm_start_selected ?
                warm_current_probability_sum -
                (1.0 - derivative.base_probability) *
                warm_probability_terminal :
                0.0
        end
        _survivor_add_one_hot_dummies!(
            model,
            next_probability_sum,
            probability_sum_recurrences,
            selected,
            team_candidate_probability_sum_lower,
            team_candidate_probability_sum_upper,
            indices,
            dummy_start_values=probability_sum_dummy_starts,
            recurrence_lower=candidate_probability_sum_lower,
            recurrence_upper=candidate_probability_sum_upper,
        )

        if position <= curvature_weeks
            current_adjusted_sum = sum(
                probability[position, loss_state] +
                0.5 * hessian[position, loss_state]
                for loss_state in state_indices
            )
            next_adjusted_sum = sum(
                probability[position + 1, loss_state] +
                0.5 * hessian[position + 1, loss_state]
                for loss_state in state_indices
            )
            adjusted_sum_recurrences = Dict{Int,Any}()
            adjusted_sum_dummy_starts = Dict{Int,Float64}()
            warm_current_adjusted_sum = sum(
                warm_start.probability[position, loss_state] +
                0.5 * warm_start.hessian[position, loss_state]
                for loss_state in state_indices
            )
            warm_probability_terminal =
                warm_start.probability[position, losses_to_elimination]
            warm_hessian_terminal =
                warm_start.hessian[position, losses_to_elimination]
            for index in indices
                derivative = inputs.derivatives[index]
                probability_terminal = probability[position, losses_to_elimination]
                hessian_terminal = hessian[position, losses_to_elimination]
                current_gradient_slot = get(
                    gradient_reference_slots[position],
                    index,
                    nothing,
                )
                if position < gradient_switch_position
                    terminal_gradient = current_gradient_slot === nothing ?
                        0.0 :
                        _survivor_parameter_gradient_contraction(
                            parameter_gradient[position],
                            losses_to_elimination,
                            index,
                            inputs,
                        )
                    warm_terminal_gradient =
                        current_gradient_slot === nothing ?
                        0.0 :
                        _survivor_parameter_gradient_contraction(
                            @view(warm_start.parameter_gradient[
                                position,
                                :,
                                :
                            ]),
                            losses_to_elimination,
                            index,
                            inputs,
                        )
                else
                    terminal_gradient =
                        current_gradient_slot === nothing ?
                        0.0 :
                        (gradient[position]::Matrix{JuMP.VariableRef})[
                            losses_to_elimination,
                            current_gradient_slot,
                        ]
                    warm_terminal_gradient =
                        current_gradient_slot === nothing ?
                        0.0 :
                        warm_start.gradient[
                            position,
                            losses_to_elimination,
                            index,
                        ]
                end
                adjusted_sum_recurrences[index] =
                    current_adjusted_sum -
                    (1.0 - derivative.base_probability) *
                    (probability_terminal + 0.5 * hessian_terminal) +
                    0.5 * derivative.hessian_covariance * probability_terminal +
                    terminal_gradient
                adjusted_sum_dummy_starts[index] =
                    index in warm_start_selected ?
                    warm_current_adjusted_sum -
                    (1.0 - derivative.base_probability) *
                    (
                        warm_probability_terminal +
                        0.5 * warm_hessian_terminal
                    ) +
                    0.5 * derivative.hessian_covariance *
                    warm_probability_terminal +
                    warm_terminal_gradient :
                    0.0
            end
            _survivor_add_one_hot_dummies!(
                model,
                next_adjusted_sum,
                adjusted_sum_recurrences,
                selected,
                team_candidate_adjusted_sum_lower,
                team_candidate_adjusted_sum_upper,
                indices,
                dummy_start_values=adjusted_sum_dummy_starts,
                recurrence_lower=candidate_adjusted_sum_lower,
                recurrence_upper=candidate_adjusted_sum_upper,
            )
        end
    end

    warm_start_objective = sum(
        warm_start.probability[position + 1, loss_state]
        for position in 1:number_of_weeks,
        loss_state in state_indices
    ) + 0.5 * sum(
        warm_start.hessian[position + 1, loss_state]
        for position in 1:curvature_weeks,
        loss_state in state_indices;
        init=0.0,
    )
    @debug "survivor MILP warm start" phase=:exact_milp objective=warm_start_objective
    _set_survivor_scalar_warm_start!(
        selected,
        probability,
        parameter_gradient,
        gradient,
        hessian,
        warm_start,
        candidate_indices,
        gradient_reference_indices,
        gradient_switch_position,
        number_of_weeks,
        losses_to_elimination,
        curvature_weeks,
    )

    objective_expression =
        sum(
            probability[position + 1, loss_state]
            for position in 1:number_of_weeks,
            loss_state in state_indices
        ) + 0.5 * sum(
            hessian[position + 1, loss_state]
            for position in 1:curvature_weeks,
            loss_state in state_indices;
            init=0.0,
        )
    @objective(model, Max, objective_expression)
    solve_started_at = time_ns()
    optimize!(model)
    _survivor_log_milp_result(
        model,
        :exact_milp,
        (time_ns() - solve_started_at) / 1.0e9,
    )
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
        curvature_weeks=curvature_weeks,
        gradient_reference_indices=gradient_reference_indices,
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
        selected_expected_weeks,
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
    data = _normalize_survivor_candidates(
        candidates,
        state,
        selection_config.through_week,
        banned_first_pick_teams=selection_config.banned_first_pick_teams,
    )
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
    method::WeibullEmpiricalBayesFit=DEFAULT_PRIOR_FIT_METHOD,
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
