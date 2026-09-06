using DataFrames
using Test
using SurvivorModel

const SURVIVOR_LIVE_REGULAR_SEASON_LENGTHS = (17, 18)

mutable struct SurvivorLiveScenario
    initial_strikes::Int
    picks_made::Dict{Int,String}
    strikes_remaining::Int
    last_survived_week::Int
    elimination_week::Union{Nothing,Int}
    elimination_reason::String
    elimination_detail::String
    records::DataFrame
end

function _survivor_live_records()
    return DataFrame(
        season=Int[],
        initial_strikes=Int[],
        week=Int[],
        game_id=String[],
        team=String[],
        opponent=String[],
        is_home=Bool[],
        win_probability=Float64[],
        discount=Float64[],
        published_home_spread=Union{Missing,Float64}[],
        picked_team_spread=Union{Missing,Float64}[],
        picked_market_status=String[],
        actual_result=Int[],
        outcome=String[],
        tie=Bool[],
        strikes_before=Int[],
        strikes_after=Int[],
        alive_after=Bool[],
    )
end

function _survivor_live_completed_seasons(schedule::AbstractDataFrame)
    regular = schedule[schedule.game_type .== "REG", :]
    seasons = Int[]
    for season in sort(unique(regular.season))
        games = regular[regular.season .== season, :]
        isempty(games) && continue
        maximum(games.week) in SURVIVOR_LIVE_REGULAR_SEASON_LENGTHS || continue
        all(.!ismissing.(games.result)) || continue
        push!(seasons, Int(season))
    end
    return seasons
end

function _survivor_live_requested_seasons(schedule::AbstractDataFrame)
    completed = _survivor_live_completed_seasons(schedule)
    isempty(completed) &&
        throw(ArgumentError("schedule has no completed regular seasons"))

    requested = strip(get(ENV, "SURVIVORMODEL_SURVIVOR_SEASONS", ""))
    if !isempty(requested)
        seasons = sort(unique([
            parse(Int, strip(value)) for value in split(requested, ',')
        ]))
        missing_seasons = setdiff(seasons, completed)
        isempty(missing_seasons) ||
            throw(ArgumentError(
                "requested seasons are not completed regular seasons: $missing_seasons",
            ))
        return seasons
    end

    recent_count = parse(
        Int,
        get(ENV, "SURVIVORMODEL_SURVIVOR_RECENT_SEASONS", "3"),
    )
    recent_count > 0 ||
        throw(ArgumentError("SURVIVORMODEL_SURVIVOR_RECENT_SEASONS must be positive"))
    length(completed) >= recent_count ||
        throw(ArgumentError(
            "only $(length(completed)) completed seasons are available, " *
            "but $recent_count were requested",
        ))
    return completed[(end - recent_count + 1):end]
end

function _survivor_live_last_week(
    schedule::AbstractDataFrame,
    season::Integer,
)
    regular = schedule[
        (schedule.game_type .== "REG") .& (schedule.season .== season),
        :,
    ]
    isempty(regular) &&
        throw(ArgumentError("schedule has no regular-season games for season $season"))
    last_week = maximum(regular.week)
    last_week in SURVIVOR_LIVE_REGULAR_SEASON_LENGTHS ||
        throw(ArgumentError(
            "season $season has unsupported final regular-season week $last_week",
        ))
    return Int(last_week)
end

function _survivor_live_weekly_probability()
    probability = parse(
        Float64,
        get(ENV, "SURVIVORMODEL_SURVIVOR_WEEKLY_SURVIVAL", "0.65"),
    )
    0.0 <= probability <= 1.0 ||
        throw(ArgumentError(
            "SURVIVORMODEL_SURVIVOR_WEEKLY_SURVIVAL must be in [0, 1]",
        ))
    return probability
end

function _survivor_live_recency_half_life()
    half_life = parse(
        Float64,
        get(ENV, "SURVIVORMODEL_SURVIVOR_RECENCY_HALF_LIFE", "1.0"),
    )
    isfinite(half_life) && half_life > 0.0 ||
        throw(ArgumentError(
            "SURVIVORMODEL_SURVIVOR_RECENCY_HALF_LIFE must be finite and positive",
        ))
    return half_life
end

function _survivor_live_game_row(
    forecast::AbstractDataFrame,
    current_pick::AbstractDataFrame,
)
    game_id = current_pick.game_id[1]
    rows = forecast[forecast.game_id .== game_id, :]
    nrow(rows) == 1 ||
        throw(ArgumentError("forecast must contain exactly one row for game $game_id"))
    return rows[1, :]
end

function _survivor_live_actual_result(
    forecast::AbstractDataFrame,
    current_pick::AbstractDataFrame,
)
    row = _survivor_live_game_row(forecast, current_pick)
    result = row.result
    ismissing(result) &&
        throw(ArgumentError(
            "game $(current_pick.game_id[1]) has no completed result",
        ))
    return Int(result)
end

function _survivor_live_record_pick!(
    scenario::SurvivorLiveScenario,
    season::Integer,
    week::Integer,
    current_pick::AbstractDataFrame,
    actual_result::Integer,
    published_home_spread,
)
    team_won = current_pick.is_home[1] ?
        actual_result > 0 :
        actual_result < 0
    strikes_before = scenario.strikes_remaining
    strikes_after = team_won ? strikes_before : max(strikes_before - 1, 0)
    alive_after = team_won || strikes_before > 0
    spread = ismissing(published_home_spread) ?
        missing :
        Float64(published_home_spread)
    picked_spread = ismissing(spread) ?
        missing :
        (current_pick.is_home[1] ? spread : -spread)
    market_status = ismissing(picked_spread) ? "unavailable" :
        picked_spread > 0 ? "favorite" :
        picked_spread < 0 ? "underdog" :
        "pickem"

    push!(
        scenario.records,
        (
            season=Int(season),
            initial_strikes=scenario.initial_strikes,
            week=Int(week),
            game_id=String(current_pick.game_id[1]),
            team=String(current_pick.team[1]),
            opponent=String(current_pick.opponent[1]),
            is_home=Bool(current_pick.is_home[1]),
            win_probability=Float64(current_pick.win_probability[1]),
            discount=Float64(current_pick.discount[1]),
            published_home_spread=spread,
            picked_team_spread=picked_spread,
            picked_market_status=market_status,
            actual_result=Int(actual_result),
            outcome=team_won ? "win" : "loss",
            tie=actual_result == 0,
            strikes_before=strikes_before,
            strikes_after=strikes_after,
            alive_after=alive_after,
        ),
    )

    scenario.picks_made[Int(week)] = String(current_pick.team[1])
    scenario.strikes_remaining = strikes_after
    if alive_after
        scenario.last_survived_week = Int(week)
    else
        scenario.elimination_week = Int(week)
        scenario.elimination_reason = "loss"
        scenario.elimination_detail = actual_result == 0 ?
            "selected game was a tie; ties are treated as losses" :
            "selected team lost with no strikes remaining"
    end
    return nothing
end

function _survivor_live_summary(
    scenario::SurvivorLiveScenario,
    season::Integer,
    last_week::Integer,
    weekly_survival_probability::Real,
    elapsed_seconds::Real,
)
    wins = count(==("win"), scenario.records.outcome)
    losses = count(==("loss"), scenario.records.outcome)
    losing_records = scenario.records[scenario.records.outcome .== "loss", :]
    return (
        season=Int(season),
        regular_season_last_week=Int(last_week),
        initial_strikes=scenario.initial_strikes,
        weekly_survival_probability=Float64(weekly_survival_probability),
        last_survived_week=scenario.last_survived_week,
        elimination_week=isnothing(scenario.elimination_week) ?
            missing :
            scenario.elimination_week,
        elimination_reason=scenario.elimination_reason,
        weeks_played=nrow(scenario.records),
        wins=wins,
        losses=losses,
        losses_as_favorite=count(
            ==("favorite"),
            losing_records.picked_market_status,
        ),
        losses_as_underdog=count(
            ==("underdog"),
            losing_records.picked_market_status,
        ),
        losses_at_pickem=count(
            ==("pickem"),
            losing_records.picked_market_status,
        ),
        losses_without_published_spread=count(
            ==("unavailable"),
            losing_records.picked_market_status,
        ),
        strikes_used=scenario.initial_strikes - scenario.strikes_remaining,
        survived_season=isnothing(scenario.elimination_week),
        elapsed_seconds=Float64(elapsed_seconds),
        elimination_detail=scenario.elimination_detail,
    )
end

function _survivor_live_prior(
    season::Integer,
    schedule::AbstractDataFrame,
    drives::AbstractDataFrame,
    max_seasons::Int,
    recency_half_life::Real,
)
    regular_drives = SurvivorModel._regular_season_drives(drives, schedule)
    historical = regular_drives[regular_drives.season .< Int(season), :]
    return fit_empirical_bayes_prior(
        historical;
        max_seasons=max_seasons,
        recency_half_life=recency_half_life,
        current_season=season,
    )
end

function _survivor_live_backtest_season(
    season::Integer,
    schedule::AbstractDataFrame,
    drives::AbstractDataFrame,
    last_week::Integer,
    initial_strikes_values,
    weekly_survival_probability::Real,
    max_seasons::Int,
    recency_half_life::Real,
)
    scenarios = [
        SurvivorLiveScenario(
            Int(strikes),
            Dict{Int,String}(),
            Int(strikes),
            0,
            nothing,
            "survived_season",
            "",
            _survivor_live_records(),
        ) for strikes in initial_strikes_values
    ]
    started_at = time()
    prior = _survivor_live_prior(
        season,
        schedule,
        drives,
        max_seasons,
        recency_half_life,
    )

    for week in 1:last_week
        active = [
            scenario for scenario in scenarios
            if isnothing(scenario.elimination_week)
        ]
        isempty(active) && break

        context = fit_regular_season_forecast(
            season;
            as_of_week=week,
            schedule=schedule,
            historical_drives=drives,
            current_drives=drives,
            max_seasons=max_seasons,
            prior=prior,
        )
        forecast = forecast_win_probabilities(
            context;
            include_completed=true,
            full_schedule=true,
        )

        for scenario in active
            state = SurvivorPoolState(
                season,
                week;
                picks_made=scenario.picks_made,
                strikes_remaining=scenario.strikes_remaining,
            )
            candidates = build_survivor_candidates(
                forecast;
                from_week=week,
                through_week=last_week,
                include_completed=true,
                picks_made=scenario.picks_made,
            )

            if !any(candidates.week .== week)
                scenario.elimination_week = week
                scenario.elimination_reason = "no_legal_pick"
                scenario.elimination_detail =
                    "all current-week teams had already been selected"
                continue
            end

            missing_future_weeks = setdiff(
                collect(week:last_week),
                sort(unique(candidates.week)),
            )
            if !isempty(missing_future_weeks)
                scenario.elimination_week = week
                scenario.elimination_reason = "no_legal_full_plan"
                scenario.elimination_detail =
                    "no unused team was available in week(s) $missing_future_weeks"
                continue
            end

            plan = optimize_survivor_pool(
                candidates,
                state;
                weekly_survival_probability=weekly_survival_probability,
                through_week=last_week,
            )
            game_row = _survivor_live_game_row(forecast, plan.current_pick)
            actual_result = _survivor_live_actual_result(
                forecast,
                plan.current_pick,
            )
            _survivor_live_record_pick!(
                scenario,
                season,
                week,
                plan.current_pick,
                actual_result,
                :spread_line in propertynames(game_row) ?
                    game_row.spread_line :
                    missing,
            )
        end
    end

    elapsed_seconds = time() - started_at
    summary = DataFrame([
        _survivor_live_summary(
            scenario,
            season,
            last_week,
            weekly_survival_probability,
            elapsed_seconds,
        ) for scenario in scenarios
    ])
    history = isempty(scenarios) ?
        _survivor_live_records() :
        vcat([scenario.records for scenario in scenarios]...; cols=:union)
    return (summary=summary, history=history)
end

function _survivor_live_load_data(
    schedule::AbstractDataFrame,
    seasons,
    max_seasons::Int,
)
    effective_max_seasons = min(max_seasons, MAX_HISTORICAL_SEASONS)
    first_data_season = max(1999, first(seasons) - effective_max_seasons)
    last_data_season = last(seasons)
    loaded = DataFrame[]
    for data_season in first_data_season:last_data_season
        drives = try
            load_drive_pbp(data_season)
        catch error
            throw(ArgumentError(
                "unable to load drive data for season $data_season: " *
                sprint(showerror, error),
            ))
        end
        nrow(drives) > 0 ||
            throw(ArgumentError(
                "drive data for season $data_season is empty",
            ))
        push!(loaded, drives)
    end
    return vcat(loaded...; cols=:union)
end

@testset "live survivor backtest" begin
    schedule = load_schedule()
    seasons = _survivor_live_requested_seasons(schedule)
    weekly_survival_probability = _survivor_live_weekly_probability()
    recency_half_life = _survivor_live_recency_half_life()
    max_seasons = parse(
        Int,
        get(ENV, "SURVIVORMODEL_SURVIVOR_MAX_SEASONS", "3"),
    )
    max_seasons > 0 ||
        throw(ArgumentError("SURVIVORMODEL_SURVIVOR_MAX_SEASONS must be positive"))
    initial_strikes_values = (1, 2)
    drives = _survivor_live_load_data(schedule, seasons, max_seasons)

    reports = [
        _survivor_live_backtest_season(
            season,
            schedule,
            drives,
            _survivor_live_last_week(schedule, season),
            initial_strikes_values,
            weekly_survival_probability,
            max_seasons,
            recency_half_life,
        ) for season in seasons
    ]
    summary = vcat([report.summary for report in reports]...)
    history = vcat([report.history for report in reports]...; cols=:union)

    @test nrow(summary) == length(seasons) * length(initial_strikes_values)
    @test all(summary.last_survived_week .>= 0)
    @test all(value -> value in initial_strikes_values, summary.initial_strikes)
    @test all(x -> x in ("win", "loss"), history.outcome)

    println("Survivor backtest summary:")
    show(stdout, summary; allrows=true, allcols=true)
    println()
    println("Survivor backtest picks:")
    show(stdout, history; allrows=true, allcols=true)
    println()
end
