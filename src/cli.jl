const SURVIVOR_CLI_APP_NAME = "survivor"

function _survivor_cli_refresh_data_enabled()
    value = lowercase(strip(get(ENV, "SURVIVORMODEL_REFRESH_DATA", "false")))
    return value in ("1", "true", "yes", "on")
end

function _survivor_cli_usage()
    return """
    Usage:
      survivor --season YEAR [--strikes N] [--objective NAME] [--timings] < picks.txt
      julia --project=. -m SurvivorModel --season YEAR [--strikes N] [--objective NAME] [--timings] < picks.txt

    Input:
      One team abbreviation per nonblank line, starting with week 1.
      The next pick is inferred from the number of nonblank lines.

    Options:
      --season YEAR       Target season (required).
      --strikes N         Initial strike count (default: 2).
      --objective NAME    Selection objective (default:
                          milp; use micp for the conic
                          expected-weeks strategy).
      --timings           Print phase timings to stderr.
      --refresh-data      Clear NFLData's raw cache and refresh summarized
                          historical drive data before running.
      --help              Show this help.
    """
end

function _parse_survivor_cli_integer(value::AbstractString, option::AbstractString)
    parsed = tryparse(Int, value)
    parsed === nothing &&
        throw(ArgumentError("$option requires an integer value"))
    return parsed
end

function _parse_survivor_cli_objective(value::AbstractString)
    normalized = lowercase(replace(strip(value), '-' => '_'))
    return _canonical_survivor_objective(Symbol(normalized))
end

function _parse_survivor_cli_args(args::AbstractVector{<:AbstractString})
    season = nothing
    initial_strikes = 2
    strikes_specified = false
    objective = DEFAULT_SURVIVOR_OBJECTIVE
    objective_specified = false
    timings = false
    timings_specified = false
    refresh_data = false
    refresh_data_specified = false
    show_help = false
    index = 1

    while index <= length(args)
        argument = String(args[index])
        if argument == "--help" || argument == "-h"
            show_help = true
            index += 1
            continue
        elseif argument == "--"
            index += 1
            index <= length(args) &&
                throw(ArgumentError("positional arguments are not supported"))
            break
        end

        if argument == "--refresh-data"
            refresh_data_specified &&
                throw(ArgumentError("--refresh-data may only be specified once"))
            refresh_data = true
            refresh_data_specified = true
            index += 1
            continue
        end
        if argument == "--timings"
            timings_specified &&
                throw(ArgumentError("--timings may only be specified once"))
            timings = true
            timings_specified = true
            index += 1
            continue
        end

        option = nothing
        value = nothing
        if argument == "--season" ||
            argument == "--strikes" ||
            argument == "--objective"
            option = argument
            index += 1
            index <= length(args) ||
                throw(ArgumentError("$option requires a value"))
            value = String(args[index])
        elseif startswith(argument, "--season=")
            option = "--season"
            value = argument[length("--season=") + 1:end]
        elseif startswith(argument, "--strikes=")
            option = "--strikes"
            value = argument[length("--strikes=") + 1:end]
        elseif startswith(argument, "--objective=")
            option = "--objective"
            value = argument[length("--objective=") + 1:end]
        else
            throw(ArgumentError("unknown option: $argument"))
        end

        isempty(value) && throw(ArgumentError("$option requires a value"))
        if option == "--season"
            parsed = _parse_survivor_cli_integer(value, option)
            season === nothing ||
                throw(ArgumentError("--season may only be specified once"))
            season = parsed
        elseif option == "--strikes"
            parsed = _parse_survivor_cli_integer(value, option)
            strikes_specified &&
                throw(ArgumentError("--strikes may only be specified once"))
            initial_strikes = parsed
            strikes_specified = true
        elseif option == "--objective"
            objective_specified &&
                throw(ArgumentError("--objective may only be specified once"))
            objective = _parse_survivor_cli_objective(value)
            objective_specified = true
        end
        index += 1
    end

    show_help && return (
        show_help=true,
        season=0,
        initial_strikes=2,
        objective=DEFAULT_SURVIVOR_OBJECTIVE,
        timings=false,
        refresh_data=false,
    )
    season === nothing && throw(ArgumentError("--season is required"))
    season > 0 || throw(ArgumentError("--season must be positive"))
    initial_strikes >= 0 ||
        throw(ArgumentError("--strikes must be nonnegative"))
    return (
        show_help=false,
        season=Int(season),
        initial_strikes=Int(initial_strikes),
        objective=objective,
        timings=timings,
        refresh_data=refresh_data,
    )
end

function _read_survivor_cli_picks(input::IO)
    picks = String[]
    for line in eachline(input)
        team = uppercase(strip(line))
        isempty(team) && continue
        occursin(r"\s", team) &&
            throw(ArgumentError("each pick must contain one team abbreviation"))
        push!(picks, team)
    end
    return picks
end

function _survivor_cli_result_value(value)
    ismissing(value) &&
        throw(ArgumentError("a previous pick refers to an uncompleted game"))
    result = value isa Real ? Float64(value) : tryparse(Float64, string(value))
    result === nothing ||
        (isfinite(result) && return result)
    throw(ArgumentError("schedule results must be finite numeric margins"))
end

function _survivor_cli_state(
    schedule::AbstractDataFrame,
    season::Integer,
    picks::AbstractVector{<:AbstractString},
    initial_strikes::Integer,
)
    initial_strikes >= 0 ||
        throw(ArgumentError("initial strikes must be nonnegative"))
    length(picks) < 18 ||
        throw(ArgumentError("there is no current week after 18 completed picks"))

    regular_schedule = _regular_season_schedule(schedule, season)
    isempty(regular_schedule) &&
        throw(ArgumentError("schedule has no regular-season games for season $season"))

    picks_made = Dict{Int,String}()
    losses = 0
    for (week, raw_team) in enumerate(picks)
        team = uppercase(strip(String(raw_team)))
        isempty(team) && throw(ArgumentError("picks cannot contain empty teams"))
        haskey(picks_made, week) &&
            throw(ArgumentError("picks cannot contain duplicate weeks"))
        team in values(picks_made) &&
            throw(ArgumentError("team $team was picked more than once"))

        matching = regular_schedule[
            (regular_schedule.week .== week) .&
            (
                (regular_schedule.away_team .== team) .|
                (regular_schedule.home_team .== team)
            ),
            :,
        ]
        nrow(matching) == 1 ||
            throw(ArgumentError(
                "pick $team is not uniquely scheduled in season $season week $week",
            ))

        result = _survivor_cli_result_value(matching.result[1])
        home_team = String(matching.home_team[1])
        away_team = String(matching.away_team[1])
        team_won = team == home_team ? result > 0 : result < 0
        losses += team_won ? 0 : 1
        picks_made[week] = team
    end

    strikes_remaining = Int(initial_strikes) - losses
    strikes_remaining >= 0 ||
        throw(ArgumentError(
            "the supplied picks contain $losses losses, exceeding the " *
            "$initial_strikes-strike count",
        ))
    losses < max(1, Int(initial_strikes)) ||
        throw(ArgumentError(
            "the supplied picks already reach the elimination loss threshold",
        ))
    return (
        current_week=length(picks) + 1,
        picks_made=picks_made,
        losses=losses,
        strikes_remaining=strikes_remaining,
    )
end

function _survivor_cli_historical_drives(
    schedule::AbstractDataFrame,
    season::Integer,
    historical_drives::AbstractDataFrame,
)
    regular = _regular_season_drives(historical_drives, schedule)
    return regular[regular.season .< Int(season), :]
end

function _survivor_cli_load_historical_drives(
    season::Integer,
    max_seasons::Int,
    historical_drives,
    cache_directory::Union{Nothing,AbstractString},
    ;
    refresh::Bool=false,
    loader::Function=load_drive_pbp,
)
    historical_drives !== nothing && return historical_drives
    season <= 1999 && return nothing

    first_season = max(1999, Int(season) - max_seasons)
    last_season = Int(season) - 1
    cached = DataFrame[
        _cached_season_drives(
            data_season;
            cache_directory=cache_directory,
            refresh=refresh,
            loader=loader,
        ).drives for data_season in first_season:last_season
    ]
    return vcat(cached...; cols=:union)
end

function _run_survivor_cli(
    args::AbstractVector{<:AbstractString};
    input::IO=stdin,
    output::IO=stdout,
    schedule=nothing,
    historical_drives=nothing,
    current_drives=nothing,
    cache_directory::Union{Nothing,AbstractString}=nothing,
    method::PriorFitMethod=DEFAULT_PRIOR_FIT_METHOD,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    through_week::Int=18,
    timing_output::IO=stderr,
)
    timing_started = time_ns()
    timing_logger = Logging.SimpleLogger(timing_output, Logging.Info)
    record_timing = function(phase::Symbol)
        timings_enabled || return nothing
        now = time_ns()
        duration = (now - timing_started) / 1.0e9
        timing_started = now
        Logging.with_logger(timing_logger) do
            @info "survivor phase complete" phase elapsed_seconds=duration
        end
        return nothing
    end

    options = _parse_survivor_cli_args(args)
    timings_enabled = options.timings
    record_timing(:parse)
    if options.show_help
        print(output, _survivor_cli_usage())
        return 0
    end
    refresh_data = options.refresh_data || _survivor_cli_refresh_data_enabled()
    refresh_data && NFLData.clear_cache()

    picks = _read_survivor_cli_picks(input)
    normalized_schedule = schedule === nothing ? load_schedule() : load_schedule(schedule)
    record_timing(:schedule)
    state = _survivor_cli_state(
        normalized_schedule,
        options.season,
        picks,
        options.initial_strikes,
    )
    record_timing(:state)
    historical_source = _survivor_cli_load_historical_drives(
        options.season,
        max_seasons,
        historical_drives,
        cache_directory;
        refresh=refresh_data,
    )
    historical, current = _load_forecast_drives(
        options.season,
        max_seasons,
        historical_source,
        current_drives,
        ;
        allow_missing_current=state.current_week == 1,
    )
    record_timing(:drive_data)
    historical = _survivor_cli_historical_drives(
        normalized_schedule,
        options.season,
        historical,
    )
    current = _regular_season_drives(current, normalized_schedule)
    cached_prior = _cached_historical_prior(
        historical;
        current_season=options.season,
        time_edges=DEFAULT_TIME_EDGES,
        max_seasons=max_seasons,
        method=method,
        cache_directory=cache_directory,
    )
    record_timing(:prior)
    context = fit_regular_season_forecast(
        options.season;
        as_of_week=state.current_week,
        schedule=normalized_schedule,
        historical_drives=historical,
        current_drives=current,
        max_seasons=max_seasons,
        method=method,
        prior=cached_prior.prior,
        _normalized_schedule=true,
        _schedule_indexed_drives=true,
    )
    record_timing(:fit)
    record_timing(:optimize_start)
    plan = optimize_survivor_pool(
        context;
        picks_made=state.picks_made,
        strikes_remaining=state.strikes_remaining,
        include_completed=true,
        selection_config=SurvivorSelectionConfig(
            objective=options.objective,
            through_week=through_week,
        ),
    )
    record_timing(:optimize)
    nrow(plan.current_pick) == 1 ||
        throw(ArgumentError("survivor optimization did not produce one current pick"))
    print(output, String(plan.current_pick.team[1]), '\n')
    record_timing(:output)
    return 0
end

function (@main)(args)
    try
        return _run_survivor_cli(args)
    catch error
        if error isa ArgumentError
            println(stderr, "$SURVIVOR_CLI_APP_NAME: ", sprint(showerror, error))
            return 2
        end
        rethrow()
    end
end
