const SURVIVOR_CLI_APP_NAME = "survivor"

function _survivor_cli_refresh_data_enabled()
    value = lowercase(strip(get(ENV, "SURVIVORMODEL_REFRESH_DATA", "false")))
    return value in ("1", "true", "yes", "on")
end

function _survivor_cli_usage()
    return """
    Usage:
      survivor --season YEAR [--ban TEAM1,TEAM2] [--branch-and-bound] [--write-model FILE.lp] [--strikes N] [--hessian-weeks N] [--timeout SECONDS] < picks.txt
      julia --project=. -m SurvivorModel --season YEAR [--ban TEAM1,TEAM2] [--branch-and-bound] [--write-model FILE.lp] [--strikes N] [--hessian-weeks N] [--timeout SECONDS] < picks.txt
      survivor --refresh-priors
      survivor --refresh-data [--refresh-priors]
      julia --project=. -m SurvivorModel --refresh-priors
      julia --project=. -m SurvivorModel --refresh-data [--refresh-priors]
      survivor --season YEAR --plot-strength WEEK
      julia --project=. -m SurvivorModel --season YEAR --plot-strength WEEK
      survivor --season YEAR --grid [--strikes N] < picks.txt
      julia --project=. -m SurvivorModel --season YEAR --grid [--strikes N] < picks.txt

    Input:
      One team abbreviation per nonblank line, starting with week 1.
      The next pick is inferred from the number of nonblank lines.

    Options:
      --season YEAR       Target season (required unless refreshing caches).
      --ban TEAMS         Comma-separated teams forbidden as the current pick.
      --branch-and-bound  Full-LP external tree certifying the current pick;
                          HiPO with crossover is used for root and child LPs.
      --write-model FILE.lp  Save the MILP with HiGHS and exit without solving.
      --strikes N         Initial strike count (default: 2).
      --hessian-weeks N  Number of future weeks with Hessian adjustments
                          for exact-milp (default: 3; 0 is linear-only).
      --timeout SECONDS  Optimization time limit in seconds (default: unlimited).
      --refresh-data      Clear NFLData's raw cache and refresh summarized
                          historical drive data before running. Without
                          --season, clear raw-data and summarized-drive
                          caches and exit.
      --refresh-priors    Clear cached historical prior fits. Without
                          --season, clear the cache and exit.
      --plot-strength WEEK  Show a plot at the start of WEEK (1-18), using
                            prior weeks only; no picks or run options, and a
                            native desktop/OpenGL display is required.
                            League-prior percentile axes and central 50%
                            posterior intervals with labeled team points.
      --grid              Print all unused teams and remaining weeks, with
                          opponents and Hessian-adjusted win percentages,
                          sorted by current-week probability; byes are blank.
                          Stars mark each week's top five unused teams;
                          horizontal rules group every five team rows.
                          No selection/export options; --strikes and cache
                          refresh flags are allowed.
      --help              Show this help.
    """
end

function _parse_survivor_cli_integer(value::AbstractString, option::AbstractString)
    parsed = tryparse(Int, value)
    parsed === nothing &&
        throw(ArgumentError("$option requires an integer value"))
    return parsed
end

function _parse_survivor_cli_real(value::AbstractString, option::AbstractString)
    parsed = tryparse(Float64, value)
    parsed === nothing ||
        (isfinite(parsed) && parsed > 0.0 && return parsed)
    throw(ArgumentError("$option requires a finite positive number of seconds"))
end

function _parse_survivor_cli_banned_teams(value::AbstractString)
    return _normalize_survivor_team_abbreviations(
        split(value, ','; keepempty=true),
        "--ban",
    )
end

function _parse_survivor_cli_args(args::AbstractVector{<:AbstractString})
    season = nothing
    initial_strikes = 2
    strikes_specified = false
    banned_first_pick_teams = String[]
    ban_specified = false
    branch_and_bound = false
    write_model_file = nothing
    write_model_specified = false
    hessian_weeks = 3
    hessian_weeks_specified = false
    timeout_seconds = nothing
    timeout_specified = false
    refresh_data = false
    refresh_data_specified = false
    refresh_priors = false
    refresh_priors_specified = false
    plot_strength_week = nothing
    plot_strength_specified = false
    grid = false
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

        if argument == "--branch-and-bound"
            branch_and_bound &&
                throw(ArgumentError("--branch-and-bound may only be specified once"))
            branch_and_bound = true
            index += 1
            continue
        end
        if argument == "--grid"
            grid && throw(ArgumentError("--grid may only be specified once"))
            grid = true
            index += 1
            continue
        end
        if argument == "--refresh-data"
            refresh_data_specified &&
                throw(ArgumentError("--refresh-data may only be specified once"))
            refresh_data = true
            refresh_data_specified = true
            index += 1
            continue
        end
        if argument == "--refresh-priors"
            refresh_priors_specified &&
                throw(ArgumentError("--refresh-priors may only be specified once"))
            refresh_priors = true
            refresh_priors_specified = true
            index += 1
            continue
        end
        option = nothing
        value = nothing
        if argument == "--season" ||
            argument == "--ban" ||
            argument == "--write-model" ||
            argument == "--strikes" ||
            argument == "--hessian-weeks" ||
            argument == "--timeout" ||
            argument == "--plot-strength"
            option = argument
            index += 1
            index <= length(args) ||
                throw(ArgumentError("$option requires a value"))
            value = String(args[index])
        elseif startswith(argument, "--season=")
            option = "--season"
            value = argument[length("--season=") + 1:end]
        elseif startswith(argument, "--ban=")
            option = "--ban"
            value = argument[length("--ban=") + 1:end]
        elseif startswith(argument, "--write-model=")
            option = "--write-model"
            value = argument[length("--write-model=") + 1:end]
        elseif startswith(argument, "--strikes=")
            option = "--strikes"
            value = argument[length("--strikes=") + 1:end]
        elseif startswith(argument, "--hessian-weeks=")
            option = "--hessian-weeks"
            value = argument[length("--hessian-weeks=") + 1:end]
        elseif startswith(argument, "--timeout=")
            option = "--timeout"
            value = argument[length("--timeout=") + 1:end]
        elseif startswith(argument, "--plot-strength=")
            option = "--plot-strength"
            value = argument[length("--plot-strength=") + 1:end]
        else
            throw(ArgumentError("unknown option: $argument"))
        end

        isempty(value) && throw(ArgumentError("$option requires a value"))
        if option == "--season"
            parsed = _parse_survivor_cli_integer(value, option)
            season === nothing ||
                throw(ArgumentError("--season may only be specified once"))
            season = parsed
        elseif option == "--ban"
            ban_specified &&
                throw(ArgumentError("--ban may only be specified once"))
            banned_first_pick_teams =
                _parse_survivor_cli_banned_teams(value)
            ban_specified = true
        elseif option == "--write-model"
            write_model_specified &&
                throw(ArgumentError(
                    "--write-model may only be specified once",
                ))
            write_model_file = _survivor_lp_output_path(value)
            write_model_specified = true
        elseif option == "--strikes"
            parsed = _parse_survivor_cli_integer(value, option)
            strikes_specified &&
                throw(ArgumentError("--strikes may only be specified once"))
            initial_strikes = parsed
            strikes_specified = true
        elseif option == "--hessian-weeks"
            hessian_weeks_specified &&
                throw(ArgumentError("--hessian-weeks may only be specified once"))
            hessian_weeks = _parse_survivor_cli_integer(value, option)
            hessian_weeks_specified = true
            hessian_weeks >= 0 ||
                throw(ArgumentError("--hessian-weeks must be nonnegative"))
        elseif option == "--timeout"
            timeout_specified &&
                throw(ArgumentError("--timeout may only be specified once"))
            timeout_seconds = _parse_survivor_cli_real(value, option)
            timeout_specified = true
        elseif option == "--plot-strength"
            plot_strength_specified &&
                throw(ArgumentError(
                    "--plot-strength may only be specified once",
                ))
            plot_strength_week = _parse_survivor_cli_integer(value, option)
            plot_strength_specified = true
            1 <= plot_strength_week <= 18 ||
                throw(ArgumentError("--plot-strength must be between 1 and 18"))
        end
        index += 1
    end

    show_help && return (
        show_help=true,
        season=nothing,
        initial_strikes=2,
        banned_first_pick_teams=String[],
        write_model_file=nothing,
        hessian_weeks=3,
        branch_and_bound=false,
        timeout_seconds=nothing,
        refresh_data=false,
        refresh_priors=false,
        plot_strength_week=nothing,
        grid=false,
    )
    plot_strength_specified && season === nothing &&
        throw(ArgumentError("--plot-strength requires --season"))
    grid && season === nothing &&
        throw(ArgumentError("--grid requires --season"))
    if season === nothing
        (refresh_data || refresh_priors) ||
            throw(ArgumentError(
                "--season is required unless --refresh-data or " *
                "--refresh-priors is specified",
            ))
        (
            ban_specified ||
            branch_and_bound ||
            write_model_specified ||
            strikes_specified ||
            hessian_weeks_specified ||
            timeout_specified
        ) && throw(ArgumentError("run options require --season"))
    else
        season > 0 || throw(ArgumentError("--season must be positive"))
    end
    if plot_strength_specified
        (
            grid ||
            ban_specified ||
            branch_and_bound ||
            write_model_specified ||
            strikes_specified ||
            hessian_weeks_specified ||
            timeout_specified
        ) && throw(ArgumentError(
            "--plot-strength cannot be combined with --grid, selection, or model-export options",
        ))
    end
    if grid
        (
            ban_specified ||
            branch_and_bound ||
            write_model_specified ||
            hessian_weeks_specified ||
            timeout_specified
        ) && throw(ArgumentError(
            "--grid cannot be combined with selection or model-export options; " *
            "all grid probabilities always include Hessian adjustments",
        ))
    end
    initial_strikes >= 0 ||
        throw(ArgumentError("--strikes must be nonnegative"))
    hessian_weeks >= 0 ||
        throw(ArgumentError("--hessian-weeks must be nonnegative"))
    return (
        show_help=false,
        season=season === nothing ? nothing : Int(season),
        initial_strikes=Int(initial_strikes),
        banned_first_pick_teams=banned_first_pick_teams,
        write_model_file=write_model_file,
        hessian_weeks=Int(hessian_weeks),
        branch_and_bound,
        timeout_seconds=timeout_seconds,
        refresh_data=refresh_data,
        refresh_priors=refresh_priors,
        plot_strength_week=plot_strength_week,
        grid=grid,
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

function _survivor_cli_schedule_and_state(
    schedule,
    season::Integer,
    picks::AbstractVector{<:AbstractString},
    initial_strikes::Integer;
    schedule_loader::Function=load_schedule,
    clear_data_cache::Function=NFLData.clear_cache,
)
    normalized_schedule = schedule === nothing ?
        schedule_loader() :
        load_schedule(schedule)
    state = try
        _survivor_cli_state(
            normalized_schedule,
            season,
            picks,
            initial_strikes,
        )
    catch error
        stale_schedule = schedule === nothing &&
            error isa ArgumentError &&
            error.msg ==
            "a previous pick refers to an uncompleted game"
        stale_schedule || rethrow()
        clear_data_cache()
        normalized_schedule = schedule_loader()
        _survivor_cli_state(
            normalized_schedule,
            season,
            picks,
            initial_strikes,
        )
    end
    return normalized_schedule, state
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

function _survivor_cli_build_forecast_context(
    season::Integer,
    as_of_week::Integer,
    normalized_schedule::AbstractDataFrame,
    historical_drives,
    current_drives,
    cache_directory::Union{Nothing,AbstractString},
    max_seasons::Int,
    refresh_data::Bool,
    log_phase_timing::Function,
)
    historical_source = _survivor_cli_load_historical_drives(
        season,
        max_seasons,
        historical_drives,
        cache_directory;
        refresh=refresh_data,
    )
    historical, current = _load_forecast_drives(
        season,
        max_seasons,
        historical_source,
        current_drives;
        allow_missing_current=as_of_week == 1,
    )
    log_phase_timing(:drive_data)
    historical = _survivor_cli_historical_drives(
        normalized_schedule,
        season,
        historical,
    )
    current = _regular_season_drives(current, normalized_schedule)
    cached_prior = _cached_historical_prior(
        historical;
        current_season=season,
        max_seasons=max_seasons,
        cache_directory=cache_directory,
    )
    log_phase_timing(:prior)
    context = fit_regular_season_forecast(
        season;
        as_of_week=as_of_week,
        schedule=normalized_schedule,
        historical_drives=historical,
        current_drives=current,
        max_seasons=max_seasons,
        prior=cached_prior.prior,
        _normalized_schedule=true,
        _schedule_indexed_drives=true,
    )
    log_phase_timing(:fit)
    return context
end

function _run_survivor_cli(
    args::AbstractVector{<:AbstractString};
    input::IO=stdin,
    output::IO=stdout,
    schedule=nothing,
    historical_drives=nothing,
    current_drives=nothing,
    cache_directory::Union{Nothing,AbstractString}=nothing,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    through_week::Int=18,
    schedule_loader::Function=load_schedule,
    clear_data_cache::Function=NFLData.clear_cache,
    show_team_strength_plot::Function=_show_team_strength_plot,
)
    phase_started = time_ns()
    log_phase_timing = function(phase::Symbol)
        now = time_ns()
        elapsed_seconds = (now - phase_started) / 1.0e9
        phase_started = now
        @debug "survivor phase complete" phase=phase elapsed_seconds=elapsed_seconds
        return nothing
    end

    options = _parse_survivor_cli_args(args)
    log_phase_timing(:parse)
    if options.show_help
        print(output, _survivor_cli_usage())
        return 0
    end
    if options.season === nothing
        if options.refresh_priors
            clear_historical_prior_cache!(; cache_directory=cache_directory)
            @info "historical prior cache cleared"
        end
        if options.refresh_data
            clear_data_cache()
            @info "NFLData raw cache cleared"
            summarized_cache_entries =
                _clear_drive_cache!(; cache_directory=cache_directory)
            @info "summarized historical-drive cache cleared" summarized_cache_entries
        end
        log_phase_timing(:cache_clear)
        return 0
    end

    if options.refresh_priors
        clear_historical_prior_cache!(; cache_directory=cache_directory)
        @info "historical prior cache cleared"
    end
    refresh_data = options.refresh_data || _survivor_cli_refresh_data_enabled()
    refresh_data && clear_data_cache()

    if options.plot_strength_week !== nothing
        normalized_schedule, _ = _survivor_cli_schedule_and_state(
            schedule,
            options.season,
            String[],
            options.initial_strikes;
            schedule_loader=schedule_loader,
            clear_data_cache=clear_data_cache,
        )
        log_phase_timing(:schedule)
        context = _survivor_cli_build_forecast_context(
            options.season,
            options.plot_strength_week,
            normalized_schedule,
            historical_drives,
            current_drives,
            cache_directory,
            max_seasons,
            refresh_data,
            log_phase_timing,
        )
        show_team_strength_plot(context)
        return 0
    end

    picks = _read_survivor_cli_picks(input)
    normalized_schedule, state = _survivor_cli_schedule_and_state(
        schedule,
        options.season,
        picks,
        options.initial_strikes;
        schedule_loader=schedule_loader,
        clear_data_cache=clear_data_cache,
    )
    log_phase_timing(:schedule)
    log_phase_timing(:state)
    context = _survivor_cli_build_forecast_context(
        options.season,
        state.current_week,
        normalized_schedule,
        historical_drives,
        current_drives,
        cache_directory,
        max_seasons,
        refresh_data,
        log_phase_timing,
    )
    if options.grid
        grid = _survivor_grid_data(context; picks_made=state.picks_made)
        _write_survivor_grid(output, grid)
        log_phase_timing(:grid)
        return 0
    end
    selection_config = SurvivorSelectionConfig(
        through_week=through_week,
        banned_first_pick_teams=options.banned_first_pick_teams,
        hessian_weeks=options.hessian_weeks,
        branch_and_bound=options.branch_and_bound,
        timeout_seconds=options.timeout_seconds,
    )
    if options.write_model_file !== nothing
        saved_path = write_survivor_pool_lp(
            options.write_model_file,
            context;
            picks_made=state.picks_made,
            strikes_remaining=state.strikes_remaining,
            selection_config=selection_config,
            include_completed=true,
        )
        log_phase_timing(:model_export)
        @info "survivor MILP saved" path=saved_path
        return 0
    end

    log_phase_timing(:optimize_start)
    solve_plan = () -> optimize_survivor_pool(
        context;
        picks_made=state.picks_made,
        strikes_remaining=state.strikes_remaining,
        include_completed=true,
        selection_config=selection_config,
    )
    plan = solve_plan()
    log_phase_timing(:optimize)
    nrow(plan.current_pick) == 1 ||
        throw(ArgumentError("survivor optimization did not produce one current pick"))
    selected_team = String(plan.current_pick.team[1])
    @info "survivor pick selected" week=state.current_week team=selected_team
    print(output, selected_team, '\n')
    log_phase_timing(:output)
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
