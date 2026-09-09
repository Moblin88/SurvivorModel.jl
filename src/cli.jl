const SURVIVOR_CLI_APP_NAME = "survivor"

function _survivor_cli_usage()
    return """
    Usage:
      survivor --season YEAR [--strikes N] < picks.txt
      julia --project=. -m SurvivorModel --season YEAR [--strikes N] < picks.txt
      survivor --benchmark [--data synthetic|real] [--season YEAR]

    Input:
      One team abbreviation per nonblank line, starting with week 1.
      The next pick is inferred from the number of nonblank lines.

    Options:
      --season YEAR       Target season. Required for survivor mode and real
                          benchmark data; synthetic benchmarks default to 2024.
      --strikes N         Initial loss allowance (default: 2).
      --benchmark         Compare all historical fitting methods.
      --data SOURCE       Benchmark data source: synthetic (default) or real.
      --scenario NAME     Benchmark scenario: performance (default) or recovery.
      --repeats N         Timed fits per method after one warm-up (default: 3).
      --max-seasons N     Historical fitting window (default: 5).
      --recovery-seasons N
                          Synthetic recovery seasons (default: 3).
      --clear-cache       Remove cached historical fits before running, then
                          exit successfully if no other mode is selected.
      --help              Show this help.
    """
end

function _parse_survivor_cli_integer(value::AbstractString, option::AbstractString)
    parsed = tryparse(Int, value)
    parsed === nothing &&
        throw(ArgumentError("$option requires an integer value"))
    return parsed
end

function _parse_survivor_cli_data(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized == "synthetic" && return :synthetic
    normalized == "real" && return :real
    throw(ArgumentError("--data must be synthetic or real"))
end

function _parse_survivor_cli_args(args::AbstractVector{<:AbstractString})
    season = nothing
    initial_strikes = 2
    strikes_specified = false
    benchmark = false
    benchmark_specified = false
    data_source = :synthetic
    data_specified = false
    benchmark_scenario = :performance
    scenario_specified = false
    repeats = 3
    repeats_specified = false
    max_seasons = DEFAULT_HISTORICAL_SEASONS
    max_seasons_specified = false
    recovery_seasons = 3
    recovery_seasons_specified = false
    clear_cache = false
    clear_cache_specified = false
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

        if argument == "--benchmark"
            benchmark_specified &&
                throw(ArgumentError("--benchmark may only be specified once"))
            benchmark = true
            benchmark_specified = true
            index += 1
            continue
        end
        if argument == "--clear-cache"
            clear_cache_specified &&
                throw(ArgumentError("--clear-cache may only be specified once"))
            clear_cache = true
            clear_cache_specified = true
            index += 1
            continue
        end

        option = nothing
        value = nothing
        if argument in (
            "--season",
            "--strikes",
            "--data",
            "--scenario",
            "--repeats",
            "--max-seasons",
            "--recovery-seasons",
        )
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
        elseif startswith(argument, "--data=")
            option = "--data"
            value = argument[length("--data=") + 1:end]
        elseif startswith(argument, "--scenario=")
            option = "--scenario"
            value = argument[length("--scenario=") + 1:end]
        elseif startswith(argument, "--repeats=")
            option = "--repeats"
            value = argument[length("--repeats=") + 1:end]
        elseif startswith(argument, "--max-seasons=")
            option = "--max-seasons"
            value = argument[length("--max-seasons=") + 1:end]
        elseif startswith(argument, "--recovery-seasons=")
            option = "--recovery-seasons"
            value = argument[length("--recovery-seasons=") + 1:end]
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
        elseif option == "--data"
            data_specified &&
                throw(ArgumentError("--data may only be specified once"))
            data_source = _parse_survivor_cli_data(value)
            data_specified = true
        elseif option == "--scenario"
            scenario_specified &&
                throw(ArgumentError("--scenario may only be specified once"))
            normalized = lowercase(strip(value))
            if normalized == "performance"
                benchmark_scenario = :performance
            elseif normalized == "recovery"
                benchmark_scenario = :recovery
            else
                throw(ArgumentError(
                    "--scenario must be performance or recovery",
                ))
            end
            scenario_specified = true
        elseif option == "--repeats"
            parsed = _parse_survivor_cli_integer(value, option)
            repeats_specified &&
                throw(ArgumentError("--repeats may only be specified once"))
            repeats = parsed
            repeats_specified = true
        else
            parsed = _parse_survivor_cli_integer(value, option)
            if option == "--max-seasons"
                max_seasons_specified &&
                    throw(ArgumentError("--max-seasons may only be specified once"))
                max_seasons = parsed
                max_seasons_specified = true
            else
                recovery_seasons_specified &&
                    throw(ArgumentError(
                        "--recovery-seasons may only be specified once",
                    ))
                recovery_seasons = parsed
                recovery_seasons_specified = true
            end
        end
        index += 1
    end

    show_help && return (
        show_help=true,
        benchmark=false,
        data_source=:synthetic,
        benchmark_scenario=:performance,
        season=0,
        initial_strikes=2,
        repeats=3,
        max_seasons=DEFAULT_HISTORICAL_SEASONS,
        recovery_seasons=3,
        clear_cache=clear_cache,
    )
    !benchmark && (
        data_specified ||
        scenario_specified ||
        repeats_specified ||
        max_seasons_specified ||
        recovery_seasons_specified
    ) &&
        throw(ArgumentError(
            "--data, --scenario, --repeats, --max-seasons, and " *
            "--recovery-seasons " *
            "require --benchmark",
        ))
    !benchmark && clear_cache && season === nothing && strikes_specified &&
        throw(ArgumentError(
            "--strikes requires --season when used with --clear-cache",
        ))
    benchmark && strikes_specified &&
        throw(ArgumentError("--strikes cannot be used with --benchmark"))
    benchmark_scenario == :recovery && data_source != :synthetic &&
        throw(ArgumentError("the recovery scenario requires synthetic data"))
    recovery_seasons_specified && benchmark_scenario != :recovery &&
        throw(ArgumentError(
            "--recovery-seasons requires --scenario recovery",
        ))
    if benchmark
        data_source == :real && season === nothing &&
            throw(ArgumentError("--season is required with --data real"))
        season = season === nothing ? 2024 : season
    elseif !clear_cache
        season === nothing && throw(ArgumentError("--season is required"))
    end
    clear_cache || season > 0 || throw(ArgumentError("--season must be positive"))
    initial_strikes >= 0 ||
        throw(ArgumentError("--strikes must be nonnegative"))
    repeats > 0 || throw(ArgumentError("--repeats must be positive"))
    max_seasons > 0 || throw(ArgumentError("--max-seasons must be positive"))
    recovery_seasons > 0 ||
        throw(ArgumentError("--recovery-seasons must be positive"))
    season_value = season === nothing ? 0 : Int(season)
    return (
        show_help=false,
        benchmark=benchmark,
        data_source=data_source,
        benchmark_scenario=benchmark_scenario,
        season=season_value,
        initial_strikes=Int(initial_strikes),
        repeats=Int(repeats),
        max_seasons=Int(max_seasons),
        recovery_seasons=Int(recovery_seasons),
        clear_cache=clear_cache,
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
            "$initial_strikes-strike allowance",
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

function _load_fit_benchmark_drives(options)
    if options.benchmark_scenario == :recovery
        first_season = options.season - options.recovery_seasons
        last_season = options.season - 1
        first_season > 0 ||
            throw(ArgumentError(
                "recovery benchmark requires enough positive seasons before " *
                "$(options.season)",
            ))
        return synthetic_fit_recovery_drives(
            seasons=first_season:last_season,
        )
    end
    options.data_source == :synthetic &&
        return synthetic_fit_benchmark_drives()

    first_season = max(1999, options.season - options.max_seasons)
    last_season = options.season - 1
    first_season <= last_season ||
        throw(ArgumentError(
            "real benchmark data requires at least one season before " *
            "$(options.season)",
        ))
    return load_drive_pbp(first_season:last_season)
end

function _fit_benchmark_season_label(drives::AbstractDataFrame)
    seasons = Int[]
    for index in 1:nrow(drives)
        season = _drive_season(drives, index)
        ismissing(season) || push!(seasons, Int(season))
    end
    isempty(seasons) && return "unknown"
    return string(minimum(seasons), "-", maximum(seasons))
end

function _fit_benchmark_number(value::Real, format::Symbol)
    isfinite(value) || return "NA"
    format === :fixed2 && return @sprintf("%.2f", value)
    format === :fixed3 && return @sprintf("%.3f", value)
    format === :general3 && return @sprintf("%.3g", value)
    format === :signed3 && return @sprintf("%+.3g", value)
    throw(ArgumentError("unknown benchmark number format: $format"))
end

function _print_fit_benchmark_table(
    result::AbstractDataFrame;
    output::IO=stdout,
    data_source::Symbol,
    drives::AbstractDataFrame,
    current_season::Integer,
    repeats::Integer,
    benchmark_scenario::Symbol=:performance,
)
    println(output, "Fitting benchmark")
    println(output, "data: ", data_source)
    println(output, "scenario: ", benchmark_scenario)
    println(output, "reference season: ", current_season)
    println(output, "drive rows: ", nrow(drives))
    println(output, "observed seasons: ", _fit_benchmark_season_label(drives))
    println(output, "timed repeats: ", repeats, " (one warm-up per method)")
    println(output)

    headers = [
        "method",
        "median_ms",
        "total_logLik",
        "gap",
        "ok",
        "status",
        "iterations",
        "evals",
        "boundary",
    ]
    values = Vector{Vector{String}}()
    for row in eachrow(result)
        push!(
            values,
            [
                string(row.method),
                _fit_benchmark_number(row.median_ms, :fixed2),
                _fit_benchmark_number(row.total_log_likelihood, :fixed3),
                _fit_benchmark_number(row.likelihood_gap, :general3),
                row.converged ? "yes" : "no",
                string(row.status),
                string(row.iterations),
                string(row.function_evaluations),
                string(row.boundary_parameters),
            ],
        )
    end
    widths = [
        maximum(length(values[row_index][column]) for row_index in eachindex(values))
        for column in eachindex(headers)
    ]
    widths = max.(widths, length.(headers))
    print_row = function(row_values)
        for column in eachindex(row_values)
            column > 1 && print(output, "  ")
            if column == 1
                print(output, rpad(row_values[column], widths[column]))
            else
                print(output, lpad(row_values[column], widths[column]))
            end
        end
        println(output)
    end
    print_row(headers)
    println(output, join(["-"^width for width in widths], "  "))
    for row_values in values
        print_row(row_values)
    end
    for row in eachrow(result)
        isempty(row.error) || println(output, "  ", row.method, ": ", row.error)
    end
    return nothing
end

function _print_fit_recovery_table(
    title::AbstractString,
    headers::AbstractVector{<:AbstractString},
    values::Vector{<:AbstractVector{<:AbstractString}};
    output::IO=stdout,
    left_columns::Int=2,
)
    isempty(values) && return nothing
    println(output)
    println(output, title)
    widths = [
        maximum(length(values[row_index][column]) for row_index in eachindex(values))
        for column in eachindex(headers)
    ]
    widths = max.(widths, length.(headers))
    print_row = function(row_values)
        for column in eachindex(row_values)
            column > 1 && print(output, "  ")
            if column <= left_columns
                print(output, rpad(row_values[column], widths[column]))
            else
                print(output, lpad(row_values[column], widths[column]))
            end
        end
        println(output)
    end
    print_row(headers)
    println(output, join(["-"^width for width in widths], "  "))
    for row_values in values
        print_row(row_values)
    end
    return nothing
end

function _print_fit_recovery_truth(truth; output::IO=stdout)
    println(output)
    println(
        output,
        "recovery truth: ",
        length(truth.seasons),
        " seasons, ",
        length(truth.teams),
        " teams, ",
        length(truth.td_hyperparameters),
        " time bins",
    )
    if hasproperty(truth, :games_per_team)
        println(
            output,
            "schedule: ",
            truth.games_per_team,
            " games per team per season, horizon=",
            truth.horizon_seconds,
            " seconds",
        )
    end
    println(
        output,
        "td home=",
        _fit_benchmark_number(truth.td_home_multiplier, :fixed3),
        ", td persistence=",
        _fit_benchmark_number(truth.td_persistence, :fixed3),
        ", defensive home=",
        _fit_benchmark_number(truth.defensive_home_multiplier, :fixed3),
        ", defensive persistence=",
        _fit_benchmark_number(truth.defensive_persistence, :fixed3),
    )
    return nothing
end

function _print_fit_recovery_results(benchmark; output::IO=stdout)
    _print_fit_recovery_truth(benchmark.truth; output=output)

    quality_values = [
        [
            string(row.method),
            string(row.kind),
            _fit_benchmark_number(row.mean_abs_relative_error, :general3),
            _fit_benchmark_number(row.variance_abs_relative_error, :general3),
            _fit_benchmark_number(row.home_abs_error, :general3),
            _fit_benchmark_number(row.persistence_abs_error, :general3),
            _fit_benchmark_number(row.max_abs_relative_error, :general3),
        ]
        for row in eachrow(benchmark.recovery_quality)
    ]
    _print_fit_recovery_table(
        "Recovery quality (absolute errors; relative for rate moments)",
        [
            "method",
            "outcome",
            "mean_abs_rel_err",
            "variance_abs_rel_err",
            "home_abs_err",
            "persistence_abs_err",
            "max_abs_rel_err",
        ],
        quality_values;
        output=output,
    )

    shared_values = [
        [
            string(row.method),
            string(row.kind),
            _fit_benchmark_number(row.fitted_home_advantage, :fixed3),
            _fit_benchmark_number(row.target_home_advantage, :fixed3),
            _fit_benchmark_number(row.home_advantage_error, :signed3),
            _fit_benchmark_number(row.fitted_persistence, :fixed3),
            _fit_benchmark_number(row.target_persistence, :fixed3),
            _fit_benchmark_number(row.persistence_error, :signed3),
        ]
        for row in eachrow(benchmark.recovery_shared)
    ]
    _print_fit_recovery_table(
        "Shared-parameter recovery (estimate, truth, estimate - truth)",
        [
            "method",
            "outcome",
            "home_est",
            "home_truth",
            "home_error",
            "persist_est",
            "persist_truth",
            "persist_error",
        ],
        shared_values;
        output=output,
    )

    bin_values = [
        [
            string(row.method),
            string(row.kind),
            string(row.time_bin),
            _fit_benchmark_number(row.fitted_mean, :general3),
            _fit_benchmark_number(row.target_mean, :general3),
            _fit_benchmark_number(row.mean_relative_error, :signed3),
            _fit_benchmark_number(row.fitted_variance, :general3),
            _fit_benchmark_number(row.target_variance, :general3),
            _fit_benchmark_number(row.variance_relative_error, :signed3),
        ]
        for row in eachrow(benchmark.recovery_bins)
    ]
    _print_fit_recovery_table(
        "Per-bin hazard recovery (estimate, truth, relative error)",
        [
            "method",
            "outcome",
            "bin",
            "mean_est",
            "mean_truth",
            "mean_rel_err",
            "variance_est",
            "variance_truth",
            "variance_rel_err",
        ],
        bin_values;
        output=output,
    )
    return nothing
end

function _print_fit_benchmark_parameters(
    parameters::AbstractDataFrame;
    output::IO=stdout,
    data_source::Symbol=:real,
)
    isempty(parameters) && return nothing
    println(output)
    println(output, "Fitted hazard parameters")
    println(output, "home_advantage is the outcome-specific home hazard multiplier")
    if data_source === :synthetic
        println(
            output,
            "synthetic target: home_advantage=",
            @sprintf("%.2f", FIT_BENCHMARK_SYNTHETIC_HOME_MULTIPLIER),
            ", persistence=",
            @sprintf("%.2f", FIT_BENCHMARK_SYNTHETIC_PERSISTENCE),
        )
    end
    headers = [
        "method",
        "outcome",
        "bin",
        "home_advantage",
        "persistence",
        "hazard_mean",
        "hazard_variance",
    ]
    values = Vector{Vector{String}}()
    for row in eachrow(parameters)
        push!(
            values,
            [
                string(row.method),
                string(row.kind),
                string(row.time_bin),
                _fit_benchmark_number(row.home_advantage, :fixed3),
                _fit_benchmark_number(row.persistence, :fixed3),
                _fit_benchmark_number(row.hazard_mean, :general3),
                _fit_benchmark_number(row.hazard_variance, :general3),
            ],
        )
    end
    widths = [
        maximum(length(values[row_index][column]) for row_index in eachindex(values))
        for column in eachindex(headers)
    ]
    widths = max.(widths, length.(headers))
    print_row = function(row_values)
        for column in eachindex(row_values)
            column > 1 && print(output, "  ")
            if column <= 2
                print(output, rpad(row_values[column], widths[column]))
            else
                print(output, lpad(row_values[column], widths[column]))
            end
        end
        println(output)
    end
    print_row(headers)
    println(output, join(["-"^width for width in widths], "  "))
    for row_values in values
        print_row(row_values)
    end
    return nothing
end

function _run_fit_benchmark_cli(
    options;
    output::IO=stdout,
    drives=nothing,
    truth=nothing,
    methods=FIT_BENCHMARK_METHODS,
)
    if options.benchmark_scenario == :recovery
        simulation = if drives === nothing
            _load_fit_benchmark_drives(options)
        else
            truth === nothing &&
                throw(ArgumentError(
                    "recovery benchmark drives require matching truth",
                ))
            (drives=drives, truth=truth)
        end
        benchmark = fit_recovery_benchmark(
            simulation.drives,
            simulation.truth;
            current_season=options.season,
            max_seasons=options.max_seasons,
            repeats=options.repeats,
            methods=methods,
        )
        _print_fit_benchmark_table(
            benchmark.summary;
            output=output,
            data_source=:synthetic,
            drives=simulation.drives,
            current_season=options.season,
            repeats=options.repeats,
            benchmark_scenario=:recovery,
        )
        _print_fit_recovery_results(benchmark; output=output)
        return 0
    end

    benchmark_drives = drives === nothing ?
        _load_fit_benchmark_drives(options) :
        drives
    benchmark = _fit_benchmark_with_parameters(
        benchmark_drives;
        current_season=options.season,
        max_seasons=options.max_seasons,
        repeats=options.repeats,
        methods=methods,
    )
    _print_fit_benchmark_table(
        benchmark.summary;
        output=output,
        data_source=options.data_source,
        drives=benchmark_drives,
        current_season=options.season,
        repeats=options.repeats,
        benchmark_scenario=:performance,
    )
    _print_fit_benchmark_parameters(
        benchmark.parameters;
        output=output,
        data_source=options.data_source,
    )
    return 0
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
)
    options = _parse_survivor_cli_args(args)
    if options.show_help
        print(output, _survivor_cli_usage())
        return 0
    end
    options.clear_cache && clear_historical_prior_cache!()
    !options.benchmark && options.clear_cache && options.season == 0 &&
        return 0
    options.benchmark &&
        return _run_fit_benchmark_cli(options; output=output)

    picks = _read_survivor_cli_picks(input)
    normalized_schedule = schedule === nothing ? load_schedule() : load_schedule(schedule)
    state = _survivor_cli_state(
        normalized_schedule,
        options.season,
        picks,
        options.initial_strikes,
    )
    historical, current = _load_forecast_drives(
        options.season,
        max_seasons,
        historical_drives,
        current_drives,
        ;
        allow_missing_current=state.current_week == 1,
    )
    historical = _survivor_cli_historical_drives(
        normalized_schedule,
        options.season,
        historical,
    )
    cached_prior = _cached_historical_prior(
        historical;
        current_season=options.season,
        time_edges=DEFAULT_TIME_EDGES,
        max_seasons=max_seasons,
        method=method,
        cache_directory=cache_directory,
    )
    context = fit_regular_season_forecast(
        options.season;
        as_of_week=state.current_week,
        schedule=normalized_schedule,
        historical_drives=historical,
        current_drives=current,
        max_seasons=max_seasons,
        method=method,
        prior=cached_prior.prior,
    )
    plan = optimize_survivor_pool(
        context;
        picks_made=state.picks_made,
        strikes_remaining=state.strikes_remaining,
        include_completed=true,
        through_week=through_week,
    )
    nrow(plan.current_pick) == 1 ||
        throw(ArgumentError("survivor optimization did not produce one current pick"))
    print(output, String(plan.current_pick.team[1]), '\n')
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
