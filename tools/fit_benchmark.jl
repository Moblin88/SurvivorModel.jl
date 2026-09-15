using DataFrames
using Printf

import SurvivorModel

const FIT_BENCHMARK_APP_NAME = "fit_benchmark"

function _fit_benchmark_usage()
    return """
    Usage:
      julia --project=. tools/fit_benchmark.jl [options]

    Options:
      --data SOURCE       Data source: synthetic (default) or real.
      --scenario NAME     Scenario: performance (default) or recovery.
      --season YEAR       Reference season (synthetic default: 2024; required
                          for real data).
      --repeats N         Timed fits per method after one warm-up (default: 3).
      --max-seasons N     Historical fitting window (default: 5).
      --recovery-seasons N
                          Synthetic recovery seasons (default: 3).
      --help              Show this help.
    """
end

function _parse_fit_benchmark_integer(
    value::AbstractString,
    option::AbstractString,
)
    parsed = tryparse(Int, value)
    parsed === nothing &&
        throw(ArgumentError("$option requires an integer value"))
    return parsed
end

function _parse_fit_benchmark_data(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized == "synthetic" && return :synthetic
    normalized == "real" && return :real
    throw(ArgumentError("--data must be synthetic or real"))
end

function _parse_fit_benchmark_scenario(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized == "performance" && return :performance
    normalized == "recovery" && return :recovery
    throw(ArgumentError("--scenario must be performance or recovery"))
end

function _parse_fit_benchmark_args(args::AbstractVector{<:AbstractString})
    season = nothing
    data_source = :synthetic
    data_specified = false
    scenario = :performance
    scenario_specified = false
    repeats = 3
    repeats_specified = false
    max_seasons = SurvivorModel.DEFAULT_HISTORICAL_SEASONS
    max_seasons_specified = false
    recovery_seasons = 3
    recovery_seasons_specified = false
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

        option = nothing
        value = nothing
        if argument in (
            "--data",
            "--scenario",
            "--season",
            "--repeats",
            "--max-seasons",
            "--recovery-seasons",
        )
            option = argument
            index += 1
            index <= length(args) ||
                throw(ArgumentError("$option requires a value"))
            value = String(args[index])
        elseif startswith(argument, "--data=")
            option = "--data"
            value = argument[length("--data=") + 1:end]
        elseif startswith(argument, "--scenario=")
            option = "--scenario"
            value = argument[length("--scenario=") + 1:end]
        elseif startswith(argument, "--season=")
            option = "--season"
            value = argument[length("--season=") + 1:end]
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
        if option == "--data"
            data_specified &&
                throw(ArgumentError("--data may only be specified once"))
            data_source = _parse_fit_benchmark_data(value)
            data_specified = true
        elseif option == "--scenario"
            scenario_specified &&
                throw(ArgumentError("--scenario may only be specified once"))
            scenario = _parse_fit_benchmark_scenario(value)
            scenario_specified = true
        elseif option == "--season"
            season === nothing ||
                throw(ArgumentError("--season may only be specified once"))
            season = _parse_fit_benchmark_integer(value, option)
        elseif option == "--repeats"
            repeats_specified &&
                throw(ArgumentError("--repeats may only be specified once"))
            repeats = _parse_fit_benchmark_integer(value, option)
            repeats_specified = true
        elseif option == "--max-seasons"
            max_seasons_specified &&
                throw(ArgumentError("--max-seasons may only be specified once"))
            max_seasons = _parse_fit_benchmark_integer(value, option)
            max_seasons_specified = true
        else
            recovery_seasons_specified &&
                throw(ArgumentError(
                    "--recovery-seasons may only be specified once",
                ))
            recovery_seasons = _parse_fit_benchmark_integer(value, option)
            recovery_seasons_specified = true
        end
        index += 1
    end

    show_help && return (
        show_help=true,
        data_source=:synthetic,
        scenario=:performance,
        season=0,
        repeats=3,
        max_seasons=SurvivorModel.DEFAULT_HISTORICAL_SEASONS,
        recovery_seasons=3,
    )

    scenario == :recovery && data_source != :synthetic &&
        throw(ArgumentError("the recovery scenario requires synthetic data"))
    recovery_seasons_specified && scenario != :recovery &&
        throw(ArgumentError(
            "--recovery-seasons requires --scenario recovery",
        ))
    data_source == :real && season === nothing &&
        throw(ArgumentError("--season is required with --data real"))
    season = season === nothing ? 2024 : season
    season > 0 || throw(ArgumentError("--season must be positive"))
    repeats > 0 || throw(ArgumentError("--repeats must be positive"))
    max_seasons > 0 || throw(ArgumentError("--max-seasons must be positive"))
    recovery_seasons > 0 ||
        throw(ArgumentError("--recovery-seasons must be positive"))

    return (
        show_help=false,
        data_source=data_source,
        scenario=scenario,
        season=Int(season),
        repeats=Int(repeats),
        max_seasons=Int(max_seasons),
        recovery_seasons=Int(recovery_seasons),
    )
end

function _load_fit_benchmark_drives(options)
    if options.scenario == :recovery
        first_season = options.season - options.recovery_seasons
        last_season = options.season - 1
        first_season > 0 ||
            throw(ArgumentError(
                "recovery benchmark requires enough positive seasons before " *
                "$(options.season)",
            ))
        return SurvivorModel.synthetic_fit_recovery_drives(
            seasons=first_season:last_season,
        )
    end
    options.data_source == :synthetic &&
        return SurvivorModel.synthetic_fit_benchmark_drives()

    first_season = max(1999, options.season - options.max_seasons)
    last_season = options.season - 1
    first_season <= last_season ||
        throw(ArgumentError(
            "real benchmark data requires at least one season before " *
            "$(options.season)",
        ))
    drives = DataFrame[
        SurvivorModel.load_drive_pbp(data_season) for
        data_season in first_season:last_season
    ]
    return vcat(drives...; cols=:union)
end

function _fit_benchmark_season_label(drives::AbstractDataFrame)
    seasons = Int[]
    for index in 1:nrow(drives)
        season = SurvivorModel._drive_season(drives, index)
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

function _print_fit_table(
    headers::AbstractVector{<:AbstractString},
    values::Vector{<:AbstractVector{<:AbstractString}};
    output::IO=stdout,
    left_columns::Int=1,
)
    isempty(values) && return nothing
    widths = [
        maximum(length(values[row_index][column]) for row_index in eachindex(values))
        for column in eachindex(headers)
    ]
    widths = max.(widths, length.(headers))
    for row_values in (headers, values...)
        for column in eachindex(row_values)
            column > 1 && print(output, "  ")
            if column <= left_columns
                print(output, rpad(row_values[column], widths[column]))
            else
                print(output, lpad(row_values[column], widths[column]))
            end
        end
        println(output)
        row_values === headers &&
            println(output, join(["-"^width for width in widths], "  "))
    end
    return nothing
end

function _print_fit_benchmark_table(
    result::AbstractDataFrame;
    output::IO=stdout,
    data_source::Symbol,
    drives::AbstractDataFrame,
    current_season::Integer,
    repeats::Integer,
    scenario::Symbol,
)
    println(output, "Fitting benchmark")
    println(output, "data: ", data_source)
    println(output, "scenario: ", scenario)
    println(output, "reference season: ", current_season)
    println(output, "drive rows: ", nrow(drives))
    println(output, "observed seasons: ", _fit_benchmark_season_label(drives))
    println(output, "timed repeats: ", repeats, " (one warm-up per method)")
    println(output)

    values = [
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
        ]
        for row in eachrow(result)
    ]
    _print_fit_table(
        [
            "method",
            "median_ms",
            "total_logLik",
            "gap",
            "ok",
            "status",
            "iterations",
            "evals",
            "boundary",
        ],
        values;
        output=output,
    )
    for row in eachrow(result)
        isempty(row.error) || println(output, "  ", row.method, ": ", row.error)
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
    println(output, "Recovery quality (absolute errors; relative for rate moments)")
    _print_fit_table(
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
        left_columns=2,
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
    println(output)
    println(output, "Shared-parameter recovery (estimate, truth, estimate - truth)")
    _print_fit_table(
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
        left_columns=2,
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
    println(output)
    println(output, "Per-bin hazard recovery (estimate, truth, relative error)")
    _print_fit_table(
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
        left_columns=2,
    )
    return nothing
end

function _print_fit_benchmark_parameters(
    parameters::AbstractDataFrame;
    output::IO=stdout,
    data_source::Symbol,
)
    isempty(parameters) && return nothing
    println(output)
    println(output, "Fitted hazard parameters")
    println(output, "home_advantage is the outcome-specific home hazard multiplier")
    if data_source === :synthetic
        println(
            output,
            "synthetic target: home_advantage=",
            @sprintf(
                "%.2f",
                SurvivorModel.FIT_BENCHMARK_SYNTHETIC_HOME_MULTIPLIER,
            ),
            ", persistence=",
            @sprintf(
                "%.2f",
                SurvivorModel.FIT_BENCHMARK_SYNTHETIC_PERSISTENCE,
            ),
        )
    end
    values = [
        [
            string(row.method),
            string(row.kind),
            string(row.time_bin),
            _fit_benchmark_number(row.home_advantage, :fixed3),
            _fit_benchmark_number(row.persistence, :fixed3),
            _fit_benchmark_number(row.hazard_mean, :general3),
            _fit_benchmark_number(row.hazard_variance, :general3),
        ]
        for row in eachrow(parameters)
    ]
    _print_fit_table(
        [
            "method",
            "outcome",
            "bin",
            "home_advantage",
            "persistence",
            "hazard_mean",
            "hazard_variance",
        ],
        values;
        output=output,
        left_columns=2,
    )
    return nothing
end

function _run_fit_benchmark(options; output::IO=stdout)
    drives = _load_fit_benchmark_drives(options)
    if options.scenario == :recovery
        benchmark = SurvivorModel.fit_recovery_benchmark(
            drives.drives,
            drives.truth;
            current_season=options.season,
            max_seasons=options.max_seasons,
            repeats=options.repeats,
            methods=SurvivorModel.FIT_BENCHMARK_METHODS,
        )
        _print_fit_benchmark_table(
            benchmark.summary;
            output=output,
            data_source=:synthetic,
            drives=drives.drives,
            current_season=options.season,
            repeats=options.repeats,
            scenario=:recovery,
        )
        _print_fit_recovery_results(benchmark; output=output)
        return 0
    end

    benchmark = SurvivorModel._fit_benchmark_with_parameters(
        drives;
        current_season=options.season,
        max_seasons=options.max_seasons,
        repeats=options.repeats,
        methods=SurvivorModel.FIT_BENCHMARK_METHODS,
    )
    _print_fit_benchmark_table(
        benchmark.summary;
        output=output,
        data_source=options.data_source,
        drives=drives,
        current_season=options.season,
        repeats=options.repeats,
        scenario=:performance,
    )
    _print_fit_benchmark_parameters(
        benchmark.parameters;
        output=output,
        data_source=options.data_source,
    )
    return 0
end

function _fit_benchmark_main(args)
    options = _parse_fit_benchmark_args(args)
    if options.show_help
        print(_fit_benchmark_usage())
        return 0
    end
    return _run_fit_benchmark(options)
end

try
    exit(_fit_benchmark_main(ARGS))
catch error
    if error isa ArgumentError
        println(stderr, "$FIT_BENCHMARK_APP_NAME: ", sprint(showerror, error))
        exit(2)
    end
    rethrow()
end
