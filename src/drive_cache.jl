const DRIVE_CACHE_SCHEMA_VERSION = 2
const DRIVE_SUMMARY_SCHEMA_VERSION = 1
const DRIVE_CACHE_DIRECTORY_NAME = "summarized_drives"
const DEFAULT_DRIVE_CACHE_CONFIG = (
    summary_schema=DRIVE_SUMMARY_SCHEMA_VERSION,
)
const DRIVE_SUMMARY_REQUIRED_COLUMNS = (
    :game_id,
    :fixed_drive,
    :posteam,
    :defteam,
    :posteam_home,
    :defteam_home,
    :drive_result,
    :time_of_possession,
    :drive_start_yards_to_goal,
    :yards_gained,
    :home_spread_change,
)

struct DriveCacheEntry
    schema_version::Int
    season::Int
    config::NamedTuple
    source_fingerprint::String
    source_timestamp::Union{Nothing,DateTime}
    data_fingerprint::String
    cached_at::DateTime
    drives::DataFrame
end

function _validate_drive_cache_season(season::Integer)
    season > 0 || throw(ArgumentError("season must be positive"))
    return Int(season)
end

function _normalize_drive_cache_config(config)
    config isa NamedTuple ||
        throw(ArgumentError("drive cache config must be a NamedTuple"))

    names = Tuple(sort!(collect(propertynames(config)); by=String))
    values = ntuple(index -> getproperty(config, names[index]), length(names))
    normalized = NamedTuple{names}(values)

    io = IOBuffer()
    try
        serialize(io, normalized)
    catch error
        throw(ArgumentError(
            "drive cache config must be serializable: " *
            sprint(showerror, error),
        ))
    end
    return normalized
end

function _drive_cache_directory()
    return Scratch.get_scratch!(@__MODULE__, DRIVE_CACHE_DIRECTORY_NAME)
end

function _drive_cache_path(cache_directory::AbstractString, season::Integer)
    season_value = _validate_drive_cache_season(season)
    return joinpath(
        String(cache_directory),
        "drive_summaries_v$(DRIVE_CACHE_SCHEMA_VERSION)_season$(season_value).jls",
    )
end

function _drive_summary_validation_error(
    drives,
    season::Union{Nothing,Integer}=nothing,
)
    drives isa AbstractDataFrame ||
        return "cached drives must be an AbstractDataFrame"

    columns = Set(Symbol.(names(drives)))
    missing_columns = setdiff(collect(DRIVE_SUMMARY_REQUIRED_COLUMNS), columns)
    isempty(missing_columns) ||
        return "summarized drives are missing required columns: " *
            join(string.(missing_columns), ", ")

    if season !== nothing && :season in columns
        for value in skipmissing(drives[!, :season])
            matches = value isa Integer && Int(value) == Int(season)
            if !matches && value isa Real
                matches = isfinite(value) && value == season
            end
            matches || return "drive data contains a season other than $(Int(season))"
        end
    end
    return nothing
end

function _validate_drive_summary(
    drives,
    season::Union{Nothing,Integer}=nothing,
)
    error_message = _drive_summary_validation_error(drives, season)
    isnothing(error_message) || throw(ArgumentError(error_message))
    return drives
end

function _drive_cache_fingerprint(drives::AbstractDataFrame)
    _validate_drive_summary(drives)
    data = DataFrame(drives)
    io = IOBuffer()
    print(io, "rows=", nrow(data), '\0', "columns=", ncol(data), '\0')
    for name in names(data)
        print(io, "column=", name, '\0', "eltype=", eltype(data[!, name]), '\0')
    end
    for row in eachrow(data)
        for name in names(data)
            value = row[name]
            if ismissing(value)
                print(io, "missing")
            else
                print(io, typeof(value), '=')
                show(io, value)
            end
            write(io, UInt8(0))
        end
    end
    bytes = take!(io)

    state = UInt64(0xcbf29ce484222325)
    prime = UInt64(0x100000001b3)
    for byte in bytes
        state = (state ⊻ UInt64(byte)) * prime
    end
    return lowercase(string(state, base=16, pad=16))
end

function _normalize_drive_cache_fingerprint(value, label::AbstractString)
    value === nothing && return nothing
    value isa AbstractString ||
        throw(ArgumentError("$label must be a string or nothing"))
    normalized = String(value)
    isempty(normalized) && throw(ArgumentError("$label must not be empty"))
    return normalized
end

function _normalize_drive_cache_timestamp(value, label::AbstractString)
    value === nothing && return nothing
    value isa DateTime ||
        throw(ArgumentError("$label must be a DateTime or nothing"))
    return value
end

function _drive_cache_loader_result(result)
    if result isa AbstractDataFrame
        return (
            drives=DataFrame(result),
            source_fingerprint=nothing,
            source_timestamp=nothing,
        )
    end

    result isa NamedTuple && hasproperty(result, :drives) ||
        throw(ArgumentError(
            "drive cache loader must return an AbstractDataFrame or " *
            "a NamedTuple with a drives field",
        ))
    drives = getproperty(result, :drives)
    _validate_drive_summary(drives)

    fingerprint = if hasproperty(result, :source_fingerprint)
        getproperty(result, :source_fingerprint)
    elseif hasproperty(result, :fingerprint)
        getproperty(result, :fingerprint)
    else
        nothing
    end
    source_timestamp = if hasproperty(result, :source_timestamp)
        getproperty(result, :source_timestamp)
    elseif hasproperty(result, :fetched_at)
        getproperty(result, :fetched_at)
    else
        nothing
    end
    return (
        drives=DataFrame(drives),
        source_fingerprint=_normalize_drive_cache_fingerprint(
            fingerprint,
            "source_fingerprint",
        ),
        source_timestamp=_normalize_drive_cache_timestamp(
            source_timestamp,
            "source_timestamp",
        ),
    )
end

function _load_drive_cache_source(loader::Function, season::Int)
    result = if applicable(loader, season)
        loader(season)
    elseif applicable(loader)
        loader()
    else
        throw(ArgumentError(
            "drive cache loader must accept the season or no arguments",
        ))
    end
    return _drive_cache_loader_result(result)
end

function _read_drive_cache(path::AbstractString)
    isfile(path) || return nothing
    try
        entry = open(path, "r") do io
            deserialize(io)
        end
        entry isa DriveCacheEntry || throw(ArgumentError(
            "serialized value has type $(typeof(entry)), expected DriveCacheEntry",
        ))
        return entry
    catch error
        throw(ArgumentError(
            "could not read drive cache $(path): " * sprint(showerror, error),
        ))
    end
end

function _drive_cache_entry_matches(
    entry,
    season::Int,
    config::NamedTuple,
    expected_source_fingerprint,
)
    entry isa DriveCacheEntry || return false
    entry.schema_version == DRIVE_CACHE_SCHEMA_VERSION || return false
    entry.season == season || return false
    isequal(entry.config, config) || return false
    isnothing(_drive_summary_validation_error(entry.drives, season)) || return false
    entry.data_fingerprint == _drive_cache_fingerprint(entry.drives) || return false
    expected_source_fingerprint === nothing ||
        entry.source_fingerprint == expected_source_fingerprint || return false
    return true
end

function _write_drive_cache(path::AbstractString, entry::DriveCacheEntry)
    cache_directory = dirname(path)
    mkpath(cache_directory)
    temporary_path = tempname(cache_directory)
    try
        open(temporary_path, "w") do io
            serialize(io, entry)
            flush(io)
        end
        mv(temporary_path, path; force=true)
    finally
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return entry
end

function _drive_cache_result(
    entry::DriveCacheEntry,
    path::AbstractString,
    cache_hit::Bool,
)
    metadata = (
        schema_version=entry.schema_version,
        season=entry.season,
        config=entry.config,
        source_fingerprint=entry.source_fingerprint,
        source_timestamp=entry.source_timestamp,
        data_fingerprint=entry.data_fingerprint,
        cached_at=entry.cached_at,
    )
    return (
        drives=entry.drives,
        cache_hit=cache_hit,
        path=String(path),
        metadata=metadata,
        schema_version=entry.schema_version,
        season=entry.season,
        config=entry.config,
        source_fingerprint=entry.source_fingerprint,
        source_timestamp=entry.source_timestamp,
        data_fingerprint=entry.data_fingerprint,
        cached_at=entry.cached_at,
    )
end

"""
    _cached_season_drives(season; kwargs...) -> NamedTuple

Load one season of summarized drive data from the package's Scratch-backed
cache. `refresh=true` always invokes `loader` and atomically replaces the
season's cache entry. `source_fingerprint` can be supplied by a caller that
knows the upstream data version; a mismatch is treated as a cache miss.

`loader` normally accepts the season and returns a `DataFrame`. It may instead
return `(drives=data, source_fingerprint=..., source_timestamp=...)` to
record upstream freshness metadata. A zero-argument loader is also accepted
for test and adapter use.
"""
function _cached_season_drives(
    season::Integer;
    refresh::Bool=false,
    source_fingerprint=nothing,
    cache_config=DEFAULT_DRIVE_CACHE_CONFIG,
    cache_directory::Union{Nothing,AbstractString}=nothing,
    loader::Function=load_drive_pbp,
)
    season_value = _validate_drive_cache_season(season)
    config = _normalize_drive_cache_config(cache_config)
    expected_fingerprint = _normalize_drive_cache_fingerprint(
        source_fingerprint,
        "source_fingerprint",
    )
    directory = cache_directory === nothing ?
        _drive_cache_directory() :
        String(cache_directory)
    path = _drive_cache_path(directory, season_value)

    if !refresh
        entry = _read_drive_cache(path)
        if _drive_cache_entry_matches(
            entry,
            season_value,
            config,
            expected_fingerprint,
        )
            return _drive_cache_result(entry, path, true)
        end
    end

    loaded = _load_drive_cache_source(loader, season_value)
    _validate_drive_summary(loaded.drives, season_value)
    loaded_fingerprint = loaded.source_fingerprint
    if expected_fingerprint !== nothing &&
        loaded_fingerprint !== nothing &&
        loaded_fingerprint != expected_fingerprint
        throw(ArgumentError(
            "drive cache loader returned source_fingerprint " *
            "different from the requested fingerprint",
        ))
    end
    data_fingerprint = _drive_cache_fingerprint(loaded.drives)
    entry = DriveCacheEntry(
        DRIVE_CACHE_SCHEMA_VERSION,
        season_value,
        config,
        something(
            loaded_fingerprint,
            something(expected_fingerprint, data_fingerprint),
        ),
        loaded.source_timestamp,
        data_fingerprint,
        now(),
        loaded.drives,
    )
    _write_drive_cache(path, entry)
    return _drive_cache_result(entry, path, false)
end

function _drive_cache_files(
    directory::AbstractString;
    season::Union{Nothing,Integer}=nothing,
)
    isdir(directory) || return String[]
    season_suffix = season === nothing ? nothing : "_season$(Int(season)).jls"
    paths = String[]
    for path in readdir(directory; join=true)
        filename = basename(path)
        startswith(filename, "drive_summaries_v") || continue
        endswith(filename, ".jls") || continue
        season_suffix === nothing || endswith(filename, season_suffix) || continue
        isfile(path) && push!(paths, path)
    end
    return paths
end

"""
    _clear_drive_cache!(; season=nothing, cache_directory=nothing) -> Int

Remove cached summarized drives. With `season`, only that season's versioned
entries are removed; without it, all package-owned drive-cache entries are
removed. The return value is the number of files removed.
"""
function _clear_drive_cache!(
    ;
    season::Union{Nothing,Integer}=nothing,
    cache_directory::Union{Nothing,AbstractString}=nothing,
)
    season_value = season === nothing ? nothing : _validate_drive_cache_season(season)
    directory = cache_directory === nothing ?
        _drive_cache_directory() :
        String(cache_directory)
    paths = _drive_cache_files(directory; season=season_value)
    for path in paths
        rm(path; force=true)
    end
    return length(paths)
end

function _invalidate_drive_cache!(; kwargs...)
    return _clear_drive_cache!(; kwargs...)
end
