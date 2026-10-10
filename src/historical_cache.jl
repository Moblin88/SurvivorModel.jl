const HISTORICAL_PRIOR_CACHE_SCHEMA = 4

struct HistoricalPriorCacheEntry
    schema_version::Int
    season::Int
    max_seasons::Int
    data_fingerprint::String
    prior::HazardPrior
end

function _historical_prior_cache_directory()
    return Scratch.get_scratch!(@__MODULE__, "historical_priors")
end

function _cache_path_token(value)
    return replace(string(value), r"[^A-Za-z0-9_.-]" => "_")
end

function _historical_prior_cache_path(
    cache_directory::AbstractString,
    season::Integer,
    max_seasons::Integer,
    data_fingerprint::AbstractString,
)
    filename = join(
        (
            "historical_prior",
            "v$(HISTORICAL_PRIOR_CACHE_SCHEMA)",
            "season$(Int(season))",
            "window$(Int(max_seasons))",
            "data$(_cache_path_token(data_fingerprint))",
        ),
        "_",
    ) * ".jls"
    return joinpath(cache_directory, filename)
end

function _historical_prior_cache_entry_matches(
    entry,
    season::Integer,
    max_seasons::Integer,
    data_fingerprint::AbstractString,
)
    entry isa HistoricalPriorCacheEntry || return false
    entry.schema_version == HISTORICAL_PRIOR_CACHE_SCHEMA || return false
    entry.season == Int(season) || return false
    entry.max_seasons == Int(max_seasons) || return false
    entry.data_fingerprint == data_fingerprint || return false
    return true
end

function _read_historical_prior_cache(
    path::AbstractString,
    season::Integer,
    max_seasons::Integer,
    data_fingerprint::AbstractString,
)
    isfile(path) || return nothing
    entry = try
        open(path, "r") do io
            deserialize(io)
        end
    catch error
        throw(ArgumentError(
            "could not read historical prior cache $(path): " *
            sprint(showerror, error),
        ))
    end
    entry isa HistoricalPriorCacheEntry || throw(ArgumentError(
        "serialized value has type $(typeof(entry)), expected " *
        "HistoricalPriorCacheEntry",
    ))
    _historical_prior_cache_entry_matches(
        entry,
        season,
        max_seasons,
        data_fingerprint,
    ) || return nothing
    return entry.prior
end

function _write_historical_prior_cache(path::AbstractString, entry)
    cache_directory = dirname(path)
    mkpath(cache_directory)
    temporary_path = tempname(cache_directory)
    try
        open(temporary_path, "w") do io
            serialize(io, entry)
        end
        mv(temporary_path, path; force=true)
    finally
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return entry.prior
end

"""
    _cached_historical_prior(historical_drives; ...)

Fit or retrieve a historical empirical-Bayes prior. The cache contains only
the fitted `HazardPrior`; current-season data are intentionally excluded so
that each forecast invocation observes newly available drives.
"""
function _cached_historical_prior(
    historical_drives::AbstractDataFrame;
    current_season::Integer,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    method::WeibullEmpiricalBayesFit=DEFAULT_PRIOR_FIT_METHOD,
    cache_directory::Union{Nothing,AbstractString}=nothing,
)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    data_fingerprint = _dataframe_fingerprint(historical_drives)

    directory = cache_directory === nothing ?
        _historical_prior_cache_directory() :
        String(cache_directory)
    mkpath(directory)
    path = _historical_prior_cache_path(
        directory,
        current_season,
        max_seasons,
        data_fingerprint,
    )
    cached_prior = _read_historical_prior_cache(
        path,
        current_season,
        max_seasons,
        data_fingerprint,
    )
    if cached_prior !== nothing
        _log_historical_prior_diagnostics(
            cached_prior;
            source=:cache,
            cache_path=path,
        )
        return (
            prior=cached_prior,
            cache_hit=true,
            path=path,
            data_fingerprint=data_fingerprint,
        )
    end

    prior = fit_empirical_bayes_prior(
        historical_drives;
        max_seasons=max_seasons,
        current_season=current_season,
        method=method,
    )
    entry = HistoricalPriorCacheEntry(
        HISTORICAL_PRIOR_CACHE_SCHEMA,
        Int(current_season),
        Int(max_seasons),
        data_fingerprint,
        prior,
    )
    _write_historical_prior_cache(path, entry)
    return (
        prior=prior,
        cache_hit=false,
        path=path,
        data_fingerprint=data_fingerprint,
    )
end

"""
    clear_historical_prior_cache!(; cache_directory=nothing)

Remove cached historical empirical-Bayes fits. By default, delete the
package-owned Scratch.jl cache; with `cache_directory`, delete only matching
prior-fit cache files in that directory.
"""
function clear_historical_prior_cache!(
    ;
    cache_directory::Union{Nothing,AbstractString}=nothing,
)
    if cache_directory === nothing
        Scratch.delete_scratch!(@__MODULE__, "historical_priors")
    else
        directory = String(cache_directory)
        if isdir(directory)
            for path in readdir(directory; join=true)
                filename = basename(path)
                startswith(filename, "historical_prior_") || continue
                endswith(filename, ".jls") || continue
                isfile(path) || continue
                rm(path; force=true)
            end
        end
    end
    return nothing
end
