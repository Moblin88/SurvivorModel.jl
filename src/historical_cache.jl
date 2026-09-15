const HISTORICAL_PRIOR_CACHE_SCHEMA = 2

struct HistoricalPriorCacheEntry
    schema_version::Int
    season::Int
    max_seasons::Int
    time_edges::Vector{Float64}
    method::Symbol
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
    time_edges::AbstractVector{<:Real},
    method::PriorFitMethod,
    data_fingerprint::AbstractString,
)
    edge_token = join(_cache_path_token.(Float64.(time_edges)), "_")
    method_token = _cache_path_token(prior_fit_method_name(method))
    filename = join(
        (
            "historical_prior",
            "v$(HISTORICAL_PRIOR_CACHE_SCHEMA)",
            "season$(Int(season))",
            "window$(Int(max_seasons))",
            "edges$(edge_token)",
            "method$(method_token)",
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
    time_edges::AbstractVector{<:Real},
    method::PriorFitMethod,
    data_fingerprint::AbstractString,
)
    entry isa HistoricalPriorCacheEntry || return false
    expected_edges = Float64.(time_edges)
    entry.schema_version == HISTORICAL_PRIOR_CACHE_SCHEMA || return false
    entry.season == Int(season) || return false
    entry.max_seasons == Int(max_seasons) || return false
    entry.time_edges == expected_edges || return false
    entry.method === prior_fit_method_name(method) || return false
    entry.data_fingerprint == data_fingerprint || return false
    entry.prior.time_edges == expected_edges || return false
    return true
end

function _read_historical_prior_cache(
    path::AbstractString,
    season::Integer,
    max_seasons::Integer,
    time_edges::AbstractVector{<:Real},
    method::PriorFitMethod,
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
        time_edges,
        method,
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
    time_edges=DEFAULT_TIME_EDGES,
    max_seasons::Int=DEFAULT_HISTORICAL_SEASONS,
    method::PriorFitMethod=DEFAULT_PRIOR_FIT_METHOD,
    cache_directory::Union{Nothing,AbstractString}=nothing,
)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    edges = Float64.(collect(time_edges))
    length(edges) >= 2 ||
        throw(ArgumentError("time_edges must contain at least two values"))
    all(isfinite, edges[1:(end - 1)]) ||
        throw(ArgumentError("finite time edges are required before the final edge"))
    isinf(edges[end]) || throw(ArgumentError("the final time edge must be Inf"))
    all(diff(edges[1:(end - 1)]) .> 0.0) ||
        throw(ArgumentError("time_edges must be strictly increasing"))
    data_fingerprint = _dataframe_fingerprint(historical_drives)

    directory = cache_directory === nothing ?
        _historical_prior_cache_directory() :
        String(cache_directory)
    mkpath(directory)
    path = _historical_prior_cache_path(
        directory,
        current_season,
        max_seasons,
        edges,
        method,
        data_fingerprint,
    )
    cached_prior = _read_historical_prior_cache(
        path,
        current_season,
        max_seasons,
        edges,
        method,
        data_fingerprint,
    )
    cached_prior !== nothing &&
        return (
            prior=cached_prior,
            cache_hit=true,
            path=path,
            data_fingerprint=data_fingerprint,
        )

    prior = fit_empirical_bayes_prior(
        historical_drives;
        time_edges=edges,
        max_seasons=max_seasons,
        current_season=current_season,
        method=method,
    )
    entry = HistoricalPriorCacheEntry(
        HISTORICAL_PRIOR_CACHE_SCHEMA,
        Int(current_season),
        Int(max_seasons),
        edges,
        prior_fit_method_name(method),
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
    clear_historical_prior_cache!()

Remove the package-owned historical empirical-Bayes fit cache.
"""
function clear_historical_prior_cache!()
    Scratch.delete_scratch!(@__MODULE__, "historical_priors")
    return nothing
end
