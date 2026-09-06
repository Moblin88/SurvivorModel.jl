const DEFAULT_CALIBRATION_CUTOFF_WEEKS = (1, 5, 10, 15)
const DEFAULT_PROBABILITY_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

"""
    CalibrationReport

Per-game calibration observations, summary metrics by season and cutoff week,
and fixed-bin reliability tables.
"""
struct CalibrationReport
    games::DataFrame
    summary::DataFrame
    reliability::DataFrame
end

function _calibration_vectors(predicted, actual)
    length(predicted) == length(actual) ||
        throw(ArgumentError("predictions and outcomes must have equal lengths"))
    isempty(predicted) &&
        throw(ArgumentError("calibration metrics require at least one observation"))

    predictions = Float64.(collect(predicted))
    outcomes = Float64.(collect(actual))
    all(isfinite, predictions) &&
        all(0.0 .<= predictions .<= 1.0) ||
        throw(ArgumentError("predictions must be finite probabilities"))
    all(isfinite, outcomes) &&
        all(0.0 .<= outcomes .<= 1.0) ||
        throw(ArgumentError("outcomes must be finite values in [0, 1]"))
    return predictions, outcomes
end

"""
    brier_score(predicted, actual) -> Float64

Compute the mean squared probability error. Ties can be represented by an
actual outcome of `0.5`.
"""
function brier_score(predicted, actual)
    predictions, outcomes = _calibration_vectors(predicted, actual)
    return mean((predictions .- outcomes) .^ 2)
end

"""
    log_loss(predicted, actual) -> Float64

Compute binary log loss, treating an actual outcome of `0.5` as half a win and
half a loss. Probabilities are bounded away from zero and one numerically.
"""
function log_loss(predicted, actual)
    predictions, outcomes = _calibration_vectors(predicted, actual)
    bounded = clamp.(predictions, eps(Float64), 1.0 - eps(Float64))
    return mean(
        -(
            outcomes .* log.(bounded) +
            (1.0 .- outcomes) .* log1p.(-bounded)
        ),
    )
end

function _validate_probability_bins(probability_bins)
    bins = Float64.(collect(probability_bins))
    length(bins) >= 2 ||
        throw(ArgumentError("probability_bins must contain at least two edges"))
    first(bins) == 0.0 ||
        throw(ArgumentError("probability_bins must start at 0.0"))
    last(bins) == 1.0 ||
        throw(ArgumentError("probability_bins must end at 1.0"))
    all(diff(bins) .> 0) ||
        throw(ArgumentError("probability_bins must be strictly increasing"))
    return bins
end

function _empty_reliability_bins(probability_bins)
    bins = _validate_probability_bins(probability_bins)
    n_bins = length(bins) - 1
    return DataFrame(
        bin_lower=bins[1:(end - 1)],
        bin_upper=bins[2:end],
        n_games=zeros(Int, n_bins),
        mean_predicted=Vector{Union{Missing,Float64}}(missing, n_bins),
        observed_rate=Vector{Union{Missing,Float64}}(missing, n_bins),
        calibration_gap=Vector{Union{Missing,Float64}}(missing, n_bins),
    )
end

"""
    reliability_bins(predicted, actual; probability_bins=DEFAULT_PROBABILITY_BINS)
        -> DataFrame

Summarize predicted and observed rates in fixed probability bins. The
`calibration_gap` is observed rate minus mean predicted probability.
"""
function reliability_bins(
    predicted,
    actual;
    probability_bins=DEFAULT_PROBABILITY_BINS,
)
    predictions, outcomes = _calibration_vectors(predicted, actual)
    bins = _validate_probability_bins(probability_bins)
    n_bins = length(bins) - 1
    bin_indices = zeros(Int, length(predictions))

    for observation in eachindex(predictions)
        probability = predictions[observation]
        for bin in 1:n_bins
            lower, upper = bins[bin], bins[bin + 1]
            in_bin = bin == n_bins ?
                lower <= probability <= upper :
                lower <= probability < upper
            if in_bin
                bin_indices[observation] = bin
                break
            end
        end
        bin_indices[observation] > 0 ||
            throw(ArgumentError("probability did not fall into a calibration bin"))
    end

    n_games = zeros(Int, n_bins)
    mean_predicted = Vector{Union{Missing,Float64}}(missing, n_bins)
    observed_rate = Vector{Union{Missing,Float64}}(missing, n_bins)
    calibration_gap = Vector{Union{Missing,Float64}}(missing, n_bins)
    for bin in 1:n_bins
        observations = findall(==(bin), bin_indices)
        n_games[bin] = length(observations)
        isempty(observations) && continue
        mean_predicted[bin] = mean(predictions[observations])
        observed_rate[bin] = mean(outcomes[observations])
        calibration_gap[bin] = observed_rate[bin] - mean_predicted[bin]
    end

    return DataFrame(
        bin_lower=bins[1:(end - 1)],
        bin_upper=bins[2:end],
        n_games=n_games,
        mean_predicted=mean_predicted,
        observed_rate=observed_rate,
        calibration_gap=calibration_gap,
    )
end

function _actual_home_outcome(result)
    ismissing(result) && return missing
    result > 0 && return 1.0
    result < 0 && return 0.0
    return 0.5
end

function _calibration_seasons(seasons)
    values = seasons isa Integer ? [Int(seasons)] : Int.(collect(seasons))
    isempty(values) && throw(ArgumentError("seasons must not be empty"))
    return sort!(unique(values))
end

function _recent_completed_calibration_seasons(
    schedule::AbstractDataFrame,
    recent_seasons::Int,
)
    recent_seasons > 0 ||
        throw(ArgumentError("recent_seasons must be positive"))
    completed = Int[]
    for season in sort(unique(Int.(schedule.season)))
        games = _regular_season_schedule(schedule, season)
        isempty(games) && continue
        all(.!ismissing.(games.result)) || continue
        push!(completed, season)
    end
    length(completed) >= recent_seasons ||
        throw(ArgumentError(
            "schedule has only $(length(completed)) completed regular seasons; " *
            "cannot select $recent_seasons",
        ))
    return completed[(end - recent_seasons + 1):end]
end

function _calibration_cutoff_weeks(cutoff_weeks)
    values = cutoff_weeks isa Integer ?
        [Int(cutoff_weeks)] : Int.(collect(cutoff_weeks))
    isempty(values) && throw(ArgumentError("cutoff_weeks must not be empty"))
    all(1 .<= values .<= 18) ||
        throw(ArgumentError("cutoff_weeks must be between 1 and 18"))
    return sort!(unique(values))
end

function _append_reliability_metadata!(
    reliability::DataFrame,
    season::Integer,
    as_of_week::Integer,
)
    reliability.season = fill(Int(season), nrow(reliability))
    reliability.as_of_week = fill(Int(as_of_week), nrow(reliability))
    return select!(
        reliability,
        [
            :season,
            :as_of_week,
            :bin_lower,
            :bin_upper,
            :n_games,
            :mean_predicted,
            :observed_rate,
            :calibration_gap,
        ],
    )
end

"""
    evaluate_calibration(
        seasons=nothing;
        cutoff_weeks=DEFAULT_CALIBRATION_CUTOFF_WEEKS,
        schedule=nothing,
        drives=nothing,
        max_seasons=3,
        recent_seasons=3,
        time_edges=DEFAULT_TIME_EDGES,
        recency_half_life=nothing,
        probability_bins=DEFAULT_PROBABILITY_BINS,
    ) -> CalibrationReport

Evaluate fixed pre-week win-probability snapshots for completed games in the
requested seasons. When `seasons` is omitted, the most recent
`recent_seasons` completed regular seasons in the schedule are selected.
Schedule and drive data can be injected to reuse data across multiple seasons
and cutoffs without downloading it repeatedly.
"""
function evaluate_calibration(
    seasons=nothing;
    cutoff_weeks=DEFAULT_CALIBRATION_CUTOFF_WEEKS,
    schedule::Union{Nothing,AbstractDataFrame}=nothing,
    drives::Union{Nothing,AbstractDataFrame}=nothing,
    max_seasons::Int=3,
    recent_seasons::Int=3,
    time_edges=DEFAULT_TIME_EDGES,
    recency_half_life::Union{Nothing,Real}=nothing,
    probability_bins=DEFAULT_PROBABILITY_BINS,
)
    cutoff_values = _calibration_cutoff_weeks(cutoff_weeks)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    recent_seasons > 0 ||
        throw(ArgumentError("recent_seasons must be positive"))
    _validate_probability_bins(probability_bins)

    normalized_schedule = schedule === nothing ? load_schedule() : load_schedule(schedule)
    season_values = isnothing(seasons) ?
        _recent_completed_calibration_seasons(
            normalized_schedule,
            recent_seasons,
        ) :
        _calibration_seasons(seasons)
    first_data_season = max(1999, first(season_values) - max_seasons)
    last_data_season = last(season_values)
    all_drives = if drives === nothing
        load_drive_pbp(first_data_season:last_data_season)
    else
        drives
    end
    regular_drives = _regular_season_drives(all_drives, normalized_schedule)

    game_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    reliability_tables = DataFrame[]

    for season in season_values
        target_schedule = _regular_season_schedule(normalized_schedule, season)
        isempty(target_schedule) &&
            throw(ArgumentError("schedule has no regular-season games for season $season"))
        historical = regular_drives[regular_drives.season .< season, :]
        current = regular_drives[regular_drives.season .== season, :]

        for as_of_week in cutoff_values
            context = fit_regular_season_forecast(
                season;
                as_of_week=as_of_week,
                schedule=normalized_schedule,
                historical_drives=historical,
                current_drives=current,
                max_seasons=max_seasons,
                time_edges=time_edges,
                recency_half_life=recency_half_life,
            )
            probabilities = forecast_win_probabilities(
                context;
                include_completed=true,
                full_schedule=false,
            )
            scored = probabilities[.!ismissing.(probabilities.result), :]
            predictions = Float64.(scored.home_win_probability)
            outcomes = Float64[
                _actual_home_outcome(result) for result in scored.result
            ]

            for row_index in 1:nrow(scored)
                row = scored[row_index, :]
                push!(
                    game_rows,
                    (
                        season=Int(season),
                        as_of_week=Int(as_of_week),
                        game_id=string(row.game_id),
                        week=Int(row.week),
                        away_team=string(row.away_team),
                        home_team=string(row.home_team),
                        home_win_probability=predictions[row_index],
                        actual_home_outcome=outcomes[row_index],
                        result=row.result,
                    ),
                )
            end

            scored_games = length(predictions)
            summary = if scored_games == 0
                (
                    season=Int(season),
                    as_of_week=Int(as_of_week),
                    forecasted_games=nrow(probabilities),
                    scored_games=0,
                    brier_score=missing,
                    log_loss=missing,
                    mean_predicted=missing,
                    observed_rate=missing,
                    calibration_gap=missing,
                )
            else
                mean_predicted = mean(predictions)
                observed_rate = mean(outcomes)
                (
                    season=Int(season),
                    as_of_week=Int(as_of_week),
                    forecasted_games=nrow(probabilities),
                    scored_games=scored_games,
                    brier_score=brier_score(predictions, outcomes),
                    log_loss=log_loss(predictions, outcomes),
                    mean_predicted=mean_predicted,
                    observed_rate=observed_rate,
                    calibration_gap=observed_rate - mean_predicted,
                )
            end
            push!(summary_rows, summary)

            reliability = scored_games == 0 ?
                _empty_reliability_bins(probability_bins) :
                reliability_bins(
                    predictions,
                    outcomes;
                    probability_bins=probability_bins,
                )
            push!(
                reliability_tables,
                _append_reliability_metadata!(
                    reliability,
                    season,
                    as_of_week,
                ),
            )
        end
    end

    games = DataFrame(game_rows)
    summary = DataFrame(summary_rows)
    reliability = isempty(reliability_tables) ?
        DataFrame() : vcat(reliability_tables...; cols=:union)
    return CalibrationReport(games, summary, reliability)
end
