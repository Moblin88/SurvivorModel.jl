const DEFAULT_CALIBRATION_CUTOFF_WEEKS = (1, 5, 10, 15)
const DEFAULT_PROBABILITY_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
const DEFAULT_SPREAD_BINS = (-Inf, -7.0, -3.0, 0.0, 3.0, 7.0, Inf)
const DEFAULT_SPREAD_INTERVAL_LEVELS = (0.5, 0.8, 0.9, 0.95)
const DEFAULT_SPREAD_TOLERANCE = 1.0e-9

"""
    CalibrationReport

Per-game probability and score-difference calibration observations, summary
metrics by season and cutoff week, fixed-bin reliability tables, and
predictive-interval coverage tables.
"""
struct CalibrationReport
    games::DataFrame
    summary::DataFrame
    reliability::DataFrame
    spread_games::DataFrame
    spread_summary::DataFrame
    spread_reliability::DataFrame
    spread_coverage::DataFrame
end

CalibrationReport(
    games::DataFrame,
    summary::DataFrame,
    reliability::DataFrame,
) = CalibrationReport(
    games,
    summary,
    reliability,
    DataFrame(),
    DataFrame(),
    DataFrame(),
    DataFrame(),
)

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

function _margin_vectors(predicted, actual)
    length(predicted) == length(actual) ||
        throw(ArgumentError("predicted and actual margins must have equal lengths"))
    isempty(predicted) &&
        throw(ArgumentError("score-difference metrics require at least one observation"))

    predictions = Float64.(collect(predicted))
    outcomes = Float64.(collect(actual))
    all(isfinite, predictions) ||
        throw(ArgumentError("predicted margins must be finite"))
    all(isfinite, outcomes) ||
        throw(ArgumentError("actual margins must be finite"))
    return predictions, outcomes
end

function _validate_spread_tolerance(tolerance)
    value = Float64(tolerance)
    isfinite(value) && value >= 0.0 ||
        throw(ArgumentError("spread_tolerance must be finite and nonnegative"))
    return value
end

function _validate_interval_levels(interval_levels)
    levels = Float64.(collect(interval_levels))
    isempty(levels) &&
        throw(ArgumentError("interval_levels must not be empty"))
    all(isfinite, levels) &&
        all(0.0 .< levels .< 1.0) ||
        throw(ArgumentError("interval_levels must be finite values in (0, 1)"))
    all(diff(levels) .> 0) ||
        throw(ArgumentError("interval_levels must be strictly increasing"))
    return levels
end

"""
    score_difference_metrics(predicted, actual; tolerance=DEFAULT_SPREAD_TOLERANCE)
        -> NamedTuple

Compute point-forecast error metrics for predicted home-team score
differences. A positive error means the actual home margin exceeded the
forecast. The above/below/push rates classify errors using `tolerance`.
"""
function score_difference_metrics(
    predicted,
    actual;
    tolerance::Real=DEFAULT_SPREAD_TOLERANCE,
)
    predictions, outcomes = _margin_vectors(predicted, actual)
    tolerance_value = _validate_spread_tolerance(tolerance)
    errors = outcomes .- predictions
    above = errors .> tolerance_value
    below = errors .< -tolerance_value
    n_games = length(errors)

    return (
        n_games=n_games,
        mean_predicted=mean(predictions),
        mean_actual=mean(outcomes),
        mean_error=mean(errors),
        mean_absolute_error=mean(abs.(errors)),
        rmse=sqrt(mean(errors .^ 2)),
        above_forecast_rate=count(above) / n_games,
        below_forecast_rate=count(below) / n_games,
        push_rate=(n_games - count(above) - count(below)) / n_games,
    )
end

function _predictive_variances(predictive_variance, n_games::Integer)
    variances = Float64.(collect(predictive_variance))
    length(variances) == n_games ||
        throw(ArgumentError("predictive variances and margins must have equal lengths"))
    all(isfinite, variances) &&
        all(>=(0.0), variances) ||
        throw(ArgumentError("predictive variances must be finite and nonnegative"))
    all(>(0.0), variances) ||
        throw(ArgumentError("predictive variances must be positive"))
    return variances
end

"""
    spread_interval_coverage(
        predicted,
        predictive_variance,
        actual;
        interval_levels=DEFAULT_SPREAD_INTERVAL_LEVELS,
    ) -> DataFrame

Measure central Normal predictive-interval coverage for score-difference
forecasts. The predictive distribution is centered at `predicted` with the
supplied variance.
"""
function spread_interval_coverage(
    predicted,
    predictive_variance,
    actual;
    interval_levels=DEFAULT_SPREAD_INTERVAL_LEVELS,
)
    predictions, outcomes = _margin_vectors(predicted, actual)
    variances = _predictive_variances(predictive_variance, length(predictions))
    levels = _validate_interval_levels(interval_levels)
    standard_deviations = sqrt.(variances)
    rows = NamedTuple[]

    for level in levels
        critical_value = quantile(Normal(), (1.0 + level) / 2.0)
        covered = abs.(outcomes .- predictions) .<=
            critical_value .* standard_deviations
        covered_games = count(covered)
        push!(
            rows,
            (
                interval_level=level,
                n_games=length(predictions),
                covered_games=covered_games,
                coverage=covered_games / length(predictions),
                coverage_gap=covered_games / length(predictions) - level,
            ),
        )
    end

    return DataFrame(rows)
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

function _validate_spread_bins(spread_bins)
    bins = Float64.(collect(spread_bins))
    length(bins) >= 2 ||
        throw(ArgumentError("spread_bins must contain at least two edges"))
    first(bins) == -Inf ||
        throw(ArgumentError("spread_bins must start at -Inf"))
    last(bins) == Inf ||
        throw(ArgumentError("spread_bins must end at Inf"))
    any(isnan, bins) &&
        throw(ArgumentError("spread_bins cannot contain NaN"))
    all(diff(bins) .> 0) ||
        throw(ArgumentError("spread_bins must be strictly increasing"))
    return bins
end

function _empty_spread_reliability_bins(spread_bins)
    bins = _validate_spread_bins(spread_bins)
    n_bins = length(bins) - 1
    return DataFrame(
        spread_lower=bins[1:(end - 1)],
        spread_upper=bins[2:end],
        n_games=zeros(Int, n_bins),
        mean_predicted_spread=Vector{Union{Missing,Float64}}(missing, n_bins),
        mean_actual_margin=Vector{Union{Missing,Float64}}(missing, n_bins),
        mean_error=Vector{Union{Missing,Float64}}(missing, n_bins),
    )
end

"""
    spread_reliability_bins(
        predicted,
        actual;
        spread_bins=DEFAULT_SPREAD_BINS,
    ) -> DataFrame

Summarize actual score differences within fixed bins of the predicted home
margin. The `mean_error` is actual margin minus predicted margin.
"""
function spread_reliability_bins(
    predicted,
    actual;
    spread_bins=DEFAULT_SPREAD_BINS,
)
    predictions, outcomes = _margin_vectors(predicted, actual)
    bins = _validate_spread_bins(spread_bins)
    n_bins = length(bins) - 1
    bin_indices = zeros(Int, length(predictions))

    for observation in eachindex(predictions)
        predicted_margin = predictions[observation]
        for bin in 1:n_bins
            lower, upper = bins[bin], bins[bin + 1]
            in_bin = bin == n_bins ?
                lower <= predicted_margin <= upper :
                lower <= predicted_margin < upper
            if in_bin
                bin_indices[observation] = bin
                break
            end
        end
        bin_indices[observation] > 0 ||
            throw(ArgumentError("predicted margin did not fall into a spread bin"))
    end

    n_games = zeros(Int, n_bins)
    mean_predicted = Vector{Union{Missing,Float64}}(missing, n_bins)
    mean_actual = Vector{Union{Missing,Float64}}(missing, n_bins)
    mean_error = Vector{Union{Missing,Float64}}(missing, n_bins)
    for bin in 1:n_bins
        observations = findall(==(bin), bin_indices)
        n_games[bin] = length(observations)
        isempty(observations) && continue
        mean_predicted[bin] = mean(predictions[observations])
        mean_actual[bin] = mean(outcomes[observations])
        mean_error[bin] = mean(
            outcomes[observations] .- predictions[observations],
        )
    end

    return DataFrame(
        spread_lower=bins[1:(end - 1)],
        spread_upper=bins[2:end],
        n_games=n_games,
        mean_predicted_spread=mean_predicted,
        mean_actual_margin=mean_actual,
        mean_error=mean_error,
    )
end

function _empty_spread_coverage(interval_levels)
    levels = _validate_interval_levels(interval_levels)
    n_levels = length(levels)
    return DataFrame(
        interval_level=levels,
        n_games=zeros(Int, n_levels),
        covered_games=zeros(Int, n_levels),
        coverage=Vector{Union{Missing,Float64}}(missing, n_levels),
        coverage_gap=Vector{Union{Missing,Float64}}(missing, n_levels),
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

function _append_spread_reliability_metadata!(
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
            :spread_lower,
            :spread_upper,
            :n_games,
            :mean_predicted_spread,
            :mean_actual_margin,
            :mean_error,
        ],
    )
end

function _append_spread_coverage_metadata!(
    coverage::DataFrame,
    season::Integer,
    as_of_week::Integer,
)
    coverage.season = fill(Int(season), nrow(coverage))
    coverage.as_of_week = fill(Int(as_of_week), nrow(coverage))
    return select!(
        coverage,
        [
            :season,
            :as_of_week,
            :interval_level,
            :n_games,
            :covered_games,
            :coverage,
            :coverage_gap,
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
        spread_bins=DEFAULT_SPREAD_BINS,
        interval_levels=DEFAULT_SPREAD_INTERVAL_LEVELS,
        spread_tolerance=DEFAULT_SPREAD_TOLERANCE,
    ) -> CalibrationReport

Evaluate fixed pre-week win-probability and score-difference snapshots for
completed games in the requested seasons. When `seasons` is omitted, the most
recent `recent_seasons` completed regular seasons in the schedule are selected.
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
    spread_bins=DEFAULT_SPREAD_BINS,
    interval_levels=DEFAULT_SPREAD_INTERVAL_LEVELS,
    spread_tolerance::Real=DEFAULT_SPREAD_TOLERANCE,
)
    cutoff_values = _calibration_cutoff_weeks(cutoff_weeks)
    max_seasons > 0 || throw(ArgumentError("max_seasons must be positive"))
    recent_seasons > 0 ||
        throw(ArgumentError("recent_seasons must be positive"))
    _validate_probability_bins(probability_bins)
    _validate_spread_bins(spread_bins)
    _validate_interval_levels(interval_levels)
    spread_tolerance_value = _validate_spread_tolerance(spread_tolerance)

    normalized_schedule = schedule === nothing ? load_schedule() : load_schedule(schedule)
    season_values = isnothing(seasons) ?
        _recent_completed_calibration_seasons(
            normalized_schedule,
            recent_seasons,
        ) :
        _calibration_seasons(seasons)
    effective_max_seasons = min(max_seasons, MAX_HISTORICAL_SEASONS)
    first_data_season = max(1999, first(season_values) - effective_max_seasons)
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
    spread_game_rows = NamedTuple[]
    spread_summary_rows = NamedTuple[]
    spread_reliability_tables = DataFrame[]
    spread_coverage_tables = DataFrame[]

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

            spreads = forecast_spreads(
                context;
                include_completed=true,
                full_schedule=false,
            )
            nrow(spreads) == nrow(probabilities) ||
                throw(ArgumentError(
                    "probability and spread forecasts returned different game counts",
                ))
            all(spreads.game_id .== probabilities.game_id) ||
                throw(ArgumentError(
                    "probability and spread forecasts returned different games",
                ))
            scored_spreads = spreads[.!ismissing.(spreads.result), :]
            expected_spreads = Float64.(scored_spreads.expected_spread)
            predictive_variances = _predictive_variances(
                scored_spreads.predictive_spread_variance,
                length(expected_spreads),
            )
            actual_margins = Float64.(scored_spreads.result)
            margin_errors = actual_margins .- expected_spreads
            standardized_errors = margin_errors ./ sqrt.(predictive_variances)

            for row_index in 1:nrow(scored_spreads)
                row = scored_spreads[row_index, :]
                error = margin_errors[row_index]
                above = error > spread_tolerance_value
                below = error < -spread_tolerance_value
                push!(
                    spread_game_rows,
                    (
                        season=Int(season),
                        as_of_week=Int(as_of_week),
                        game_id=string(row.game_id),
                        week=Int(row.week),
                        away_team=string(row.away_team),
                        home_team=string(row.home_team),
                        expected_spread=expected_spreads[row_index],
                        predictive_spread_variance=predictive_variances[row_index],
                        actual_home_margin=actual_margins[row_index],
                        spread_error=error,
                        standardized_error=standardized_errors[row_index],
                        home_above_forecast=above,
                        away_above_forecast=below,
                        model_line_push=!(above || below),
                        result=row.result,
                    ),
                )
            end

            scored_spread_games = length(expected_spreads)
            spread_summary = if scored_spread_games == 0
                (
                    season=Int(season),
                    as_of_week=Int(as_of_week),
                    forecasted_games=nrow(spreads),
                    scored_games=0,
                    mean_predicted_spread=missing,
                    mean_actual_margin=missing,
                    mean_error=missing,
                    mean_absolute_error=missing,
                    rmse=missing,
                    mean_predictive_standard_deviation=missing,
                    mean_standardized_error=missing,
                    standardized_error_sd=missing,
                    above_forecast_rate=missing,
                    below_forecast_rate=missing,
                    push_rate=missing,
                )
            else
                metrics = score_difference_metrics(
                    expected_spreads,
                    actual_margins;
                    tolerance=spread_tolerance_value,
                )
                (
                    season=Int(season),
                    as_of_week=Int(as_of_week),
                    forecasted_games=nrow(spreads),
                    scored_games=scored_spread_games,
                    mean_predicted_spread=metrics.mean_predicted,
                    mean_actual_margin=metrics.mean_actual,
                    mean_error=metrics.mean_error,
                    mean_absolute_error=metrics.mean_absolute_error,
                    rmse=metrics.rmse,
                    mean_predictive_standard_deviation=mean(
                        sqrt.(predictive_variances)
                    ),
                    mean_standardized_error=mean(standardized_errors),
                    standardized_error_sd=std(
                        standardized_errors;
                        corrected=false,
                    ),
                    above_forecast_rate=metrics.above_forecast_rate,
                    below_forecast_rate=metrics.below_forecast_rate,
                    push_rate=metrics.push_rate,
                )
            end
            push!(spread_summary_rows, spread_summary)

            spread_reliability = scored_spread_games == 0 ?
                _empty_spread_reliability_bins(spread_bins) :
                spread_reliability_bins(
                    expected_spreads,
                    actual_margins;
                    spread_bins=spread_bins,
                )
            push!(
                spread_reliability_tables,
                _append_spread_reliability_metadata!(
                    spread_reliability,
                    season,
                    as_of_week,
                ),
            )

            spread_coverage = scored_spread_games == 0 ?
                _empty_spread_coverage(interval_levels) :
                spread_interval_coverage(
                    expected_spreads,
                    predictive_variances,
                    actual_margins;
                    interval_levels=interval_levels,
                )
            push!(
                spread_coverage_tables,
                _append_spread_coverage_metadata!(
                    spread_coverage,
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
    spread_games = DataFrame(spread_game_rows)
    spread_summary = DataFrame(spread_summary_rows)
    spread_reliability = isempty(spread_reliability_tables) ?
        DataFrame() : vcat(spread_reliability_tables...; cols=:union)
    spread_coverage = isempty(spread_coverage_tables) ?
        DataFrame() : vcat(spread_coverage_tables...; cols=:union)
    return CalibrationReport(
        games,
        summary,
        reliability,
        spread_games,
        spread_summary,
        spread_reliability,
        spread_coverage,
    )
end
