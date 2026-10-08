const TEAM_STRENGTH_TEAM_COLOR_HEX = Dict(
    "ARI" => "#97233F",
    "ATL" => "#91172A",
    "BAL" => "#241773",
    "BUF" => "#00338D",
    "CAR" => "#006A9F",
    "CHI" => "#0B162A",
    "CIN" => "#A1360D",
    "CLE" => "#311D00",
    "DAL" => "#004C9A",
    "DEN" => "#A83F0F",
    "DET" => "#0076B6",
    "GB" => "#203731",
    "HOU" => "#03202F",
    "IND" => "#002C5F",
    "JAX" => "#006778",
    "KC" => "#E31837",
    "LAC" => "#005C94",
    "LAR" => "#003594",
    "LV" => "#6B7074",
    "MIA" => "#005778",
    "MIN" => "#4F2683",
    "NE" => "#002244",
    "NO" => "#6A5834",
    "NYG" => "#0B2265",
    "NYJ" => "#003F2D",
    "PHI" => "#004C54",
    "PIT" => "#000000",
    "SEA" => "#345F14",
    "SF" => "#AA0000",
    "TB" => "#A71930",
    "TEN" => "#0C2340",
    "WAS" => "#5A1414",
)

const TEAM_STRENGTH_TEAM_COLOR_ALIASES = Dict(
    "ARZ" => "ARI",
    "JAC" => "JAX",
    "LA" => "LAR",
    "OAK" => "LV",
    "SD" => "LAC",
    "STL" => "LAR",
    "WFT" => "WAS",
    "WSH" => "WAS",
)

const TEAM_STRENGTH_UNKNOWN_TEAM_COLOR = "#595959"

function _team_strength_team_colors(teams::AbstractVector{<:AbstractString})
    colors = Vector{String}(undef, length(teams))
    unknown_teams = Set{String}()
    for (index, raw_team) in enumerate(teams)
        team = uppercase(strip(String(raw_team)))
        abbreviation = get(TEAM_STRENGTH_TEAM_COLOR_ALIASES, team, team)
        color = get(TEAM_STRENGTH_TEAM_COLOR_HEX, abbreviation, nothing)
        if color === nothing
            colors[index] = TEAM_STRENGTH_UNKNOWN_TEAM_COLOR
            push!(unknown_teams, team)
        else
            colors[index] = color
        end
    end
    for team in sort!(collect(unknown_teams))
        @warn "unknown team abbreviation in team-strength plot; using neutral gray" team
    end
    return colors
end

"""
    _team_strength_plot_data(context)

Return a team-sorted `DataFrame` of neutral-site posterior mean rates and
central 80% posterior interval endpoints for touchdown offense and
defensive events.
"""
function _team_strength_plot_data(
    context::RegularSeasonForecastContext,
)
    schedule = _regular_season_schedule(context.schedule, context.season)
    isempty(schedule) &&
        throw(ArgumentError(
            "schedule has no regular-season games for season $(context.season)",
        ))
    teams = sort!(unique(vcat(
        String.(schedule.away_team),
        String.(schedule.home_team),
    )))

    offense_mean = Float64[]
    offense_lower = Float64[]
    offense_upper = Float64[]
    defense_mean = Float64[]
    defense_lower = Float64[]
    defense_upper = Float64[]
    for team in teams
        offense = hazard_posterior(context.model, :td, team)
        defense = hazard_posterior(context.model, :defensive, team)
        team_offense_mean = _gamma_mixture_mean(offense)
        team_defense_mean = _gamma_mixture_mean(defense)
        for (cause, mean) in (
            (:td, team_offense_mean),
            (:defensive, team_defense_mean),
        )
            isfinite(mean) && mean >= 0.0 ||
                throw(ArgumentError(
                    "$cause posterior mean for team $team must be finite and nonnegative",
                ))
        end
        intervals = (
            (
                :td,
                _gamma_mixture_quantile(offense, 0.1),
                _gamma_mixture_quantile(offense, 0.9),
            ),
            (
                :defensive,
                _gamma_mixture_quantile(defense, 0.1),
                _gamma_mixture_quantile(defense, 0.9),
            ),
        )
        for (cause, lower, upper) in intervals
            lower <= upper ||
                throw(ArgumentError(
                    "$cause posterior interval for team $team must have ordered endpoints",
                ))
        end
        push!(offense_mean, team_offense_mean)
        push!(offense_lower, intervals[1][2])
        push!(offense_upper, intervals[1][3])
        push!(defense_mean, team_defense_mean)
        push!(defense_lower, intervals[2][2])
        push!(defense_upper, intervals[2][3])
    end

    return DataFrame(
        team=teams,
        offense_rate=offense_mean,
        offense_lower=offense_lower,
        offense_upper=offense_upper,
        defense_rate=defense_mean,
        defense_lower=defense_lower,
        defense_upper=defense_upper,
    )
end

function _load_glmakie()
    if !isdefined(@__MODULE__, :GLMakie)
        try
            Base.eval(@__MODULE__, :(import GLMakie))
        catch error
            throw(ErrorException(
                "could not load GLMakie for the interactive team-strength " *
                "plot: $(sprint(showerror, error))",
            ))
        end
    end
    return Base.eval(@__MODULE__, :(GLMakie))
end

function _team_strength_prior_percentile(prior::GammaParams, rate::Real)
    value = Float64(rate)
    isfinite(value) && value >= 0.0 ||
        throw(ArgumentError("team-strength rates must be finite and nonnegative"))
    scale = inv(prior.rate)
    isfinite(scale) && scale > 0.0 ||
        throw(ArgumentError("league Gamma prior scale must be finite and positive"))
    probability = cdf(Gamma(prior.shape, scale), value)
    isfinite(probability) && 0.0 <= probability <= 1.0 ||
        throw(ArgumentError("league Gamma prior CDF must be finite and within [0, 1]"))
    return 100.0 * probability
end

function _team_strength_plot_percentiles(
    data::AbstractDataFrame,
    prior::HazardPrior,
)
    percentiles = DataFrame(team=String.(data.team))
    for (reference, columns) in (
        (
            prior.td_hyperparameters,
            (
                :offense_rate => :offense_percentile,
                :offense_lower => :offense_lower_percentile,
                :offense_upper => :offense_upper_percentile,
            ),
        ),
        (
            prior.defensive_hyperparameters,
            (
                :defense_rate => :defense_percentile,
                :defense_lower => :defense_lower_percentile,
                :defense_upper => :defense_upper_percentile,
            ),
        ),
    )
        for (source, destination) in columns
            percentiles[!, destination] = [
                _team_strength_prior_percentile(reference, value)
                for value in data[!, source]
            ]
        end
    end
    all(percentiles.offense_lower_percentile .<= percentiles.offense_upper_percentile) &&
        all(percentiles.defense_lower_percentile .<= percentiles.defense_upper_percentile) ||
        throw(ArgumentError("team-strength intervals must have ordered endpoints"))
    return percentiles
end

function _team_strength_figure(
    makie::Module,
    data::AbstractDataFrame,
    context::RegularSeasonForecastContext,
)
    percentiles = _team_strength_plot_percentiles(data, context.model.prior)
    team_colors = makie.Makie.to_color.(
        _team_strength_team_colors(percentiles.team)
    )
    title = "Season $(context.season) team strengths as of start of week " *
        "$(context.as_of_week)"
    figure = makie.Figure(size=(1100, 760), figure_padding=24)
    axis = makie.Axis(
        figure[1, 1];
        title,
        xlabel="Offense rate percentile (league touchdown Gamma prior)",
        ylabel="Defense rate percentile (league defensive-event Gamma prior)",
        xticks=(0:10:100, ["$value%" for value in 0:10:100]),
        yticks=(0:10:100, ["$value%" for value in 0:10:100]),
    )
    makie.scatter!(
        axis,
        percentiles.offense_percentile,
        percentiles.defense_percentile;
        color=team_colors,
        markersize=12,
        strokecolor=:white,
        strokewidth=1,
    )
    makie.rangebars!(
        axis,
        percentiles.defense_percentile,
        percentiles.offense_lower_percentile,
        percentiles.offense_upper_percentile;
        direction=:x,
        color=team_colors,
        whiskerwidth=10,
    )
    makie.rangebars!(
        axis,
        percentiles.offense_percentile,
        percentiles.defense_lower_percentile,
        percentiles.defense_upper_percentile;
        direction=:y,
        color=team_colors,
        whiskerwidth=10,
    )
    makie.xlims!(axis, 0, 100)
    makie.ylims!(axis, 0, 100)
    makie.Label(
        figure[2, 1],
        "Neutral-site mean rates and central 80% posterior intervals from full team mixtures.\n" *
        "Mapped to cause-specific league Gamma prior percentiles; higher is stronger.",
        tellwidth=false,
        fontsize=13,
    )
    _team_strength_plot_labels!(makie, axis, percentiles, team_colors)
    return figure
end

function _show_team_strength_plot(
    context::RegularSeasonForecastContext;
    on_screen::Function=(_screen -> nothing),
)
    data = _team_strength_plot_data(context)
    makie = _load_glmakie()
    return Base.invokelatest(
        _show_team_strength_plot_latest!,
        makie,
        data,
        context;
        on_screen,
    )
end

function _show_team_strength_plot_latest!(
    makie::Module,
    data::AbstractDataFrame,
    context::RegularSeasonForecastContext;
    on_screen::Function=(_screen -> nothing),
)
    figure = _team_strength_figure(makie, data, context)
    try
        screen = try
            makie.activate!()
            makie.Screen(
                figure.scene;
                start_renderloop=false,
                visible=true,
                title="SurvivorModel team strength",
            )
        catch error
            throw(ErrorException(
                "could not open the GLMakie team-strength window; a working " *
                "desktop/OpenGL display is required: $(sprint(showerror, error))",
            ))
        end
        try
            Base.invokelatest(on_screen, screen)
            if isopen(screen)
                makie.start_renderloop!(screen)
                wait(screen)
            end
        finally
            isopen(screen) && close(screen)
        end
    finally
        makie.empty!(figure)
    end
    return nothing
end
