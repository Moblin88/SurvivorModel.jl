const RENEWAL_SIMULATION_KINDS = (:td, :defensive)

function _sample_renewal_gamma_mixture(
    rng::AbstractRNG,
    mixture::GammaMixture,
)
    component = mixture.components[
        rand(rng, Categorical(mixture.weights))
    ]
    return rand(rng, Gamma(component.shape, 1.0 / component.rate))
end

function _renewal_simulation_hyperparameters(
    model::HazardModel,
    kind::Symbol,
)
    kind === :td && return model.prior.td_hyperparameters
    kind === :defensive && return model.prior.defensive_hyperparameters
    throw(ArgumentError("kind must be :td or :defensive; got $kind"))
end

function _renewal_simulation_persistence(
    model::HazardModel,
    kind::Symbol,
)
    return hazard_persistence(model.prior, kind)
end

function _renewal_simulation_sample_rates!(
    rates::Dict{Tuple{Symbol,String,Int},Float64},
    model::HazardModel,
    teams::AbstractVector{<:AbstractString},
    season::Integer,
    rng::AbstractRNG,
    season_values::Vector{Int},
    season_teams::Vector{String},
    season_kinds::Vector{Symbol},
    season_bins::Vector{Int},
    season_rates::Vector{Float64},
)
    n_bins = length(model.time_edges) - 1
    for team in teams, kind in RENEWAL_SIMULATION_KINDS, time_bin in 1:n_bins
        mixture = hazard_posterior(model, kind, team, time_bin)
        rate = _sample_renewal_gamma_mixture(rng, mixture)
        rates[(kind, String(team), time_bin)] = rate
        push!(season_values, Int(season))
        push!(season_teams, String(team))
        push!(season_kinds, kind)
        push!(season_bins, time_bin)
        push!(season_rates, rate)
    end
    return rates
end

function _renewal_simulation_advance_rates!(
    rates::Dict{Tuple{Symbol,String,Int},Float64},
    model::HazardModel,
    teams::AbstractVector{<:AbstractString},
    rng::AbstractRNG,
)
    n_bins = length(model.time_edges) - 1
    for team in teams, kind in RENEWAL_SIMULATION_KINDS, time_bin in 1:n_bins
        key = (kind, String(team), time_bin)
        persistence = _renewal_simulation_persistence(model, kind)
        if rand(rng) >= persistence
            hyperparameters = _renewal_simulation_hyperparameters(model, kind)
            parameter = hyperparameters[time_bin]
            rates[key] = rand(rng, Gamma(parameter.shape, 1.0 / parameter.rate))
        end
    end
    return rates
end

function _renewal_simulation_record_rates!(
    season_values::Vector{Int},
    season_teams::Vector{String},
    season_kinds::Vector{Symbol},
    season_bins::Vector{Int},
    season_rates::Vector{Float64},
    rates::Dict{Tuple{Symbol,String,Int},Float64},
    teams::AbstractVector{<:AbstractString},
    model::HazardModel,
    season::Integer,
)
    n_bins = length(model.time_edges) - 1
    for team in teams, kind in RENEWAL_SIMULATION_KINDS, time_bin in 1:n_bins
        push!(season_values, Int(season))
        push!(season_teams, String(team))
        push!(season_kinds, kind)
        push!(season_bins, time_bin)
        push!(season_rates, rates[(kind, String(team), time_bin)])
    end
    return nothing
end

function _renewal_simulation_horizon(horizon::Real)
    value = Float64(horizon)
    isfinite(value) && value > 0.0 ||
        throw(ArgumentError("horizon must be finite and positive"))
    value == floor(value) ||
        throw(ArgumentError("horizon must be an integer number of seconds"))
    value <= typemax(Int) ||
        throw(ArgumentError("horizon is too large for integer-second simulation"))
    return Int(value)
end

function _renewal_simulation_opening_possession(
    rng::AbstractRNG,
    opening_possession::Symbol,
)
    opening_possession === :home && return true
    opening_possession === :away && return false
    opening_possession === :random && return rand(rng, Bool)
    throw(ArgumentError(
        "opening_possession must be :random, :home, or :away",
    ))
end

function _renewal_simulation_rate(
    rates::Dict{Tuple{Symbol,String,Int},Float64},
    model::HazardModel,
    kind::Symbol,
    team::AbstractString,
    time_bin::Int,
    ;
    home::Bool,
)
    baseline = rates[(kind, String(team), time_bin)]
    multiplier = home ? home_multiplier(model.prior, kind) : 1.0
    return baseline * multiplier
end

function _sample_renewal_drive(
    model::HazardModel,
    rates::Dict{Tuple{Symbol,String,Int},Float64},
    home_team::AbstractString,
    away_team::AbstractString,
    possessing_home::Bool,
    remaining_seconds::Int,
    rng::AbstractRNG,
)
    remaining_seconds > 0 ||
        throw(ArgumentError("remaining game time must be positive"))
    posteam = possessing_home ? home_team : away_team
    defteam = possessing_home ? away_team : home_team
    n_bins = length(model.time_edges) - 1
    elapsed = 0.0

    for time_bin in 1:n_bins
        width = model.time_edges[time_bin + 1] -
            model.time_edges[time_bin]
        available_width = isinf(width) ?
            remaining_seconds - elapsed :
            min(width, remaining_seconds - elapsed)
        available_width > 0.0 ||
            return (duration=remaining_seconds, event=:censored)
        td_rate = _renewal_simulation_rate(
            rates,
            model,
            :td,
            posteam,
            time_bin;
            home=possessing_home,
        )
        defensive_rate = _renewal_simulation_rate(
            rates,
            model,
            :defensive,
            defteam,
            time_bin;
            home=!possessing_home,
        )
        total_rate = td_rate + defensive_rate
        isfinite(total_rate) && total_rate > 0.0 ||
            throw(ArgumentError("simulated hazards must be finite and positive"))

        waiting = randexp(rng) / total_rate
        if waiting <= available_width
            raw_duration = elapsed + waiting
            duration = clamp(
                max(1, ceil(Int, raw_duration)),
                1,
                remaining_seconds,
            )
            event = rand(rng) < td_rate / total_rate ?
                :td :
                :defensive
            return (duration=duration, event=event)
        end
        if isinf(width) || available_width < width
            return (duration=remaining_seconds, event=:censored)
        end
        elapsed += width
    end

    return (duration=remaining_seconds, event=:censored)
end

function _renewal_simulation_event_label(event::Symbol)
    event === :td && return "Touchdown"
    event === :defensive && return "Punt"
    event === :censored && return "End of half"
    throw(ArgumentError("unknown simulated event: $event"))
end

"""
    simulate_renewal_schedule(model, schedule; kwargs...) -> NamedTuple

Simulate the abstract two-outcome renewal process over a hypothetical schedule.
The schedule is normalized with `load_schedule` and may contain multiple
seasons; its `result` values may be missing. The first simulated season samples
one latent Gamma-mixture hazard per team, outcome, and time bin from the
model's current posterior. Those rates are reused across every game in that
season. Each later season retains or redraws each realized rate according to
the fitted outcome-specific persistence probability and Gamma hyperparameter.

Each game starts with a 50/50 random opening possession by default. Drives
alternate between the home and away teams until the integer-second horizon is
exhausted. Drive durations are rounded up to seconds, and a final interval
without a terminal event is emitted as `End of half`, which is compatible with
`build_exposure_data` and `fit_hazard_model`. Score rewards are intentionally
not simulated.
"""
function simulate_renewal_schedule(
    model::HazardModel,
    schedule::AbstractDataFrame;
    rng::AbstractRNG=Random.default_rng(),
    horizon::Real=GAME_CLOCK_SECONDS,
    opening_possession::Symbol=:random,
)
    horizon_seconds = _renewal_simulation_horizon(horizon)
    opening_possession in (:random, :home, :away) ||
        throw(ArgumentError(
            "opening_possession must be :random, :home, or :away",
        ))

    normalized_schedule = load_schedule(schedule)
    nrow(normalized_schedule) > 0 ||
        throw(ArgumentError("schedule must contain at least one game"))
    any(
        normalized_schedule.home_team .==
        normalized_schedule.away_team,
    ) &&
        throw(ArgumentError("a scheduled game must have distinct teams"))

    games = sort(
        DataFrame(normalized_schedule),
        [:season, :week, :game_id],
    )
    seasons = sort(unique(Int.(games.season)))
    teams = sort!(
        unique(
            vcat(
                String.(games.home_team),
                String.(games.away_team),
            ),
        ),
    )

    rates = Dict{Tuple{Symbol,String,Int},Float64}()
    latent_seasons = Int[]
    latent_teams = String[]
    latent_kinds = Symbol[]
    latent_bins = Int[]
    latent_values = Float64[]
    _renewal_simulation_sample_rates!(
        rates,
        model,
        teams,
        first(seasons),
        rng,
        latent_seasons,
        latent_teams,
        latent_kinds,
        latent_bins,
        latent_values,
    )

    drive_data = DataFrame(
        game_id=String[],
        fixed_drive=Int[],
        season=Int[],
        week=Int[],
        game_type=String[],
        posteam=String[],
        defteam=String[],
        posteam_home=Bool[],
        defteam_home=Bool[],
        drive_result=String[],
        time_of_possession=Second[],
    )
    game_data = DataFrame(
        game_id=String[],
        season=Int[],
        week=Int[],
        game_type=String[],
        away_team=String[],
        home_team=String[],
        opening_possession=Symbol[],
        drive_count=Int[],
        home_drive_count=Int[],
        away_drive_count=Int[],
        simulated_seconds=Int[],
        terminal_censored=Bool[],
    )

    previous_season = first(seasons)
    for season in seasons
        if season != first(seasons)
            for _ in 1:(season - previous_season)
                _renewal_simulation_advance_rates!(
                    rates,
                    model,
                    teams,
                    rng,
                )
            end
            _renewal_simulation_record_rates!(
                latent_seasons,
                latent_teams,
                latent_kinds,
                latent_bins,
                latent_values,
                rates,
                teams,
                model,
                season,
            )
        end

        season_games = games[games.season .== season, :]
        for row in eachrow(season_games)
            possessing_home = _renewal_simulation_opening_possession(
                rng,
                opening_possession,
            )
            opening = possessing_home ? :home : :away
            remaining_seconds = horizon_seconds
            fixed_drive = 0
            home_drive_count = 0
            away_drive_count = 0
            terminal_censored = false

            while remaining_seconds > 0
                fixed_drive += 1
                sample = _sample_renewal_drive(
                    model,
                    rates,
                    row.home_team,
                    row.away_team,
                    possessing_home,
                    remaining_seconds,
                    rng,
                )
                duration = sample.duration
                event = sample.event
                push!(
                    drive_data,
                    (
                        String(row.game_id),
                        fixed_drive,
                        Int(row.season),
                        Int(row.week),
                        String(row.game_type),
                        possessing_home ?
                            String(row.home_team) :
                            String(row.away_team),
                        possessing_home ?
                            String(row.away_team) :
                            String(row.home_team),
                        possessing_home,
                        !possessing_home,
                        _renewal_simulation_event_label(event),
                        Second(duration),
                    ),
                )
                if possessing_home
                    home_drive_count += 1
                else
                    away_drive_count += 1
                end
                remaining_seconds -= duration
                if event === :censored
                    terminal_censored = true
                    break
                end
                possessing_home = !possessing_home
            end

            push!(
                game_data,
                (
                    String(row.game_id),
                    Int(row.season),
                    Int(row.week),
                    String(row.game_type),
                    String(row.away_team),
                    String(row.home_team),
                    opening,
                    fixed_drive,
                    home_drive_count,
                    away_drive_count,
                    horizon_seconds - remaining_seconds,
                    terminal_censored,
                ),
            )
        end
        previous_season = season
    end

    latent_rates = DataFrame(
        season=latent_seasons,
        team=latent_teams,
        outcome=latent_kinds,
        time_bin=latent_bins,
        baseline_rate=latent_values,
    )
    return (
        drives=drive_data,
        games=game_data,
        latent_rates=latent_rates,
    )
end
