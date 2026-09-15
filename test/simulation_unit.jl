using DataFrames
using Dates
using Random
using Test
import SurvivorModel: simulate_renewal_schedule
using SurvivorModel

function _renewal_simulation_model_fixture(
    ;
    td_persistence=1.0,
    defensive_persistence=0.0,
    rate=0.01,
)
    time_edges = [0.0, 30.0, Inf]
    hyperparameters = [
        GammaParams(2.0, 2.0 / rate),
        GammaParams(2.0, 2.0 / rate),
    ]
    prior = HazardPrior(
        time_edges,
        hyperparameters,
        hyperparameters,
        Dict{String,Vector{GammaMixture}}(),
        Dict{String,Vector{GammaMixture}}(),
        1.0,
        1.0,
        td_persistence,
        defensive_persistence,
        Int[],
        nothing,
        nothing,
    )
    empty_drives = DataFrame(
        game_id=String[],
        fixed_drive=Int[],
        posteam=String[],
        defteam=String[],
        posteam_home=Bool[],
        defteam_home=Bool[],
        drive_result=String[],
        time_of_possession=Second[],
    )
    return fit_hazard_model(
        empty_drives;
        prior=prior,
        time_edges=time_edges,
    )
end

function _renewal_simulation_schedule_fixture()
    return DataFrame(
        game_id=["2022-2", "2021-2", "2022-1", "2021-1"],
        season=[2022, 2021, 2022, 2021],
        game_type=["REG", "REG", "REG", "REG"],
        week=[2, 2, 1, 1],
        away_team=["A", "B", "A", "B"],
        home_team=["B", "A", "B", "A"],
        result=Union{Missing,Int}[missing, missing, missing, missing],
    )
end

@testset "abstract schedule renewal simulation" begin
    model = _renewal_simulation_model_fixture()
    schedule = _renewal_simulation_schedule_fixture()
    simulation = simulate_renewal_schedule(
        model,
        schedule;
        rng=MersenneTwister(17),
        horizon=120,
        opening_possession=:home,
    )

    @test simulation isa NamedTuple
    @test simulation.games.season == [2021, 2021, 2022, 2022]
    @test simulation.games.week == [1, 2, 1, 2]
    @test nrow(simulation.games) == 4
    @test all(simulation.games.simulated_seconds .== 120)
    @test all(
        simulation.games.drive_count .==
        simulation.games.home_drive_count .+
        simulation.games.away_drive_count,
    )
    @test all(
        simulation.games.opening_possession .== fill(:home, nrow(simulation.games)),
    )

    for game in groupby(simulation.drives, :game_id)
        @test sum(Dates.value.(game.time_of_possession)) == 120
        @test all(
            result -> result in ("Touchdown", "Punt", "End of half"),
            game.drive_result,
        )
        @test all(Dates.value.(game.time_of_possession) .>= 1)
        @test all(game.posteam .!= game.defteam)
        if nrow(game) > 1
            @test all(game.posteam[2:end] .== game.defteam[1:(end - 1)])
            @test all(game.defteam[2:end] .== game.posteam[1:(end - 1)])
        end
    end

    @test simulation.drives isa DataFrame
    @test all(
        name in propertynames(simulation.drives)
        for name in (
            :game_id,
            :fixed_drive,
            :season,
            :posteam,
            :defteam,
            :posteam_home,
            :defteam_home,
            :drive_result,
            :time_of_possession,
        )
    )
    @test nrow(simulation.latent_rates) == 2 * 2 * 2 * 2
    @test sort(unique(simulation.latent_rates.season)) == [2021, 2022]
    @test all(
        count(==(season), simulation.latent_rates.season) == 8
        for season in (2021, 2022)
    )

    td_rates = simulation.latent_rates[
        simulation.latent_rates.outcome .== :td,
        :,
    ]
    for team in unique(td_rates.team)
        team_rates = td_rates[td_rates.team .== team, :]
        @test team_rates.baseline_rate[1] == team_rates.baseline_rate[3]
        @test team_rates.baseline_rate[2] == team_rates.baseline_rate[4]
    end

    defensive_rates = simulation.latent_rates[
        simulation.latent_rates.outcome .== :defensive,
        :,
    ]
    @test any(
        defensive_rates.baseline_rate[1:2] .!=
        defensive_rates.baseline_rate[3:4],
    )

    fitted = fit_hazard_model(
        simulation.drives;
        prior=model.prior,
        time_edges=model.time_edges,
    )
    @test fitted isa HazardModel
    @test nrow(build_exposure_data(simulation.drives; time_edges=model.time_edges)[1]) > 0

    random_one = simulate_renewal_schedule(
        model,
        schedule;
        rng=MersenneTwister(31),
        horizon=30,
        opening_possession=:random,
    )
    random_two = simulate_renewal_schedule(
        model,
        schedule;
        rng=MersenneTwister(31),
        horizon=30,
        opening_possession=:random,
    )
    @test random_one.games.opening_possession == random_two.games.opening_possession
    @test all(
        possession -> possession in (:home, :away),
        random_one.games.opening_possession,
    )

    censored_model = _renewal_simulation_model_fixture(rate=1.0e-12)
    censored = simulate_renewal_schedule(
        censored_model,
        schedule[1:1, :];
        rng=MersenneTwister(1),
        horizon=10,
        opening_possession=:home,
    )
    @test only(censored.drives.drive_result) == "End of half"
    @test only(censored.drives.time_of_possession) == Second(10)

    @test_throws ArgumentError simulate_renewal_schedule(
        model,
        schedule;
        horizon=1.5,
    )
    invalid_schedule = copy(schedule)
    invalid_schedule.home_team[1] = invalid_schedule.away_team[1]
    @test_throws ArgumentError simulate_renewal_schedule(model, invalid_schedule)
end
