using DataFrames
using Dates
using Test
using SurvivorModel

@testset "survivor command-line application" begin
    @testset "argument and stdin parsing" begin
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023"],
        ) == (
            show_help=false,
            benchmark=false,
            data_source=:synthetic,
            benchmark_scenario=:performance,
            season=2023,
            initial_strikes=2,
            repeats=3,
            max_seasons=5,
            recovery_seasons=3,
            clear_cache=false,
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--strikes=4"],
        ).initial_strikes == 4
        @test SurvivorModel._parse_survivor_cli_args(
            ["--benchmark", "--data", "real", "--season", "2023", "--repeats", "2"],
        ) == (
            show_help=false,
            benchmark=true,
            data_source=:real,
            benchmark_scenario=:performance,
            season=2023,
            initial_strikes=2,
            repeats=2,
            max_seasons=5,
            recovery_seasons=3,
            clear_cache=false,
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--benchmark"],
        ).season == 2024
        @test SurvivorModel._parse_survivor_cli_args(
            ["--benchmark", "--scenario", "recovery"],
        ).benchmark_scenario == :recovery
        @test SurvivorModel._parse_survivor_cli_args(
            [
                "--benchmark",
                "--scenario",
                "recovery",
                "--recovery-seasons",
                "5",
            ],
        ).recovery_seasons == 5
        @test SurvivorModel._read_survivor_cli_picks(
            IOBuffer("KC\n\n sf \n"),
        ) == ["KC", "SF"]
        @test SurvivorModel._parse_survivor_cli_args(["--help"]).show_help
        @test SurvivorModel._parse_survivor_cli_args(
            ["--clear-cache"],
        ) == (
            show_help=false,
            benchmark=false,
            data_source=:synthetic,
            benchmark_scenario=:performance,
            season=0,
            initial_strikes=2,
            repeats=3,
            max_seasons=5,
            recovery_seasons=3,
            clear_cache=true,
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--clear-cache", "--season", "2023"],
        ).clear_cache
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--clear-cache", "--clear-cache"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--clear-cache", "--strikes", "1"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--strikes", "2"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--benchmark", "--data", "real"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--data", "synthetic", "--season", "2023"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--benchmark", "--data", "real", "--scenario", "recovery", "--season", "2023"],
        )
        @test_throws ArgumentError SurvivorModel._read_survivor_cli_picks(
            IOBuffer("KC SF\n"),
        )
    end

    @testset "synthetic fitting benchmark" begin
        options = SurvivorModel._parse_survivor_cli_args(
            ["--benchmark", "--repeats", "1"],
        )
        output = IOBuffer()
        synthetic_drives = synthetic_fit_benchmark_drives(
            drives_per_team=10,
        )
        exit_code = SurvivorModel._run_fit_benchmark_cli(
            options;
            output=output,
            drives=synthetic_drives,
            methods=(MomentFit(),),
        )
        rendered = String(take!(output))
        @test exit_code == 0
        @test length(unique(synthetic_drives.defteam)) > 1
        @test occursin("Fitting benchmark", rendered)
        @test occursin("median_ms", rendered)
        @test occursin("moment", rendered)
        @test occursin("Fitted hazard parameters", rendered)
        @test occursin("synthetic target", rendered)
        @test occursin("home_advantage", rendered)
        @test occursin("persistence", rendered)
        @test occursin("hazard_mean", rendered)
        @test occursin("hazard_variance", rendered)
    end

    @testset "synthetic recovery benchmark" begin
        @test SurvivorModel.FIT_RECOVERY_GAMES_PER_TEAM == 17
        @test SurvivorModel.FIT_RECOVERY_APPROX_DRIVES_PER_TEAM_GAME == 11
        options = SurvivorModel._parse_survivor_cli_args(
            ["--benchmark", "--scenario", "recovery", "--repeats", "1"],
        )
        simulation = synthetic_fit_recovery_drives(
            games_per_team=8,
            seed=17,
        )
        @test length(simulation.truth.seasons) == 3
        @test length(simulation.truth.teams) == 32
        @test length(simulation.truth.td_hyperparameters) == 3
        @test nrow(simulation.schedule) == 3 * 8 * 16
        @test nrow(simulation.games) == nrow(simulation.schedule)
        @test all(simulation.games.drive_count .> 0)
        output = IOBuffer()
        exit_code = SurvivorModel._run_fit_benchmark_cli(
            options;
            output=output,
            drives=simulation.drives,
            truth=simulation.truth,
            methods=(MomentFit(),),
        )
        rendered = String(take!(output))
        @test exit_code == 0
        @test occursin("scenario: recovery", rendered)
        @test occursin("recovery truth", rendered)
        @test occursin("Recovery quality", rendered)
        @test occursin("Shared-parameter recovery", rendered)
        @test occursin("Per-bin hazard recovery", rendered)
        @test occursin("mean_rel_err", rendered)
    end

    @testset "longer synthetic recovery history" begin
        simulation = synthetic_fit_recovery_drives(
            seasons=2019:2023,
            games_per_team=1,
            seed=23,
        )
        @test simulation.truth.seasons == collect(2019:2023)
        @test nrow(simulation.schedule) == 5 * 1 * 16

        benchmark = fit_recovery_benchmark(
            simulation.drives,
            simulation.truth;
            max_seasons=5,
            current_season=2024,
            repeats=1,
            methods=(MomentFit(),),
        )
        @test nrow(benchmark.summary) == 1
        @test only(benchmark.summary.converged)
        @test !isempty(benchmark.recovery_quality)
    end

    @testset "schedule-derived survivor state" begin
        schedule, historical, _ = _survivor_context_fixture()
        normalized = SurvivorModel.load_schedule(schedule)
        state = SurvivorModel._survivor_cli_state(
            normalized,
            2023,
            ["a"],
            2,
        )
        @test state.current_week == 2
        @test state.losses == 1
        @test state.strikes_remaining == 1
        @test state.picks_made == Dict(1 => "A")
        @test_throws ArgumentError SurvivorModel._survivor_cli_state(
            normalized,
            2023,
            ["A"],
            0,
        )

        tie_schedule = copy(schedule)
        tie_schedule.result[
            (tie_schedule.season .== 2023) .&
            (tie_schedule.week .== 1),
        ] .= 0
        tie_state = SurvivorModel._survivor_cli_state(
            SurvivorModel.load_schedule(tie_schedule),
            2023,
            ["A"],
            2,
        )
        @test tie_state.losses == 1

        postseason_schedule = vcat(
            schedule,
            DataFrame(
                game_id=["2022_postseason"],
                season=[2022],
                game_type=["POST"],
                week=[1],
                away_team=["A"],
                home_team=["B"],
                away_score=Union{Missing,Int}[0],
                home_score=Union{Missing,Int}[35],
                result=Union{Missing,Int}[35],
            );
            cols=:union,
        )
        postseason_drives = vcat(
            historical,
            DataFrame(
                game_id=["2022_postseason"],
                fixed_drive=[1],
                posteam=["A"],
                defteam=["B"],
                posteam_home=[false],
                defteam_home=[true],
                drive_result=["Touchdown"],
                time_of_possession=Second.([60]),
                home_spread_change=[0.0],
            );
            cols=:union,
        )
        regular_historical = SurvivorModel._survivor_cli_historical_drives(
            SurvivorModel.load_schedule(postseason_schedule),
            2023,
            postseason_drives,
        )
        @test !any(regular_historical.game_id .== "2022_postseason")

        @test_throws ArgumentError SurvivorModel._survivor_cli_state(
            normalized,
            2023,
            ["A", "A"],
            2,
        )
        @test_throws ArgumentError SurvivorModel._survivor_cli_state(
            normalized,
            2023,
            ["A", "A", "A"],
            2,
        )
    end

    @testset "historical prior cache" begin
        _, historical, _ = _survivor_context_fixture()
        mktempdir() do cache_directory
            first = SurvivorModel._cached_historical_prior(
                historical;
                current_season=2023,
                method=MomentFit(),
                cache_directory=cache_directory,
            )
            @test !first.cache_hit
            @test isfile(first.path)

            second = SurvivorModel._cached_historical_prior(
                historical;
                current_season=2023,
                method=MomentFit(),
                cache_directory=cache_directory,
            )
            @test second.cache_hit
            @test second.path == first.path
            @test second.prior.time_edges == first.prior.time_edges
        end
    end

    @testset "fixture-backed current pick" begin
        schedule, historical, current = _survivor_context_fixture()
        mktempdir() do cache_directory
            output = IOBuffer()
            exit_code = SurvivorModel._run_survivor_cli(
                ["--season", "2023"];
                input=IOBuffer("A\n"),
                output=output,
                schedule=schedule,
                historical_drives=historical,
                current_drives=current,
                cache_directory=cache_directory,
                method=MomentFit(),
                through_week=2,
            )
            @test exit_code == 0
            @test String(take!(output)) in ("C\n", "D\n")
        end
    end
end
