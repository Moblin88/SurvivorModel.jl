using DataFrames
using Dates
using Test
using SurvivorModel

if !isdefined(Main, :_survivor_context_fixture)
    function _survivor_context_fixture()
        schedule = DataFrame(
            game_id=["2022_01_AB", "2023_01_AB", "2023_02_CD"],
            season=[2022, 2023, 2023],
            game_type=["REG", "REG", "REG"],
            week=[1, 1, 2],
            away_team=["A", "A", "C"],
            home_team=["B", "B", "D"],
            away_score=Union{Missing,Int}[14, 17, missing],
            home_score=Union{Missing,Int}[21, 24, missing],
            result=Union{Missing,Int}[7, 7, missing],
        )
        historical = DataFrame(
            game_id=["2022_01_AB", "2022_01_AB", "2022_01_AB", "2022_01_AB"],
            fixed_drive=[1, 2, 3, 4],
            posteam=["B", "A", "B", "A"],
            defteam=["A", "B", "A", "B"],
            posteam_home=[true, false, true, false],
            defteam_home=[false, true, false, true],
            drive_result=["Touchdown", "Punt", "Field goal", "Punt"],
            time_of_possession=Second.([60, 60, 60, 60]),
            home_spread_change=[7.0, 0.0, 3.0, 0.0],
        )
        current = DataFrame(
            game_id=["2023_01_AB", "2023_01_AB"],
            fixed_drive=[1, 2],
            posteam=["B", "A"],
            defteam=["A", "B"],
            posteam_home=[true, false],
            defteam_home=[false, true],
            drive_result=["Touchdown", "Punt"],
            time_of_possession=Second.([60, 60]),
            home_spread_change=[7.0, 0.0],
        )
        return schedule, historical, current
    end
end

@testset "survivor command-line application" begin
    @testset "argument and stdin parsing" begin
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023"],
        ) == (
            show_help=false,
            season=2023,
            initial_strikes=2,
            objective=:exact_milp,
            hessian_weeks=3,
            timeout_seconds=nothing,
            timings=false,
            refresh_data=false,
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--strikes=4"],
        ).initial_strikes == 4
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--objective", "exact-milp"],
        ).objective == :exact_milp
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--objective", "fixed-exact-milp"],
        ).objective == :fixed_exact_milp
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--timeout", "12.5"],
        ).timeout_seconds == 12.5
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--timeout=12.5"],
        ).timeout_seconds == 12.5
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--hessian-weeks", "6"],
        ).hessian_weeks == 6
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--hessian-weeks=6"],
        ).hessian_weeks == 6
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--timings"],
        ).timings
        @test SurvivorModel._read_survivor_cli_picks(
            IOBuffer("KC\n\n sf \n"),
        ) == ["KC", "SF"]
        @test SurvivorModel._parse_survivor_cli_args(["--help"]).show_help
        usage = SurvivorModel._survivor_cli_usage()
        @test !occursin("--benchmark", usage)
        @test !occursin("--clear-cache", usage)
        @test occursin("exact-milp", usage)
        @test !occursin("micp", usage)
        @test occursin("--timings", usage)
        @test occursin("--timeout", usage)
        @test occursin("--hessian-weeks", usage)
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--refresh-data"],
        ).refresh_data
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--refresh-data", "--refresh-data"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--timings", "--timings"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--timeout", "0"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--timeout", "NaN"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--timeout", "Inf"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--timeout", "2", "--timeout", "3"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--hessian-weeks", "-1"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--hessian-weeks", "2", "--hessian-weeks", "3"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--strikes", "2"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--objective", "unknown"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--objective", "discounted_expected_wins"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--objective", "expected_weeks_before_elimination"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--objective", "micp"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--objective", "milp",
             "--objective", "exact-milp"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--benchmark", "--season", "2023"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--clear-cache", "--season", "2023"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--data", "synthetic", "--season", "2023"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--scenario", "recovery", "--season", "2023"],
        )
        @test_throws ArgumentError SurvivorModel._read_survivor_cli_picks(
            IOBuffer("KC SF\n"),
        )
    end

    @testset "synthetic fitting benchmark API" begin
        synthetic_drives = SurvivorModel.synthetic_fit_benchmark_drives(
            drives_per_team=10,
        )
        benchmark = SurvivorModel.fit_benchmark(
            synthetic_drives;
            current_season=2024,
            repeats=1,
            methods=(SurvivorModel.MomentFit(),),
        )
        @test nrow(benchmark) == 1
        @test length(unique(synthetic_drives.defteam)) > 1
        @test only(benchmark.method) == "moment"
    end

    @testset "synthetic recovery benchmark API" begin
        @test SurvivorModel.FIT_RECOVERY_GAMES_PER_TEAM == 17
        @test SurvivorModel.FIT_RECOVERY_APPROX_DRIVES_PER_TEAM_GAME == 11
        simulation = SurvivorModel.synthetic_fit_recovery_drives(
            games_per_team=8,
            seed=17,
        )
        @test length(simulation.truth.seasons) == 3
        @test length(simulation.truth.teams) == 32
        @test length(simulation.truth.td_hyperparameters) == 3
        @test nrow(simulation.schedule) == 3 * 8 * 16
        @test nrow(simulation.games) == nrow(simulation.schedule)
        @test all(simulation.games.drive_count .> 0)
        benchmark = SurvivorModel.fit_recovery_benchmark(
            simulation.drives,
            simulation.truth;
            current_season=2024,
            repeats=1,
            methods=(SurvivorModel.MomentFit(),),
        )
        @test nrow(benchmark.summary) == 1
        @test !isempty(benchmark.recovery_quality)
    end

    @testset "longer synthetic recovery history" begin
        simulation = SurvivorModel.synthetic_fit_recovery_drives(
            seasons=2019:2023,
            games_per_team=1,
            seed=23,
        )
        @test simulation.truth.seasons == collect(2019:2023)
        @test nrow(simulation.schedule) == 5 * 1 * 16

        benchmark = SurvivorModel.fit_recovery_benchmark(
            simulation.drives,
            simulation.truth;
            max_seasons=5,
            current_season=2024,
            repeats=1,
            methods=(SurvivorModel.MomentFit(),),
        )
        @test nrow(benchmark.summary) == 1
        @test only(benchmark.summary.converged)
        @test !isempty(benchmark.recovery_quality)
    end

    @testset "developer benchmark tool" begin
        project_directory = dirname(@__DIR__)
        tool_path = joinpath(project_directory, "tools", "fit_benchmark.jl")
        help = read(
            `$(Base.julia_cmd()) --project=$project_directory $tool_path --help`,
            String,
        )
        @test occursin("tools/fit_benchmark.jl", help)
        @test occursin("--scenario NAME", help)
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
        two_loss_schedule = copy(schedule)
        two_loss_schedule.result[two_loss_schedule.week .== 2] .= 7
        @test_throws ArgumentError SurvivorModel._survivor_cli_state(
            SurvivorModel.load_schedule(two_loss_schedule),
            2023,
            ["A", "C"],
            2,
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
                method=SurvivorModel.MomentFit(),
                cache_directory=cache_directory,
            )
            @test !first.cache_hit
            @test isfile(first.path)

            second = SurvivorModel._cached_historical_prior(
                historical;
                current_season=2023,
                method=SurvivorModel.MomentFit(),
                cache_directory=cache_directory,
            )
            @test second.cache_hit
            @test second.path == first.path
            @test second.prior.time_edges == first.prior.time_edges
            @test second.data_fingerprint == first.data_fingerprint

            changed_historical = copy(historical)
            changed_historical.drive_result[1] =
                changed_historical.drive_result[1] == "Touchdown" ?
                "Turnover" :
                "Touchdown"
            changed = SurvivorModel._cached_historical_prior(
                changed_historical;
                current_season=2023,
                method=SurvivorModel.MomentFit(),
                cache_directory=cache_directory,
            )
            @test !changed.cache_hit
            @test changed.data_fingerprint != first.data_fingerprint
            @test changed.path != first.path

            open(first.path, "w") do io
                write(io, "not a serialized cache")
            end
            @test_throws ArgumentError SurvivorModel._cached_historical_prior(
                historical;
                current_season=2023,
                method=SurvivorModel.MomentFit(),
                cache_directory=cache_directory,
            )
        end
    end

    @testset "historical drive summary cache integration" begin
        _, historical, _ = _survivor_context_fixture()
        summarized = DataFrame(historical, copycols=true)
        summarized.drive_start_yards_to_goal = fill(50, nrow(summarized))
        summarized.yards_gained = zeros(nrow(summarized))
        calls = Ref(0)
        loader = _ -> begin
            calls[] += 1
            summarized
        end
        mktempdir() do cache_directory
            first = SurvivorModel._survivor_cli_load_historical_drives(
                2023,
                1,
                nothing,
                cache_directory;
                loader=loader,
            )
            second = SurvivorModel._survivor_cli_load_historical_drives(
                2023,
                1,
                nothing,
                cache_directory;
                loader=loader,
            )
            @test calls[] == 1
            @test first == second
        end
    end

    @testset "fixture-backed current pick" begin
        schedule, historical, current = _survivor_context_fixture()
        mktempdir() do cache_directory
            output = IOBuffer()
            timing_output = IOBuffer()
            exit_code = SurvivorModel._run_survivor_cli(
                ["--season", "2023", "--timings"];
                input=IOBuffer("A\n"),
                output=output,
                timing_output=timing_output,
                schedule=schedule,
                historical_drives=historical,
                current_drives=current,
                cache_directory=cache_directory,
                method=SurvivorModel.MomentFit(),
                through_week=2,
            )
            @test exit_code == 0
            @test String(take!(output)) in ("C\n", "D\n")
            timing_text = String(take!(timing_output))
            @test occursin("survivor phase complete", timing_text)
            @test occursin("optimize_start", timing_text)
            @test occursin("optimize", timing_text)
            @test occursin("survivor MILP solve complete", timing_text)
        end

        completed_schedule = copy(schedule)
        completed_schedule.result = [7, 7, 3]
        mktempdir() do cache_directory
            output = IOBuffer()
            exit_code = SurvivorModel._run_survivor_cli(
                ["--season", "2023"];
                input=IOBuffer(),
                output=output,
                schedule=completed_schedule,
                historical_drives=historical,
                current_drives=current,
                cache_directory=cache_directory,
                method=SurvivorModel.MomentFit(),
                through_week=2,
            )
            @test exit_code == 0
            @test !isempty(strip(String(take!(output))))
        end
    end
end
