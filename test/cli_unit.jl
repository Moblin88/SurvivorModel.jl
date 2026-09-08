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
            season=2023,
            initial_strikes=2,
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--strikes=4"],
        ).initial_strikes == 4
        @test SurvivorModel._read_survivor_cli_picks(
            IOBuffer("KC\n\n sf \n"),
        ) == ["KC", "SF"]
        @test SurvivorModel._parse_survivor_cli_args(["--help"]).show_help
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--strikes", "2"],
        )
        @test_throws ArgumentError SurvivorModel._read_survivor_cli_picks(
            IOBuffer("KC SF\n"),
        )
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
