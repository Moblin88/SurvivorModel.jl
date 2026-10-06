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

@testset "survivor HiGHS LP export" begin
    schedule, historical, current = _survivor_context_fixture()
    context = fit_regular_season_forecast(
        2023;
        as_of_week=1,
        schedule=schedule,
        historical_drives=historical,
        current_drives=current,
    )
    selection_config = SurvivorSelectionConfig(
        minimum_favorite_spread=nothing,
        market_guard_weeks=0,
        through_week=2,
        hessian_weeks=2,
        benders_weeks=1,
    )
    mktempdir() do directory
        path = joinpath(directory, "survivor_2023.lp")
        @test SurvivorModel.write_survivor_pool_lp(
            path,
            context;
            selection_config=selection_config,
            include_completed=true,
        ) == path
        @test isfile(path)
        @test filesize(path) > 0
        plain_path = joinpath(directory, "plain.lp")
        tree_path = joinpath(directory, "tree.lp")
        for (output_path, branch_and_bound) in ((plain_path, false), (tree_path, true))
            @test SurvivorModel.write_survivor_pool_lp(
                output_path, context;
                selection_config=SurvivorSelectionConfig(
                    ; minimum_favorite_spread=nothing, market_guard_weeks=0,
                    through_week=2, hessian_weeks=2, branch_and_bound,
                ),
                include_completed=true,
            ) == output_path
        end
        @test read(plain_path, String) == read(tree_path, String)
        @test_throws ArgumentError SurvivorModel.write_survivor_pool_lp(
            joinpath(directory, "survivor_2023.mps"),
            context;
            selection_config=selection_config,
            include_completed=true,
        )
    end
end

@testset "survivor command-line application" begin
    @testset "argument and stdin parsing" begin
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--branch-and-bound", "--hessian-weeks=0"],
        ).branch_and_bound
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--branch-and-bound", "--branch-and-bound"],
        )
        for args in (
            ["--branch-and-bound", "--benders-weeks=0"],
            ["--benders-weeks", "2", "--branch-and-bound"],
        )
            @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(["--season=2023"; args])
        end
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023"],
        ) == (
            show_help=false,
            season=2023,
            initial_strikes=2,
            banned_first_pick_teams=String[],
            write_model_file=nothing,
            hessian_weeks=3,
            benders_weeks=nothing,
            branch_and_bound=false,
            timeout_seconds=nothing,
            refresh_data=false,
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--benders-weeks", "5"],
        ).benders_weeks == 5
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--benders-weeks=5"],
        ).benders_weeks == 5
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--benders-weeks=0"],
        ).benders_weeks == 0
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--strikes=4"],
        ).initial_strikes == 4
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--ban", " kc, SF,KC "],
        ).banned_first_pick_teams == ["KC", "SF"]
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--ban=KC,SF"],
        ).banned_first_pick_teams == ["KC", "SF"]
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--write-model", "survivor.lp"],
        ).write_model_file == "survivor.lp"
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--write-model=survivor.lp"],
        ).write_model_file == "survivor.lp"
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
        @test SurvivorModel._read_survivor_cli_picks(
            IOBuffer("KC\n\n sf \n"),
        ) == ["KC", "SF"]
        @test SurvivorModel._parse_survivor_cli_args(["--help"]).show_help
        usage = SurvivorModel._survivor_cli_usage()
        @test !occursin("--benchmark", usage)
        @test !occursin("--clear-cache", usage)
        @test occursin("exact-milp", usage)
        @test !occursin("micp", usage)
        @test !occursin("--timings", usage)
        @test occursin("--timeout", usage)
        @test occursin("--ban", usage)
        @test occursin("--benders-weeks", usage)
        @test !occursin("--benders ", usage)
        @test occursin("--write-model", usage)
        @test occursin("--hessian-weeks", usage)
        @test !occursin("--objective", usage)
        @test !occursin("--prove-first-pick", usage)
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--refresh-data"],
        ).refresh_data
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--refresh-data", "--refresh-data"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--timings"],
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
            ["--season", "2023", "--benders"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--benders-weeks"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--benders-weeks", "-1"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            [
                "--season",
                "2023",
                "--benders-weeks",
                "2",
                "--benders-weeks=3",
            ],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--ban"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--ban", "KC,"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--ban", ",KC"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--ban", "KC", "--ban=SF"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--write-model"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--write-model", "survivor.mps"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            [
                "--season",
                "2023",
                "--write-model",
                "one.lp",
                "--write-model=two.lp",
            ],
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
            ["--season", "2023", "--objective", "fixed-exact-milp"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--prove-first-pick"],
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
                cache_directory=cache_directory,
            )
            @test !first.cache_hit
            @test isfile(first.path)

            second = SurvivorModel._cached_historical_prior(
                historical;
                current_season=2023,
                cache_directory=cache_directory,
            )
            @test second.cache_hit
            @test second.path == first.path
            @test second.prior.td_shape == first.prior.td_shape
            @test second.prior.defensive_shape == first.prior.defensive_shape
            @test second.data_fingerprint == first.data_fingerprint

            changed_historical = copy(historical)
            changed_historical.time_of_possession[1] += Second(1)
            changed = SurvivorModel._cached_historical_prior(
                changed_historical;
                current_season=2023,
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
        for H in (0, 1, 2, 18)
            mktempdir() do cache_directory
                output = IOBuffer()
                args = ["--season=2023", "--branch-and-bound", "--hessian-weeks=$H"]
                @test SurvivorModel._parse_survivor_cli_args(args).hessian_weeks == H
                @test SurvivorModel._run_survivor_cli(
                    args;
                    input=IOBuffer("A\n"), output,
                    schedule, historical_drives=historical, current_drives=current,
                    cache_directory, through_week=2,
                ) == 0
                @test String(take!(output)) in ("C\n", "D\n")
                path = joinpath(cache_directory, "external.lp")
                @test SurvivorModel._run_survivor_cli(
                    [args; "--write-model"; path];
                    input=IOBuffer(), output,
                    schedule, historical_drives=historical, current_drives=current,
                    cache_directory, through_week=2,
                ) == 0
                @test isempty(String(take!(output)))
                @test isfile(path) && filesize(path) > 0
            end
        end
        mktempdir() do cache_directory
            output = IOBuffer()
            log_output = IOBuffer()
            exit_code = SurvivorModel.Logging.with_logger(
                SurvivorModel.Logging.SimpleLogger(
                    log_output,
                    SurvivorModel.Logging.Debug,
                ),
            ) do
                SurvivorModel._run_survivor_cli(
                    ["--season", "2023", "--benders-weeks", "1"];
                    input=IOBuffer("A\n"),
                    output=output,
                    schedule=schedule,
                    historical_drives=historical,
                    current_drives=current,
                    cache_directory=cache_directory,
                    through_week=2,
                )
            end
            @test exit_code == 0
            selected_team = strip(String(take!(output)))
            @test selected_team in ("C", "D")
            log_text = String(take!(log_output))
            @test occursin("survivor pick selected", log_text)
            @test occursin("team = $selected_team", log_text)
            @test occursin("survivor phase complete", log_text)
            @test occursin("optimize_start", log_text)
            @test occursin("optimize", log_text)
            @test occursin("survivor Benders first pick forced", log_text)
        end

        mktempdir() do cache_directory
            output = IOBuffer()
            exit_code = SurvivorModel._run_survivor_cli(
                ["--season", "2023", "--ban", "c"];
                input=IOBuffer("A\n"),
                output=output,
                schedule=schedule,
                historical_drives=historical,
                current_drives=current,
                cache_directory=cache_directory,
                through_week=2,
            )
            @test exit_code == 0
            @test strip(String(take!(output))) == "D"
        end

        mktempdir() do cache_directory
            output = IOBuffer()
            model_path = joinpath(cache_directory, "survivor_2023.lp")
            exit_code = SurvivorModel._run_survivor_cli(
                [
                    "--season",
                    "2023",
                    "--hessian-weeks",
                    "2",
                    "--benders-weeks",
                    "1",
                    "--write-model",
                    model_path,
                ];
                input=IOBuffer(),
                output=output,
                schedule=schedule,
                historical_drives=historical,
                current_drives=current,
                cache_directory=cache_directory,
                through_week=2,
            )
            @test exit_code == 0
            @test isfile(model_path)
            @test filesize(model_path) > 0
            @test isempty(strip(String(take!(output))))
        end

        completed_schedule = copy(schedule)
        completed_schedule.result = [7, 7, 3]
        stale_schedule = copy(schedule)
        stale_schedule.result[
            (stale_schedule.season .== 2023) .&
            (stale_schedule.week .== 1),
        ] .= missing
        schedule_calls = Ref(0)
        cache_clear_calls = Ref(0)
        schedule_loader = () -> begin
            schedule_calls[] += 1
            current_schedule = schedule_calls[] == 1 ?
                stale_schedule :
                completed_schedule
            return SurvivorModel.load_schedule(current_schedule)
        end
        mktempdir() do cache_directory
            output = IOBuffer()
            exit_code = SurvivorModel._run_survivor_cli(
                ["--season", "2023"];
                input=IOBuffer("A\n"),
                output=output,
                schedule_loader=schedule_loader,
                clear_data_cache=() -> (cache_clear_calls[] += 1),
                historical_drives=historical,
                current_drives=current,
                cache_directory=cache_directory,
                through_week=2,
            )
            @test exit_code == 0
            @test !isempty(strip(String(take!(output))))
            @test schedule_calls[] == 2
            @test cache_clear_calls[] == 1
        end

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
                through_week=2,
            )
            @test exit_code == 0
            @test !isempty(strip(String(take!(output))))
        end
    end
end
