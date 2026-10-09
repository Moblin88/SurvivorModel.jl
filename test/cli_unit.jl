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
        @test SurvivorModel._parse_survivor_cli_args(
            [
                "--season=2023", "--branch-and-bound-workers=2",
                "--branch-and-bound",
            ],
        ).branch_and_bound_workers == 2
        @test SurvivorModel._parse_survivor_cli_args(
            [
                "--season", "2023", "--branch-and-bound",
                "--branch-and-bound-workers", "3",
            ],
        ).branch_and_bound_workers == 3
        for flags in (
            ["--branch-and-bound-workers"],
            ["--branch-and-bound-workers=0"],
            ["--branch-and-bound-workers=-1"],
            ["--branch-and-bound-workers=two"],
            ["--branch-and-bound-workers=2", "--branch-and-bound-workers=3"],
            ["--branch-and-bound-workers=2"],
        )
            @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
                ["--season=2023"; flags],
            )
        end
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            [
                "--season=2023", "--branch-and-bound",
                "--branch-and-bound-workers=2", "--write-model=tree.lp",
            ],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--branch-and-bound", "--branch-and-bound"],
        )
        for flags in (
            ["--simplex"],
            ["--simplex", "--branch-and-bound"],
            ["--branch-and-bound", "--simplex"],
            ["--simplex=on"],
            ["--branch-and-bound", "--simplex=on"],
            ["--benders-weeks", "2"],
            ["--benders-weeks=2"],
            ["--branch-and-bound", "--benders-weeks", "2"],
            ["--branch-and-bound", "--benders-weeks=2"],
            ["--benders-weeks=2", "--branch-and-bound"],
        )
            @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
                ["--season=2023"; flags],
            )
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
            branch_and_bound=false,
            branch_and_bound_workers=nothing,
            timeout_seconds=nothing,
            refresh_data=false,
            refresh_priors=false,
            plot_strength_week=nothing,
            grid=false,
        )
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
        @test !occursin("--simplex", usage)
        @test !occursin("--benchmark", usage)
        @test !occursin("--clear-cache", usage)
        @test occursin("exact-milp", usage)
        @test !occursin("micp", usage)
        @test !occursin("--timings", usage)
        @test occursin("--timeout", usage)
        @test occursin("--ban", usage)
        @test !occursin("--benders", usage)
        @test occursin("--write-model", usage)
        @test occursin("--hessian-weeks", usage)
        @test occursin("--branch-and-bound-workers", usage)
        @test occursin("--refresh-priors", usage)
        @test occursin("--plot-strength WEEK", usage)
        @test occursin("--grid", usage)
        @test !occursin("--as-of-week", usage)
        @test !occursin("--objective", usage)
        @test !occursin("--prove-first-pick", usage)
        @test SurvivorModel._parse_survivor_cli_args(
            ["--refresh-data"],
        ) == (
            show_help=false,
            season=nothing,
            initial_strikes=2,
            banned_first_pick_teams=String[],
            write_model_file=nothing,
            hessian_weeks=3,
            branch_and_bound=false,
            branch_and_bound_workers=nothing,
            timeout_seconds=nothing,
            refresh_data=true,
            refresh_priors=false,
            plot_strength_week=nothing,
            grid=false,
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--grid"],
        ).grid
        @test SurvivorModel._parse_survivor_cli_args(
            [
                "--season=2023", "--grid", "--strikes=4",
                "--refresh-data", "--refresh-priors",
            ],
        ).initial_strikes == 4
        @test !SurvivorModel._parse_survivor_cli_args(["--help", "--grid"]).grid
        for flags in (
            ["--grid"],
            ["--grid", "--refresh-priors"],
            ["--grid", "--refresh-data"],
        )
            @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(flags)
        end
        for flags in (
            ["--grid"],
            ["--grid=true"],
            ["true"],
            ["--plot-strength=5"],
            ["--ban=KC"],
            ["--branch-and-bound"],
            ["--write-model=grid.lp"],
            ["--hessian-weeks=3"],
            ["--hessian-weeks=0"],
            ["--timeout=10"],
        )
            @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
                ["--season=2023"; "--grid"; flags],
            )
        end
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--grid"],
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength", "5"],
        ).plot_strength_week == 5
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=18"],
        ).plot_strength_week == 18
        @test SurvivorModel._parse_survivor_cli_args(
            [
                "--season=2023",
                "--plot-strength=3",
                "--refresh-data",
                "--refresh-priors",
            ],
        ).plot_strength_week == 3
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--plot-strength", "5"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength"],
        )
        for invalid_week in ("zero", "0", "19", "-1")
            @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
                ["--season=2023", "--plot-strength=$invalid_week"],
            )
        end
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--plot-strength", "6"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--ban=KC"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--branch-and-bound"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--write-model", "out.lp"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--strikes=2"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--hessian-weeks=3"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--timeout=10"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season=2023", "--plot-strength=5", "--as-of-week=5"],
        )
        @test SurvivorModel._parse_survivor_cli_args(
            ["--refresh-priors"],
        ).refresh_priors
        @test SurvivorModel._parse_survivor_cli_args(
            ["--refresh-data", "--refresh-priors"],
        ).refresh_priors
        @test SurvivorModel._parse_survivor_cli_args(
            ["--refresh-data", "--"],
        ).season === nothing
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--refresh-data"],
        ).refresh_data
        @test SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--refresh-priors"],
        ).refresh_priors
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--refresh-data", "--refresh-data"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--refresh-priors", "--refresh-priors"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--refresh-priors=true"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--refresh-priors", "true"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            String[],
        )
        for run_options in (
            ["--ban=KC"],
            ["--branch-and-bound"],
            ["--write-model", "survivor.lp"],
            ["--strikes", "2"],
            ["--hessian-weeks", "3"],
            ["--timeout", "1"],
            ["--branch-and-bound-workers", "2"],
        )
            @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
                ["--refresh-priors"; run_options],
            )
        end
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
            ["--season", "2023", "--simplex"],
        )
        @test_throws ArgumentError SurvivorModel._parse_survivor_cli_args(
            ["--season", "2023", "--benders-weeks"],
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

    @testset "historical prior cache clearing" begin
        mktempdir() do cache_directory
            current_cache = SurvivorModel._historical_prior_cache_path(
                cache_directory,
                2023,
                12,
                "current-fingerprint",
            )
            old_schema_cache = joinpath(
                cache_directory,
                "historical_prior_v3_season2022_window8_dataold-fingerprint.jls",
            )
            drive_cache = joinpath(
                cache_directory,
                "drive_summaries_v1_season2022.jls",
            )
            unrelated_file = joinpath(cache_directory, "keep.txt")
            cache_named_directory = joinpath(
                cache_directory,
                "historical_prior_v2_directory.jls",
            )
            for path in (
                current_cache,
                old_schema_cache,
                drive_cache,
                unrelated_file,
            )
                write(path, "cached")
            end
            mkpath(cache_named_directory)
            nested_file = joinpath(cache_named_directory, "nested.jls")
            write(nested_file, "cached")

            @test SurvivorModel.clear_historical_prior_cache!(
                ; cache_directory=cache_directory,
            ) === nothing
            @test !isfile(current_cache)
            @test !isfile(old_schema_cache)
            @test isfile(drive_cache)
            @test isfile(unrelated_file)
            @test isdir(cache_named_directory)
            @test isfile(nested_file)
            @test SurvivorModel.clear_historical_prior_cache!(
                ; cache_directory=cache_directory,
            ) === nothing

            missing_directory = joinpath(cache_directory, "missing")
            @test SurvivorModel.clear_historical_prior_cache!(
                ; cache_directory=missing_directory,
            ) === nothing
            @test !ispath(missing_directory)
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

    @testset "standalone cache refresh" begin
        seed_cache_files = function (cache_directory)
            paths = (
                prior=joinpath(
                    cache_directory,
                    "historical_prior_v4_season2023_window12_datafingerprint.jls",
                ),
                old_prior=joinpath(
                    cache_directory,
                    "historical_prior_v3_season2022_window8_dataother-fingerprint.jls",
                ),
                drive_2022=joinpath(
                    cache_directory,
                    "drive_summaries_v1_season2022.jls",
                ),
                drive_2023=joinpath(
                    cache_directory,
                    "drive_summaries_v2_season2023.jls",
                ),
                unrelated=joinpath(cache_directory, "keep.txt"),
            )
            for path in paths
                write(path, "cached")
            end
            return paths
        end
        never_load_schedule = () -> error("standalone refresh loaded a schedule")

        mktempdir() do cache_directory
            files = seed_cache_files(cache_directory)
            input = IOBuffer("A\n")
            output = IOBuffer()
            raw_cache_clears = Ref(0)
            @test SurvivorModel._run_survivor_cli(
                ["--refresh-priors"];
                input,
                output,
                cache_directory,
                schedule_loader=never_load_schedule,
                clear_data_cache=() -> (raw_cache_clears[] += 1),
            ) == 0
            @test position(input) == 0
            @test isempty(String(take!(output)))
            @test raw_cache_clears[] == 0
            @test !isfile(files.prior)
            @test !isfile(files.old_prior)
            @test isfile(files.drive_2022)
            @test isfile(files.drive_2023)
            @test isfile(files.unrelated)

            files = seed_cache_files(cache_directory)
            input = IOBuffer("A\n")
            output = IOBuffer()
            @test SurvivorModel._run_survivor_cli(
                ["--help", "--refresh-priors", "--refresh-data"];
                input,
                output,
                cache_directory,
                schedule_loader=never_load_schedule,
                clear_data_cache=() -> error("help cleared the raw cache"),
            ) == 0
            @test position(input) == 0
            @test occursin("--refresh-priors", String(take!(output)))
            @test isfile(files.prior)
            @test isfile(files.drive_2022)

            raw_cache_clears = Ref(0)
            withenv("SURVIVORMODEL_REFRESH_DATA" => "true") do
                @test_throws ArgumentError SurvivorModel._run_survivor_cli(
                    String[];
                    cache_directory,
                    schedule_loader=never_load_schedule,
                    clear_data_cache=() -> (raw_cache_clears[] += 1),
                )
            end
            @test raw_cache_clears[] == 0
            @test isfile(files.prior)
            @test isfile(files.drive_2022)

            @test_throws ArgumentError SurvivorModel._run_survivor_cli(
                ["--refresh-data", "--strikes", "2"];
                cache_directory,
                schedule_loader=never_load_schedule,
                clear_data_cache=() -> error("invalid arguments cleared the raw cache"),
            )
            @test isfile(files.prior)
            @test isfile(files.drive_2022)
        end

        mktempdir() do cache_directory
            files = seed_cache_files(cache_directory)
            input = IOBuffer("A\n")
            output = IOBuffer()
            raw_cache_clears = Ref(0)
            @test SurvivorModel._run_survivor_cli(
                ["--refresh-data"];
                input,
                output,
                cache_directory,
                schedule_loader=never_load_schedule,
                clear_data_cache=() -> (raw_cache_clears[] += 1),
            ) == 0
            @test position(input) == 0
            @test isempty(String(take!(output)))
            @test raw_cache_clears[] == 1
            @test isfile(files.prior)
            @test isfile(files.old_prior)
            @test !isfile(files.drive_2022)
            @test !isfile(files.drive_2023)
            @test isfile(files.unrelated)

            @test SurvivorModel._run_survivor_cli(
                ["--refresh-data"];
                input,
                output,
                cache_directory,
                schedule_loader=never_load_schedule,
                clear_data_cache=() -> (raw_cache_clears[] += 1),
            ) == 0
            @test raw_cache_clears[] == 2
        end

        mktempdir() do cache_directory
            files = seed_cache_files(cache_directory)
            input = IOBuffer("A\n")
            output = IOBuffer()
            raw_cache_clears = Ref(0)
            @test SurvivorModel._run_survivor_cli(
                ["--refresh-priors", "--refresh-data"];
                input,
                output,
                cache_directory,
                schedule_loader=never_load_schedule,
                clear_data_cache=() -> (raw_cache_clears[] += 1),
            ) == 0
            @test position(input) == 0
            @test isempty(String(take!(output)))
            @test raw_cache_clears[] == 1
            @test !isfile(files.prior)
            @test !isfile(files.old_prior)
            @test !isfile(files.drive_2022)
            @test !isfile(files.drive_2023)
            @test isfile(files.unrelated)
        end
    end

    @testset "plot-only CLI mode" begin
        schedule, historical, current = _survivor_context_fixture()
        input = IOBuffer("this is not a pick\n")
        output = IOBuffer()
        contexts = SurvivorModel.RegularSeasonForecastContext[]
        raw_cache_clears = Ref(0)
        display_plot = context -> begin
            push!(contexts, context)
            return nothing
        end
        @test !isdefined(SurvivorModel, :GLMakie)
        mktempdir() do cache_directory
            exit_code = withenv(
                "SURVIVORMODEL_REFRESH_DATA" => "false",
            ) do
                SurvivorModel._run_survivor_cli(
                    [
                        "--season=2023",
                        "--plot-strength",
                        "3",
                        "--refresh-priors",
                        "--refresh-data",
                    ];
                    input,
                    output,
                    schedule,
                    historical_drives=historical,
                    current_drives=current,
                    cache_directory,
                    through_week=2,
                    clear_data_cache=() -> (raw_cache_clears[] += 1),
                    show_team_strength_plot=display_plot,
                )
            end
            @test exit_code == 0
            @test position(input) == 0
            @test isempty(String(take!(output)))
            @test raw_cache_clears[] == 1
            @test length(contexts) == 1
            context = only(contexts)
            @test context.season == 2023
            @test context.as_of_week == 3
            @test get(context.model.stats.defensive.counts, "B", 0) == 1
            @test isfile(
                SurvivorModel._historical_prior_cache_path(
                    cache_directory,
                    2023,
                    SurvivorModel.DEFAULT_HISTORICAL_SEASONS,
                    SurvivorModel._dataframe_fingerprint(
                        SurvivorModel._survivor_cli_historical_drives(
                            SurvivorModel.load_schedule(schedule),
                            2023,
                            historical,
                        ),
                    ),
                ),
            )
            @test !isdefined(SurvivorModel, :GLMakie)
        end
    end

    @testset "grid-only CLI mode" begin
        schedule, historical, current = _survivor_context_fixture()
        mktempdir() do cache_directory
            stale_prior = joinpath(cache_directory, "historical_prior_v0_stale.jls")
            write(stale_prior, "cached")
            raw_cache_clears = Ref(0)
            for (picks, week, strikes, refresh) in (
                ("", 1, 2, true),
                ("A\n", 2, 4, false),
            )
                input = IOBuffer(picks)
                output = IOBuffer()
                args = ["--season=2023", "--grid", "--strikes=$strikes"]
                refresh && append!(args, ["--refresh-data", "--refresh-priors"])
                exit_code = withenv("SURVIVORMODEL_REFRESH_DATA" => "false") do
                    SurvivorModel._run_survivor_cli(
                        args;
                        input,
                        output,
                        schedule,
                        historical_drives=historical,
                        current_drives=current,
                        cache_directory,
                        through_week=1,
                        clear_data_cache=() -> (raw_cache_clears[] += 1),
                        show_team_strength_plot=_ -> error("grid opened a plot"),
                    )
                end
                @test exit_code == 0
                @test eof(input)
                @test raw_cache_clears[] == 1
                @test !isfile(stale_prior)
                @test count(
                    name -> startswith(name, "historical_prior_"),
                    readdir(cache_directory),
                ) == 1

                context = fit_regular_season_forecast(
                    2023;
                    as_of_week=week,
                    schedule,
                    historical_drives=historical,
                    current_drives=current,
                )
                expected = IOBuffer()
                grid = SurvivorModel._survivor_grid_data(
                    context;
                    picks_made=isempty(picks) ? Dict{Int,String}() : Dict(1 => "A"),
                )
                SurvivorModel._write_survivor_grid(expected, grid)
                text = String(take!(output))
                @test text == String(take!(expected))
                @test occursin("start of week $week", text)
                @test occursin("W18", text)
                @test occursin("Hessian-adjusted", text)
                @test length(grid.teams) == (week == 1 ? 4 : 3)
                @test week == 1 || !("A" in grid.teams)
                @test !isdefined(SurvivorModel, :GLMakie)
                @test !any(package -> package.name == "GLMakie", keys(Base.loaded_modules))
            end

            write(stale_prior, "cached")
            input = IOBuffer("A\n")
            output = IOBuffer()
            @test_throws ArgumentError SurvivorModel._run_survivor_cli(
                [
                    "--season=2023", "--grid", "--hessian-weeks=3",
                    "--refresh-data", "--refresh-priors",
                ];
                input,
                output,
                cache_directory,
                schedule_loader=() -> error("invalid grid loaded a schedule"),
                clear_data_cache=() -> error("invalid grid cleared the raw cache"),
            )
            @test position(input) == 0
            @test isempty(String(take!(output)))
            @test isfile(stale_prior)
            @test SurvivorModel._run_survivor_cli(
                ["--help", "--grid", "--refresh-data", "--refresh-priors"];
                input,
                output,
                cache_directory,
                schedule_loader=() -> error("help loaded a schedule"),
                clear_data_cache=() -> error("help cleared the raw cache"),
            ) == 0
            @test position(input) == 0
            @test occursin("--grid", String(take!(output)))
            @test isfile(stale_prior)
            @test !isdefined(SurvivorModel, :GLMakie)
            @test_throws ArgumentError SurvivorModel._run_survivor_cli(
                ["--season=2023", "--grid", "--strikes=0"];
                input=IOBuffer("A\n"),
                output,
                schedule,
                historical_drives=historical,
                current_drives=current,
                cache_directory,
            )
            @test isempty(String(take!(output)))
        end
    end

    @testset "fixture-backed current pick" begin
        schedule, historical, current = _survivor_context_fixture()
        for H in (0, 1, 2, 18), branch_and_bound in (false, true)
            mktempdir() do cache_directory
                output = IOBuffer()
                args = ["--season=2023", "--hessian-weeks=$H"]
                branch_and_bound && push!(args, "--branch-and-bound")
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
            normalized_schedule = SurvivorModel.load_schedule(schedule)
            cli_historical = SurvivorModel._survivor_cli_historical_drives(
                normalized_schedule,
                2023,
                historical,
            )
            cached = SurvivorModel._cached_historical_prior(
                cli_historical;
                current_season=2023,
                cache_directory=cache_directory,
            )
            open(cached.path, "w") do io
                write(io, "not a serialized cache")
            end
            drive_cache_path = joinpath(
                cache_directory,
                "drive_summaries_v1_season2022.jls",
            )
            write(drive_cache_path, "keep")
            raw_cache_clears = Ref(0)
            debug_log = IOBuffer()
            debug_logger = SurvivorModel.Logging.ConsoleLogger(
                debug_log,
                SurvivorModel.Logging.Debug,
            )
            exit_code = withenv("SURVIVORMODEL_REFRESH_DATA" => "false") do
                SurvivorModel.Logging.with_logger(debug_logger) do
                    SurvivorModel._run_survivor_cli(
                        [
                            "--season",
                            "2023",
                            "--refresh-priors",
                            "--hessian-weeks",
                            "2",
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
                        clear_data_cache=() -> (raw_cache_clears[] += 1),
                    )
                end
            end
            debug_output = String(take!(debug_log))
            @test exit_code == 0
            @test isfile(model_path)
            @test filesize(model_path) > 0
            @test isempty(strip(String(take!(output))))
            @test occursin(
                "historical empirical-Bayes prior cause diagnostics",
                debug_output,
            )
            @test occursin("weibull_shape", debug_output)
            @test occursin("gamma_mean_cumulative_hazard", debug_output)
            @test raw_cache_clears[] == 0
            @test isfile(drive_cache_path)
            @test SurvivorModel._cached_historical_prior(
                cli_historical;
                current_season=2023,
                cache_directory=cache_directory,
            ).cache_hit

            combined_model_path = joinpath(
                cache_directory,
                "survivor_2023_refreshed.lp",
            )
            combined_raw_cache_clears = Ref(0)
            combined_exit_code = withenv(
                "SURVIVORMODEL_REFRESH_DATA" => "false",
            ) do
                SurvivorModel._run_survivor_cli(
                    [
                        "--season",
                        "2023",
                        "--refresh-data",
                        "--refresh-priors",
                        "--write-model",
                        combined_model_path,
                    ];
                    input=IOBuffer(),
                    output=output,
                    schedule=schedule,
                    historical_drives=historical,
                    current_drives=current,
                    cache_directory=cache_directory,
                    through_week=2,
                    clear_data_cache=() -> (combined_raw_cache_clears[] += 1),
                )
            end
            @test combined_exit_code == 0
            @test isfile(combined_model_path)
            @test combined_raw_cache_clears[] == 1
            @test isfile(drive_cache_path)
            @test SurvivorModel._cached_historical_prior(
                cli_historical;
                current_season=2023,
                cache_directory=cache_directory,
            ).cache_hit
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
