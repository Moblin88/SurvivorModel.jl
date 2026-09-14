using DataFrames
using Dates
using Test
using SurvivorModel

function _drive_cache_fixture(; season=2024, marker=1)
    return DataFrame(
        game_id=["$(season)_01_TEST"],
        fixed_drive=[marker],
        posteam=["HOME"],
        defteam=["AWAY"],
        posteam_home=[true],
        defteam_home=[false],
        drive_result=["Punt"],
        time_of_possession=[Second(60)],
        drive_start_yards_to_goal=[75],
        yards_gained=[10.0 + marker],
        home_spread_change=[0.0],
    )
end

@testset "summarized drive cache" begin
    mktempdir() do cache_directory
        calls = Ref(0)
        loader = season -> begin
            calls[] += 1
            _drive_cache_fixture(season=season, marker=calls[])
        end

        first_load = SurvivorModel._cached_season_drives(
            2024;
            cache_directory=cache_directory,
            loader=loader,
        )
        @test first_load.cache_hit == false
        @test first_load.season == 2024
        @test first_load.metadata.schema_version ==
            SurvivorModel.DRIVE_CACHE_SCHEMA_VERSION
        @test first_load.data_fingerprint == SurvivorModel._drive_cache_fingerprint(
            first_load.drives,
        )
        @test calls[] == 1
        @test isfile(first_load.path)

        second_load = SurvivorModel._cached_season_drives(
            2024;
            cache_directory=cache_directory,
            loader=loader,
        )
        @test second_load.cache_hit
        @test second_load.drives == first_load.drives
        @test calls[] == 1
        @test isempty(filter(
            path -> startswith(basename(path), "jl_"),
            readdir(cache_directory; join=true),
        ))

        refreshed = SurvivorModel._cached_season_drives(
            2024;
            cache_directory=cache_directory,
            loader=loader,
            refresh=true,
        )
        @test !refreshed.cache_hit
        @test calls[] == 2
        @test refreshed.drives != first_load.drives
        @test refreshed.metadata.cached_at >= first_load.metadata.cached_at

        stale_fingerprint = SurvivorModel._cached_season_drives(
            2024;
            cache_directory=cache_directory,
            loader=loader,
            source_fingerprint="new-upstream-release",
        )
        @test !stale_fingerprint.cache_hit
        @test calls[] == 3
        @test stale_fingerprint.source_fingerprint == "new-upstream-release"
    end

    @testset "loader freshness metadata and validation" begin
        mktempdir() do cache_directory
            loaded_at = DateTime(2026, 9, 14, 12)
            result = SurvivorModel._cached_season_drives(
                2025;
                cache_directory=cache_directory,
                loader=season -> (
                    drives=_drive_cache_fixture(season=season),
                    source_fingerprint="nflverse-2025-week-2",
                    source_timestamp=loaded_at,
                ),
            )
            @test result.source_fingerprint == "nflverse-2025-week-2"
            @test result.source_timestamp == loaded_at

            @test SurvivorModel._cached_season_drives(
                2025;
                cache_directory=cache_directory,
                source_fingerprint="nflverse-2025-week-2",
                loader=season -> error("cache should have satisfied the request"),
            ).cache_hit

            config_miss = SurvivorModel._cached_season_drives(
                2025;
                cache_directory=cache_directory,
                cache_config=(summary_schema=99,),
                loader=season -> _drive_cache_fixture(season=season, marker=9),
            )
            @test !config_miss.cache_hit

            @test_throws ArgumentError SurvivorModel._cached_season_drives(
                2025;
                cache_directory=cache_directory,
                loader=season -> DataFrame(game_id=["bad"]),
            )
        end
    end

    @testset "schema mismatch and corruption are explicit" begin
        mktempdir() do cache_directory
            path = SurvivorModel._drive_cache_path(cache_directory, 2023)
            mkpath(dirname(path))
            old_entry = SurvivorModel.DriveCacheEntry(
                SurvivorModel.DRIVE_CACHE_SCHEMA_VERSION - 1,
                2023,
                SurvivorModel.DEFAULT_DRIVE_CACHE_CONFIG,
                "old",
                nothing,
                "old",
                now(),
                _drive_cache_fixture(season=2023),
            )
            SurvivorModel._write_drive_cache(path, old_entry)

            calls = Ref(0)
            replacement = SurvivorModel._cached_season_drives(
                2023;
                cache_directory=cache_directory,
                loader=season -> begin
                    calls[] += 1
                    _drive_cache_fixture(season=season, marker=7)
                end,
            )
            @test !replacement.cache_hit
            @test calls[] == 1

            open(path, "w") do io
                write(io, "not a serialized cache")
            end
            @test_throws ArgumentError SurvivorModel._cached_season_drives(
                2023;
                cache_directory=cache_directory,
                loader=season -> error("corrupt cache must not be silently ignored"),
            )
        end
    end

    @testset "season and full invalidation" begin
        mktempdir() do cache_directory
            for season in (2022, 2023)
                SurvivorModel._cached_season_drives(
                    season;
                    cache_directory=cache_directory,
                    loader=s -> _drive_cache_fixture(season=s),
                )
            end
            unrelated = joinpath(cache_directory, "leave-me.txt")
            open(unrelated, "w") do io
                write(io, "unrelated")
            end

            @test SurvivorModel._clear_drive_cache!(
                season=2022,
                cache_directory=cache_directory,
            ) == 1
            @test !isfile(
                SurvivorModel._drive_cache_path(cache_directory, 2022),
            )
            @test isfile(
                SurvivorModel._drive_cache_path(cache_directory, 2023),
            )

            @test SurvivorModel._invalidate_drive_cache!(
                cache_directory=cache_directory,
            ) == 1
            @test !isfile(
                SurvivorModel._drive_cache_path(cache_directory, 2023),
            )
            @test isfile(unrelated)
        end
    end
end
