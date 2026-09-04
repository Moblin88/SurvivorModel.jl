using SurvivorModel
using DataFrames
using Dates
using Test

@testset "SurvivorModel.jl" begin
    @testset "_first_nonmissing" begin
        @test SurvivorModel._first_nonmissing([missing, missing, "a", "b"]) == "a"
        @test SurvivorModel._first_nonmissing(["a", "b"]) == "a"
        @test ismissing(SurvivorModel._first_nonmissing([missing, missing]))
    end

    @testset "_parse_time_of_possession" begin
        @test SurvivorModel._parse_time_of_possession("4:01") == Second(241)
        @test SurvivorModel._parse_time_of_possession("10:00") == Minute(10)
        @test SurvivorModel._parse_time_of_possession("0:00") == Second(0)
        @test ismissing(SurvivorModel._parse_time_of_possession(missing))
    end

    @testset "summarize_drives" begin
        # A hand-built play-by-play DataFrame exercising the tricky cases:
        #   - a "GAME" marker row with missing posteam/defteam/etc. (drive 1)
        #   - a normal field-goal drive (drive 2)
        #   - a drive where the possession team at the end differs from the
        #     start, because the defense returned a turnover for a touchdown
        #     ("Opp touchdown", drive 3)
        #   - a drive with no recorded yardage at all (drive 4)
        #   - a second game reusing drive numbers, with an away-team
        #     touchdown (missed PAT) to check negative spread changes
        pbp = DataFrame(
            game_id=[
                "2023_01_TEST_GAME", "2023_01_TEST_GAME", "2023_01_TEST_GAME",
                "2023_01_TEST_GAME", "2023_01_TEST_GAME", "2023_01_TEST_GAME",
                "2023_01_TEST_GAME",
                "2023_01_TEST_GAME2", "2023_01_TEST_GAME2",
            ],
            fixed_drive=[1, 1, 1, 2, 2, 3, 3, 1, 1],
            home_team=[
                "HOME", "HOME", "HOME", "HOME", "HOME", "HOME", "HOME",
                "Y", "Y",
            ],
            away_team=[
                "AWAY", "AWAY", "AWAY", "AWAY", "AWAY", "AWAY", "AWAY",
                "X", "X",
            ],
            posteam=[
                missing, "AWAY", "AWAY", "HOME", "HOME", "HOME", "AWAY",
                "X", "X",
            ],
            defteam=[
                missing, "HOME", "HOME", "AWAY", "AWAY", "AWAY", "HOME",
                "Y", "Y",
            ],
            fixed_drive_result=[
                "Punt", "Punt", "Punt", "Field goal", "Field goal",
                "Opp touchdown", "Opp touchdown",
                "Touchdown", "Touchdown",
            ],
            drive_time_of_possession=[
                missing, "2:00", "2:00", "3:30", "3:30", "1:15", "1:15",
                "0:45", "0:45",
            ],
            yardline_100=[
                missing, 75.0, 70.0, 60.0, 55.0, 80.0, 20.0,
                65.0, 60.0,
            ],
            yards_gained=[
                missing, 5.0, 3.0, 10.0, 20.0, -5.0, missing,
                40.0, 35.0,
            ],
            total_home_score=[
                0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0,
                0.0, 0.0,
            ],
            total_away_score=[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0,
                0.0, 6.0,
            ],
        )

        drives = summarize_drives(pbp)
        @test nrow(drives) == 4
        drives = sort(drives, [:game_id, :fixed_drive])

        # Drive 1: GAME marker row plus two punt plays.
        d1 = drives[1, :]
        @test d1.game_id == "2023_01_TEST_GAME"
        @test d1.fixed_drive == 1
        @test d1.posteam == "AWAY"
        @test d1.defteam == "HOME"
        @test d1.posteam_home == false
        @test d1.defteam_home == true
        @test d1.drive_result == "Punt"
        @test d1.time_of_possession == Second(120)
        @test d1.yardline_100 == 75.0 # first non-missing (GAME row is missing)
        @test d1.yards_gained == 8.0
        @test d1.home_spread_change == 0.0

        # Drive 2: home team kicks a field goal.
        d2 = drives[2, :]
        @test d2.fixed_drive == 2
        @test d2.posteam == "HOME"
        @test d2.defteam == "AWAY"
        @test d2.posteam_home == true
        @test d2.defteam_home == false
        @test d2.drive_result == "Field goal"
        @test d2.time_of_possession == Second(210)
        @test d2.yardline_100 == 60.0
        @test d2.yards_gained == 30.0
        @test d2.home_spread_change == 3.0

        # Drive 3: home team drives, but the away defense returns a turnover
        # for a touchdown (plus PAT). Possession at the start is still home.
        d3 = drives[3, :]
        @test d3.fixed_drive == 3
        @test d3.posteam == "HOME"
        @test d3.defteam == "AWAY"
        @test d3.posteam_home == true
        @test d3.defteam_home == false
        @test d3.drive_result == "Opp touchdown"
        @test d3.time_of_possession == Second(75)
        @test d3.yardline_100 == 80.0
        @test d3.yards_gained == -5.0 # missing play skipped
        @test d3.home_spread_change == -7.0

        # Second game: away team scores a touchdown with a missed PAT.
        d4 = drives[4, :]
        @test d4.game_id == "2023_01_TEST_GAME2"
        @test d4.fixed_drive == 1
        @test d4.posteam == "X"
        @test d4.defteam == "Y"
        @test d4.posteam_home == false
        @test d4.defteam_home == true
        @test d4.drive_result == "Touchdown"
        @test d4.time_of_possession == Second(45)
        @test d4.yardline_100 == 65.0
        @test d4.yards_gained == 75.0
        @test d4.home_spread_change == -6.0
    end

    @testset "summarize_drives: all-missing yardage/time" begin
        pbp = DataFrame(
            game_id=["2023_01_TEST_GAME3"],
            fixed_drive=[1],
            home_team=["HOME"],
            away_team=["AWAY"],
            posteam=["HOME"],
            defteam=["AWAY"],
            fixed_drive_result=["End of half"],
            drive_time_of_possession=[missing],
            yardline_100=[missing],
            yards_gained=[missing],
            total_home_score=[3.0],
            total_away_score=[7.0],
        )

        drives = summarize_drives(pbp)
        @test nrow(drives) == 1
        d = drives[1, :]
        @test d.posteam == "HOME"
        @test d.defteam == "AWAY"
        @test d.posteam_home == true
        @test d.defteam_home == false
        @test ismissing(d.time_of_possession)
        @test ismissing(d.yardline_100)
        @test ismissing(d.yards_gained)
        @test d.home_spread_change == 0.0
    end
end
