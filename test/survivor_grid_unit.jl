using SurvivorModel
using DataFrames
using Dates
using Test

if !isdefined(Main, :_forecast_fixture)
    include("forecast_unit.jl")
end

function _survivor_grid_fixture()
    schedule, historical, current = _forecast_fixture()
    for (week, away, home) in ((4, "BYE1", "BYE2"), (18, "HOME", "AWAY"))
        push!(schedule, (
            game_id="2023_$(week)_$(away)_$(home)",
            season=2023,
            game_type="REG",
            week=week,
            gameday=Date(2023, 10, 1),
            away_team=away,
            home_team=home,
            away_score=missing,
            home_score=missing,
            result=missing,
        ))
    end
    return schedule, historical, current
end

@testset "survivor grid" begin
    schedule, historical, current = _survivor_grid_fixture()
    context = fit_regular_season_forecast(
        2023;
        as_of_week=2,
        schedule,
        historical_drives=historical,
        current_drives=current,
    )
    forecast = forecast_win_probabilities(context)
    grid = SurvivorModel._survivor_grid_data(context)
    @test grid.season == 2023
    @test grid.current_week == 2
    @test grid.weeks == collect(2:18)
    @test Set(grid.teams) == Set(["AWAY", "HOME", "BYE1", "BYE2"])
    @test grid.teams[end - 1:end] == ["BYE1", "BYE2"]
    @test size(grid.cells) == (4, 17)
    @test size(grid.top_five) == size(grid.cells)
    @test grid.top_five == .!isnothing.(grid.cells)
    @test !isdefined(SurvivorModel, :CairoMakie)

    @testset "opponents, byes, and every adjusted probability" begin
        corrections = Float64[]
        for game in eachrow(forecast)
            home_index = findfirst(==(game.home_team), grid.teams)
            away_index = findfirst(==(game.away_team), grid.teams)
            column = game.week - context.as_of_week + 1
            home_cell = grid.cells[home_index, column]
            away_cell = grid.cells[away_index, column]
            @test home_cell.opponent == game.away_team
            @test !home_cell.away
            @test away_cell.opponent == game.home_team
            @test away_cell.away
            @test home_cell.win_probability ≈ game.home_win_probability
            @test away_cell.win_probability ≈ game.away_win_probability
            @test home_cell.win_probability + away_cell.win_probability ≈ 1.0

            theta = SurvivorModel.hazard_theta(
                context.model,
                game.home_team,
                game.away_team,
            )
            derivatives = SurvivorModel._game_probability_derivatives(
                theta.log_mean,
                context.model.prior,
                context.marks,
            )
            adjusted = clamp(
                derivatives.probability +
                0.5 * sum(derivatives.hessian .* transpose(theta.covariance)),
                0.0,
                1.0,
            )
            @test home_cell.win_probability ≈ adjusted
            push!(corrections, abs(adjusted - derivatives.probability))
            if game.week == 18
                @test abs(adjusted - derivatives.probability) > 1e-6
            end
        end
        @test any(>(1e-6), corrections)
        @test all(isnothing, grid.cells[:, 5])
        @test all(isnothing, grid.cells[end - 1:end, 1])
        @test count(!isnothing, grid.cells[:, 1]) == 2
        @test !ismissing(only(context.games[context.games.week .== 2, :result]))

        picks = Dict(1 => "HOME")
        unused = SurvivorModel._survivor_grid_data(context; picks_made=picks)
        @test Set(unused.teams) == Set(["AWAY", "BYE1", "BYE2"])
        @test unused.teams[1] == "AWAY"
        @test picks == Dict(1 => "HOME")
        @test SurvivorModel._survivor_grid_data(
            context;
            picks_made=Dict(index => team for (index, team) in enumerate(grid.teams)),
        ).teams == String[]
    end

    @testset "unrounded sorting, ties, and zero probability" begin
        sorting_forecast = copy(forecast)
        sorting_forecast.home_win_probability[sorting_forecast.week .== 2] .= 0.49996
        sorting_forecast.away_win_probability[sorting_forecast.week .== 2] .= 0.50004
        sorted = SurvivorModel._survivor_grid_data(
            context,
            sorting_forecast,
            Dict{Int,String}(),
        )
        @test sorted.teams == ["HOME", "AWAY", "BYE1", "BYE2"]
        @test SurvivorModel._survivor_grid_cell_text(sorted.cells[1, 1]) == "@AWAY 50.0%"
        @test SurvivorModel._survivor_grid_cell_text(sorted.cells[2, 1]) == "HOME 50.0%"

        sorting_forecast.home_win_probability[sorting_forecast.week .== 2] .= 0.5
        sorting_forecast.away_win_probability[sorting_forecast.week .== 2] .= 0.5
        @test SurvivorModel._survivor_grid_data(
            context,
            sorting_forecast,
            Dict{Int,String}(),
        ).teams == ["AWAY", "HOME", "BYE1", "BYE2"]
        sorting_forecast.home_win_probability[sorting_forecast.week .== 2] .= 0.0
        sorting_forecast.away_win_probability[sorting_forecast.week .== 2] .= 1.0
        @test SurvivorModel._survivor_grid_data(
            context,
            sorting_forecast,
            Dict{Int,String}(),
        ).teams == ["HOME", "AWAY", "BYE1", "BYE2"]
    end

    @testset "complete stdout table" begin
        output = IOBuffer()
        @test SurvivorModel._write_survivor_grid(output, grid) === nothing
        text = String(take!(output))
        lines = split(chomp(text), '\n')
        @test lines[1] == "Survivor grid: season 2023, start of week 2"
        @test occursin("Hessian-adjusted", lines[3])
        @test occursin("* = top five unused teams that week", lines[3])
        @test occursin("@OPP = away; OPP = home; blank = bye", lines[2])
        @test strip.(split(lines[4], '|')) == ["Team"; ["W$week" for week in 2:18]]
        @test length(lines) == length(grid.teams) + 5
        @test !occursin("\u2026", text)
        for (index, team) in enumerate(grid.teams)
            cells = strip.(split(lines[index + 5], '|'))
            @test cells[1] == team
            @test length(cells) == 18
            for column in eachindex(grid.weeks)
                expected = SurvivorModel._survivor_grid_cell_text(grid.cells[index, column])
                marked = grid.top_five[index, column] ? "$expected *" : expected
                @test join(split(cells[column + 1]), " ") == marked
            end
        end
        @test SurvivorModel._survivor_grid_cell_text(nothing) == ""
        narrow_output = IOBuffer()
        SurvivorModel._write_survivor_grid(
            IOContext(narrow_output, :displaysize => (5, 20), :limit => true),
            grid,
        )
        @test String(take!(narrow_output)) == text
        empty = SurvivorModel._survivor_grid_data(
            context;
            picks_made=Dict(index => team for (index, team) in enumerate(grid.teams)),
        )
        SurvivorModel._write_survivor_grid(output, empty)
        @test length(split(chomp(String(take!(output))), '\n')) == 5
    end

    @testset "weekly top-five indicators and grid alignment" begin
        for count in (0, 1, 4, 5, 6, 10, 11, 32)
            teams = ["T$(lpad(index, 2, '0'))" for index in 1:count]
            cells = Matrix{Union{Nothing,SurvivorModel._SurvivorGridCell}}(undef, count, 3)
            for row in 1:count
                cells[row, 1] = SurvivorModel._SurvivorGridCell(
                    isodd(row) ? "SF" : "LONG", iseven(row), (row - 1) / max(1, count - 1),
                )
                cells[row, 2] = SurvivorModel._SurvivorGridCell("KC", false, 1.0 - cells[row, 1].win_probability)
                cells[row, 3] = nothing
            end
            top_five = SurvivorModel._survivor_grid_top_five(teams, cells)
            @test vec(sum(top_five; dims=1)) == [min(5, count), min(5, count), 0]
            @test findall(top_five[:, 1]) == collect(max(1, count - 4):count)
            @test findall(top_five[:, 2]) == collect(1:min(5, count))
            data = (; season=2023, current_week=1, teams, weeks=[1, 2, 3], cells, top_five)
            output = IOBuffer()
            SurvivorModel._write_survivor_grid(output, data)
            text = String(take!(output))
            lines = split(chomp(text), '\n')
            rule = lines[5]
            groups = count == 0 ? 0 : div(count - 1, 5)
            @test length(lines) == count + groups + 5
            @test findall(==(rule), lines) == [5; [5 + 6 * group for group in 1:groups]]
            @test count == 0 || lines[end] != rule
            rows = filter(!=(rule), lines[6:end])
            pipe_positions = findall(==('|'), lines[4])
            @test all(findall(==('|'), line) == pipe_positions for line in rows)
            @test findall(==('+'), rule) == pipe_positions
            @test all(length(line) == length(lines[4]) for line in rows)
            for column in 1:2
                fields = [split(line, '|')[column + 1] for line in rows]
                @test length(unique(findfirst(==('%'), field) for field in fields)) <= 1
                @test length(unique(findfirst(==('.'), field) for field in fields)) <= 1
                for (row, field) in enumerate(fields)
                    @test occursin('*', field) == top_five[row, column]
                    if top_five[row, column]
                        @test findfirst(==('*'), field) == findfirst(==('%'), field) + 2
                    end
                end
            end
            @test all(isempty(strip(split(line, '|')[4])) for line in rows)
        end

        teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
        cells = [
            SurvivorModel._SurvivorGridCell("KC", false, probability)
            for probability in (0.9, 0.8, 0.7, 0.6, 0.50003, 0.50004, 0.50003, 0.1)
        ]
        mask = SurvivorModel._survivor_grid_top_five(teams, reshape(cells, :, 1))
        @test teams[findall(mask[:, 1])] == ["A", "B", "C", "D", "F"]
        cells[6] = SurvivorModel._SurvivorGridCell("KC", false, 0.50003)
        mask = SurvivorModel._survivor_grid_top_five(teams, reshape(cells, :, 1))
        @test teams[findall(mask[:, 1])] == ["A", "B", "C", "D", "E"]
        @test SurvivorModel._survivor_grid_aligned_cell(
            SurvivorModel._SurvivorGridCell("KC", true, 1.0), 4, true,
        ) == "@KC  100.0% *"
        @test SurvivorModel._survivor_grid_aligned_cell(
            SurvivorModel._SurvivorGridCell("SF", false, 0.0), 4, false,
        ) == "SF     0.0%  "
    end

    @testset "replay cutoffs and invalid games" begin
        before_week_two = fit_regular_season_forecast(
            2023;
            as_of_week=2,
            schedule,
            historical_drives=historical,
            current_drives=current[1:2, :],
            prior=context.model.prior,
        )
        replay = SurvivorModel._survivor_grid_data(before_week_two)
        @test replay.teams == grid.teams
        for index in eachindex(grid.cells)
            if grid.cells[index] !== nothing
                @test replay.cells[index].win_probability ≈
                    grid.cells[index].win_probability
            end
        end
        duplicate = vcat(forecast, forecast[1:1, :])
        @test_throws ArgumentError SurvivorModel._survivor_grid_data(
            context,
            duplicate,
            Dict{Int,String}(),
        )
        invalid = copy(forecast)
        invalid.home_win_probability[1] = NaN
        @test_throws ArgumentError SurvivorModel._survivor_grid_data(
            context,
            invalid,
            Dict{Int,String}(),
        )
        invalid = copy(forecast)
        invalid.home_team[1] = "UNKNOWN"
        @test_throws ArgumentError SurvivorModel._survivor_grid_data(
            context,
            invalid,
            Dict{Int,String}(),
        )
        last_week = fit_regular_season_forecast(
            2023;
            as_of_week=18,
            schedule,
            historical_drives=historical,
            current_drives=current,
            prior=context.model.prior,
        )
        last_grid = SurvivorModel._survivor_grid_data(last_week)
        @test last_grid.weeks == [18]
        @test size(last_grid.cells) == (4, 1)
        @test count(!isnothing, last_grid.cells) == 2
    end
end
