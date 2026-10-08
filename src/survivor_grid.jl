struct _SurvivorGridCell
    opponent::String
    away::Bool
    win_probability::Float64
end

function _survivor_grid_top_five(teams, cells)
    top_five = falses(size(cells))
    for column in axes(cells, 2)
        available = [
            row for row in eachindex(teams) if cells[row, column] !== nothing
        ]
        sort!(available; by=row -> (-cells[row, column].win_probability, teams[row]))
        for row in Iterators.take(available, 5)
            top_five[row, column] = true
        end
    end
    return top_five
end

"""
    _survivor_grid_data(context; picks_made=Dict{Int,String}())

Return unused teams and remaining weeks with Hessian-adjusted game
probabilities, sorted by current-week probability with byes last.
"""
function _survivor_grid_data(
    context::RegularSeasonForecastContext;
    picks_made::AbstractDict{<:Integer,<:AbstractString}=Dict{Int,String}(),
)
    forecast = forecast_win_probabilities(context; include_completed=true)
    return _survivor_grid_data(context, forecast, picks_made)
end

function _survivor_grid_data(
    context::RegularSeasonForecastContext,
    forecast::AbstractDataFrame,
    picks_made::AbstractDict{<:Integer,<:AbstractString},
)
    schedule = _regular_season_schedule(context.schedule, context.season)
    isempty(schedule) &&
        throw(ArgumentError(
            "schedule has no regular-season games for season $(context.season)",
        ))
    used_teams = Set(values(picks_made))
    teams = sort!(filter(
        team -> !(team in used_teams),
        unique(vcat(String.(schedule.away_team), String.(schedule.home_team))),
    ))
    weeks = collect(context.as_of_week:18)
    team_indices = Dict(team => index for (index, team) in enumerate(teams))
    cells = fill!(
        Matrix{Union{Nothing,_SurvivorGridCell}}(undef, length(teams), length(weeks)),
        nothing,
    )
    seen = Set{Tuple{String,Int}}()
    for game in eachrow(forecast)
        context.as_of_week <= game.week <= 18 ||
            throw(ArgumentError(
                "grid games must be within the remaining regular-season weeks",
            ))
        for (team, opponent, away, probability) in (
            (game.home_team, game.away_team, false, game.home_win_probability),
            (game.away_team, game.home_team, true, game.away_win_probability),
        )
            key = (String(team), Int(game.week))
            key in seen &&
                throw(ArgumentError("team $team has multiple games in week $(game.week)"))
            push!(seen, key)
            team in used_teams && continue
            haskey(team_indices, team) ||
                throw(ArgumentError("grid game contains unknown team $team"))
            cells[team_indices[team], game.week - context.as_of_week + 1] =
                _SurvivorGridCell(
                    String(opponent),
                    away,
                    _validate_win_probability(probability),
                )
        end
    end
    order = sortperm(eachindex(teams); by=index -> begin
        current = cells[index, 1]
        return (
            current === nothing,
            current === nothing ? 0.0 : -current.win_probability,
            teams[index],
        )
    end)
    sorted_teams = teams[order]
    sorted_cells = cells[order, :]
    return (
        season=context.season,
        current_week=context.as_of_week,
        teams=sorted_teams,
        weeks=weeks,
        cells=sorted_cells,
        top_five=_survivor_grid_top_five(sorted_teams, sorted_cells),
    )
end

_survivor_grid_cell_text(::Nothing) = ""

_survivor_grid_cell_parts(::Nothing) = ("", "")

function _survivor_grid_cell_parts(cell::_SurvivorGridCell)
    opponent = (cell.away ? "@" : "") * cell.opponent
    percentage = string(round(100.0 * cell.win_probability; digits=1))
    return opponent, "$percentage%"
end

function _survivor_grid_cell_text(cell::_SurvivorGridCell)
    opponent, percentage = _survivor_grid_cell_parts(cell)
    return "$opponent $percentage"
end

function _survivor_grid_aligned_cell(cell, opponent_width::Int, top_five::Bool)
    opponent, percentage = _survivor_grid_cell_parts(cell)
    indicator = top_five ? "*" : " "
    return rpad(opponent, opponent_width) * " " * lpad(percentage, 6) * " " * indicator
end

function _write_survivor_grid(output::IO, grid)
    opponent_widths = [
        max(3, maximum(
            (length(_survivor_grid_cell_parts(cell)[1]) for cell in grid.cells[:, column]);
            init=0,
        ))
        for column in eachindex(grid.weeks)
    ]
    team_width = max(4, maximum(length, grid.teams; init=0))
    widths = [team_width; opponent_widths .+ 9]
    table = Matrix{String}(undef, length(grid.teams) + 1, length(grid.weeks) + 1)
    table[1, 1] = rpad("Team", team_width)
    for (column, week) in enumerate(grid.weeks)
        header = "W$week"
        padding = div(widths[column + 1] - length(header), 2)
        table[1, column + 1] = rpad(repeat(" ", padding) * header, widths[column + 1])
    end
    for (row, team) in enumerate(grid.teams)
        table[row + 1, 1] = rpad(team, team_width)
        for column in eachindex(grid.weeks)
            table[row + 1, column + 1] =
                _survivor_grid_aligned_cell(
                    grid.cells[row, column],
                    opponent_widths[column],
                    grid.top_five[row, column],
                )
        end
    end
    rule = join((repeat("-", width) for width in widths), "-+-")
    println(
        output,
        "Survivor grid: season $(grid.season), start of week $(grid.current_week)",
    )
    println(
        output,
        "@OPP = away; OPP = home; blank = bye.",
    )
    println(
        output,
        "* = top five unused teams that week; all win probabilities are Hessian-adjusted.",
    )
    for row in axes(table, 1)
        println(output, join(
            (table[row, column] for column in axes(table, 2)),
            " | ",
        ))
        if row == 1 || ((row - 1) % 5 == 0 && row <= length(grid.teams))
            println(output, rule)
        end
    end
    return nothing
end
