using Test
using SurvivorModel
using Logging

include("forecast_unit.jl")

pop!(ENV, "DISPLAY", nothing)
pop!(ENV, "WAYLAND_DISPLAY", nothing)
using CairoMakie

function _test_png(path, width, height)
    bytes = read(path)
    @test length(bytes) > 1000
    @test bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
    @test String(bytes[13:16]) == "IHDR"
    @test foldl((value, byte) -> 256value + Int(byte), bytes[17:20]; init=0) == width
    @test foldl((value, byte) -> 256value + Int(byte), bytes[21:24]; init=0) == height
    return nothing
end

function _test_saved_labels(figure, mapped, width, height)
    path = joinpath(@__DIR__, "cairo-layout-$(getpid()).png")
    try
        CairoMakie.save(path, figure; px_per_unit=1)
        _test_png(path, width, height)
        _test_rendered_team_labels(figure.content[1], mapped)
    finally
        rm(path; force=true)
    end
    return nothing
end

schedule, historical, current = _forecast_fixture()
schedule.away_team = replace.(schedule.away_team, "AWAY" => "KC", "HOME" => "JAX")
schedule.home_team = replace.(schedule.home_team, "AWAY" => "KC", "HOME" => "JAX")
historical.posteam = replace.(historical.posteam, "AWAY" => "KC", "HOME" => "JAX")
historical.defteam = replace.(historical.defteam, "AWAY" => "KC", "HOME" => "JAX")
current.posteam = replace.(current.posteam, "AWAY" => "KC", "HOME" => "JAX")
current.defteam = replace.(current.defteam, "AWAY" => "KC", "HOME" => "JAX")
context = fit_regular_season_forecast(
    2023;
    as_of_week=2,
    schedule=schedule,
    historical_drives=historical,
    current_drives=current,
)
data = SurvivorModel._team_strength_plot_data(context)
percentiles = SurvivorModel._team_strength_plot_percentiles(data, context.model.prior)
figure = SurvivorModel._team_strength_figure(
    CairoMakie,
    data,
    context,
)
axis = figure.content[1]
plots = axis.scene.plots

function _test_team_strength_plot_colors(axis, teams)
    colors = CairoMakie.Makie.to_color.(
        SurvivorModel._team_strength_team_colors(teams)
    )
    plots = axis.scene.plots
    @test plots[1].color[] == colors
    @test plots[2].color[] == colors
    @test plots[3].color[] == colors
    labels = filter(plot -> plot isa CairoMakie.TextLabel, plots)
    @test [plot.text[] for plot in labels] == teams
    @test [plot.text_color[] for plot in labels] == colors
    @test [plot.strokecolor[] for plot in labels] == colors
    leaders = only(filter(plot -> plot isa CairoMakie.LineSegments, plots))
    @test leaders.color[] == repeat(colors; inner=2)
    return colors
end

@testset "CairoMakie team-strength figure" begin
    @test length(plots) == nrow(data) + 4
    team_colors = _test_team_strength_plot_colors(axis, data.team)
    scatter = plots[1][1][]
    @test length(scatter) == nrow(data)
    @test all(
        point[1] == percentiles.offense_percentile[index] &&
        point[2] == percentiles.defense_percentile[index]
        for (index, point) in enumerate(scatter)
    )
    @test plots[2].direction[] === :x
    @test length(plots[2][1][]) == nrow(data)
    @test all(
        interval[1] == percentiles.defense_percentile[index] &&
        interval[2] == percentiles.offense_lower_percentile[index] &&
        interval[3] == percentiles.offense_upper_percentile[index]
        for (index, interval) in enumerate(plots[2][1][])
    )
    @test plots[3].direction[] === :y
    @test all(
        interval[1] == percentiles.offense_percentile[index] &&
        interval[2] == percentiles.defense_lower_percentile[index] &&
        interval[3] == percentiles.defense_upper_percentile[index]
        for (index, interval) in enumerate(plots[3][1][])
    )
    labels = filter(plot -> plot isa CairoMakie.TextLabel, plots)
    @test [plot.text[] for plot in labels] == data.team
    @test all(plot.fontsize[] >= 17 for plot in labels)
    @test all(plot.background_color[] === :white for plot in labels)
    @test [plot.text_color[] for plot in labels] == team_colors
    @test axis.title[] ==
        "Season 2023 team strengths as of start of week 2"
    limits = axis.limits[]
    @test limits == ((0, 100), (0, 100))
    @test axis.xticks[][2] == ["$value%" for value in 0:10:100]
    @test axis.yticks[][2] == ["$value%" for value in 0:10:100]
    @test occursin("central 50% posterior intervals", figure.content[2].text[])
    @test occursin("Neutral-site mean rates", figure.content[2].text[])
    @test occursin("league Gamma prior percentiles", figure.content[2].text[])
    @test occursin("league touchdown Gamma prior", axis.xlabel[])
    @test occursin("league defensive-event Gamma prior", axis.ylabel[])

    reordered_data = data[[2, 1], :]
    reordered_data.team = ["OAK", "SD"]
    reordered_figure = SurvivorModel._team_strength_figure(
        CairoMakie,
        reordered_data,
        context,
    )
    _test_team_strength_plot_colors(
        reordered_figure.content[1],
        reordered_data.team,
    )
end

@testset "plot means can lie outside central intervals" begin
    skewed_data = copy(data)
    skewed_data.offense_rate[1] = 4.0 * maximum(data.offense_upper)
    skewed_data.defense_rate[1] = 0.5 * minimum(data.defense_lower)
    skewed_figure = SurvivorModel._team_strength_figure(CairoMakie, skewed_data, context)
    skewed_axis = skewed_figure.content[1]
    mapped = SurvivorModel._team_strength_plot_percentiles(skewed_data, context.model.prior)
    @test skewed_axis.scene.plots[1][1][][1][1] == mapped.offense_percentile[1]
    @test skewed_axis.scene.plots[2][1][][1][3] == mapped.offense_upper_percentile[1]
    @test skewed_axis.scene.plots[3][1][][1][2] == mapped.defense_lower_percentile[1]
    @test mapped.offense_percentile[1] >= mapped.offense_upper_percentile[1]
    @test mapped.defense_percentile[1] <= mapped.defense_lower_percentile[1]
end

function _rendered_team_label_boxes(axis)
    labels = filter(plot -> plot isa CairoMakie.TextLabel, axis.scene.plots)
    return [
        begin
            background = only(filter(child -> child isa CairoMakie.Poly, label.plots))
            box = CairoMakie.boundingbox(background)
            (
                box.origin[1], box.origin[2],
                box.origin[1] + box.widths[1], box.origin[2] + box.widths[2],
            )
        end
        for label in labels
    ]
end

function _test_rendered_team_labels(axis, mapped)
    width, height = CairoMakie.viewport(axis.scene)[].widths
    boxes = _rendered_team_label_boxes(axis)
    @test length(boxes) == nrow(mapped)
    segments = only(filter(plot -> plot isa CairoMakie.LineSegments, axis.scene.plots))[1][]
    @test length(segments) == 2 * nrow(mapped)
    anchors = [
        CairoMakie.Makie.project(
            axis.scene, :data, :pixel,
            Point2f(row.offense_percentile, row.defense_percentile),
        ) for row in eachrow(mapped)
    ]
    for (index, box) in enumerate(boxes)
        @test 4.0 - 1e-3 <= box[1] < box[3] <= width - 4.0 + 1e-3
        @test 4.0 - 1e-3 <= box[2] < box[4] <= height - 4.0 + 1e-3
        for other in boxes[index + 1:end]
            @test SurvivorModel._team_strength_label_overlap(box, other) == 0.0
            @test SurvivorModel._team_strength_label_overlap(box, other, 4.0) <= 0.02
        end
        for anchor in anchors
            marker = SurvivorModel._team_strength_label_box(anchor, (16.0, 16.0))
            @test SurvivorModel._team_strength_label_overlap(box, marker) == 0.0
        end
        start = segments[2 * index - 1]
        target = segments[2 * index]
        @test target[1] ≈ anchors[index][1] atol=1e-3
        @test target[2] ≈ anchors[index][2] atol=1e-3
        @test any(isapprox(start[coordinate], box[edge]; atol=1e-3)
            for (coordinate, edge) in ((1, 1), (1, 3), (2, 2), (2, 4)))
    end
    return nothing
end

teams = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
offense_reference = context.model.prior.td_hyperparameters
defense_reference = context.model.prior.defensive_hyperparameters
offense_distribution = SurvivorModel.Gamma(offense_reference.shape, inv(offense_reference.rate))
defense_distribution = SurvivorModel.Gamma(defense_reference.shape, inv(defense_reference.rate))
dense_data = DataFrame(
    team=teams,
    offense_rate=fill(offense_reference.shape / offense_reference.rate, 32),
    offense_lower=fill(SurvivorModel.quantile(offense_distribution, 0.25), 32),
    offense_upper=fill(SurvivorModel.quantile(offense_distribution, 0.75), 32),
    defense_rate=fill(defense_reference.shape / defense_reference.rate, 32),
    defense_lower=fill(SurvivorModel.quantile(defense_distribution, 0.25), 32),
    defense_upper=fill(SurvivorModel.quantile(defense_distribution, 0.75), 32),
)
dense_figure = SurvivorModel._team_strength_figure(CairoMakie, dense_data, context)
dense_axis = dense_figure.content[1]
dense_mapped = SurvivorModel._team_strength_plot_percentiles(dense_data, context.model.prior)
@testset "team colors follow all dense rows" begin
    colors = _test_team_strength_plot_colors(dense_axis, dense_data.team)
    @test length(colors) == 32
end

@testset "dense team labels and resize" begin
    _test_saved_labels(dense_figure, dense_mapped, 1100, 760)
    old_positions = [box[1:2] for box in _rendered_team_label_boxes(dense_axis)]
    CairoMakie.resize!(dense_figure, 680, 500)
    _test_saved_labels(dense_figure, dense_mapped, 680, 500)
    @test old_positions != [box[1:2] for box in _rendered_team_label_boxes(dense_axis)]
    CairoMakie.resize!(dense_figure, 1100, 760)
    _test_saved_labels(dense_figure, dense_mapped, 1100, 760)
    image = get(ENV, "TEAM_STRENGTH_PLOT_SMOKE_IMAGE", "")
    isempty(image) || CairoMakie.save(image, dense_figure)

    edge_data = copy(dense_data)
    edge_data.offense_rate = repeat([0.0, 1e308], 16)
    edge_data.defense_rate = repeat([0.0, 0.0, 1e308, 1e308], 8)
    edge_data.offense_lower .= 0.0
    edge_data.offense_upper .= 1e308
    edge_data.defense_lower .= 0.0
    edge_data.defense_upper .= 1e308
    edge_figure = SurvivorModel._team_strength_figure(CairoMakie, edge_data, context)
    _test_saved_labels(
        edge_figure,
        SurvivorModel._team_strength_plot_percentiles(edge_data, context.model.prior),
        1100, 760,
    )
end

@testset "headless PNG save and overwrite" begin
    directory = mkdir(joinpath(@__DIR__, "cairo-rendering-$(getpid())"))
    try
        path = joinpath(directory, "strength.png")
        write(path, "replace me")
        @test SurvivorModel._save_team_strength_plot(context, path) === nothing
        _test_png(path, 1100, 760)
        @test SurvivorModel._save_team_strength_plot(context, path) === nothing
        _test_png(path, 1100, 760)
        @test_throws Exception SurvivorModel._save_team_strength_plot(
            context, joinpath(directory, "missing", "strength.png"),
        )
        for (name, plotted, mapped) in (
            ("dense", dense_figure, dense_mapped),
        )
            path = joinpath(directory, "$name.png")
            CairoMakie.save(path, plotted; px_per_unit=1)
            _test_png(path, 1100, 760)
            _test_rendered_team_labels(plotted.content[1], mapped)
        end
        cd(directory) do
            for args in (
                ["--season=2023", "--plot-strength=2"],
                ["--season=2023", "--plot-strength=2", "--plot-output", "custom.png"],
                ["--season=2023", "--plot-strength=2", "--plot-output=custom.png"],
            )
                input = IOBuffer("not a pick\n")
                output = IOBuffer()
                logger = Test.TestLogger()
                result = with_logger(logger) do
                    return SurvivorModel._run_survivor_cli(
                        args; input, output, schedule,
                        historical_drives=historical,
                        current_drives=current,
                        cache_directory=joinpath(pwd(), "cache"),
                    )
                end
                @test result == 0
                @test position(input) == 0
                @test isempty(String(take!(output)))
                path = length(args) == 2 ? "team-strength-2023-week-2.png" : "custom.png"
                _test_png(path, 1100, 760)
                @test any(record -> record.level == Logging.Info &&
                    record.message == "team-strength PNG saved" &&
                    record.kwargs[:path] == abspath(path), logger.logs)
            end
        end
    finally
        rm(directory; recursive=true)
    end
end
