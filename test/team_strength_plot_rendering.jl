using Test
using SurvivorModel
using Logging

include("forecast_unit.jl")

using GLMakie

schedule, historical, current = _forecast_fixture()
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
    GLMakie,
    data,
    context,
)
axis = figure.content[1]
plots = axis.scene.plots

@testset "GLMakie team-strength figure" begin
    @test length(plots) == nrow(data) + 4
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
    labels = filter(plot -> plot isa GLMakie.TextLabel, plots)
    @test [plot.text[] for plot in labels] == data.team
    @test all(plot.fontsize[] >= 17 for plot in labels)
    @test all(plot.background_color[] === :white for plot in labels)
    @test all(plot.text_color[] === :black for plot in labels)
    @test axis.title[] ==
        "Season 2023 team strengths as of start of week 2"
    limits = axis.limits[]
    @test limits == ((0, 100), (0, 100))
    @test axis.xticks[][2] == ["$value%" for value in 0:10:100]
    @test axis.yticks[][2] == ["$value%" for value in 0:10:100]
    @test occursin("central 80% posterior intervals", figure.content[2].text[])
    @test occursin("Neutral-site mean rates", figure.content[2].text[])
    @test occursin("league Gamma prior percentiles", figure.content[2].text[])
    @test occursin("league touchdown Gamma prior", axis.xlabel[])
    @test occursin("league defensive-event Gamma prior", axis.ylabel[])
end

@testset "plot means can lie outside central intervals" begin
    skewed_data = copy(data)
    skewed_data.offense_rate[1] = 4.0 * maximum(data.offense_upper)
    skewed_data.defense_rate[1] = 0.5 * minimum(data.defense_lower)
    skewed_figure = SurvivorModel._team_strength_figure(GLMakie, skewed_data, context)
    skewed_axis = skewed_figure.content[1]
    mapped = SurvivorModel._team_strength_plot_percentiles(skewed_data, context.model.prior)
    @test skewed_axis.scene.plots[1][1][][1][1] == mapped.offense_percentile[1]
    @test skewed_axis.scene.plots[2][1][][1][3] == mapped.offense_upper_percentile[1]
    @test skewed_axis.scene.plots[3][1][][1][2] == mapped.defense_lower_percentile[1]
    @test mapped.offense_percentile[1] >= mapped.offense_upper_percentile[1]
    @test mapped.defense_percentile[1] <= mapped.defense_lower_percentile[1]
end

function _rendered_team_label_boxes(axis)
    labels = filter(plot -> plot isa GLMakie.TextLabel, axis.scene.plots)
    return [
        begin
            background = only(filter(child -> child isa GLMakie.Poly, label.plots))
            box = GLMakie.boundingbox(background)
            (
                box.origin[1], box.origin[2],
                box.origin[1] + box.widths[1], box.origin[2] + box.widths[2],
            )
        end
        for label in labels
    ]
end

function _test_rendered_team_labels(axis, mapped)
    width, height = GLMakie.viewport(axis.scene)[].widths
    boxes = _rendered_team_label_boxes(axis)
    @test length(boxes) == nrow(mapped)
    segments = only(filter(plot -> plot isa GLMakie.LineSegments, axis.scene.plots))[1][]
    @test length(segments) == 2 * nrow(mapped)
    anchors = [
        GLMakie.Makie.project(
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
    offense_lower=fill(SurvivorModel.quantile(offense_distribution, 0.1), 32),
    offense_upper=fill(SurvivorModel.quantile(offense_distribution, 0.9), 32),
    defense_rate=fill(defense_reference.shape / defense_reference.rate, 32),
    defense_lower=fill(SurvivorModel.quantile(defense_distribution, 0.1), 32),
    defense_upper=fill(SurvivorModel.quantile(defense_distribution, 0.9), 32),
)
dense_figure = SurvivorModel._team_strength_figure(GLMakie, dense_data, context)
dense_axis = dense_figure.content[1]
dense_mapped = SurvivorModel._team_strength_plot_percentiles(dense_data, context.model.prior)

@testset "dense team labels and resize" begin
    _test_rendered_team_labels(dense_axis, dense_mapped)
    old_positions = [box[1:2] for box in _rendered_team_label_boxes(dense_axis)]
    GLMakie.resize!(dense_figure, 680, 500)
    _test_rendered_team_labels(dense_axis, dense_mapped)
    @test old_positions != [box[1:2] for box in _rendered_team_label_boxes(dense_axis)]
    GLMakie.resize!(dense_figure, 1100, 760)
    _test_rendered_team_labels(dense_axis, dense_mapped)
    image = get(ENV, "TEAM_STRENGTH_PLOT_SMOKE_IMAGE", "")
    isempty(image) || GLMakie.save(image, dense_figure)

    edge_data = copy(dense_data)
    edge_data.offense_rate = repeat([0.0, 1e308], 16)
    edge_data.defense_rate = repeat([0.0, 0.0, 1e308, 1e308], 8)
    edge_data.offense_lower .= 0.0
    edge_data.offense_upper .= 1e308
    edge_data.defense_lower .= 0.0
    edge_data.defense_upper .= 1e308
    edge_figure = SurvivorModel._team_strength_figure(GLMakie, edge_data, context)
    _test_rendered_team_labels(
        edge_figure.content[1],
        SurvivorModel._team_strength_plot_percentiles(edge_data, context.model.prior),
    )
end

@testset "GLMakie popup lifecycle" begin
    opened_screen = Ref{Any}(nothing)
    owned_scene = Ref{Union{Nothing,GLMakie.Scene}}(nothing)
    close_timer = Ref{Union{Nothing,Timer}}(nothing)
    closing_task = Ref{Union{Nothing,Task}}(nothing)
    lifecycle_events = Symbol[]
    on_screen = screen -> begin
        opened_screen[] = screen
        owned_scene[] = screen.scene
        @test isopen(screen)
        @test !isempty(screen.scene.children)
        close_timer[] = Timer(1.0) do _
            closing_task[] = current_task()
            push!(lifecycle_events, :closing)
            isopen(screen) && close(screen)
        end
        return nothing
    end

    @test SurvivorModel._show_team_strength_plot(
        context;
        on_screen,
    ) === nothing
    push!(lifecycle_events, :returned)
    @test lifecycle_events == [:closing, :returned]
    @test closing_task[] !== nothing
    closing_task[] !== nothing && wait(closing_task[])
    @test opened_screen[] !== nothing
    @test !isopen(opened_screen[])
    @test isempty(owned_scene[].children)
    close_timer[] !== nothing && close(close_timer[])

    failed_screen = Ref{Any}(nothing)
    failed_scene = Ref{Union{Nothing,GLMakie.Scene}}(nothing)
    on_failed_screen = screen -> begin
        failed_screen[] = screen
        failed_scene[] = screen.scene
        throw(ErrorException("test plot callback failure"))
    end
    @test_throws ErrorException("test plot callback failure") SurvivorModel._show_team_strength_plot(
        context;
        on_screen=on_failed_screen,
    )
    @test failed_screen[] !== nothing
    @test !isopen(failed_screen[])
    @test isempty(failed_scene[].children)

    immediately_closed_scene = Ref{Union{Nothing,GLMakie.Scene}}(nothing)
    on_immediate_close = screen -> begin
        immediately_closed_scene[] = screen.scene
        close(screen)
        return nothing
    end
    @test SurvivorModel._show_team_strength_plot(
        context;
        on_screen=on_immediate_close,
    ) === nothing
    @test isempty(immediately_closed_scene[].children)
    @test_logs min_level=Logging.Warn GLMakie.closeall()
end
