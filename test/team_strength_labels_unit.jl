using SurvivorModel
using Test

@testset "team-strength label layout" begin
    for viewport in ((960.0, 580.0), (560.0, 330.0))
        width, height = viewport
        for anchors in (
            fill((width / 2, height / 2), 32),
            fill((0.0, 0.0), 32),
            fill((width, height), 32),
            [(width * column / 9, height * row / 5) for row in 1:4 for column in 1:8],
        )
            sizes = [(index % 3 == 0 ? 50.0 : 40.0, 29.0) for index in 1:32]
            layout = SurvivorModel._team_strength_label_layout(anchors, sizes, viewport)
            @test !layout.crowded
            @test length(layout.centers) == length(anchors)
            @test layout == SurvivorModel._team_strength_label_layout(anchors, sizes, viewport)
            for (index, box) in enumerate(layout.boxes)
                @test 0.0 <= box[1] < box[3] <= width
                @test 0.0 <= box[2] < box[4] <= height
                for other in layout.boxes[index + 1:end]
                    @test SurvivorModel._team_strength_label_overlap(box, other, 4.0) == 0.0
                end
                for anchor in anchors
                    marker = SurvivorModel._team_strength_label_box(anchor, (16.0, 16.0))
                    @test SurvivorModel._team_strength_label_overlap(box, marker, 4.0) == 0.0
                end
                start, target = layout.leaders[index]
                @test target == anchors[index]
                @test box[1] - 1e-8 <= start[1] <= box[3] + 1e-8
                @test box[2] - 1e-8 <= start[2] <= box[4] + 1e-8
                @test any(isapprox(start[coordinate], box[edge]; atol=1e-8)
                    for (coordinate, edge) in ((1, 1), (1, 3), (2, 2), (2, 4)))
            end
        end
    end
    initial = SurvivorModel._team_strength_label_layout(
        [(480.0, 290.0)], [(40.0, 29.0)], (960.0, 580.0),
    )
    resized = SurvivorModel._team_strength_label_layout(
        [(280.0, 165.0)], [(40.0, 29.0)], (560.0, 330.0),
    )
    @test initial.centers != resized.centers
    tiny = SurvivorModel._team_strength_label_layout(
        [(10.0, 10.0)], [(80.0, 40.0)], (20.0, 20.0),
    )
    @test tiny.crowded
    @test length(tiny.centers) == 1
    @test isempty(SurvivorModel._team_strength_label_layout([], [], (100.0, 100.0)).centers)
    @test_throws ArgumentError SurvivorModel._team_strength_label_layout(
        [(0.0, 0.0)], [], (100.0, 100.0),
    )
    for viewport in ((0.0, 100.0), (100.0, Inf), (NaN, 100.0))
        @test_throws ArgumentError SurvivorModel._team_strength_label_layout(
            [(0.0, 0.0)], [(40.0, 29.0)], viewport,
        )
    end
    @test_throws ArgumentError SurvivorModel._team_strength_label_layout(
        [(NaN, 0.0)], [(40.0, 29.0)], (100.0, 100.0),
    )
    @test_throws ArgumentError SurvivorModel._team_strength_label_layout(
        [(0.0, 0.0)], [(0.0, 29.0)], (100.0, 100.0),
    )
    @test_throws ArgumentError SurvivorModel._team_strength_label_layout(
        [(0.0, 0.0)], [(40.0, 29.0)], (100.0, 100.0); padding=-1.0,
    )
    @test !isdefined(SurvivorModel, :GLMakie)
end
