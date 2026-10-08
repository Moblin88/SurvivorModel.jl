function _team_strength_label_box(center, size)
    return (
        center[1] - size[1] / 2,
        center[2] - size[2] / 2,
        center[1] + size[1] / 2,
        center[2] + size[2] / 2,
    )
end

function _team_strength_label_overlap(first, second, padding::Real=0.0)
    width = max(0.0, min(first[3] + padding, second[3]) -
        max(first[1] - padding, second[1]))
    height = max(0.0, min(first[4] + padding, second[4]) -
        max(first[2] - padding, second[2]))
    return width * height
end

function _team_strength_label_leader(center, size, anchor)
    dx = anchor[1] - center[1]
    dy = anchor[2] - center[2]
    dx == 0.0 && dy == 0.0 && return center, anchor
    scale = min(
        dx == 0.0 ? Inf : size[1] / (2 * abs(dx)),
        dy == 0.0 ? Inf : size[2] / (2 * abs(dy)),
    )
    return (center[1] + scale * dx, center[2] + scale * dy), anchor
end

function _team_strength_label_layout(
    anchors,
    sizes,
    viewport::Tuple{<:Real,<:Real};
    padding::Real=4.0,
    marker_radius::Real=8.0,
)
    length(anchors) == length(sizes) ||
        throw(ArgumentError("label anchors and sizes must have equal lengths"))
    width, height = Float64.(viewport)
    all(value -> isfinite(value) && value > 0.0, (width, height)) ||
        throw(ArgumentError("label viewport dimensions must be finite and positive"))
    isfinite(padding) && padding >= 0.0 &&
        isfinite(marker_radius) && marker_radius >= 0.0 ||
        throw(ArgumentError("label padding and marker radius must be finite and nonnegative"))
    all(anchor -> length(anchor) == 2 && all(isfinite, anchor), anchors) ||
        throw(ArgumentError("label anchors must contain two finite coordinates"))
    all(size -> length(size) == 2 &&
        all(value -> isfinite(value) && value > 0.0, size), sizes) ||
        throw(ArgumentError("label dimensions must be finite and positive"))
    centers = NTuple{2,Float64}[]
    boxes = NTuple{4,Float64}[]
    leaders = Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    crowded = false
    isempty(anchors) && return (; centers, boxes, leaders, crowded)

    margin = Float64(padding)
    max_width = maximum(size[1] for size in sizes)
    max_height = maximum(size[2] for size in sizes)
    step = max(4.0, min(12.0, max_width / 3, max_height / 2))
    grid = [
        (x, y)
        for x in (margin + max_width / 2):step:(width - margin - max_width / 2)
        for y in (margin + max_height / 2):step:(height - margin - max_height / 2)
    ]
    marker_boxes = [
        _team_strength_label_box(anchor, (2 * marker_radius, 2 * marker_radius))
        for anchor in anchors
    ]
    for (anchor, size) in zip(anchors, sizes)
        xmin, xmax = margin + size[1] / 2, width - margin - size[1] / 2
        ymin, ymax = margin + size[2] / 2, height - margin - size[2] / 2
        candidates = copy(grid)
        for ring in 1:3, (dx, dy) in (
            (0, 1), (1, 0), (-1, 0), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
        )
            x = anchor[1] + ring * dx * (size[1] / 2 + marker_radius + margin)
            y = anchor[2] + ring * dy * (size[2] / 2 + marker_radius + margin)
            push!(candidates, (
                xmin <= xmax ? clamp(x, xmin, xmax) : width / 2,
                ymin <= ymax ? clamp(y, ymin, ymax) : height / 2,
            ))
        end
        sort!(candidates; by=center -> (
            (center[1] - anchor[1])^2 + (center[2] - anchor[2])^2,
            center[1],
            center[2],
        ))
        chosen = nothing
        for center in candidates
            box = _team_strength_label_box(center, size)
            inside = box[1] >= margin && box[2] >= margin &&
                box[3] <= width - margin && box[4] <= height - margin
            if inside &&
                all(other -> _team_strength_label_overlap(box, other, margin) == 0.0, boxes) &&
                all(marker -> _team_strength_label_overlap(box, marker, margin) == 0.0, marker_boxes)
                chosen = center
                break
            end
        end
        if chosen === nothing
            crowded = true
            chosen = candidates[argmin([
                sum(_team_strength_label_overlap(
                    _team_strength_label_box(center, size), other, margin,
                ) for other in boxes; init=0.0) +
                sum(_team_strength_label_overlap(
                    _team_strength_label_box(center, size), marker, margin,
                ) for marker in marker_boxes; init=0.0)
                for center in candidates
            ])]
        end
        push!(centers, chosen)
        push!(boxes, _team_strength_label_box(chosen, size))
        push!(leaders, _team_strength_label_leader(chosen, size, anchor))
    end
    return (; centers, boxes, leaders, crowded)
end

function _team_strength_plot_labels!(
    makie::Module,
    axis,
    data::AbstractDataFrame,
)
    labels = map(data.team) do team
        position = makie.Observable(makie.Point2f(0, 0))
        plot = makie.textlabel!(
            axis.scene,
            position;
            text=team,
            space=:pixel,
            fontsize=17,
            font=:bold,
            padding=5,
            text_color=:black,
            background_color=:white,
            strokecolor=:gray55,
            strokewidth=0.7,
            cornerradius=3,
        )
        # TextLabel's data bounds contain only its anchor, not its pixel background.
        background = only(filter(child -> child isa makie.Poly, plot.plots))
        bounds = makie.boundingbox(background)
        size = (Float64(bounds.widths[1]), Float64(bounds.widths[2]))
        return (; position, plot, size)
    end
    segments = makie.Observable(makie.Point2f[])
    leader_plot = makie.linesegments!(
        axis.scene,
        segments;
        space=:pixel,
        color=:gray45,
        linewidth=0.9,
        depth_shift=0.001,
    )
    viewport = makie.viewport(axis.scene)
    was_crowded = Ref(false)
    update_labels = function(_values...)
        width, height = Float64.(viewport[].widths)
        if width <= 0.0 || height <= 0.0
            @debug "team-strength labels awaiting a usable viewport" width height
            return nothing
        end
        anchors = [
            begin
                projected = makie.Makie.project(
                    axis.scene,
                    :data,
                    :pixel,
                    makie.Point2f(row.offense_percentile, row.defense_percentile),
                )
                (Float64(projected[1]), Float64(projected[2]))
            end
            for row in eachrow(data)
        ]
        layout = _team_strength_label_layout(
            anchors,
            [label.size for label in labels],
            (width, height),
        )
        for (label, center) in zip(labels, layout.centers)
            label.position[] = makie.Point2f(center)
        end
        segments[] = [
            makie.Point2f(point)
            for leader in layout.leaders for point in leader
        ]
        if layout.crowded && !was_crowded[]
            @warn "team-strength labels are crowded; enlarge the plot window" width height
        end
        was_crowded[] = layout.crowded
        return nothing
    end
    makie.onany(
        update_labels,
        axis.scene,
        viewport,
        axis.scene.camera.projectionview,
        axis.scene.camera.resolution;
        update=true,
    )
    return (labels=[label.plot for label in labels], leaders=leader_plot)
end
