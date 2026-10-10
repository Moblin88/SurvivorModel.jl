using Test

@testset "SurvivorModel.jl" begin
    include("docker_startup_unit.jl")
    include("unit.jl")
    include("drive_cache_unit.jl")
    include("model_unit.jl")
    include("forecast_unit.jl")
    include("team_strength_labels_unit.jl")
    include("survivor_grid_unit.jl")
    include("survivor_unit.jl")
    include("survivor_branch_and_bound_unit.jl")
    include("cli_unit.jl")
    include("headless_cli_unit.jl")
end
