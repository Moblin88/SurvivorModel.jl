using Test

@testset "SurvivorModel.jl" begin
    include("unit.jl")
    include("drive_cache_unit.jl")
    include("model_unit.jl")
    include("forecast_unit.jl")
    include("survivor_unit.jl")
    include("cli_unit.jl")
end
