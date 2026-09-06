using Test

@testset "SurvivorModel.jl" begin
    include("unit.jl")
    include("drive_data_sanity.jl")
    include("model_unit.jl")
    include("model_sanity.jl")
    include("forecast_unit.jl")
end
