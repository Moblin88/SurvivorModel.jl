using Test

@testset "SurvivorModel.jl" begin
    include("unit.jl")
    include("drive_cache_unit.jl")
    include("model_unit.jl")
    include("forecast_unit.jl")
    include("survivor_unit.jl")
    include("cli_unit.jl")
    if get(ENV, "SURVIVORMODEL_RUN_LIVE_SANITY", "false") == "true"
        include("drive_data_sanity.jl")
    end
end
