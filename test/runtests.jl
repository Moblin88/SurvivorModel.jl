using Test

@testset "SurvivorModel.jl" begin
    include("unit.jl")
    include("drive_data_sanity.jl")
    include("model_unit.jl")
    include("model_sanity.jl")
    include("forecast_unit.jl")
    include("calibration_unit.jl")
    if get(ENV, "SURVIVORMODEL_RUN_CALIBRATION", "false") == "true"
        include("calibration_live.jl")
    end
end
