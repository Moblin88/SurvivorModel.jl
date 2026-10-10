using Test

# Suppress host-specific initialization; exercise selection and loading explicitly.
source = read(joinpath(@__DIR__, "..", "docker", "startup.jl"), String)
isolated_source = replace(source, "load_backend(backend(cpuinfo))" => "nothing")
include_string(Main, isolated_source, "docker/startup.jl")

@testset "Docker CPU-selected BLAS startup" begin
    startup = SurvivorDockerStartup
    @test startup.backend("vendor_id\t: AuthenticAMD\n", :x86_64; linux=true) == :AOCL
    @test startup.backend("vendor_id : GenuineIntel\n", :x86_64; linux=true) == :MKL
    @test startup.backend("vendor_id : Other\n", :x86_64; linux=true) === nothing
    @test startup.backend("", :x86_64; linux=true) === nothing
    @test startup.backend("model name : Intel processor\n", :x86_64; linux=true) === nothing
    @test startup.backend("vendor_id : AuthenticAMD\n", :aarch64; linux=true) === nothing
    @test startup.backend("vendor_id : GenuineIntel\n", :x86_64; linux=false) === nothing

    original_path = copy(LOAD_PATH)
    original_project = Base.active_project()
    selected = Symbol[]
    for name in (:AOCL, :MKL, nothing)
        startup.load_backend(name; environment="/test/blas", loader=backend -> begin
            @test first(LOAD_PATH) == "/test/blas"
            @test Base.active_project() == original_project
            push!(selected, backend)
        end)
        @test LOAD_PATH == original_path
        @test Base.active_project() == original_project
    end
    @test selected == [:AOCL, :MKL]
    mktempdir() do environment
        for name in ("AOCL", "MKL")
            directory = joinpath(environment, name, "src")
            mkpath(directory)
            write(joinpath(directory, "$name.jl"), "module $name\nend\n")
        end
        for name in (:AOCL, :MKL)
            startup.load_backend(name; environment)
            @test any(id -> id.name == String(name), keys(Base.loaded_modules))
            @test LOAD_PATH == original_path
            @test Base.active_project() == original_project
        end
    end
    @test_throws ErrorException startup.load_backend(:MKL; loader=_ -> error("load failed"))
    @test LOAD_PATH == original_path
    mktemp() do _, output
        redirect_stdout(output) do
            startup.load_backend(nothing)
        end
        @test position(output) == 0
    end
end
