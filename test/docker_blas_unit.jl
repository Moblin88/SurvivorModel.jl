include(joinpath(@__DIR__, "..", "docker", "blas_backend.jl"))

@testset "Docker BLAS backend selection" begin
    docker_blas = SurvivorModelDockerBLAS
    intel_cpuinfo = "vendor_id : GenuineIntel\n"
    amd_cpuinfo = "vendor_id : AuthenticAMD\n"

    @test docker_blas.cpu_vendor(intel_cpuinfo) == :intel
    @test docker_blas.cpu_vendor(amd_cpuinfo) == :amd
    @test docker_blas.cpu_vendor("model name : unknown\n") == :unknown
    @test docker_blas.select_backend(:Linux, :x86_64, intel_cpuinfo) == :mkl
    @test docker_blas.select_backend(:Linux, :x86_64, amd_cpuinfo) == :aocl
    @test docker_blas.select_backend(
        :Linux, :aarch64, "model name : ARM\n",
    ) == :openblas
    @test docker_blas.select_backend(
        :Linux, :x86_64, "vendor_id : GenuineUnknown\n",
    ) == :openblas
    @test docker_blas.select_backend(:Darwin, :x86_64, intel_cpuinfo) == :openblas

    @testset "backend defaults and overrides" begin
        selected = Ref{Union{Nothing,Symbol}}(nothing)
        intel_env = Dict{String,String}()
        @test docker_blas.initialize!(
            ;
            kernel=:Linux,
            architecture=:x86_64,
            cpuinfo=intel_cpuinfo,
            env=intel_env,
            backend_loader=backend -> (selected[] = backend),
        ) == :mkl
        @test selected[] == :mkl
        @test intel_env["MKL_NUM_THREADS"] == "1"
        @test intel_env["SURVIVORMODEL_BLAS_THREADS"] == "1"

        amd_env = Dict{String,String}()
        @test docker_blas.initialize!(
            ;
            kernel=:Linux,
            architecture=:x86_64,
            cpuinfo=amd_cpuinfo,
            env=amd_env,
            backend_loader=backend -> (selected[] = backend),
        ) == :aocl
        @test selected[] == :aocl
        @test amd_env["OMP_NUM_THREADS"] == "1"
        @test amd_env["SURVIVORMODEL_BLAS_THREADS"] == "1"

        arm_env = Dict{String,String}()
        @test docker_blas.initialize!(
            ;
            kernel=:Linux,
            architecture=:aarch64,
            cpuinfo="model name : ARM\n",
            env=arm_env,
            backend_loader=backend -> error("OpenBLAS should not load a vendor package"),
        ) == :openblas
        @test arm_env["OPENBLAS_NUM_THREADS"] == "1"
        @test arm_env["SURVIVORMODEL_BLAS_THREADS"] == "1"

        override_env = Dict("OMP_NUM_THREADS" => "4")
        @test docker_blas.initialize!(
            ;
            kernel=:Linux,
            architecture=:x86_64,
            cpuinfo=intel_cpuinfo,
            env=override_env,
            backend_loader=backend -> nothing,
        ) == :mkl
        @test override_env == Dict("OMP_NUM_THREADS" => "4")

        app_override_env = Dict(
            "SURVIVORMODEL_BLAS_THREADS" => "2",
            "OMP_NUM_THREADS" => "4",
        )
        docker_blas.initialize!(
            ;
            kernel=:Linux,
            architecture=:x86_64,
            cpuinfo=amd_cpuinfo,
            env=app_override_env,
            backend_loader=backend -> nothing,
        )
        @test app_override_env["OMP_NUM_THREADS"] == "2"
        @test app_override_env["SURVIVORMODEL_BLAS_THREADS"] == "2"

        @test_throws ArgumentError docker_blas.initialize!(
            ;
            kernel=:Linux,
            architecture=:x86_64,
            cpuinfo=intel_cpuinfo,
            env=Dict("SURVIVORMODEL_BLAS_THREADS" => "invalid"),
            backend_loader=backend -> nothing,
        )
    end

    @testset "backend load validation and load-path cleanup" begin
        mktempdir() do environment
            write(joinpath(environment, "Project.toml"), "[deps]\n")
            load_path = ["@v#.#", "@stdlib"]
            loaded_package = Ref{Union{Nothing,Base.PkgId}}(nothing)
            @test isnothing(docker_blas.load_backend!(
                :mkl;
                package_environment=environment,
                load_path,
                package_loader=package_id -> (loaded_package[] = package_id),
                blas_configuration=() -> "LBTConfig(libmkl_rt.so)",
            ))
            @test loaded_package[].name == "MKL"
            @test environment ∉ load_path

            @test_throws ErrorException docker_blas.load_backend!(
                :aocl;
                package_environment=environment,
                load_path,
                package_loader=package_id -> nothing,
                blas_configuration=() -> "LBTConfig(libopenblas.so)",
            )
            @test environment ∉ load_path

            pushfirst!(load_path, environment)
            docker_blas.load_backend!(
                :mkl;
                package_environment=environment,
                load_path,
                package_loader=package_id -> nothing,
                blas_configuration=() -> "LBTConfig(libmkl_rt.so)",
            )
            @test count(==(environment), load_path) == 1
        end
    end

    @test_throws ErrorException docker_blas.initialize!(
        ;
        kernel=:Linux,
        architecture=:x86_64,
        cpuinfo=intel_cpuinfo,
        env=Dict{String,String}(),
        backend_loader=backend -> error("missing test backend"),
    )
end
