using Test

function _headless_cli_process(project, directory; depot=nothing, marker=nothing)
    script = """
        using SurvivorModel, Test
        @test !any(id -> id.name in ("Makie", "CairoMakie", "GLMakie", "GLFW"), keys(Base.loaded_modules))
        output = IOBuffer()
        @test SurvivorModel._run_survivor_cli(["--help"]; output) == 0
        @test occursin("--plot-output", String(take!(output)))
        include($(repr(joinpath(@__DIR__, "cli_unit.jl"))))
        @test !any(id -> id.name in ("Makie", "CairoMakie", "GLMakie", "GLFW"), keys(Base.loaded_modules))
        $(marker === nothing ? "" : "@test SurvivorModel._headless_cache_marker == $(repr(marker))")
        println("HEADLESS_CLI_OK")
        """
    stdout_path = joinpath(directory, "stdout.log")
    stderr_path = joinpath(directory, "stderr.log")
    command = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $script`
    environment = copy(ENV)
    pop!(environment, "DISPLAY", nothing)
    pop!(environment, "WAYLAND_DISPLAY", nothing)
    environment["TMPDIR"] = directory
    depot === nothing || (environment["JULIA_DEPOT_PATH"] = depot)
    process = open(stdout_path, "w") do output
        open(stderr_path, "w") do errors
            return run(pipeline(ignorestatus(setenv(command, environment)); stdout=output, stderr=errors))
        end
    end
    stdout_text = read(stdout_path, String)
    stderr_text = read(stderr_path, String)
    if !success(process)
        @error "headless CLI subprocess failed" stdout_text stderr_text
    end
    @test success(process)
    @test occursin("HEADLESS_CLI_OK", stdout_text)
    @test !occursin(r"(?i)GLFW|GLMakie|OpenGL|could not.*display", stderr_text)
    return nothing
end

@testset "fresh-process non-plot CLI with warm, cold, and stale caches" begin
    root = dirname(@__DIR__)
    directory = mkdir(joinpath(@__DIR__, "headless-cli-$(getpid())"))
    try
        _headless_cli_process(root, directory)
        project = mkdir(joinpath(directory, "project"))
        cp(joinpath(root, "Project.toml"), joinpath(project, "Project.toml"))
        cp(joinpath(root, "Manifest.toml"), joinpath(project, "Manifest.toml"))
        cp(joinpath(root, "src"), joinpath(project, "src"))
        mkdir(joinpath(project, "test"))
        cp(joinpath(@__DIR__, "Project.toml"), joinpath(project, "test", "Project.toml"))
        depot = mkdir(joinpath(directory, "depot"))
        # Reuse dependency caches, but compile the copied package into an isolated depot.
        depot_path = join(vcat([depot], DEPOT_PATH), Sys.iswindows() ? ';' : ':')
        source = joinpath(project, "src", "SurvivorModel.jl")
        write(source, replace(read(source, String),
            "module SurvivorModel" => "module SurvivorModel\nconst _headless_cache_marker = :cold";
            count=1,
        ))
        _headless_cli_process(project, directory; depot=depot_path, marker=:cold)
        compiled = joinpath(depot, "compiled", "v$(VERSION.major).$(VERSION.minor)", "SurvivorModel")
        @test isdir(compiled)
        @test any(endswith(".ji"), readdir(compiled))
        source_text = read(source, String)
        write(source, replace(source_text, "_headless_cache_marker = :cold" => "_headless_cache_marker = :stale"))
        _headless_cli_process(project, directory; depot=depot_path, marker=:stale)
    finally
        rm(directory; recursive=true)
    end
end
