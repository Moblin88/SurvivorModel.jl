module SurvivorModelDockerBLAS

const MKL_UUID = Base.UUID("33e6dc65-8f57-5167-99aa-e5a354878fb2")
const AOCL_UUID = Base.UUID("f52889c3-24a9-4d04-95e9-8e965a4fc4c7")
const LINEAR_ALGEBRA_UUID = Base.UUID("37e2e46d-f89d-539d-b4ee-838fcccc9c8e")
const BLAS_PACKAGE_ENV = "/opt/survivormodel-blas"

function cpu_vendor(cpuinfo::AbstractString)
    vendor = match(r"(?m)^vendor_id\s*:\s*(\S+)", cpuinfo)
    vendor === nothing && return :unknown
    vendor_id = vendor.captures[1]
    vendor_id == "GenuineIntel" && return :intel
    vendor_id == "AuthenticAMD" && return :amd
    return :unknown
end

function select_backend(
    kernel::Symbol,
    architecture::Symbol,
    cpuinfo::AbstractString,
)
    kernel === :Linux && architecture === :x86_64 || return :openblas
    vendor = cpu_vendor(cpuinfo)
    vendor === :intel && return :mkl
    vendor === :amd && return :aocl
    return :openblas
end

function _thread_environment_keys(backend::Symbol)
    if backend === :mkl
        return (
            "SURVIVORMODEL_BLAS_THREADS",
            "MKL_NUM_THREADS",
            "MKL_DOMAIN_NUM_THREADS",
            "OMP_NUM_THREADS",
        )
    elseif backend === :aocl
        return (
            "SURVIVORMODEL_BLAS_THREADS",
            "AOCL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "OMP_NUM_THREADS",
        )
    end
    return (
        "SURVIVORMODEL_BLAS_THREADS",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
    )
end

function _native_thread_environment_key(backend::Symbol)
    return backend === :mkl ? "MKL_NUM_THREADS" :
           backend === :aocl ? "OMP_NUM_THREADS" : "OPENBLAS_NUM_THREADS"
end

function _parse_app_thread_override(env)
    value = get(env, "SURVIVORMODEL_BLAS_THREADS", nothing)
    value === nothing && return nothing
    parsed = tryparse(Int, strip(value))
    parsed !== nothing && 1 <= parsed <= typemax(Int32) ||
        throw(ArgumentError(
            "SURVIVORMODEL_BLAS_THREADS must be a positive integer no greater than $(typemax(Int32))",
        ))
    return string(parsed)
end

function set_default_thread_environment!(backend::Symbol, env=ENV)
    app_thread_override = _parse_app_thread_override(env)
    native_key = _native_thread_environment_key(backend)
    if app_thread_override !== nothing
        env[native_key] = app_thread_override
        return false
    end
    any(
        key -> key != "SURVIVORMODEL_BLAS_THREADS" && haskey(env, key),
        _thread_environment_keys(backend),
    ) && return false
    env[native_key] = "1"
    env["SURVIVORMODEL_BLAS_THREADS"] = "1"
    return true
end

function _package_identity(backend::Symbol)
    if backend === :mkl
        return Base.PkgId(MKL_UUID, "MKL"), "mkl"
    elseif backend === :aocl
        return Base.PkgId(AOCL_UUID, "AOCL"), "aocl"
    end
    throw(ArgumentError("no Docker vendor package for BLAS backend $backend"))
end

function _load_package(package_id::Base.PkgId)
    return Base.require(package_id)
end

function _blas_configuration()
    linear_algebra = Base.require(
        Base.PkgId(LINEAR_ALGEBRA_UUID, "LinearAlgebra"),
    )
    return sprint(show, getproperty(linear_algebra, :BLAS).get_config())
end

function load_backend!(
    backend::Symbol;
    package_environment::AbstractString=BLAS_PACKAGE_ENV,
    load_path=LOAD_PATH,
    package_loader=_load_package,
    blas_configuration=_blas_configuration,
)
    package_id, expected_backend = _package_identity(backend)
    isfile(joinpath(package_environment, "Project.toml")) ||
        throw(ErrorException(
            "Docker BLAS package environment is missing at $package_environment",
        ))
    already_on_load_path = package_environment in load_path
    if !already_on_load_path
        pushfirst!(load_path, package_environment)
    end
    try
        package_loader(package_id)
        configuration = lowercase(blas_configuration())
        occursin(expected_backend, configuration) ||
            throw(ErrorException(
                "$backend loaded but did not activate; active BLAS is $configuration",
            ))
    catch error
        error isa InterruptException && rethrow()
        throw(ErrorException(
            "Docker-selected $(package_id.name) BLAS failed to initialize: " *
            sprint(showerror, error),
        ))
    finally
        if !already_on_load_path
            index = findfirst(==(package_environment), load_path)
            index === nothing || deleteat!(load_path, index)
        end
    end
    return nothing
end

function _cpuinfo()
    path = "/proc/cpuinfo"
    return isfile(path) ? read(path, String) : ""
end

function initialize!(
    ;
    kernel::Symbol=Sys.KERNEL,
    architecture::Symbol=Sys.ARCH,
    cpuinfo::AbstractString=_cpuinfo(),
    env=ENV,
    backend_loader=load_backend!,
)
    backend = select_backend(kernel, architecture, cpuinfo)
    set_default_thread_environment!(backend, env)
    if backend === :mkl || backend === :aocl
        backend_loader(backend)
    end
    return backend
end

end
