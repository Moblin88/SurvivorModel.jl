module SurvivorDockerStartup

function backend(cpuinfo::AbstractString, arch::Symbol=Sys.ARCH; linux::Bool=Sys.islinux())
    linux && arch == :x86_64 || return nothing
    vendor = match(r"(?m)^vendor_id\s*:\s*(\S+)", cpuinfo)
    vendor === nothing && return nothing
    vendor[1] == "AuthenticAMD" && return :AOCL
    vendor[1] == "GenuineIntel" && return :MKL
    return nothing
end

function load_backend(
    selected;
    environment::AbstractString="/opt/julia-blas",
    loader::Function=name -> Base.require(@__MODULE__, name),
)
    selected === nothing && return nothing
    original_load_path = copy(LOAD_PATH)
    try
        pushfirst!(LOAD_PATH, environment)
        loader(selected)
    finally
        empty!(LOAD_PATH)
        append!(LOAD_PATH, original_load_path)
    end
    return nothing
end

if Sys.islinux() && Sys.ARCH == :x86_64
    cpuinfo = isfile("/proc/cpuinfo") ? read("/proc/cpuinfo", String) : ""
    load_backend(backend(cpuinfo))
end

end
