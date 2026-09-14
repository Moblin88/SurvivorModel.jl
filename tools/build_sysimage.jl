const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_SYSIMAGE_PATH = joinpath(
    homedir(),
    ".cache",
    "SurvivorModel",
    "survivor.dylib",
)

sysimage_path = get(
    ENV,
    "SURVIVORMODEL_SYSIMAGE_PATH",
    DEFAULT_SYSIMAGE_PATH,
)

try
    @eval using PackageCompiler
catch error
    throw(ArgumentError(
        "PackageCompiler is required; install it in the active Julia " *
        "environment before running this script: " *
        sprint(showerror, error),
    ))
end

mkpath(dirname(sysimage_path))
PackageCompiler.create_sysimage(
    [:SurvivorModel];
    sysimage_path=sysimage_path,
    project=PROJECT_ROOT,
)
println("Wrote SurvivorModel sysimage to ", sysimage_path)
