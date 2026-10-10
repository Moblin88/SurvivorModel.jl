using LinearAlgebra
using SurvivorModel

length(ARGS) in (1, 2) ||
    error("usage: smoke.jl <blas-threads> [julia-threads]")
expected_blas_threads = parse(Int, first(ARGS))
SurvivorModel._survivor_configure_blas_threads!()
actual_blas_threads = BLAS.get_num_threads()
thread_environment = filter(
    pair -> first(pair) in (
        "SURVIVORMODEL_BLAS_THREADS",
        "BLAS_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
    ),
    collect(ENV),
)
actual_blas_threads == expected_blas_threads ||
    error(
        "expected $expected_blas_threads BLAS threads, got $actual_blas_threads " *
        "with $(BLAS.get_config()) and environment $thread_environment",
    )

matrix = [2.0 1.0; 1.0 3.0]
vector = [1.0, 2.0]
@assert matrix * vector == [4.0, 7.0]
@assert matrix \ vector ≈ [0.2, 0.6]
@assert Threads.nthreads(:default) > 0
if length(ARGS) > 1
    expected_julia_threads = parse(Int, ARGS[2])
    @assert Threads.nthreads(:default) == expected_julia_threads
end
