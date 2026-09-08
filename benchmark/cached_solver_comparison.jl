using DataFrames
using Dates
using Printf
using Statistics
using SurvivorModel

const _CACHED_METHODS = (
    EMECMEFit(),
    EMLBFGSFit(),
    DirectLBFGSFit(),
    DirectBFGSFit(),
    MomentFit(),
    MomentLBFGSFit(),
    HybridFit(),
    BlockNewtonFit(),
    SchurNewtonFit(),
)

function _cached_em_solver(
    cells,
    initial_values,
    n_bins::Int;
    method::Symbol,
    polish::Bool,
)
    hyperparameters, home_multiplier, persistence =
        SurvivorModel._unpack_reset_parameters(initial_values, n_bins)
    log_likelihood = SurvivorModel._reset_event_log_likelihood(
        cells,
        hyperparameters,
        home_multiplier,
        persistence,
    )
    function_evaluations = 1
    iterations = 0
    converged = false

    for iteration in 1:SurvivorModel.RESET_EM_MAX_ITERATIONS
        expectations = SurvivorModel._reset_em_expectations(
            cells,
            hyperparameters,
            home_multiplier,
            persistence,
        )
        function_evaluations += expectations.function_evaluations
        em_hyperparameters, em_home_multiplier, em_persistence =
            SurvivorModel._reset_em_maximize(
                expectations,
                hyperparameters,
                home_multiplier,
                persistence,
            )
        ecme = SurvivorModel._reset_ecme_update(
            cells,
            em_hyperparameters,
            em_home_multiplier,
            em_persistence;
            method=method,
        )
        function_evaluations += ecme.function_evaluations

        next_hyperparameters = ecme.hyperparameters
        next_home_multiplier = ecme.home_multiplier
        next_persistence = ecme.persistence
        next_log_likelihood = SurvivorModel._reset_event_log_likelihood(
            cells,
            next_hyperparameters,
            next_home_multiplier,
            next_persistence,
        )
        function_evaluations += 1
        iterations = iteration

        hyperparameters = next_hyperparameters
        home_multiplier = next_home_multiplier
        persistence = next_persistence
        if abs(next_log_likelihood - log_likelihood) <=
           SurvivorModel.RESET_EM_ABSOLUTE_TOLERANCE +
           SurvivorModel.RESET_EM_RELATIVE_TOLERANCE *
           max(1.0, abs(log_likelihood))
            log_likelihood = next_log_likelihood
            converged = true
            break
        end
        log_likelihood = next_log_likelihood
    end

    if polish
        packed = SurvivorModel._pack_reset_parameters(
            hyperparameters,
            home_multiplier,
            persistence,
        )
        direct = SurvivorModel._reset_optimize_joint_parameters(
            cells,
            packed,
            n_bins;
            method=:lbfgs,
        )
        hyperparameters = direct.hyperparameters
        home_multiplier = direct.home_multiplier
        persistence = direct.persistence
        log_likelihood = direct.log_likelihood
        iterations += direct.iterations
        function_evaluations += direct.function_evaluations
        converged = true
    end

    return (
        hyperparameters=hyperparameters,
        home_multiplier=home_multiplier,
        persistence=persistence,
        log_likelihood=log_likelihood,
        iterations=iterations,
        function_evaluations=function_evaluations,
        hessian_evaluations=0,
        converged=converged,
        status=converged ? :converged : :failed,
    )
end

function _cached_solver(
    cells,
    initial_values,
    moment_initial_values,
    moment_blocks,
    n_bins::Int,
    method::PriorFitMethod,
)
    solver = prior_fit_method_name(method)
    if solver === :em_ecme
        return _cached_em_solver(
            cells,
            initial_values,
            n_bins;
            method=:nelder_mead,
            polish=false,
        )
    elseif solver === :em_lbfgs
        return _cached_em_solver(
            cells,
            initial_values,
            n_bins;
            method=:lbfgs,
            polish=false,
        )
    elseif solver === :direct_lbfgs
        result = SurvivorModel._reset_optimize_joint_parameters(
            cells,
            initial_values,
            n_bins;
            method=:lbfgs,
        )
        return merge(
            result,
            (
                hessian_evaluations=0,
                converged=true,
                status=:converged,
            ),
        )
    elseif solver === :direct_bfgs
        result = SurvivorModel._reset_optimize_joint_parameters(
            cells,
            initial_values,
            n_bins;
            method=:bfgs,
        )
        return merge(
            result,
            (
                hessian_evaluations=0,
                converged=true,
                status=:converged,
            ),
        )
    elseif solver === :moment
        moment_fit = SurvivorModel._reset_iterated_moment_parameters(
            moment_blocks,
        )
        packed = SurvivorModel._reset_moment_parameter_vector(moment_fit)
        hyperparameters, home_multiplier, persistence =
            SurvivorModel._unpack_reset_parameters(packed, n_bins)
        log_likelihood = SurvivorModel._reset_event_log_likelihood(
            cells,
            hyperparameters,
            home_multiplier,
            persistence,
        )
        return (
            hyperparameters=hyperparameters,
            home_multiplier=home_multiplier,
            persistence=persistence,
            log_likelihood=log_likelihood,
            iterations=moment_fit.iterations,
            function_evaluations=1,
            hessian_evaluations=0,
            converged=moment_fit.converged,
            status=moment_fit.converged ? :moment : :max_iterations,
        )
    elseif solver === :moment_lbfgs
        result = SurvivorModel._reset_optimize_joint_parameters(
            cells,
            moment_initial_values,
            n_bins;
            method=:lbfgs,
        )
        return merge(
            result,
            (
                hessian_evaluations=0,
                converged=true,
                status=:converged,
            ),
        )
    elseif solver === :hybrid
        return _cached_em_solver(
            cells,
            moment_initial_values,
            n_bins;
            method=:lbfgs,
            polish=true,
        )
    elseif solver === :block_newton
        result = SurvivorModel._reset_optimize_block_newton(
            cells,
            moment_initial_values,
            n_bins,
        )
        return merge(
            result,
            (
                hessian_evaluations=result.solver_metrics.hessian_evaluations,
                status=result.solver_metrics.status,
            ),
        )
    elseif solver === :schur_newton
        result = SurvivorModel._reset_optimize_schur_newton(
            cells,
            moment_initial_values,
            n_bins,
        )
        return merge(
            result,
            (
                hessian_evaluations=result.solver_metrics.hessian_evaluations,
                status=result.solver_metrics.status,
            ),
        )
    end
    throw(ArgumentError("unsupported cached benchmark solver: $solver"))
end

function _cached_solver_comparison(
    historical_drives;
    time_edges=[0, 120, 240, Inf],
    max_seasons::Int=3,
    repeats::Int=3,
)
    repeats > 0 || throw(ArgumentError("repeats must be positive"))
    data, edges = SurvivorModel.build_exposure_data(
        historical_drives;
        time_edges=time_edges,
    )
    byseason = SurvivorModel._season_stats(data)
    seasons = SurvivorModel._historical_seasons(byseason, max_seasons)
    n_bins = length(edges) - 1
    preparation_times = Float64[]
    for _ in 1:repeats
        preparation_start = time_ns()
        prepared_data, _ = SurvivorModel.build_exposure_data(
            historical_drives;
            time_edges=time_edges,
        )
        SurvivorModel._season_stats(prepared_data)
        push!(
            preparation_times,
            (time_ns() - preparation_start) / 1.0e6,
        )
    end
    sort!(preparation_times)
    preparation_ms = median(preparation_times)

    rows = NamedTuple[]
    for kind in (:td, :defensive)
        cells = SurvivorModel._reset_likelihood_cells(
            byseason,
            seasons,
            kind,
            n_bins,
        )
        moment_blocks = SurvivorModel._reset_moment_blocks(
            byseason,
            seasons,
            kind,
            n_bins,
        )
        initial_values = SurvivorModel._initial_reset_parameter_vector(
            byseason,
            seasons,
            kind,
            n_bins,
        )
        moment_initial_values =
            SurvivorModel._reset_moment_parameter_vector(
                SurvivorModel._reset_iterated_moment_parameters(moment_blocks),
            )

        for method in _CACHED_METHODS
            method_name = prior_fit_method_name(method)
            warmup = try
                _cached_solver(
                    cells,
                    initial_values,
                    moment_initial_values,
                    moment_blocks,
                    n_bins,
                    method,
                )
            catch error
                error isa ArgumentError || rethrow()
                nothing
            end

            if isnothing(warmup)
                push!(
                    rows,
                    (
                        kind=kind,
                        method=method_name,
                        cells=length(cells),
                        median_ms=NaN,
                        p90_ms=NaN,
                        iterations=0,
                        function_evaluations=0,
                        hessian_evaluations=0,
                        log_likelihood=NaN,
                        likelihood_gap=NaN,
                        converged_gap=NaN,
                        gradient_norm=NaN,
                        boundary_parameters=Symbol[],
                        status=:failed,
                        error="warm-up failed",
                    ),
                )
                continue
            end

            times = Float64[]
            result = warmup
            error_message = ""
            for _ in 1:repeats
                try
                    started = time_ns()
                    result = _cached_solver(
                        cells,
                        initial_values,
                        moment_initial_values,
                        moment_blocks,
                        n_bins,
                        method,
                    )
                    push!(times, (time_ns() - started) / 1.0e6)
                catch error
                    error isa ArgumentError || rethrow()
                    error_message = sprint(showerror, error)
                    break
                end
            end

            if isempty(times)
                push!(
                    rows,
                    (
                        kind=kind,
                        method=method_name,
                        cells=length(cells),
                        median_ms=NaN,
                        p90_ms=NaN,
                        iterations=0,
                        function_evaluations=0,
                        hessian_evaluations=0,
                        log_likelihood=NaN,
                        likelihood_gap=NaN,
                        converged_gap=NaN,
                        gradient_norm=NaN,
                        boundary_parameters=Symbol[],
                        status=:failed,
                        error=error_message,
                    ),
                )
                continue
            end

            sort!(times)
            final_gradient = SurvivorModel._reset_event_log_likelihood_with_gradient(
                cells,
                result.hyperparameters,
                result.home_multiplier,
                result.persistence,
            )
            metrics = hasproperty(result, :solver_metrics) ?
                result.solver_metrics : nothing
            status = isnothing(metrics) ? result.status : metrics.status
            push!(
                rows,
                (
                    kind=kind,
                    method=method_name,
                    cells=length(cells),
                    median_ms=median(times),
                    p90_ms=times[ceil(Int, 0.9 * length(times))],
                    iterations=result.iterations,
                    function_evaluations=result.function_evaluations,
                    hessian_evaluations=result.hessian_evaluations,
                    log_likelihood=final_gradient.log_likelihood,
                    likelihood_gap=NaN,
                    converged_gap=NaN,
                    gradient_norm=maximum(abs, final_gradient.gradient),
                    boundary_parameters=SurvivorModel._reset_boundary_parameters(
                        result.hyperparameters,
                        result.home_multiplier,
                        result.persistence,
                    ),
                    status=status,
                    error=error_message,
                ),
            )
        end
    end

    result = DataFrame(rows)
    for kind in (:td, :defensive)
        finite_rows = filter(
            row ->
                row.kind === kind &&
                isfinite(row.log_likelihood),
            eachrow(result),
        )
        isempty(finite_rows) && continue
        best_finite_log_likelihood = maximum(
            row.log_likelihood for row in finite_rows
        )
        converged_rows = filter(
            row -> row.status !== :failed,
            finite_rows,
        )
        best_converged_log_likelihood = isempty(converged_rows) ?
            NaN :
            maximum(row.log_likelihood for row in converged_rows)
        for row_index in axes(result, 1)
            result.kind[row_index] === kind || continue
            isfinite(result.log_likelihood[row_index]) || continue
            result.likelihood_gap[row_index] =
                best_finite_log_likelihood -
                result.log_likelihood[row_index]
            isfinite(best_converged_log_likelihood) &&
                (result.converged_gap[row_index] =
                    best_converged_log_likelihood -
                    result.log_likelihood[row_index])
        end
    end

    metadata = (
        drive_rows=nrow(historical_drives),
        exposure_rows=nrow(data),
        seasons=seasons,
        preparation_ms=preparation_ms,
    )
    return result, metadata
end

function main()
    historical_drives = load_drive_pbp(2021:2023)
    result, metadata = _cached_solver_comparison(
        historical_drives;
        repeats=3,
    )
    println(
        "drive_rows=", metadata.drive_rows,
        " exposure_rows=", metadata.exposure_rows,
        " seasons=", metadata.seasons,
        " preparation_ms=", round(metadata.preparation_ms; digits=3),
    )
    println(
        "kind       method             cells   median_ms     p90_ms  " *
        "iters  evals  hess    log_likelihood        gap      status " *
        "boundaries",
    )
    println("-"^126)
    for row in eachrow(result)
        boundaries = isempty(row.boundary_parameters) ?
            "-" :
            join(string.(row.boundary_parameters), ",")
        @printf(
            "%-10s %-18s %5d %11.3f %10.3f %6d %6d %5d %16.6f %10.3g %-12s %s\n",
            string(row.kind),
            string(row.method),
            row.cells,
            row.median_ms,
            row.p90_ms,
            row.iterations,
            row.function_evaluations,
            row.hessian_evaluations,
            row.log_likelihood,
            row.likelihood_gap,
            string(row.status),
            boundaries,
        )
    end
    return result
end

if basename(PROGRAM_FILE) == "cached_solver_comparison.jl"
    main()
end
