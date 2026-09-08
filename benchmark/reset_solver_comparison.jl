using DataFrames
using Dates
using Printf
using SurvivorModel

function _synthetic_reset_drives(
    ;
    seasons=2021:2023,
    teams=["A", "B", "C", "D", "E", "F"],
    drives_per_team=60,
    touchdown_probability=0.12,
    home_multiplier=1.35,
)
    rows = DataFrame(
        game_id=String[],
        fixed_drive=Int[],
        posteam=String[],
        defteam=String[],
        posteam_home=Bool[],
        defteam_home=Bool[],
        drive_result=String[],
        time_of_possession=Second[],
        home_spread_change=Float64[],
    )
    for season in seasons
        for team in teams
            for index in 1:drives_per_team
                posteam_home = iseven(index + season + length(team))
                probability = posteam_home ?
                    touchdown_probability * home_multiplier :
                    touchdown_probability
                touchdown = mod(index + season + length(team), 1000) <
                    round(Int, 1000 * probability)
                push!(
                    rows,
                    (
                        "$(season)_$(team)_$(index)",
                        1,
                        team,
                        "DEFENSE",
                        posteam_home,
                        !posteam_home,
                        touchdown ? "Touchdown" : "Punt",
                        Second(60 + 60 * mod(index + season, 5)),
                        touchdown && posteam_home ? 7.0 : 0.0,
                    ),
                )
            end
        end
    end
    return rows
end

function reset_solver_comparison()
    scenarios = [
        (
            name=:balanced,
            drives=_synthetic_reset_drives(),
        ),
        (
            name=:sparse,
            drives=_synthetic_reset_drives(
                teams=["A", "B", "C"],
                drives_per_team=24,
                touchdown_probability=0.04,
            ),
        ),
        (
            name=:one_season,
            drives=_synthetic_reset_drives(
                seasons=2023:2023,
                teams=["A", "B", "C", "D"],
                drives_per_team=48,
            ),
        ),
        (
            name=:strong_home,
            drives=_synthetic_reset_drives(
                teams=["A", "B", "C", "D"],
                drives_per_team=48,
                touchdown_probability=0.08,
                home_multiplier=2.0,
            ),
        ),
    ]
    methods = (
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
    rows = NamedTuple[]
    for scenario in scenarios
        for method in methods
            started = time_ns()
            try
                fitted = fit_empirical_bayes_prior(
                    scenario.drives;
                    time_edges=[0, 120, 240, Inf],
                    current_season=2024,
                    method=method,
                    _return_solver_metrics=true,
                )
                prior = fitted.prior
                elapsed_ms = (time_ns() - started) / 1.0e6
                for kind in (:td, :defensive)
                    diagnostics = likelihood_fit_diagnostics(prior, kind)
                    metrics = fitted.solver_metrics[kind]
                    push!(
                        rows,
                        (
                            scenario=scenario.name,
                            method=prior_fit_method_name(method),
                            kind=kind,
                            milliseconds=elapsed_ms,
                            log_likelihood=diagnostics.log_likelihood,
                            iterations=diagnostics.iterations,
                            function_evaluations=diagnostics.function_evaluations,
                            boundary_parameters=diagnostics.boundary_parameters,
                            gradient_evaluations=metrics.gradient_evaluations,
                            hessian_evaluations=metrics.hessian_evaluations,
                            newton_iterations=metrics.newton_iterations,
                            damping_steps=metrics.damping_steps,
                            backtracking_steps=metrics.backtracking_steps,
                            schur_corrections=metrics.schur_corrections,
                            fallback_count=metrics.fallback_count,
                            active_coordinates=metrics.active_coordinates,
                            gradient_norm=metrics.gradient_norm,
                            parameter_step_norm=metrics.parameter_step_norm,
                            solver_status=metrics.status,
                            error="",
                        ),
                    )
                end
            catch err
                err isa ArgumentError || rethrow()
                elapsed_ms = (time_ns() - started) / 1.0e6
                for kind in (:td, :defensive)
                    push!(
                        rows,
                        (
                            scenario=scenario.name,
                            method=prior_fit_method_name(method),
                            kind=kind,
                            milliseconds=elapsed_ms,
                            log_likelihood=NaN,
                            iterations=0,
                            function_evaluations=0,
                            boundary_parameters=Symbol[],
                            gradient_evaluations=0,
                            hessian_evaluations=0,
                            newton_iterations=0,
                            damping_steps=0,
                            backtracking_steps=0,
                            schur_corrections=0,
                            fallback_count=0,
                            active_coordinates=0,
                            gradient_norm=NaN,
                            parameter_step_norm=NaN,
                            solver_status=:failed,
                            error=sprint(showerror, err),
                        ),
                    )
                end
            end
        end
    end
    result = DataFrame(rows)
    best_likelihoods = Dict{Tuple{Symbol,Symbol},Float64}()
    for row in rows
        isfinite(row.log_likelihood) || continue
        key = (row.scenario, row.kind)
        best_likelihoods[key] = max(
            get(best_likelihoods, key, -Inf),
            row.log_likelihood,
        )
    end
    result.likelihood_gap = [
        isfinite(row.log_likelihood) ?
            best_likelihoods[(row.scenario, row.kind)] -
            row.log_likelihood :
            NaN
        for row in eachrow(result)
    ]
    return result
end

function _print_reset_solver_comparison(result)
    println(
        "scenario    kind       method             ms       log_likelihood " *
        "       gap  status       error",
    )
    println("-"^122)
    for row in eachrow(result)
        @printf(
            "%-11s %-10s %-18s %8.2f %16.6f %10.3g %-12s %s\n",
            string(row.scenario),
            string(row.kind),
            string(row.method),
            row.milliseconds,
            row.log_likelihood,
            row.likelihood_gap,
            string(row.solver_status),
            row.error,
        )
    end
    return nothing
end

if basename(PROGRAM_FILE) == "reset_solver_comparison.jl"
    _print_reset_solver_comparison(reset_solver_comparison())
end
