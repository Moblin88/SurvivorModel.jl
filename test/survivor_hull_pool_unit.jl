function _pool_basis_dual_residual(model, basis)
    optimizer = BNB.JuMP.backend(model)
    n, m = length(basis.columns), length(basis.rows)
    costs = zeros(n)
    objective = BNB.JuMP.objective_function(model)
    for (c, v) in BNB.JuMP.linear_terms(objective)
        costs[Int(BNB.HiGHS.column(optimizer, BNB.JuMP.index(v))) + 1] = c
    end
    basic = zeros(BNB.HiGHS.HighsInt, m)
    @test BNB.HiGHS.Highs_getBasicVariables(optimizer.inner, basic) == BNB.HiGHS.kHighsStatusOk
    rhs = [i >= 0 ? costs[i + 1] : 0.0 for i in basic]
    multipliers = zeros(m)
    @test BNB.HiGHS.Highs_getBasisTransposeSolve(
        optimizer.inner, rhs, multipliers, C_NULL, C_NULL,
    ) == BNB.HiGHS.kHighsStatusOk
    reduced = copy(costs)
    residual = 0.0
    function violation(status, value, lower, upper)
        lower == upper && return 0.0
        status == BNB.HiGHS.kHighsBasisStatusBasic && return abs(value)
        status == BNB.HiGHS.kHighsBasisStatusLower && return max(0.0, -value)
        status == BNB.HiGHS.kHighsBasisStatusUpper && return max(0.0, value)
        return abs(value)
    end
    for (F, S) in BNB.JuMP.list_of_constraint_types(model)
        F == BNB.JuMP.AffExpr || continue
        for ref in BNB.JuMP.all_constraints(model, F, S)
            row = Int(BNB.HiGHS.row(optimizer, BNB.JuMP.index(ref))) + 1
            object = BNB.JuMP.constraint_object(ref)
            for (c, v) in BNB.JuMP.linear_terms(object.func)
                reduced[Int(BNB.HiGHS.column(optimizer, BNB.JuMP.index(v))) + 1] -= c * multipliers[row]
            end
            lower = object.set isa BNB.JuMP.MOI.LessThan ? -Inf :
                object.set isa BNB.JuMP.MOI.EqualTo ? object.set.value : object.set.lower
            upper = object.set isa BNB.JuMP.MOI.GreaterThan ? Inf :
                object.set isa BNB.JuMP.MOI.EqualTo ? object.set.value : object.set.upper
            residual = max(residual, violation(basis.rows[row], multipliers[row], lower, upper))
        end
    end
    for variable in BNB.JuMP.all_variables(model)
        column = Int(BNB.HiGHS.column(optimizer, BNB.JuMP.index(variable))) + 1
        residual = max(residual, violation(basis.columns[column], reduced[column],
            BNB.JuMP.lower_bound(variable), BNB.JuMP.upper_bound(variable)))
    end
    return residual
end

function _pool_solve!(tree)
    for (option, value) in (("solver", "simplex"), ("simplex_strategy", 1),
                            ("presolve", "choose"), ("objective_bound", Inf))
        BNB.JuMP.set_optimizer_attribute(tree.model, option, value)
    end
    BNB.JuMP.optimize!(tree.model)
    @test BNB.JuMP.termination_status(tree.model) == BNB.JuMP.MOI.OPTIMAL
end

function _pool_discarded_snapshot(tree, pool)
    return WeakRef(BNB._survivor_hull_parent(tree, pool, Int[], true))
end

@testset "stable simplex hull pool" begin
    @testset "parent, sibling, grandchild and exhaustive LP agreement" begin
        for repeated in (false, true), losses in (0, 2), H in 0:3
            data, inputs = _bnb_fixture(; repeated)
            state = SurvivorPoolState(2025, 1; strikes_remaining=losses)
            config = _bnb_config(; simplex=true, hessian_weeks=H)
            tree = BNB._build_survivor_full_model(data, state, config, inputs)
            direct = BNB._build_survivor_full_model(data, state, config, inputs)
            BNB.JuMP.set_objective_function(tree.model, -BNB.JuMP.objective_function(tree.model))
            BNB.JuMP.set_objective_sense(tree.model, BNB.JuMP.MOI.MIN_SENSE)
            BNB.JuMP.set_objective_function(direct.model, -BNB.JuMP.objective_function(direct.model))
            BNB.JuMP.set_objective_sense(direct.model, BNB.JuMP.MOI.MIN_SENSE)
            pool = BNB._survivor_hull_pool(tree)
            _pool_solve!(tree)
            root = BNB._survivor_hull_parent(tree, pool, Int[], true)
            original = _bnb_numeric_snapshot(tree)
            @test BNB._survivor_hull_restore!(tree, pool, root; expired=() -> true) === nothing
            @test BNB._survivor_hull_strengthen!(tree, pool, root.basis; expired=() -> true) === nothing
            parents = Dict{Int,BNB.SurvivorHullParent}()
            for path in ([1], [2], [1, 4], [2, 3], [4], [4, 1], [1], [2])
                repeated && path == [2, 3] && continue
                parent = length(path) == 1 || first(path) == 4 ? root : parents[first(path)]
                basis = BNB._survivor_hull_restore!(tree, pool, parent)
                @test _pool_basis_dual_residual(tree.model, basis) <= 1e-7
                @test BNB._survivor_tree_apply_path!(tree, path; rewrite_hulls=false)
                basis = BNB._survivor_hull_strengthen!(tree, pool, basis)
                @test _pool_basis_dual_residual(tree.model, basis) <= 1e-7
                @test BNB._survivor_tree_apply_path!(direct, path)
                _pool_solve!(tree)
                _pool_solve!(direct)
                @test BNB.JuMP.objective_value(tree.model) ≈ BNB.JuMP.objective_value(direct.model) atol=1e-7
                @test isfinite(BNB._survivor_tree_upper_bound(tree.model))
                parents[first(path)] = BNB._survivor_hull_parent(tree, pool, path, true)
                @test _pool_basis_dual_residual(tree.model, parents[first(path)].basis) <= 1e-7
            end
            for picks in Iterators.product((1, 2), (3, 4), (5, 6))
                path = collect(picks)
                length(unique(data.team[path])) == 3 || continue
                basis = BNB._survivor_hull_restore!(tree, pool, root)
                @test BNB._survivor_tree_apply_path!(tree, path; rewrite_hulls=false)
                basis = BNB._survivor_hull_strengthen!(tree, pool, basis)
                @test _pool_basis_dual_residual(tree.model, basis) <= 1e-7
                _pool_solve!(tree)
                plan = BNB._survivor_tree_plan(tree, path)
                @test -BNB.JuMP.objective_value(tree.model) ≈ plan.objective_value atol=1e-7
            end
            basis = BNB._survivor_hull_restore!(tree, pool, root)
            @test _pool_basis_dual_residual(tree.model, basis) <= 1e-7
            @test _bnb_numeric_snapshot(tree)[1] == original[1]
            for slot in eachindex(root.versions)
                @test pool.versions[slot] == root.versions[slot]
            end
            @test all(v -> !v.enabled, pool.versions[(length(root.versions) + 1):end])
            if length(basis.rows) > length(root.basis.rows)
                @test_throws ErrorException BNB._survivor_restore_basis!(tree.model, root.basis)
            else
                @test BNB._survivor_restore_basis!(tree.model, root.basis) === nothing
            end
            extra = BNB.JuMP.@constraint(tree.model, tree.selected[1] <= 1)
            @test_throws ErrorException BNB._survivor_hull_extend_basis(tree.model, pool, root.basis)
            BNB.JuMP.delete(tree.model, extra)
        end

        @testset "basic slot reuse and nonbasic retention" begin
            model = BNB._survivor_direct_milp_model(_bnb_config(; simplex=true); lp_relaxation=true)
            BNB.JuMP.@variable(model, 0 <= x <= 1)
            BNB.JuMP.@variable(model, 0.1 <= y <= 0.9)
            BNB.JuMP.@variable(model, 0 <= z <= 0.9)
            rows = (
                BNB.JuMP.@constraint(model, z >= 0.1x),
                BNB.JuMP.@constraint(model, z <= 0.9x),
                BNB.JuMP.@constraint(model, z - y >= -0.9 + 0.9x),
                BNB.JuMP.@constraint(model, z - y <= -0.1 + 0.1x),
            )
            hull = (; key=(:candidate_probability, 1), index=1, dummy=z, selected=x,
                    lower_row=rows[1], upper_row=rows[2],
                    other_upper_row=rows[3], other_lower_row=rows[4])
            model.ext[:survivor_node_hulls] = [hull]
            interval = (; lower=fill(0.2, 1, 1), upper=fill(0.8, 1, 1))
            model.ext[:survivor_node_bounds] = (
                candidate_probability=interval, team_conditioned=(candidate_probability=interval,),
            )
            BNB.JuMP.@objective(model, Min, 0.2x - z)
            tree = (; model)
            pool = BNB._survivor_hull_pool(tree)
            _pool_solve!(tree)
            parent = BNB._survivor_hull_parent(tree, pool, Int[], true)
            retained = [i for i in 1:4 if parent.basis.rows[pool.native_rows[i]] !=
                        BNB.HiGHS.kHighsBasisStatusBasic]
            original_rows = [BNB.JuMP.constraint_object(ref) for ref in rows]
            @test !isempty(retained)
            basis = BNB._survivor_hull_strengthen!(tree, pool, parent.basis)
            @test pool.reused > 0
            @test pool.added > 0
            @test pool.deactivated > 0
            @test _pool_basis_dual_residual(model, basis) <= 1e-7
            for slot in retained
                @test pool.versions[slot] == parent.versions[slot]
                @test pool.versions[slot].enabled
                object = BNB.JuMP.constraint_object(rows[slot])
                @test object.func == original_rows[slot].func
                @test object.set == original_rows[slot].set
            end
            _pool_solve!(tree)
            @test BNB.JuMP.objective_value(model) ≈ -0.6 atol=1e-7
            certificate = BNB._survivor_tree_upper_bound(model)
            child = BNB._survivor_hull_parent(tree, pool, [1], true)
            @test isfinite(certificate)
            @test _pool_basis_dual_residual(model, child.basis) <= 1e-7
            restored = BNB._survivor_hull_restore!(tree, pool, parent)
            @test _pool_basis_dual_residual(model, restored) <= 1e-7
            @test pool.versions[1:4] == parent.versions
            @test all(v -> !v.enabled, pool.versions[5:end])
            @test isfinite(BNB._survivor_tree_lagrangian_bound(model; row_dual=ref -> 0.0))
            _pool_solve!(tree)
            discarded = _pool_discarded_snapshot(tree, pool)
            GC.gc()
            @test discarded.value === nothing
        end

        @testset "inactive nonbasic contents and unrelated mutation rejection" begin
            model = BNB._survivor_direct_milp_model(_bnb_config(; simplex=true); lp_relaxation=true)
            BNB.JuMP.@variable(model, 0 <= x <= 1)
            BNB.JuMP.@variable(model, 0 <= z <= 1)
            ref = BNB.JuMP.add_constraint(model, BNB.JuMP.ScalarConstraint(
                z - 0.1x, BNB.JuMP.MOI.GreaterThan(-Inf),
            ))
            BNB.JuMP.@objective(model, Min, 0.0x + 0.0z)
            expr = 1.0z
            version = BNB.SurvivorHullVersion(1, 1, 0.1, false)
            pool = BNB.SurvivorHullPool(
                BNB.JuMP.ConstraintRef[ref], [version], [(expr, expr, expr, expr)],
                [x], [[1]], [1], 0, 0, 0, 1,
            )
            optimizer = BNB.JuMP.backend(model)
            basis = BNB.SurvivorNativeBasis(
                optimizer, BNB.HiGHS.HighsInt[BNB.HiGHS.kHighsBasisStatusLower,
                                             BNB.HiGHS.kHighsBasisStatusBasic],
                BNB.HiGHS.HighsInt[BNB.HiGHS.kHighsBasisStatusZero],
                BNB._survivor_basis_mapping(model, optimizer),
                BNB._survivor_basis_row_mapping(model, optimizer),
            )
            BNB._survivor_restore_basis!(model, basis)
            parent = BNB.SurvivorHullParent(basis, [version], Int[], true, [(0.0, 1.0), (0.0, 1.0)])
            BNB._survivor_hull_write!(model, pool, 1, BNB.SurvivorHullVersion(1, 1, 0.3, false))
            restored = BNB._survivor_hull_restore!((; model), pool, parent)
            @test BNB.JuMP.normalized_coefficient(ref, x) == -0.1
            @test restored.rows == basis.rows
            @test _pool_basis_dual_residual(model, restored) == 0.0
            extra = BNB.JuMP.@constraint(model, x == z)
            @test_throws ErrorException BNB._survivor_hull_extend_basis(model, pool, basis)
            BNB.JuMP.delete(model, extra)
            BNB.JuMP.@variable(model, 0 <= new_column <= 1)
            @test_throws ErrorException BNB._survivor_hull_extend_basis(model, pool, basis)
        end
    end
end
