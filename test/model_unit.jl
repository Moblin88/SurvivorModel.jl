using SurvivorModel
using DataFrames
using Dates
using Distributions
using Random
using Statistics
using Test
import SurvivorModel: ScoreMarks, fit_score_marks, hazard_theta

@testset "model unit tests" begin
    @testset "_classify_event" begin
        @test SurvivorModel._classify_event("Touchdown") === :td
        @test SurvivorModel._classify_event("End of half") === :censored
        for result in [
            "Field goal", "Punt", "Turnover", "Turnover on downs",
            "Missed field goal", "Safety", "Opp touchdown",
        ]
            @test SurvivorModel._classify_event(result) === :defensive
        end
    end

    @testset "_validate_time_edges" begin
        @test SurvivorModel._validate_time_edges([0, 120, 240, 360, Inf]) ==
            [0.0, 120.0, 240.0, 360.0, Inf]
        @test_throws ArgumentError SurvivorModel._validate_time_edges([0, 10, 5, Inf])
        @test_throws ArgumentError SurvivorModel._validate_time_edges([1, 10, Inf])
        @test_throws ArgumentError SurvivorModel._validate_time_edges([0, 10, 20])
    end

    @testset "_drive_exposure_records" begin
        edges = [0.0, 10.0, 20.0, Inf]

        records = SurvivorModel._drive_exposure_records(15.0, edges, :td)
        @test records == [
            (time_bin=1, exposure=10.0, event=:none),
            (time_bin=2, exposure=5.0, event=:td),
        ]

        records = SurvivorModel._drive_exposure_records(25.0, edges, :censored)
        @test all(r -> r.event === :none, records)
        @test sum(r -> r.exposure, records) == 25.0

        records = SurvivorModel._drive_exposure_records(10.0, edges, :defensive)
        @test records == [(time_bin=1, exposure=10.0, event=:defensive)]

        records = SurvivorModel._drive_exposure_records(100.0, edges, :td)
        @test records[end] == (time_bin=3, exposure=80.0, event=:td)
    end

    function _make_drives()
        n = 5
        DataFrame(
            game_id=["2023_01_$i" for i in 1:n],
            fixed_drive=ones(Int, n),
            posteam=[isodd(i) ? "HOME" : "AWAY" for i in 1:n],
            defteam=[isodd(i) ? "AWAY" : "HOME" for i in 1:n],
            posteam_home=[isodd(i) for i in 1:n],
            defteam_home=[iseven(i) for i in 1:n],
            drive_result=["Touchdown", "Punt", "Field goal", "End of half", "Punt"],
            time_of_possession=[Second(120), Second(180), Second(240), Second(60), Second(150)],
            drive_start_yards_to_goal=Union{Missing,Int}[missing, 60, 40, 20, 95],
            yards_gained=[75.0, 10.0, 30.0, 0.0, 5.0],
            home_spread_change=[7.0, 0.0, 3.0, 0.0, 0.0],
        )
    end

    @testset "build_exposure_data" begin
        drives = _make_drives()
        data, edges = build_exposure_data(drives; time_edges=[0, 120, 240, Inf])
        @test edges == [0.0, 120.0, 240.0, Inf]
        @test issubset(
            ["game_id", "fixed_drive", "season", "posteam", "defteam", "time_bin",
                "posteam_home", "defteam_home", "exposure", "td", "defensive"],
            names(data),
        )
        @test !("pos_bin" in names(data))
        @test sum(data.td) == 1
        @test sum(data.defensive) == 3
        censored_rows = filter(:game_id => ==("2023_01_4"), data)
        @test all(==(0), censored_rows.td)
        @test all(==(0), censored_rows.defensive)
        @test sum(censored_rows.exposure) == 60.0
    end

    @testset "home-aware sufficient statistics" begin
        data, _ = build_exposure_data(_make_drives(); time_edges=[0, 120, 240, Inf])
        stats = SurvivorModel._season_stats(data)[2023]

        @test stats.td.home_counts[("HOME", 1)] == 1.0
        @test get(stats.td.away_counts, ("HOME", 1), 0.0) == 0.0
        @test stats.td.home_exposure[("HOME", 1)] == 360.0
        @test stats.defensive.home_counts[("HOME", 2)] == 1.0
        @test stats.defensive.away_exposure[("AWAY", 1)] == 360.0
    end

    @testset "Gamma posterior updates" begin
        drives = _make_drives()
        model = fit_hazard_model(drives; time_edges=[0, 120, 240, Inf])

        td_posterior = hazard_posterior(model, :td, "HOME", 1)
        @test length(td_posterior.components) == 1
        @test td_posterior.components[1].shape > 1.0
        @test td_posterior.components[1].rate > 100.0
        @test hazard_rate(model, :td, "HOME", 1) ==
            td_posterior.components[1].shape / td_posterior.components[1].rate

        previous_shape = td_posterior.components[1].shape
        previous_rate = td_posterior.components[1].rate
        update_hazard_model!(model, drives[1:1, :])
        updated = hazard_posterior(model, :td, "HOME", 1)
        @test updated.components[1].shape == previous_shape + 1.0
        @test updated.components[1].rate == previous_rate + 120.0
    end

    @testset "Gamma mixture primitives" begin
        mixture = GammaMixture(
            [1.0, 3.0],
            [GammaParams(2.0, 4.0), GammaParams(6.0, 9.0)];
            source_seasons=[2022, 2023],
        )
        @test mixture.weights ≈ [0.25, 0.75]
        @test SurvivorModel._gamma_mixture_mean(mixture) ≈
            0.25 * (2.0 / 4.0) + 0.75 * (6.0 / 9.0)
        @test SurvivorModel._gamma_mixture_variance(mixture) > 0.0
        log_mean, log_variance =
            SurvivorModel._gamma_mixture_log_moments(mixture)
        @test isfinite(log_mean)
        @test log_variance > 0.0

        home = SurvivorModel._gamma_mixture_home_adjusted(mixture, 2.0)
        @test home.source_seasons == mixture.source_seasons
        @test all(
            home_component.rate == mixture_component.rate / 2.0
            for (home_component, mixture_component) in
                zip(home.components, mixture.components)
        )

        updated = SurvivorModel._update_gamma_mixture(mixture, 2, 10.0)
        @test all(
            updated_component.shape == mixture_component.shape + 2.0 &&
            updated_component.rate == mixture_component.rate + 10.0
            for (updated_component, mixture_component) in
                zip(updated.components, mixture.components)
        )
        @test sum(updated.weights) ≈ 1.0
        transitioned = SurvivorModel._transition_gamma_mixture(
            updated,
            0.7,
            GammaParams(4.0, 8.0),
            2024,
        )
        @test length(transitioned.components) == 3
        @test transitioned.source_seasons == [2022, 2023, 2024]
        @test sum(transitioned.weights) ≈ 1.0

        @test_throws ArgumentError GammaMixture(
            [0.0, 0.0],
            [GammaParams(1.0, 1.0), GammaParams(1.0, 1.0)],
        )
        @test_throws ArgumentError SurvivorModel._update_gamma_mixture(
            mixture,
            0.5,
            10.0,
        )
        @test_throws ArgumentError SurvivorModel._transition_gamma_mixture(
            mixture,
            1.1,
            GammaParams(1.0, 1.0),
            2024,
        )
    end

    @testset "event-process Gamma marginal likelihood" begin
        component = GammaParams(2.0, 4.0)
        expected = 2.0 * log(4.0) + log(2.0) - 3.0 * log(7.0)
        @test SurvivorModel._log_gamma_event_marginal(
            component,
            1,
            3.0,
        ) ≈ expected
        @test SurvivorModel._log_gamma_event_marginal(
            component,
            0,
            0.0,
        ) == 0.0

        cell = (
            time_bin=1,
            counts=(1.0, 0.0, 2.0),
            home_counts=(0.0, 0.0, 0.0),
            away_exposures=(3.0, 4.0, 5.0),
            home_exposures=(0.0, 0.0, 0.0),
        )
        rho = 0.25
        m1 = SurvivorModel._log_gamma_event_marginal(component, 1, 3.0)
        m2 = SurvivorModel._log_gamma_event_marginal(component, 0, 4.0)
        m3 = SurvivorModel._log_gamma_event_marginal(component, 2, 5.0)
        m12 = SurvivorModel._log_gamma_event_marginal(component, 1, 7.0)
        m23 = SurvivorModel._log_gamma_event_marginal(component, 2, 9.0)
        m123 = SurvivorModel._log_gamma_event_marginal(component, 3, 12.0)
        expected_four_path = SurvivorModel._logsumexp((
            2.0 * log1p(-rho) + m1 + m2 + m3,
            log(rho) + log1p(-rho) + m12 + m3,
            log1p(-rho) + log(rho) + m1 + m23,
            2.0 * log(rho) + m123,
        ))
        @test SurvivorModel._log_reset_partition_marginal(
            cell,
            component,
            1.0,
            rho,
        ) ≈ expected_four_path

        base_values = SurvivorModel._gamma_shape_special_values(2.5)
        shifted_values = SurvivorModel._gamma_shape_shifted_special_values(
            2.5,
            8,
            base_values,
        )
        @test length(shifted_values) == 2
        @test shifted_values[1] ≈ SurvivorModel.SpecialFunctions.loggamma(10.5)
        @test shifted_values[2] ≈ SurvivorModel.SpecialFunctions.digamma(10.5)
        differences = SurvivorModel._gamma_shape_difference_values(
            2.5,
            8,
            base_values,
        )
        @test differences[1] ≈
            SurvivorModel.SpecialFunctions.loggamma(10.5) -
            SurvivorModel.SpecialFunctions.loggamma(2.5)
        @test differences[2] ≈
            SurvivorModel.SpecialFunctions.digamma(10.5) -
            SurvivorModel.SpecialFunctions.digamma(2.5)
        @test length(differences) == 2
        @test SurvivorModel._gamma_shape_shifted_special_values(
            2.5,
            0,
            base_values,
        ) === base_values
    end

    @testset "event-process likelihood gradient" begin
        cell = (
            time_bin=1,
            counts=(1.0, 0.0, 2.0),
            home_counts=(0.0, 1.0, 0.0),
            away_exposures=(3.0, 4.0, 5.0),
            home_exposures=(1.0, 2.0, 1.0),
        )
        values = [
            log(0.35),
            log(2.5),
            log(0.4 / 0.6),
            log(1.3),
        ]
        hyperparameters, home_multiplier, persistence =
            SurvivorModel._unpack_reset_parameters(values, 1)
        result = SurvivorModel._reset_event_log_likelihood_with_gradient(
            [cell],
            hyperparameters,
            home_multiplier,
            persistence,
        )
        for index in eachindex(values)
            step = 1.0e-6
            lower = copy(values)
            upper = copy(values)
            lower[index] -= step
            upper[index] += step
            lower_hyper, lower_home, lower_persistence =
                SurvivorModel._unpack_reset_parameters(lower, 1)
            upper_hyper, upper_home, upper_persistence =
                SurvivorModel._unpack_reset_parameters(upper, 1)
            finite_difference = (
                SurvivorModel._reset_event_log_likelihood(
                    [cell],
                    upper_hyper,
                    upper_home,
                    upper_persistence,
                ) -
                SurvivorModel._reset_event_log_likelihood(
                    [cell],
                    lower_hyper,
                    lower_home,
                    lower_persistence,
                )
            ) / (2.0 * step)
            @test result.gradient[index] ≈ finite_difference atol=2.0e-6
        end
    end

    @testset "probabilistic reset prior and moments" begin
        Random.seed!(17)
        historical = DataFrame(
            game_id=String[],
            fixed_drive=Int[],
            posteam=String[],
            defteam=String[],
            drive_result=String[],
            time_of_possession=Second[],
            posteam_home=Bool[],
            defteam_home=Bool[],
        )
        for season in 2021:2023
            for team in ["A", "B", "C", "D"]
                for index in 1:120
                    posteam_home = iseven(index + season)
                    base_probability = Dict(
                        "A" => 0.18,
                        "B" => 0.12,
                        "C" => 0.08,
                        "D" => 0.05,
                    )[team]
                    probability = posteam_home ?
                        1.6 * base_probability : base_probability
                    push!(
                        historical,
                        (
                            "$(season)_$(team)_$(index)",
                            1,
                            team,
                            "DEFENSE",
                            rand() < probability ? "Touchdown" : "Punt",
                            Second(60),
                            posteam_home,
                            !posteam_home,
                        ),
                    )
                end
            end
        end

        prior = fit_empirical_bayes_prior(
            historical;
            time_edges=[0, Inf],
            current_season=2024,
        )
        @test prior.historical_seasons == [2021, 2022, 2023]
        @test likelihood_fit_diagnostics(prior, :td).converged
        @test likelihood_fit_diagnostics(prior, :defensive).converged
        @test likelihood_fit_diagnostics(prior, :td).iterations > 0
        @test likelihood_fit_diagnostics(prior, :defensive).iterations > 0
        @test likelihood_fit_diagnostics(prior, :td).function_evaluations > 0
        @test likelihood_fit_diagnostics(prior, :defensive).function_evaluations > 0
        @test isfinite(likelihood_fit_diagnostics(prior, :td).log_likelihood)
        @test isfinite(
            likelihood_fit_diagnostics(prior, :defensive).log_likelihood,
        )
        @test all(p -> p.shape > 0 && p.rate > 0, prior.td_hyperparameters)
        @test all(p -> p.shape > 0 && p.rate > 0, prior.defensive_hyperparameters)
        @test haskey(prior.td_team_mixtures, "A")
        @test haskey(prior.defensive_team_mixtures, "B")
        td_mixture = prior.td_team_mixtures["A"][1]
        @test length(td_mixture.components) == 4
        @test td_mixture.source_seasons == [2021, 2022, 2023, 2024]
        @test sum(td_mixture.weights) ≈ 1.0
        @test 0.0 <= hazard_persistence(prior, :td) <= 1.0
        @test 0.0 <= hazard_persistence(prior, :defensive) <= 1.0
        @test prior.td_home_multiplier > 0.0
        @test prior.defensive_home_multiplier > 0.0

        @test_throws ArgumentError fit_empirical_bayes_prior(
            historical;
            time_edges=[0, Inf],
            max_seasons=0,
        )
        @test_throws ArgumentError fit_empirical_bayes_prior(
            historical[1:0, :];
            time_edges=[0, Inf],
        )

        empty_model = fit_hazard_model(
            historical[1:0, :];
            prior=prior,
            time_edges=[0, Inf],
        )
        posterior = hazard_posterior(empty_model, :td, "A", 1)
        @test posterior.weights ≈ td_mixture.weights
        @test posterior.source_seasons == td_mixture.source_seasons

        current = historical[1:1, :]
        before = hazard_posterior(empty_model, :td, "A", 1)
        update_hazard_model!(empty_model, current)
        after = hazard_posterior(empty_model, :td, "A", 1)
        expected_count = current.drive_result[1] == "Touchdown" ? 1.0 : 0.0
        @test all(
            after.components[index].shape ==
                before.components[index].shape + expected_count
            for index in eachindex(before.components)
        )
    end

    @testset "arbitrary historical season windows" begin
        long_historical = DataFrame(
            game_id=String[],
            fixed_drive=Int[],
            posteam=String[],
            defteam=String[],
            posteam_home=Bool[],
            defteam_home=Bool[],
            drive_result=String[],
            time_of_possession=Second[],
        )
        teams = ["A", "B", "C", "D"]
        for season in 2019:2023
            for (team_index, team) in enumerate(teams)
                for index in 1:40
                    posteam_home = iseven(index + season + team_index)
                    touchdown = mod(index + season + team_index, 5) == 0
                    push!(
                        long_historical,
                        (
                            "$(season)_$(team)_$(index)",
                            1,
                            team,
                            "DEFENSE",
                            posteam_home,
                            !posteam_home,
                            touchdown ? "Touchdown" : "Punt",
                            Second(60),
                        ),
                    )
                end
            end
        end

        long_prior = fit_empirical_bayes_prior(
            long_historical;
            time_edges=[0, Inf],
            max_seasons=5,
            current_season=2024,
        )
        @test long_prior.historical_seasons == collect(2019:2023)
        long_mixture = long_prior.td_team_mixtures["A"][1]
        @test length(long_mixture.components) == 6
        @test long_mixture.source_seasons == [2019, 2020, 2021, 2022, 2023, 2024]

        recent_prior = fit_empirical_bayes_prior(
            long_historical;
            time_edges=[0, Inf],
            max_seasons=2,
            current_season=2024,
        )
        @test recent_prior.historical_seasons == [2022, 2023]
    end

    @testset "empirical-Bayes home multipliers" begin
        Random.seed!(11)
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
        for season in 2021:2023
            for i in 1:1200
                posteam_home = iseven(i)
                td_rate = posteam_home ? 0.012 : 0.006
                defensive_rate = posteam_home ? 0.008 : 0.016
                td_time = randexp() / td_rate
                defensive_time = randexp() / defensive_rate
                touchdown = td_time < defensive_time
                duration = max(1, ceil(Int, min(td_time, defensive_time)))
                push!(
                    rows,
                    (
                        "$(season)_$(i)",
                        1,
                        "OFFENSE",
                        "DEFENSE",
                        posteam_home,
                        !posteam_home,
                        touchdown ? "Touchdown" : "Punt",
                        Second(duration),
                        touchdown ? 7.0 : 0.0,
                    ),
                )
            end
        end

        prior = fit_empirical_bayes_prior(
            rows;
            time_edges=[0, Inf],
            current_season=2024,
        )
        @test prior.td_home_multiplier ≈ 2.0 rtol=0.4
        @test prior.defensive_home_multiplier ≈ 2.0 rtol=0.4
        @test home_multiplier(prior, :td) == prior.td_home_multiplier
        @test home_multiplier(prior, :defensive) == prior.defensive_home_multiplier

        model = fit_hazard_model(rows[1:20, :]; prior=prior, time_edges=[0, Inf])
        td_away = hazard_rate(model, :td, "OFFENSE", 1; home=false)
        td_home = hazard_rate(model, :td, "OFFENSE", 1; home=true)
        @test td_home / td_away ≈ prior.td_home_multiplier
        defensive_away = hazard_rate(model, :defensive, "DEFENSE", 1; home=false)
        defensive_home = hazard_rate(model, :defensive, "DEFENSE", 1; home=true)
        @test defensive_home / defensive_away ≈ prior.defensive_home_multiplier

        td_multiplier = model.prior.td_home_multiplier
        defensive_multiplier = model.prior.defensive_home_multiplier
        update_hazard_model!(model, rows[21:40, :])
        @test model.prior.td_home_multiplier == td_multiplier
        @test model.prior.defensive_home_multiplier == defensive_multiplier
    end

    @testset "fit_score_marks" begin
        drives = DataFrame(
            posteam_home=[true, false, true, false, true],
            drive_result=["Touchdown", "Touchdown", "Punt", "Field goal", "End of half"],
            home_spread_change=[7.0, -7.0, 0.0, 3.0, 0.0],
        )
        marks = fit_score_marks(drives)
        @test marks.mean_td == 7.0
        @test marks.var_td == 0.0
        @test marks.mean_defensive == -1.5
        @test marks.var_defensive == 2.25
    end

    @testset "time-binned score marks" begin
        drives = DataFrame(
            posteam_home=[true, true, false, false],
            drive_result=["Touchdown", "Touchdown", "Punt", "Punt"],
            time_of_possession=[Second(30), Second(150), Second(30), Second(150)],
            home_spread_change=[7.0, 3.0, 0.0, -2.0],
        )
        marks = fit_score_marks(drives; time_edges=[0, 120, Inf])
        @test marks.mean_td_by_bin == [7.0, 3.0]
        @test marks.var_td_by_bin == [0.0, 0.0]
        @test marks.mean_defensive_by_bin == [0.0, 2.0]
        @test marks.var_defensive_by_bin == [0.0, 0.0]
    end

    @testset "matchup log-hazard theta" begin
        drives = vcat(_make_drives(), _make_drives(), _make_drives())
        model = fit_hazard_model(drives; time_edges=[0, 120, 240, Inf])
        theta = hazard_theta(model, "HOME", "AWAY")

        @test length(theta.log_mean) == 12
        @test size(theta.covariance) == (12, 12)
        @test theta.labels[1:3] == [:home_td_1, :home_td_2, :home_td_3]
        @test theta.labels[4:6] == [
            :away_defensive_1, :away_defensive_2, :away_defensive_3,
        ]
        @test theta.labels[7:9] == [:away_td_1, :away_td_2, :away_td_3]
        @test theta.labels[10:12] == [
            :home_defensive_1, :home_defensive_2, :home_defensive_3,
        ]

        posterior = hazard_posterior(model, :td, "HOME", 1; home=true)
        posterior_log_mean, posterior_log_variance =
            SurvivorModel._gamma_mixture_log_moments(posterior)
        @test theta.log_mean[1] ≈
            posterior_log_mean
        @test theta.covariance[1, 1] ≈ posterior_log_variance
        @test all(
            theta.covariance[i, i] > 0
            for i in axes(theta.covariance, 1)
        )
        @test all(
            theta.covariance[i, j] == 0
            for i in axes(theta.covariance, 1), j in axes(theta.covariance, 2)
            if i != j
        )
    end

    @testset "posterior win probability" begin
        drives = vcat(_make_drives(), _make_drives(), _make_drives())
        model = fit_hazard_model(drives; time_edges=[0, 120, 240, Inf])
        marks = fit_score_marks(drives)
        probability = expected_game_win_probability(
            model,
            marks,
            "HOME",
            "AWAY";
            horizon=60.0,
        )

        @test isfinite(probability)
        @test 0 < probability < 1

        empty_model = fit_hazard_model(
            drives[1:0, :];
            time_edges=[0, Inf],
        )
        symmetric = expected_game_win_probability(
            empty_model,
            ScoreMarks(7.0, 0.0, 0.0, 0.0),
            "HOME",
            "AWAY";
            horizon=60.0,
        )
        @test symmetric ≈ 0.5 atol=1e-10
    end

    @testset "second-order win probability versus posterior simulation" begin
        drives = vcat([_make_drives() for _ in 1:30]...)
        model = fit_hazard_model(drives; time_edges=[0, Inf])
        marks = fit_score_marks(drives)
        probability = expected_game_win_probability(
            model,
            marks,
            "HOME",
            "AWAY";
            horizon=60.0,
        )
        posteriors, _, _ =
            SurvivorModel._matchup_theta_posteriors(model, "HOME", "AWAY")
        n_samples = 4000
        samples = [
            [
                begin
                    component = posterior.components[
                        rand(Categorical(posterior.weights))
                    ]
                    rand(Gamma(component.shape, 1 / component.rate))
                end
                for _ in 1:n_samples
            ]
            for posterior in posteriors
        ]
        win_samples = zeros(n_samples)
        for i in 1:n_samples
            theta_sample = [log(samples[j][i]) for j in eachindex(samples)]
            sample_metrics = SurvivorModel._game_metrics_from_theta(
                theta_sample,
                model.time_edges,
                marks;
                horizon=60.0,
            )
            win_samples[i] = sample_metrics.win_probability
        end

        @test mean(win_samples) ≈ probability atol=0.03
    end

    @testset "two-outcome synthetic race" begin
        lambda_td, lambda_defensive = 0.01, 0.015
        marks = ScoreMarks(7.0, 0.0, 0.0, 0.0)
        moments = SurvivorModel._drive_moments_from_hazards(
            [0.0, Inf],
            marks,
            [lambda_td],
            [lambda_defensive],
        )
        @test moments.p_td ≈ lambda_td / (lambda_td + lambda_defensive) atol=0.02
        @test moments.p_defensive ≈ lambda_defensive /
            (lambda_td + lambda_defensive) atol=0.02
        @test moments.mean_T ≈ 1 / (lambda_td + lambda_defensive) rtol=0.05
        @test abs(moments.cov_TS) <= 1.0e-12
    end
end
