using SurvivorModel
using DataFrames
using Dates
using Distributions
using Random
using Statistics
using Test

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
    end

    @testset "pooled reset moment identity" begin
        moments = (
            away_mean=1.0,
            home_mean=2.0,
            away_factorial=1.25,
            home_factorial=5.0,
            same_season_product=2.5,
            sequential_away_away=1.1,
            sequential_away_home=2.2,
            sequential_home_away=2.2,
            sequential_home_home=4.4,
            n_away_mean=1,
            n_home_mean=1,
            n_away_factorial=1,
            n_home_factorial=1,
            n_same_season_product=1,
            n_sequential_away_away=1,
            n_sequential_away_home=1,
            n_sequential_home_away=1,
            n_sequential_home_home=1,
        )
        means, variances, fitted_home_multiplier, fitted_persistence =
            SurvivorModel._reset_moment_parameters(
                [(time_bin=1, moments=moments)],
            )

        @test means ≈ [1.0]
        @test variances ≈ [0.25]
        @test fitted_home_multiplier ≈ 2.0
        @test fitted_persistence ≈ 0.4

        @test SurvivorModel._shared_moment_count((10, 10, 40)) == 10.0
    end

    @testset "season-centered cross moments" begin
        base_moments = (
            away_mean=1.5,
            home_mean=1.5,
            away_factorial=2.5,
            home_factorial=2.5,
            same_season_product=2.5,
            sequential_away_away=2.1,
            sequential_away_home=2.1,
            sequential_home_away=2.1,
            sequential_home_home=2.1,
            n_away_mean=1,
            n_home_mean=1,
            n_away_factorial=1,
            n_home_factorial=1,
            n_same_season_product=1,
            n_sequential_away_away=1,
            n_sequential_away_home=1,
            n_sequential_home_away=1,
            n_sequential_home_home=1,
        )
        season_moments = [
            (
                season=2021,
                moments=merge(
                    base_moments,
                    (away_mean=1.0, home_mean=1.0),
                ),
            ),
            (
                season=2022,
                moments=merge(
                    base_moments,
                    (away_mean=2.0, home_mean=2.0),
                ),
            ),
        ]
        transition_moments = [
            (
                previous_season=2021,
                current_season=2022,
                moments=base_moments,
            ),
        ]
        _, _, _, fitted_persistence = SurvivorModel._reset_moment_parameters(
            [(
                time_bin=1,
                moments=base_moments,
                season_moments=season_moments,
                transition_moments=transition_moments,
            )],
        )

        @test fitted_persistence ≈ 0.4

        finite_sample_moments = merge(
            base_moments,
            (
                sequential_away_away=2.09,
                sequential_away_home=2.09,
                sequential_home_away=2.09,
                sequential_home_home=2.09,
                n_sequential_away_away=10,
                n_sequential_away_home=10,
                n_sequential_home_away=10,
                n_sequential_home_home=10,
            ),
        )
        _, _, _, corrected_persistence =
            SurvivorModel._reset_moment_parameters(
                [(
                    time_bin=1,
                    moments=base_moments,
                    season_moments=season_moments,
                    transition_moments=[(
                        previous_season=2021,
                        current_season=2022,
                        moments=finite_sample_moments,
                    )],
                )],
            )
        @test corrected_persistence ≈ 0.4
    end

    @testset "synthetic reset moment recovery" begin
        Random.seed!(29)
        seasons = 2021:2023
        teams = ["T$(index)" for index in 1:120]
        byseason = Dict{Int,SurvivorModel.HazardSufficientStats}()
        mean_rate = 0.01
        shape = 4.0
        rate = shape / mean_rate
        persistence = 0.7
        home_multiplier = 1.5
        previous_rates = Dict{String,Float64}()

        for season in seasons
            stats = SurvivorModel.HazardSufficientStats()
            for team in teams
                latent_rate = if haskey(previous_rates, team) &&
                    rand() < persistence
                    previous_rates[team]
                else
                    rand(Gamma(shape, 1 / rate))
                end
                previous_rates[team] = latent_rate

                key = (team, 1)
                exposure = 10_000.0
                away_count = rand(Poisson(latent_rate * exposure))
                home_count = rand(
                    Poisson(latent_rate * home_multiplier * exposure),
                )
                stats.td.away_exposure[key] = exposure
                stats.td.home_exposure[key] = exposure
                stats.td.away_counts[key] = away_count
                stats.td.home_counts[key] = home_count
                stats.td.exposure[key] = 2 * exposure
                stats.td.counts[key] = away_count + home_count
            end
            byseason[season] = stats
        end

        fitted, fitted_home_multiplier, fitted_persistence =
            SurvivorModel._fit_reset_outcome_parameters(
                byseason,
                collect(seasons),
                [0.0, Inf],
                :td,
            )
        fitted_mean = fitted[1].shape / fitted[1].rate
        @test fitted_mean ≈ mean_rate rtol=0.2
        @test fitted[1].shape ≈ shape rtol=0.3
        @test fitted_home_multiplier ≈ home_multiplier rtol=0.15
        @test fitted_persistence ≈ persistence rtol=0.2
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

        marks = fit_score_marks(rows)
        home_moments = drive_moments(
            model,
            marks,
            "OFFENSE",
            "DEFENSE";
            posteam_home=true,
        )
        away_moments = drive_moments(
            model,
            marks,
            "OFFENSE",
            "DEFENSE";
            posteam_home=false,
        )
        @test home_moments.p_td > away_moments.p_td

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

    @testset "two-outcome drive moments" begin
        drives = vcat(_make_drives(), _make_drives(), _make_drives())
        model = fit_hazard_model(drives; time_edges=[0, 120, 240, Inf])
        marks = fit_score_marks(drives)
        moments = drive_moments(
            model,
            marks,
            "HOME",
            "AWAY";
            posteam_home=true,
        )

        @test 0 < moments.p_td < 1
        @test 0 < moments.p_defensive < 1
        @test moments.p_td + moments.p_defensive ≈ 1.0
        @test moments.mean_T > 0
        @test moments.var_T > 0
        @test moments.var_S >= 0
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

    @testset "uncertainty-aware game metrics" begin
        drives = vcat(_make_drives(), _make_drives(), _make_drives())
        model = fit_hazard_model(drives; time_edges=[0, 120, 240, Inf])
        marks = fit_score_marks(drives)
        metrics = expected_game_metrics(
            model,
            marks,
            "HOME",
            "AWAY";
            horizon=60.0,
        )

        @test metrics isa ExpectedGameMetrics
        @test isfinite(metrics.expected_spread)
        @test 0 < metrics.expected_win_probability < 1
        @test metrics.predictive_spread_variance > 0

        empty_model = fit_hazard_model(
            drives[1:0, :];
            time_edges=[0, Inf],
        )
        symmetric = expected_game_metrics(
            empty_model,
            ScoreMarks(7.0, 0.0, 0.0, 0.0),
            "HOME",
            "AWAY";
            horizon=60.0,
        )
        @test symmetric.expected_spread ≈ 0.0 atol=1e-10
        @test symmetric.expected_win_probability ≈ 0.5 atol=1e-10
    end

    @testset "second-order metrics versus posterior simulation" begin
        drives = vcat([_make_drives() for _ in 1:30]...)
        model = fit_hazard_model(drives; time_edges=[0, Inf])
        marks = fit_score_marks(drives)
        metrics = expected_game_metrics(
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
        spread_samples = zeros(n_samples)
        win_samples = zeros(n_samples)
        for i in 1:n_samples
            theta_sample = [log(samples[j][i]) for j in eachindex(samples)]
            sample_metrics = SurvivorModel._game_metrics_from_theta(
                theta_sample,
                model.time_edges,
                marks;
                horizon=60.0,
            )
            spread_samples[i] = sample_metrics.mean_spread
            win_samples[i] = sample_metrics.win_probability
        end

        @test mean(spread_samples) ≈ metrics.expected_spread atol=0.05
        @test mean(win_samples) ≈ metrics.expected_win_probability atol=0.03
    end

    @testset "two-outcome synthetic race" begin
        Random.seed!(1234)
        lambda_td, lambda_defensive = 0.01, 0.015
        n = 50_000
        times = Vector{Float64}(undef, n)
        results = Vector{String}(undef, n)
        for i in 1:n
            td_time = randexp() / lambda_td
            defensive_time = randexp() / lambda_defensive
            if td_time < defensive_time
                times[i], results[i] = td_time, "Touchdown"
            else
                times[i], results[i] = defensive_time, "Punt"
            end
        end
        drives = DataFrame(
            game_id=["2024_$i" for i in 1:n],
            fixed_drive=ones(Int, n),
            posteam=fill("OFFENSE", n),
            defteam=fill("DEFENSE", n),
            posteam_home=fill(true, n),
            defteam_home=fill(false, n),
            drive_result=results,
            time_of_possession=Second.(round.(Int, times)),
            drive_start_yards_to_goal=fill(50, n),
            yards_gained=zeros(n),
            home_spread_change=[r == "Touchdown" ? 7.0 : 0.0 for r in results],
        )

        model = fit_hazard_model(drives; time_edges=[0, Inf])
        marks = fit_score_marks(drives)
        moments = drive_moments(model, marks, "OFFENSE", "DEFENSE")
        @test moments.p_td ≈ lambda_td / (lambda_td + lambda_defensive) atol=0.02
        @test moments.p_defensive ≈ lambda_defensive /
            (lambda_td + lambda_defensive) atol=0.02
        @test moments.mean_T ≈ 1 / (lambda_td + lambda_defensive) rtol=0.05
        @test abs(moments.cov_TS) < 0.05 * sqrt(moments.var_T * moments.var_S)
    end

    @testset "game_spread_distribution" begin
        home = DriveMoments(0.3, 0.7, 30.0, 100.0, 2.0, 5.0, 1.0)
        away = DriveMoments(0.3, 0.7, 30.0, 100.0, 2.0, 5.0, 1.0)
        dist = game_spread_distribution(home, away; horizon=60.0)
        @test dist isa Normal
        @test mean(dist) ≈ 0.0 atol=1e-9
        @test var(dist) > 0
    end
end
