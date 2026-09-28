using SurvivorModel
using DataFrames
using Dates
using Distributions
using Random
using Test
import SurvivorModel: ScoreMarks, fit_score_marks, hazard_theta

function _make_drives()
    n = 5
    return DataFrame(
        game_id=["2023_01_$i" for i in 1:n],
        fixed_drive=ones(Int, n),
        posteam=[isodd(i) ? "HOME" : "AWAY" for i in 1:n],
        defteam=[isodd(i) ? "AWAY" : "HOME" for i in 1:n],
        posteam_home=[isodd(i) for i in 1:n],
        defteam_home=[iseven(i) for i in 1:n],
        drive_result=["Touchdown", "Punt", "Field goal", "End of half", "Punt"],
        time_of_possession=[Second(120), Second(180), Second(240), Second(60), Second(150)],
        home_spread_change=[7.0, 0.0, 3.0, 0.0, 0.0],
    )
end

function _test_prior(;
    td_shape=2.0,
    defensive_shape=1.5,
    td_hyperparameter=GammaParams(40.0, 40.0 / 0.03),
    defensive_hyperparameter=GammaParams(40.0, 40.0 / 0.10),
    td_home_multiplier=1.0,
    defensive_home_multiplier=1.0,
)
    return HazardPrior(
        td_shape,
        defensive_shape,
        td_hyperparameter,
        defensive_hyperparameter,
        Dict{String,GammaMixture}(),
        Dict{String,GammaMixture}(),
        td_home_multiplier,
        defensive_home_multiplier,
        0.8,
        0.8,
        Int[],
        nothing,
        nothing,
    )
end

function _synthetic_weibull_drives()
    rng = MersenneTwister(42)
    teams = ["A", "B", "C", "D"]
    rows = NamedTuple[]
    for season in 2021:2024, index in 1:180
        offense = teams[mod1(index + season, length(teams))]
        defense = teams[mod1(index + season + 1, length(teams))]
        offense_home = iseven(index)
        defense_home = !offense_home
        offense_rate = 0.025 *
            (1 + 0.1 * (findfirst(==(offense), teams) - 1))
        defense_rate = 0.10 *
            (1 + 0.08 * (findfirst(==(defense), teams) - 1))
        td_shape = 2.1
        defensive_shape = 1.4
        td_time = rand(
            rng,
            Weibull(
                td_shape,
                (offense_rate * (offense_home ? 1.08 : 1.0))^(-1 / td_shape),
            ),
        )
        defensive_time = rand(
            rng,
            Weibull(
                defensive_shape,
                (defense_rate * (defense_home ? 1.05 : 1.0))^(-1 / defensive_shape),
            ),
        )
        duration = min(td_time, defensive_time)
        event = td_time < defensive_time ? "Touchdown" : "Interception"
        duration > 14.0 && (duration = 14.0; event = "End of half")
        push!(
            rows,
            (
                season=season,
                game_id="$(season)_$(index)",
                fixed_drive=index,
                drive_result=event,
                time_of_possession=Second(max(1, round(Int, duration * 60))),
                posteam=offense,
                defteam=defense,
                posteam_home=offense_home,
                defteam_home=defense_home,
                home_spread_change=0.0,
            ),
        )
    end
    return DataFrame(rows)
end

@testset "Weibull competing-risk model" begin
    @testset "event classification and continuous drive data" begin
        @test SurvivorModel._classify_event("Touchdown") === :td
        @test SurvivorModel._classify_event("End of half") === :censored
        for result in [
            "Field goal",
            "Punt",
            "Turnover",
            "Turnover on downs",
            "Missed field goal",
            "Safety",
            "Opp touchdown",
        ]
            @test SurvivorModel._classify_event(result) === :defensive
        end

        data = build_drive_data(_make_drives())
        @test data.season == fill(2023, 5)
        @test data.duration ≈ [2.0, 3.0, 4.0, 1.0, 2.5]
        @test data.event == [:td, :defensive, :defensive, :censored, :defensive]
        @test data.posteam_home == [true, false, true, false, true]

        zero_resolution = _make_drives()
        zero_resolution.time_of_possession[1] = Second(0)
        @test build_drive_data(zero_resolution).duration[1] == 1.0 / 60.0

        negative_duration = _make_drives()
        negative_duration.time_of_possession[1] = Second(-1)
        @test_throws ArgumentError build_drive_data(negative_duration)
    end

    @testset "Gamma mixture updates and reset transitions" begin
        mixture = GammaMixture(
            [1.0, 3.0],
            [GammaParams(2.0, 4.0), GammaParams(6.0, 9.0)];
            source_seasons=[2022, 2023],
        )
        @test mixture.weights ≈ [0.25, 0.75]
        @test SurvivorModel._gamma_mixture_mean(mixture) ≈
            0.25 * (2.0 / 4.0) + 0.75 * (6.0 / 9.0)
        @test SurvivorModel._gamma_mixture_variance(mixture) > 0.0
        mean_log, variance_log = SurvivorModel._gamma_mixture_log_moments(mixture)
        @test isfinite(mean_log)
        @test variance_log > 0.0

        home_mixture = SurvivorModel._gamma_mixture_home_adjusted(mixture, 2.0)
        @test home_mixture.source_seasons == mixture.source_seasons
        @test all(
            home.rate == away.rate / 2.0
            for (home, away) in zip(home_mixture.components, mixture.components)
        )
        updated = SurvivorModel._update_gamma_mixture(mixture, 2, 10.0)
        @test all(
            updated.components[index].shape ==
                mixture.components[index].shape + 2.0 &&
            updated.components[index].rate ==
                mixture.components[index].rate + 10.0
            for index in eachindex(mixture.components)
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

        expected_marginal = 2.0 * log(4.0) + log(2.0) - 3.0 * log(7.0)
        @test SurvivorModel._log_gamma_process_marginal(
            GammaParams(2.0, 4.0),
            1,
            3.0,
        ) ≈ expected_marginal
        @test SurvivorModel._log_gamma_process_marginal(
            GammaParams(2.0, 4.0),
            0,
            0.0,
        ) == 0.0
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

    @testset "conjugate cumulative-hazard updates" begin
        prior = _test_prior(
            td_shape=2.0,
            defensive_shape=1.5,
            td_hyperparameter=GammaParams(2.0, 4.0),
            defensive_hyperparameter=GammaParams(3.0, 7.0),
            td_home_multiplier=1.5,
            defensive_home_multiplier=1.25,
        )
        drives = _make_drives()
        model = fit_hazard_model(drives[1:0, :]; prior=prior)
        update_hazard_model!(model, drives[1:1, :])

        td_posterior = hazard_posterior(model, :td, "HOME")
        @test only(td_posterior.components).shape == 3.0
        @test only(td_posterior.components).rate == 10.0
        home_td_posterior = hazard_posterior(model, :td, "HOME"; home=true)
        @test only(home_td_posterior.components).rate ≈ 10.0 / 1.5
        @test hazard_rate(model, :td, "HOME", 1.0; home=true) ≈ 0.9
        @test hazard_rate(model, :td, "HOME", 0.0; home=true) == 0.0
        @test home_multiplier(prior, :td) == 1.5

        defensive_posterior = hazard_posterior(model, :defensive, "AWAY")
        @test only(defensive_posterior.components).shape == 3.0
        @test only(defensive_posterior.components).rate ≈ 7.0 + 2.0^1.5

        censored_model = fit_hazard_model(drives[1:0, :]; prior=prior)
        update_hazard_model!(censored_model, drives[4:4, :])
        td_after_censor = hazard_posterior(censored_model, :td, "AWAY")
        defensive_after_censor =
            hazard_posterior(censored_model, :defensive, "HOME")
        @test only(td_after_censor.components).shape == 2.0
        @test only(td_after_censor.components).rate == 5.0
        @test only(defensive_after_censor.components).shape == 3.0
        @test only(defensive_after_censor.components).rate ≈ 8.25
    end

    @testset "empirical-Bayes Weibull shapes and reset mixtures" begin
        historical = _synthetic_weibull_drives()
        prior = fit_empirical_bayes_prior(
            historical;
            max_seasons=4,
            current_season=2025,
        )
        @test prior.historical_seasons == collect(2021:2024)
        @test 1.5 < weibull_shape(prior, :td) < 2.7
        @test 0.9 < weibull_shape(prior, :defensive) < 1.9
        @test likelihood_fit_diagnostics(prior, :td).converged
        @test likelihood_fit_diagnostics(prior, :defensive).converged
        @test isfinite(likelihood_fit_diagnostics(prior, :td).log_likelihood)
        @test isfinite(
            likelihood_fit_diagnostics(prior, :defensive).log_likelihood,
        )
        @test all(
            parameter.shape > 0.0 && parameter.rate > 0.0
            for parameter in (
                prior.td_hyperparameters,
                prior.defensive_hyperparameters,
            )
        )
        @test 0.0 <= hazard_persistence(prior, :td) <= 1.0
        @test 0.0 <= hazard_persistence(prior, :defensive) <= 1.0
        @test home_multiplier(prior, :td) > 0.0
        @test home_multiplier(prior, :defensive) > 0.0
        @test length(prior.td_team_mixtures["A"].components) == 5
        @test prior.td_team_mixtures["A"].source_seasons ==
            [2021, 2022, 2023, 2024, 2025]

        @test_throws ArgumentError fit_empirical_bayes_prior(
            historical;
            max_seasons=0,
        )
        @test_throws ArgumentError fit_empirical_bayes_prior(
            historical;
            current_season=2024,
        )
    end

    @testset "cause-specific score marks" begin
        drives = DataFrame(
            posteam_home=[true, false, true, false, true],
            drive_result=[
                "Touchdown",
                "Touchdown",
                "Punt",
                "Field goal",
                "End of half",
            ],
            home_spread_change=[7.0, -7.0, 0.0, 3.0, 0.0],
        )
        marks = fit_score_marks(drives)
        @test marks.mean_td == 7.0
        @test marks.var_td == 0.0
        @test marks.mean_defensive == -1.5
        @test marks.var_defensive == 2.25
    end

    @testset "analytic equal-shape Weibull race" begin
        td_rate, defensive_rate, shape = 0.2, 0.3, 1.0
        integrals = SurvivorModel._weibull_race_integrals(
            td_rate,
            shape,
            defensive_rate,
            shape,
        )
        total_rate = td_rate + defensive_rate
        @test integrals.values[1] ≈ td_rate / total_rate atol=1e-10
        @test integrals.values[4] ≈ defensive_rate / total_rate atol=1e-10
        @test integrals.values[2] ≈ td_rate / total_rate / total_rate atol=1e-10
        @test integrals.values[5] ≈
            defensive_rate / total_rate / total_rate atol=1e-10
        @test integrals.values[3] ≈ 2td_rate / total_rate^3 atol=1e-10
        @test integrals.values[6] ≈ 2defensive_rate / total_rate^3 atol=1e-10
        @test integrals.values[1] + integrals.values[4] ≈ 1.0 atol=1e-10
    end

    @testset "unequal-shape integral derivatives and mixed moments" begin
        td_rate, defensive_rate = 0.18, 0.42
        td_shape, defensive_shape = 2.2, 1.4
        integrals = SurvivorModel._weibull_race_integrals(
            td_rate,
            td_shape,
            defensive_rate,
            defensive_shape,
        )
        @test integrals.values[1] + integrals.values[4] ≈ 1.0 atol=1e-8

        step = 1.0e-4
        plus = SurvivorModel._weibull_race_integrals(
            td_rate * exp(step),
            td_shape,
            defensive_rate,
            defensive_shape,
        )
        minus = SurvivorModel._weibull_race_integrals(
            td_rate * exp(-step),
            td_shape,
            defensive_rate,
            defensive_shape,
        )
        @test integrals.jacobian[1, 1] ≈
            (plus.values[1] - minus.values[1]) / (2step) atol=1e-6
        @test integrals.hessians[1, 1, 1] ≈
            (plus.jacobian[1, 1] - minus.jacobian[1, 1]) / (2step) atol=1e-5
        @test integrals.hessians[1, 1, 2] ≈
            (plus.jacobian[1, 2] - minus.jacobian[1, 2]) / (2step) atol=1e-5

        marks = ScoreMarks(7.0, 2.0, -1.0, 3.0)
        moments = SurvivorModel._drive_moments_from_weibull(
            td_rate,
            td_shape,
            defensive_rate,
            defensive_shape,
            marks,
        )
        expected_mean_time = integrals.values[2] + integrals.values[5]
        expected_mean_score =
            integrals.values[1] * marks.mean_td +
            integrals.values[4] * marks.mean_defensive
        expected_time_score =
            integrals.values[2] * marks.mean_td +
            integrals.values[5] * marks.mean_defensive
        @test moments.mean_T ≈ expected_mean_time
        @test moments.mean_S ≈ expected_mean_score
        @test moments.cov_TS ≈
            expected_time_score - expected_mean_time * expected_mean_score
        @test abs(moments.cov_TS) > 1e-4
    end

    @testset "game probability gradients and Hessians" begin
        prior = _test_prior(td_shape=2.2, defensive_shape=1.4)
        marks = ScoreMarks(6.5, 4.0, -0.5, 2.0)
        theta = log.([0.03, 0.09, 0.028, 0.11])
        analytic = SurvivorModel._game_probability_derivatives(
            theta,
            prior,
            marks;
            horizon=3600.0,
        )
        probability = value -> SurvivorModel._game_metrics_from_theta(
            value,
            prior,
            marks;
            horizon=3600.0,
        ).win_probability
        @test analytic.probability ≈ probability(theta) atol=1e-10

        step = 1.0e-4
        finite_gradient = zeros(4)
        finite_hessian = zeros(4, 4)
        for parameter in 1:4
            upper = copy(theta)
            lower = copy(theta)
            upper[parameter] += step
            lower[parameter] -= step
            finite_gradient[parameter] =
                (probability(upper) - probability(lower)) / (2step)
            upper_derivatives = SurvivorModel._game_probability_derivatives(
                upper,
                prior,
                marks;
                horizon=3600.0,
            )
            lower_derivatives = SurvivorModel._game_probability_derivatives(
                lower,
                prior,
                marks;
                horizon=3600.0,
            )
            finite_hessian[:, parameter] =
                (upper_derivatives.gradient - lower_derivatives.gradient) /
                (2step)
        end
        @test analytic.gradient ≈ finite_gradient atol=2e-5 rtol=2e-4
        @test analytic.hessian ≈ finite_hessian atol=5e-4 rtol=2e-3
    end

    @testset "matchup posterior and game probability" begin
        prior = _test_prior()
        model = fit_hazard_model(_make_drives()[1:0, :]; prior=prior)
        theta = hazard_theta(model, "HOME", "AWAY")
        @test length(theta.log_mean) == 4
        @test size(theta.covariance) == (4, 4)
        @test theta.labels == [
            :home_td,
            :away_defensive,
            :away_td,
            :home_defensive,
        ]
        @test all(theta.covariance[index, index] > 0.0 for index in 1:4)
        @test all(
            theta.covariance[first, second] == 0.0
            for first in 1:4, second in 1:4
            if first != second
        )

        marks = ScoreMarks(7.0, 2.0, -1.0, 3.0)
        probability = expected_game_win_probability(
            model,
            marks,
            "HOME",
            "AWAY";
            horizon=3600.0,
        )
        @test isfinite(probability)
        @test 0.0 <= probability <= 1.0

        symmetric_prior = _test_prior(td_home_multiplier=1.0, defensive_home_multiplier=1.0)
        symmetric_model = fit_hazard_model(_make_drives()[1:0, :]; prior=symmetric_prior)
        symmetric_marks = ScoreMarks(1.0, 2.0, -1.0, 3.0)
        symmetric = expected_game_win_probability(
            symmetric_model,
            symmetric_marks,
            "HOME",
            "AWAY";
            horizon=3600.0,
        )
        @test symmetric ≈ 0.5 atol=1e-9
    end
end
