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
        @test td_posterior.shape > 1.0
        @test td_posterior.rate > 100.0
        @test hazard_rate(model, :td, "HOME", 1) ==
            td_posterior.shape / td_posterior.rate

        previous_shape = td_posterior.shape
        previous_rate = td_posterior.rate
        update_hazard_model!(model, drives[1:1, :])
        updated = hazard_posterior(model, :td, "HOME", 1)
        @test updated.shape == previous_shape + 1.0
        @test updated.rate == previous_rate + 120.0
    end

    @testset "empirical-Bayes prior and recency calibration" begin
        historical = vcat(
            DataFrame(
                game_id=["2018_1", "2019_1", "2020_1"],
                fixed_drive=ones(Int, 3),
                posteam=fill("A", 3),
                defteam=fill("B", 3),
                drive_result=fill("Punt", 3),
                time_of_possession=Second.([60, 60, 60]),
                posteam_home=fill(true, 3),
                defteam_home=fill(false, 3),
                home_spread_change=zeros(3),
            ),
            DataFrame(
                game_id=["2021_1", "2021_2", "2022_1", "2022_2", "2023_1", "2023_2"],
                fixed_drive=ones(Int, 6),
                posteam=fill("A", 6),
                defteam=fill("B", 6),
                drive_result=["Touchdown", "Punt", "Touchdown", "Punt", "Touchdown", "Punt"],
                time_of_possession=Second.([60, 120, 60, 120, 60, 120]),
                posteam_home=fill(true, 6),
                defteam_home=fill(false, 6),
                home_spread_change=[7.0, 0.0, 7.0, 0.0, 7.0, 0.0],
            ),
            DataFrame(
                game_id=["2021_3", "2022_3", "2023_3"],
                fixed_drive=ones(Int, 3),
                posteam=fill("C", 3),
                defteam=fill("D", 3),
                drive_result=fill("Punt", 3),
                time_of_possession=Second.([180, 180, 180]),
                posteam_home=fill(true, 3),
                defteam_home=fill(false, 3),
                home_spread_change=zeros(3),
            ),
        )
        prior = fit_empirical_bayes_prior(
            historical;
            time_edges=[0, 120, 240, Inf],
            max_seasons=99,
            half_life_candidates=[0.5, 1.0, Inf],
            current_season=2024,
        )
        @test prior.historical_seasons == [2021, 2022, 2023]
        @test prior.recency_half_life == DEFAULT_RECENCY_HALF_LIFE
        @test SurvivorModel.MIN_RECENCY_HALF_LIFE <= prior.recency_half_life <=
            SurvivorModel.MAX_RECENCY_HALF_LIFE
        @test all(p -> p.shape > 0 && p.rate > 0, prior.td_hyperparameters)
        @test haskey(prior.td_team_priors, "A")
        @test haskey(prior.defensive_team_priors, "B")
        weights = recency_weights(prior)
        @test weights[2024] == 1.0
        @test weights[2023] ≈ exp(-1 / prior.recency_half_life)
        @test weights[2022] ≈ exp(-2 / prior.recency_half_life)
        @test weights[2021] ≈ exp(-3 / prior.recency_half_life)

        current = historical[4:4, :]
        model = fit_hazard_model(
            current;
            prior=prior,
            time_edges=[0, 120, 240, Inf],
        )
        @test hazard_posterior(model, :td, "A", 1).shape >
            prior.td_team_priors["A"][1].shape

        short_prior = fit_empirical_bayes_prior(
            historical[1:2, :];
            time_edges=[0, 120, 240, Inf],
            current_season=2023,
        )
        @test short_prior.recency_half_life == DEFAULT_RECENCY_HALF_LIFE
        @test_throws ArgumentError fit_empirical_bayes_prior(
            historical;
            time_edges=[0, 120, 240, Inf],
            max_seasons=0,
        )
        automatic_prior = fit_empirical_bayes_prior(
            historical;
            time_edges=[0, Inf],
            max_seasons=99,
            recency_half_life=nothing,
            current_season=2024,
        )
        @test SurvivorModel.MIN_RECENCY_HALF_LIFE <=
            automatic_prior.recency_half_life <=
            SurvivorModel.MAX_RECENCY_HALF_LIFE
        mle_prior = fit_empirical_bayes_prior(
            historical;
            time_edges=[0, Inf],
            max_seasons=99,
            recency_half_life=Inf,
            fit_strategy=:mle,
            current_season=2024,
        )
        @test all(p -> p.shape > 0 && p.rate > 0, mle_prior.td_hyperparameters)
        @test all(p -> p.shape > 0 && p.rate > 0, mle_prior.defensive_hyperparameters)
        @test_throws ArgumentError fit_empirical_bayes_prior(
            historical;
            fit_strategy=:invalid,
        )

        function _moment_drives(results_by_season)
            rows = DataFrame(
                game_id=String[],
                fixed_drive=Int[],
                posteam=String[],
                defteam=String[],
                drive_result=String[],
                time_of_possession=Second[],
                posteam_home=Bool[],
                defteam_home=Bool[],
            )
            teams = ["A", "B", "C", "D"]
            for (season, results) in zip(2021:2023, results_by_season)
                for (index, result) in enumerate(results)
                    push!(
                        rows,
                        (
                            "$(season)_$(index)",
                            1,
                            teams[index],
                            "DEFENSE",
                            result,
                            Second(60),
                            true,
                            false,
                        ),
                    )
                end
            end
            return rows
        end

        positive_data, positive_edges = build_exposure_data(
            _moment_drives([
                ["Touchdown", "Touchdown", "Punt", "Punt"],
                ["Touchdown", "Touchdown", "Punt", "Punt"],
                ["Touchdown", "Touchdown", "Punt", "Punt"],
            ]);
            time_edges=[0, Inf],
        )
        negative_data, negative_edges = build_exposure_data(
            _moment_drives([
                ["Touchdown", "Touchdown", "Punt", "Punt"],
                ["Punt", "Punt", "Touchdown", "Touchdown"],
                ["Touchdown", "Touchdown", "Punt", "Punt"],
            ]);
            time_edges=[0, Inf],
        )
        positive_half_life = SurvivorModel._calibrate_recency_half_life(
            SurvivorModel._season_stats(positive_data),
            [2021, 2022, 2023],
            positive_edges,
        )
        negative_half_life = SurvivorModel._calibrate_recency_half_life(
            SurvivorModel._season_stats(negative_data),
            [2021, 2022, 2023],
            negative_edges,
        )
        @test positive_half_life ≈ SurvivorModel.MAX_RECENCY_HALF_LIFE
        @test negative_half_life ≈ SurvivorModel.MIN_RECENCY_HALF_LIFE
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
            recency_half_life=Inf,
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
        @test theta.log_mean[1] ≈
            SurvivorModel.SpecialFunctions.digamma(posterior.shape) -
            log(posterior.rate)
        @test theta.covariance[1, 1] ≈
            SurvivorModel.SpecialFunctions.trigamma(posterior.shape)
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
            rand(Gamma(posterior.shape, 1 / posterior.rate), n_samples)
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
