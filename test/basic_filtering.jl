using GeometricKalman, Test, LinearAlgebra, Distributions
using Manifolds
using RecursiveArrayTools

using GeometricKalman: default_discretization, car_f, car_h, gen_car_data

using LieGroups

@testset "Basic filtering" begin
    dt = 0.01
    vt = 5
    times, samples, controls, measurements = gen_car_data(;
        vt=vt,
        N=200,
        noise_f_distr=MvNormal([0.0, 0.0, 0.0], 1e3 * diagm([1e-10, 1.0, 0.001])),
        noise_h_distr=MvNormal([0.0, 0.0], diagm([0.001, 0.001])),
    )
    M = SpecialEuclideanGroup(2)
    M_obs = Euclidean(2)
    p0 = ArrayPartition([1.0 0.0; 0.0 1.0], [0.0, 0.0])
    P0 = diagm([0.1, 0.1, 0.1])
    Q = diagm([0.5, 0.5, 2])
    R = diagm([0.001, 0.001])
    car_f_adapted(p, q, noise, t::Real) = car_f(p, q, noise, t; vt=vt)
    f_tilde = default_discretization(M, car_f_adapted; dt=dt)

    sp = WanMerweSigmaPoints(; α=1.0)
    params = [
        (
            "EKF",
            (; propagator=EKFPropagator(M, f_tilde), updater=EKFUpdater(M, M_obs, car_h)),
        ),
        (
            "UKF",
            (;
                propagator=UnscentedPropagator(; sigma_points=sp),
                updater=UnscentedUpdater(; sigma_points=sp),
            ),
        ),
        (
            "EKF adaptive α=0.99",
            (;
                propagator=EKFPropagator(M, f_tilde),
                updater=EKFUpdater(M, M_obs, car_h),
                measurement_covariance_adapter=CovarianceMatchingMeasurementCovarianceAdapter(
                    0.99,
                ),
                process_covariance_adapter=CovarianceMatchingProcessCovarianceAdapter(0.99),
            ),
        ),
    ]

    for (name, filter_kwargs) in params
        kf = discrete_kalman_filter_manifold(
            M,
            M_obs,
            p0,
            f_tilde,
            car_h,
            P0,
            copy(Q),
            copy(R);
            filter_kwargs...,
        )

        for i in eachindex(samples)
            @test is_point(M, kf.p_n)
            predict!(kf, controls[i])
            GeometricKalman.update!(kf, controls[i], measurements[i])
        end
    end
end
