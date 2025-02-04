using GeometricKalman, Test, LinearAlgebra, Distributions
using Manifolds
using RecursiveArrayTools

using GeometricKalman:
    default_discretization, earth_f, earth_h, earth_control, RotEarthManifold, gen_data

@testset "Basic filtering" begin
    M = RotEarthManifold

    # (position and orientation) × (joint orientation) × (velocity) × (angular velocity) × (acceleration) × (gyroscope bias A) × (gyroscope bias B) × (accelerometer bias A) × (accelerometer bias B)
    pos_orient_0 = ArrayPartition([0.0, 0.0, 0.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    joint_0 = [1.0, 0.0, 0.0]
    vel_0 = [0.0, 0.0, 0.0]
    ang_vel_0 = [0.0, 0.0, 0.0]
    acc_0 = [0.0, 0.0, 0.0]
    p0 = ArrayPartition(
        pos_orient_0,
        joint_0,
        vel_0,
        ang_vel_0,
        acc_0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    M_dim = manifold_dimension(M)
    P0 = zeros(M_dim, M_dim)
    view(P0, diagind(P0)) .= 1e-5
    view(P0, 1:3, 1:3) .= diagm([0.1, 0.1, 0.1])

    Q = sqrt.(P0)
    R = diagm([0.001, 0.001])
    dt = 0.01
    N = 100
    earth_f_adapted(p, q, noise, t::Real) = earth_f(p, q, noise, t)
    f_tilde = default_discretization(M, earth_f_adapted; dt=dt)

    samples, controls, measurements = gen_data(
        M,
        p0,
        earth_f,
        earth_h,
        earth_control,
        noise_f_distr,
        noise_h_distr;
        N=N,
        dt=dt,
    )

    sp = WanMerweSigmaPoints(; α=1.0)
    params = [
        (
            "EKF",
            (; propagator=EKFPropagator(M, f_tilde), updater=EKFUpdater(M, M_obs, earth_h)),
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
                updater=EKFUpdater(M, M_obs, earth_h),
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
            earth_h,
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
