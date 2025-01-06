using GeometricKalman, Test, LinearAlgebra, Distributions
using Manifolds
using RecursiveArrayTools

using GeometricKalman: gen_car_data, default_discretization, car_h, car_f, car_control

using GeometricKalman: make_kalman_parameter_fitting_objective

@testset "Parameter fitting" begin
    dt = 0.01
    vt = 5
    samples, controls, measurements = gen_car_data(;
        vt=vt,
        N=200,
        noise_f_distr=MvNormal([0.0, 0.0, 0.0], 1e3 * diagm([0.001, 1e-10, 1.0])),
        noise_h_distr=MvNormal([0.0, 0.0], diagm([0.001, 0.001])),
    )
    M = SpecialEuclidean(2)
    M_obs = Euclidean(2)
    p0 = ArrayPartition([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    P0 = diagm([0.1, 0.1, 0.1])
    Q = diagm([0.5, 0.5, 2])
    R = diagm([0.001, 0.001])
    car_f_adapted(p, q, noise, t::Real) = car_f(p, q, noise, t::Real; vt=vt)
    f_tilde = default_discretization(M, car_f_adapted; dt=dt)

    M_obj = Euclidean(2)
    M_fit = ProductManifold(M, SymmetricPositiveDefinite(3))
    ref_samples = [s.x[1] for s in samples]

    sp = WanMerweSigmaPoints(; α=1.0)
    params = [
        (
            "EKF",
            (;
                propagator=EKFPropagator(M, f_tilde, DefaultOrthonormalBasis()),
                updater=EKFUpdater(
                    M,
                    M_obs,
                    car_h,
                    DefaultOrthonormalBasis(),
                    DefaultOrthonormalBasis(),
                ),
            ),
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
                propagator=EKFPropagator(M, f_tilde, DefaultOrthonormalBasis()),
                updater=EKFUpdater(
                    M,
                    M_obs,
                    car_h,
                    DefaultOrthonormalBasis(),
                    DefaultOrthonormalBasis(),
                ),
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

        param = InitialConditionKFOParametrization(Q, R)
        pfo = make_kalman_parameter_fitting_objective(
            M,
            M_obs,
            M_obj,
            M_fit,
            f_tilde,
            car_h,
            filter_kwargs,
            param,
            kf -> kf.p_n.x[1],
            ref_samples,
            controls,
            measurements,
        )
        p_obj_0 = ArrayPartition(p0, P0)
        @test GeometricKalman.objective(pfo, p_obj_0) isa Real
        @test GeometricKalman.residuals(pfo, p_obj_0) isa Vector{<:Real}
        if name in ["EKF", "UKF"]
            @test GeometricKalman.jacobian(pfo, p_obj_0) isa Matrix{<:Real}
        end
    end
end
