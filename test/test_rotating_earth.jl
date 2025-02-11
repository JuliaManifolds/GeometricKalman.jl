using GeometricKalman, Test, LinearAlgebra, Distributions
using Manifolds
using Rotations
using RecursiveArrayTools
using LinearAlgebra

using GeometricKalman:
    default_discretization,
    EarthModel,
    earth_h,
    earth_control,
    RotEarthManifold,
    gen_data,
    RotEarthRetraction,
    RotEarthProcessNoiseDimensionality

function joint_rotation_matrix_ref(joint::AbstractVector)
    jy = atan(joint[3], joint[2])
    jz = acos(joint[1])
    return Rotations.RotXYZ(0.0, jy, jz)
end

@testset "Basic methods" begin
    @test GeometricKalman.joint_rotation_matrix([1.0, 0.0, 0.0]) ≈ I(3)
    ry = 0.2
    rz = 0.3
    joint_pos = GeometricKalman.angles_to_joint_position(ry, rz)
    @test joint_pos ≈ [0.955336489125606, 0.28962947762551555, 0.05871080169382652]
    @test GeometricKalman.joint_rotation_matrix(joint_pos) ≈ Rotations.RotXYZ(0.0, ry, rz)
end

@testset "Basic filtering" begin
    M = RotEarthManifold
    M_obs = GeometricKalman.RotEarthObsManifold

    M_dim = manifold_dimension(M)
    M_obs_dim = manifold_dimension(M_obs)

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

    P0 = zeros(M_dim, M_dim)
    view(P0, diagind(P0)) .= 1e-5
    view(P0, 1:3, 1:3) .= diagm([0.1, 0.1, 0.1])

    Q = zeros(RotEarthProcessNoiseDimensionality, RotEarthProcessNoiseDimensionality)
    view(Q, diagind(Q)) .= 0.1

    R = diagm(fill(0.001, M_obs_dim))
    dt = 0.01
    N = 100
    em = EarthModel()
    f_tilde = default_discretization(M, em; dt=dt)

    noise_f_distr = MvNormal(zeros(RotEarthProcessNoiseDimensionality), Q)
    noise_h_distr = MvNormal(zeros(M_obs_dim), R)

    samples, controls, measurements = gen_data(
        M,
        p0,
        em,
        earth_h,
        earth_control,
        noise_f_distr,
        noise_h_distr;
        N=N,
        dt=dt,
        retraction=RotEarthRetraction,
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
            process_noise_dimensionality=RotEarthProcessNoiseDimensionality,
            filter_kwargs...,
        )

        for i in eachindex(samples)
            @test is_point(M, kf.p_n)
            predict!(kf, controls[i])
            GeometricKalman.update!(kf, controls[i], measurements[i])
        end
    end
end
