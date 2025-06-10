# GeometricKalman.jl

Kalman filters on manifolds with an affine connection, unifying the Lie group and Riemannian approaches.

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamanifolds.github.io/GeometricKalman.jl/dev/)

arXiv preprint: https://arxiv.org/abs/2506.01086

Usage example:

```julia


# Basic setting (example: a car driving on a sphere)

using Manifolds
using RecursiveArrayTools
using LinearAlgebra

using Distributions

using GeometricKalman
using GeometricKalman: gen_car_sphere_data, car_sphere_f, car_sphere_h

M = TangentBundle(Manifolds.Sphere(2)) # state manifold
M_obs = Manifolds.Sphere(2) # observation manifold
retraction = Manifolds.FiberBundleProductRetraction()
inverse_retraction = Manifolds.FiberBundleInverseProductRetraction()

# generate data
dt = 0.01
vt = 5

times, samples, controls, measurements = gen_car_sphere_data(;
    vt = vt,
    N = 200,
    noise_f_distr = MvNormal(
        [0.0, 0.0, 0.0, 0.0],
        1e4 * diagm([1e-3, 1e-3, 1e-2, 1e-2]),
    ),
    noise_h_distr = MvNormal([0.0, 0.0], diagm([0.01, 0.01])),
    retraction = retraction,
)


# general initial conditions for filtering
p0 = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
P0 = diagm([0.1, 0.1, 0.1, 0.1])
Q = diagm([0.1, 0.1, 0.01, 0.01])
R = diagm([0.01, 0.01])

# system dynamics
car_f_adapted(p, q, noise, t::Real) = car_sphere_f(p, q, noise, t::Real; vt = vt)
f_tilde = GeometricKalman.default_discretization(
    M,
    car_f_adapted;
    dt = dt,
    retraction = retraction,
)
# Filter-specific settings

sp = WanMerweSigmaPoints(; α = 1.0)
filter_params = [
    ( # extended Kalman filter
        "EKF",
        (;
            propagator = EKFPropagator(M, f_tilde; B_M = DefaultOrthonormalBasis()),
            updater = EKFUpdater(
                M,
                M_obs,
                car_sphere_h;
                B_M = DefaultOrthonormalBasis(),
                B_M_obs = DefaultOrthonormalBasis(),
            ),
        ),
    ),
    ( # unscented Kalman filter
        "UKF", 
        (;
            propagator = UnscentedPropagator(
                M;
                sigma_points = sp,
                inverse_retraction_method = inverse_retraction,
            ),
            updater = UnscentedUpdater(; sigma_points = sp),
        ),
    ),
    ( # adaptive extended Kalman filter
        "EKF adaptive M α=0.99",
        (;
            propagator = EKFPropagator(M, f_tilde; B_M = DefaultOrthonormalBasis()),
            updater = EKFUpdater(
                M,
                M_obs,
                car_sphere_h;
                B_M = DefaultOrthonormalBasis(),
                B_M_obs = DefaultOrthonormalBasis(),
            ),
            measurement_covariance_adapter = CovarianceMatchingMeasurementCovarianceAdapter(
                0.99,
            ),
        ),
    ),
]

# running the filters -- results will be saved in `reconstructions`
reconstructions = NamedTuple[]

for (name, filter_kwargs) in filter_params
    kf = discrete_kalman_filter_manifold(
        M,
        M_obs,
        p0,
        f_tilde,
        car_sphere_h,
        P0,
        copy(Q),
        copy(R);
        filter_kwargs...,
    )

    samples_kalman = []
    for i in eachindex(samples)
        GeometricKalman.update!(kf, controls[i], measurements[i])
        push!(samples_kalman, kf.p_n)
        predict!(kf, controls[i])
    end
    push!(reconstructions, (; data = samples_kalman, label = name))
end
```
