# GeometricKalman

Kalman filters on manifolds with an affine connection, unifying the Lie group and Riemannian approaches.

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamanifolds.github.io/GeometricKalman.jl/dev/)

arXiv preprint: https://arxiv.org/abs/2506.01086

## Getting started

Basic setting (example: a car driving on a sphere)

```@example 1

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
```

Generating data using the `gen_car_sphere_data` function.

```@example 1
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
```

Setting initial conditions for filters

```@example 1
p0 = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
P0 = diagm([0.1, 0.1, 0.1, 0.1])
Q = diagm([0.1, 0.1, 0.01, 0.01])
R = diagm([0.01, 0.01])

```

Adapting system dynamics to the interface expected by Kalman filters.

```@example 1

car_f_adapted(p, q, noise, t::Real) = car_sphere_f(p, q, noise, t::Real; vt = vt)
f_tilde = GeometricKalman.default_discretization(
    M,
    car_f_adapted;
    dt = dt,
    retraction = retraction,
)

```

Filter-specific settings

```@example 1

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

```

Running the filters. Results will be saved in `reconstructions`.

```@example 1
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

Plotting the estimated trajectory and measurements.

```@example 1

using Plots

function trajectory_plot3d(
    p0,
    samples::Vector,
    reconstructions::Vector{<:NamedTuple},
    measurements::Vector,
)
    fig = plot(
        [s.x[1][1] for s in samples],
        [s.x[1][2] for s in samples],
        [s.x[1][3] for s in samples];
        label = "original",
        linewidth = 5.0,
    )
    scatter3d!(map(v -> [v], p0.x[1])..., markersize = 15, label = "Starting point")

    for rec in reconstructions
        plot!(
            [s.x[1][1] for s in rec.data],
            [s.x[1][2] for s in rec.data],
            [s.x[1][3] for s in rec.data];
            label = rec.label,
        )
    end

    scatter!(
        [s[1] for s in measurements],
        [s[2] for s in measurements],
        [s[3] for s in measurements];
        label = "measurements",
    )
    return fig
end


trajectory_plot3d(p0, samples, reconstructions, measurements)
```
