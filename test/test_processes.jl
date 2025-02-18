using GeometricKalman, Test, LinearAlgebra, Distributions
using Manifolds
using RecursiveArrayTools

using GeometricKalman: car_f, car_h, car_control, gen_data

@testset "Planar car" begin
    M = SpecialEuclidean(2)
    N = 200
    dt = 0.01
    p0 = identity_element(M)
    noise_f_distr = MvNormal([0.0, 0.0, 0.0], 1e3 * diagm([0.001, 1e-10, 1.0]))
    noise_h_distr = MvNormal([0.0, 0.0], diagm([0.001, 0.001]))
    vt = 0.2

    times, samples, controls, measurements = gen_data(
        M,
        p0,
        car_f,
        car_h,
        car_control,
        noise_f_distr,
        noise_h_distr;
        N=N,
        dt=dt,
        f_kwargs=(; vt=vt),
    )

    for p in samples
        @test is_point(M, p)
    end
end
