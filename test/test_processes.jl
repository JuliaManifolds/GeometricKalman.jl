using GeometricKalman, Test, LinearAlgebra, Distributions
using Manifolds
using RecursiveArrayTools

using GeometricKalman: gen_car_data

@testset "Planar car" begin
    samples, controls, measurements = gen_car_data(;
        vt=5,
        N=200,
        noise_f_distr=MvNormal([0.0, 0.0, 0.0], 1e3 * diagm([0.001, 1e-10, 1.0])),
    )
    M = SpecialEuclidean(2)
    for p in samples
        @test is_point(M, p)
    end
end
