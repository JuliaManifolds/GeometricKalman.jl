module GeometricKalman

using LinearAlgebra

using ManifoldsBase, Manifolds
using ManifoldsBase: exp_fused, exp_fused!, retract_fused, retract_fused!
using Manifolds: FiberBundleProductRetraction

using LieGroups

using Manopt

using StatsBase
import StatsBase: predict!

using Distributions

using RecursiveArrayTools
using ForwardDiff
using FiniteDifferences

using Rotations: Rotations

include("jacobian_tools.jl")
include("filters.jl")
include("parameter_fitting.jl")

include("processes/general.jl")
include("processes/car_2d.jl")
include("processes/car_sphere.jl")
include("processes/rotating_earth.jl")

export CovarianceMatchingMeasurementCovarianceAdapter,
    CovarianceMatchingProcessCovarianceAdapter,
    EKFPropagator,
    EKFUpdater,
    KalmanState,
    UnscentedPropagator,
    UnscentedUpdater,
    WanMerweSigmaPoints

export InitialConditionKFOParametrization

export discrete_kalman_filter_manifold, predict!, update!

end # module GeometricKalman
