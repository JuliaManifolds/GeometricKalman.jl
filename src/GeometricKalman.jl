module GeometricKalman

using LinearAlgebra

using ManifoldsBase, Manifolds
using Manopt

using StatsBase
import StatsBase: predict!

using Distributions

using RecursiveArrayTools
using ForwardDiff

include("jacobian_tools.jl")
include("filters.jl")
include("parameter_fitting.jl")

include("processes/general.jl")
include("processes/car_2d.jl")

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
