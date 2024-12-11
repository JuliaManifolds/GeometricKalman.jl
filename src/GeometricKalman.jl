module GeometricKalman

using LinearAlgebra

using Manifolds
using Manopt

using StatsBase
import StatsBase: predict!

using ForwardDiff

include("jacobian_tools.jl")
include("filters.jl")

export CovarianceMatchingMeasurementCovarianceAdapter,
    CovarianceMatchingProcessCovarianceAdapter,
    EKFPropagator,
    EKFUpdater,
    KalmanState,
    UnscentedPropagator,
    UnscentedUpdater,
    WanMerweSigmaPoints

export discrete_kalman_filter_manifold, predict!, update!

end # module GeometricKalman
