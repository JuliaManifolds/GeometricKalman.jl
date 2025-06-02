using Test
using GeometricKalman

using Manifolds

include("test_processes.jl")

include("basic_filtering.jl")

include("llpf_comparison.jl")
include("parameter_fitting.jl")

# this example doesn't consistently work yet
# include("test_rotating_earth.jl")
