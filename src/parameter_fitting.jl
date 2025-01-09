
"""
    abstract type AbstractKFOParametrization end

Represent a nonlinear least squares fitting problem for parameters of a Kalman filter.
The parameters may refer to initial conditions, noise covariance matrices, process
parameters (like weight of a certain factor) or measurement parameters (like sensor bias).

The following functionality is expected:
* `get_inits(pfo::AbstractKFOParametrization, p_opt)`
* `residuals(pfo::AbstractKFOParametrization, p_opt)`
"""
abstract type AbstractKFOParametrization end

"""
    get_inits(pfo::AbstractKFOParametrization, p_opt)

Get filter initialization parameters: `p0`, `P0`, `Q` and `R`, where `p_opt` is
the set of parameters over which optimization is performed.
"""
get_inits(pfo::AbstractKFOParametrization, p_opt)

struct InitialConditionKFOParametrization{TAM<:AbstractMatrix{<:Real}} <:
       AbstractKFOParametrization
    Q::TAM
    R::TAM
end

function get_inits(pfo::InitialConditionKFOParametrization, p_opt::ArrayPartition)
    return p_opt.x[1], p_opt.x[2], pfo.Q, pfo.R
end

struct KalmanParameterFittingObjective{
    TM<:AbstractManifold,
    TM_obs<:AbstractManifold,
    TM_obj<:AbstractManifold,
    TM_fit<:AbstractManifold,
    TB<:AbstractBasis,
    TFT,
    TH,
    TFKW,
    TP<:AbstractKFOParametrization,
    TOE,
    TRS,
    TRC,
    TRM,
    TZMC,
}
    M::TM
    M_obs::TM_obs
    M_obj::TM_obj
    M_fit::TM_fit
    jacobian_basis_arg::TB
    f_tilde::TFT
    h::TH
    filter_kwargs::TFKW
    kf_parametrization::TP
    obj_extractor::TOE
    ref_obj_vals::TRS
    ref_controls::TRC
    ref_measurements::TRM
    zero_M_fit_coordinates::TZMC
end

"""
"""
function make_kalman_parameter_fitting_objective(
    M::AbstractManifold, # where state evolves
    M_obs::AbstractManifold, # where observation happen
    M_obj::AbstractManifold, # where ref_obj_vals and values returned by obj_extractor live
    M_fit::AbstractManifold, # where arguments to the objective live
    f_tilde,
    h,
    filter_kwargs,
    kf_parametrization,
    obj_extractor,
    ref_obj_vals,
    ref_controls,
    ref_measurements;
    jacobian_basis_arg::AbstractBasis=DefaultOrthonormalBasis(),
)
    zero_M_fit_coordinates = zeros(manifold_dimension(M_fit))
    return KalmanParameterFittingObjective(
        M,
        M_obs,
        M_obj,
        M_fit,
        jacobian_basis_arg,
        f_tilde,
        h,
        filter_kwargs,
        kf_parametrization,
        obj_extractor,
        ref_obj_vals,
        ref_controls,
        ref_measurements,
        zero_M_fit_coordinates,
    )
end

function objective(pfo::KalmanParameterFittingObjective, p_opt)
    resid = residuals(pfo, p_opt)
    return (norm(resid)^2) / 2
end

function jacobian(pfo::KalmanParameterFittingObjective, p_opt)
    jac = ForwardDiff.jacobian(
        pc -> residuals(
            pfo,
            exp(pfo.M_fit, p_opt, get_vector(pfo.M_fit, p_opt, pc, pfo.jacobian_basis_arg)),
        ),
        pfo.zero_M_fit_coordinates,
    )
    return jac
end

function jacobian!(pfo::KalmanParameterFittingObjective, jac, p_opt)
    ForwardDiff.jacobian!(
        jac,
        pc -> residuals(
            pfo,
            exp(pfo.M_fit, p_opt, get_vector(pfo.M_fit, p_opt, pc, pfo.jacobian_basis_arg)),
        ),
        pfo.zero_M_fit_coordinates,
    )
    return jac
end

function residuals(pfo::KalmanParameterFittingObjective, p_opt)
    res =
        Vector{Base.promote_op(+, number_eltype(p_opt), number_eltype(pfo.ref_obj_vals[1]))}(
            undef,
            length(pfo.ref_controls),
        )
    residuals!(pfo, res, p_opt)
    return res
end

distance_squared(M::AbstractManifold, p, q) = distance(M, p, q)^2
# square root can't be effectively AD'd through at 0
function distance_squared(M::Euclidean, p, q)
    # Inspired by euclidean distance calculation in Distances.jl
    # Much faster for large p, q than a naive implementation
    @boundscheck if axes(p) != axes(q)
        throw(DimensionMismatch("At last one of $p and $q does not belong to $M"))
    end
    s = zero(eltype(p))
    @inbounds begin
        @simd for I in eachindex(p, q)
            p_i = p[I]
            q_i = q[I]
            s += abs2(p_i - q_i)
        end
    end
    return s
end

function residuals!(pfo::KalmanParameterFittingObjective, res, p_opt)
    p0, P0, Q, R = get_inits(pfo.kf_parametrization, p_opt)
    kf = discrete_kalman_filter_manifold(
        pfo.M,
        pfo.M_obs,
        p0,
        pfo.f_tilde,
        pfo.h,
        P0,
        copy(Q),
        copy(R);
        pfo.filter_kwargs...,
    )

    for i in eachindex(pfo.ref_obj_vals)
        obj_kf = pfo.obj_extractor(kf)
        obj_ref = pfo.ref_obj_vals[i]
        res[i] = distance_squared(pfo.M_obj, obj_ref, obj_kf)
        predict!(kf, pfo.ref_controls[i])
        update!(kf, pfo.ref_controls[i], pfo.ref_measurements[i])
    end
    return res
end
