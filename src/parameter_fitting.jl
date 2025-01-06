
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
    TFT,
    TH,
    TFKW,
    TP<:AbstractKFOParametrization,
    TOE,
    TRS,
    TRC,
    TRM,
}
    M::TM
    M_obs::TM_obs
    M_obj::TM_obj
    f_tilde::TFT
    h::TH
    filter_kwargs::TFKW
    kf_parametrization::TP
    obj_extractor::TOE
    ref_obj_vals::TRS
    ref_controls::TRC
    ref_measurements::TRM
end

function objective(pfo::KalmanParameterFittingObjective, p_opt)
    resid = residuals(pfo, p_opt)
    return (norm(resid)^2) / 2
end

function jacobian(pfo::KalmanParameterFittingObjective, p_opt)
    jac = ForwardDiff.jacobian(p -> residuals(pfo, p), p_opt)
    return jac
end

function residuals(pfo::KalmanParameterFittingObjective, p_opt)
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

    res =
        Vector{Base.promote_op(+, number_eltype(p_opt), number_eltype(pfo.ref_obj_vals[1]))}(
            undef,
            length(pfo.ref_controls),
        )
    for i in eachindex(pfo.ref_obj_vals)
        obj_kf = pfo.obj_extractor(kf)
        obj_ref = pfo.ref_obj_vals[i]
        res[i] = distance(pfo.M_obj, obj_ref, obj_kf)
        predict!(kf, pfo.ref_controls[i])
        update!(kf, pfo.ref_controls[i], pfo.ref_measurements[i])
    end
    return res
end
