
abstract type AbstractKFPropagator end

function instantiate_propagator(::AbstractManifold, propagator::AbstractKFPropagator, p0)
    return propagator
end

abstract type AbstractKFUpdater end

function instantiate_updater(
    ::AbstractManifold,
    ::AbstractManifold,
    updater::AbstractKFUpdater,
    p0,
    h,
    control_prototype,
    zero_noise_obs,
    t0::Real,
)
    return updater
end

abstract type AbstractMeasurementCovarianceAdapter end

struct ConstantMeasurementCovarianceAdapter <: AbstractMeasurementCovarianceAdapter end

"""

HPHT is innovation covariance without measurement noise.
"""
adapt_covariance!(
    R::AbstractMatrix,
    ::ConstantMeasurementCovarianceAdapter,
    W_n,
    HPHT,
    S_n,
    residual,
) = R

abstract type AbstractProcessCovarianceAdapter end

struct ConstantProcessCovarianceAdapter <: AbstractProcessCovarianceAdapter end

adapt_covariance!(Q::AbstractMatrix, ::ConstantProcessCovarianceAdapter, Kyc, L_n) = Q

"""

Corresponds to Eq. (11) from [AkhlaghiZhouHuang:2017](@cite) with added `W_n`.
"""
struct CovarianceMatchingMeasurementCovarianceAdapter{T<:Real} <:
       AbstractMeasurementCovarianceAdapter
    α::T
end

function adapt_covariance!(
    R::AbstractMatrix,
    ca::CovarianceMatchingMeasurementCovarianceAdapter,
    W_n,
    HPHT,
    S_n,
    residual,
)
    R .*= ca.α
    R .+= (1 - ca.α) .* (W_n \ (residual * residual' + HPHT) / (W_n'))
    Manifolds.symmetrize!(R, R)
    return R
end

"""

Corresponds to Eq. (15) from [AkhlaghiZhouHuang:2017](@cite) with added `L_n`.
"""
struct CovarianceMatchingProcessCovarianceAdapter{T<:Real} <:
       AbstractProcessCovarianceAdapter
    α::T
end

function adapt_covariance!(
    Q::AbstractMatrix,
    ca::CovarianceMatchingProcessCovarianceAdapter,
    Kyc,
    L_n,
)
    Q .*= ca.α
    Q .+= (1 - ca.α) .* (L_n \ (Kyc * Kyc') / (L_n'))
    Manifolds.symmetrize!(Q, Q)
    return Q
end

@doc raw"""
    mutable struct KalmanState end

Manifold Kalman filter state.

The following deterministic dynamical system is discretized:

````math
D_t y(t) = f(y(t), q(t), t) \\
z(t) = h(p(t), q(t))
````

where ``y(t) \in \mathcal{M}`` is state at time `t`, ``q(t) \in Q`` are
control parameters,
``f\colon \mathcal{M} × \mathcal{N} × \mathbb{R} \to T\mathbb{M}`` represents
state transition function and ``h \colon \mathcal{M} \times Q \to \mathcal{M}_{obs}`` is
the measurement function.

Notably, when ``f`` is given by an action of a Lie group  on ``y(t)``, i.e.
``f(y, q, t) = g(y, q, t) ∘ y(t)`` for some function ``gf\colon \mathcal{M} \times \mathbb{R} \to \mathbb{G}``
and an action ``\circ`` of group ``\mathbb{G}`` on manifold ``\mathbb{M}``, the system can be solved using
RKMK integrators.

Such system is a generalization of the IEKF from [BarrauBonnabel:2015](@cite),
where ``\mathcal{M} = \mathcal{G}`` and action represents either left or right group
operation action.

Discrete equations are as follows:

````math

\begin{align*}
\tilde{f}(p, q_n, w_n, t_n) &= \exp_p(\Delta t f(p, q_n, w_n, t_n)) \quad & \text{discretization} \\
p_{n|n-1} &= \tilde{f}(p_{n-1|n-1}, q_n, w_n, t_n) \quad &\text{mean propagation} \\
\hat{f}(p_c, q_n, w_n, t_n) &= \tilde{f}(\phi^{-1}_p(p_c), q_n, w_n, t_n) \quad & \text{Jacobian parametrization} \\
F_n &= D_p \tilde{f}(p_{n-1|n-1}, q_{n}, w_n) \quad& \text{Jacobian of } \hat{f} \text{ wrt. } p_c \text{ at } p_{n-1|n-1} \\
L_n &= D_w \tilde{f}(p_{n-1|n-1}, q_{n}, w_n) \quad& \text{Jacobian of } \tilde{f} \text{ wrt. } w \text{ at } p_{n-1|n-1} \\
P_{n|n-1} &= F_n P_{n-1|n-1} F_n^{T} + L_n Q_n(p_{n|n-1}, q_n) L_n^{T} \quad& \text{covariance propagation} \\
P_{n|n-1} &  & \text{a linear operator on }T_{p_{n|n-1}}\mathcal{M} \\
z_n & \quad& \text{actual measurement}\\
y_n &= \log_{h(p_{n|n-1}, q_n)}(z_n) \quad & \text{measurement residual} \\
H_n &= D_p h(p_{n|n-1}, q_n) \quad& \text{Jacobian of } h \text{ wrt. } p \text{ at } p_{n|n-1} \\
H_n &  & \text{a linear operator from }T_{p_{n|n-1}}\mathcal{M} \text{ to } T_{h(p_{n|n-1}, q_n)} \mathcal{N} \\
R_n & \quad & \text{covariance matrix of the observation noise}
S_n &= H_n P_{n|n-1} H_n^{T} + R_n \quad& \text{innovation covariance}\\
S_n &  & \text{a linear operator on }T_{h(p_{n|n-1}, q_n)}\mathcal{N} \\
K_n &= P_{n|n-1} H_n^T S_n^{-1} \quad& \text{Kalman gain}\\
K_n &  & \text{a linear operator from }T_{h(p_{n|n-1}, q_n)}\mathcal{N} \text{ to } T_{p_{n|n-1}} \mathcal{M} \\
p_{n|n} &= \exp_{p_{n|n-1}}(K_n y_n) \quad& \text{updated state estimate} \\
P_{n|n} &= \operatorname{PT}_{p_{n|n-1} \to p_{n|n}} P_{n|n-1} - K_n S_n K_n^T \quad& \text{updated covariance estimate} 
\end{align*}
````

where ``p_{n|m}`` denotes estimate of ``p`` at time step ``n`` given samples up to and
including the one at time ``m``.
``\operatorname{PT}`` represents transporting covaraince from propagated state to the one
updated through measurement. It is sometimes known as "covariance reset",
see [GeGoorMahony:2023](@cite) and [MahonyGoorTarek:2022](@cite).
``\phi_p`` is a chart around point `p`, ``\phi_p \colon \mathcal{M} \to \mathbb{R}^k``,
and ``p_c = \phi_p(p)``. 

Note: for linear noise, ``L_n`` is the identity matrix.

## Fields

* `p`: filter state estimate
* `P`: state covariance matrix coordinates
* `Q`: process noise
* `R`: measurement noise
* `f_tilde`: discretized transition function
* `jacobian_f_tilde`: Jacobian of the discretized transition function
* `B`: basis type in which covariance and Jacobian matrices are computed
"""
mutable struct KalmanState{
    TM<:AbstractManifold,
    TMobs<:AbstractManifold,
    TB<:AbstractBasis{ℝ},
    TBobs<:AbstractBasis{ℝ},
    TVT<:AbstractVectorTransportMethod,
    TProp<:AbstractKFPropagator,
    TUpd<:AbstractKFUpdater,
    TOIR<:AbstractInverseRetractionMethod,
    TSR<:AbstractRetractionMethod,
    TMCA<:AbstractMeasurementCovarianceAdapter,
    TPCA<:AbstractProcessCovarianceAdapter,
    TCov_P<:AbstractMatrix{<:Real},
    TCov_Q<:AbstractMatrix{<:Real},
    TCov_R<:AbstractMatrix{<:Real},
    TP,
    TF,
    TH,
    TZN,
    TZNobs,
    TJacWF,
    TJacWH,
}
    M::TM
    M_obs::TMobs
    B_state::TB
    B_obs::TBobs
    vt::TVT
    propagator::TProp
    updater::TUpd
    p_n::TP
    t::Float64
    dt::Float64
    obs_inv_retr::TOIR
    state_retr::TSR
    measurement_covariance_adapter::TMCA
    process_covariance_adapter::TPCA
    P_n::TCov_P
    Q::TCov_Q
    R::TCov_R
    f_tilde::TF
    h::TH
    zero_noise::TZN
    zero_noise_obs::TZNobs
    jacobian_w_f_tilde::TJacWF
    jacobian_w_h::TJacWH
end

"""
    discrete_kalman_filter_manifold(
        M::AbstractManifold,
        M_obs::AbstractManifold,
        p0,
        f_tilde,
        h,
        P0,
        Q,
        R;
        kwargs...
    )

Construct a Kalman filter on manifold `M`.

# Keyword arguments

* `control_prototype=nothing`: prototype of the control parameter for `h` and `f_tilde`.
  Can be used for constructing caches by some algorithms.
"""
function discrete_kalman_filter_manifold(
    M::AbstractManifold,
    M_obs::AbstractManifold,
    p0,
    f_tilde,
    h,
    P0,
    Q,
    R;
    B_M::AbstractBasis=DefaultOrthonormalBasis(),
    B_M_obs::AbstractBasis=DefaultOrthonormalBasis(),
    vt::AbstractVectorTransportMethod=default_vector_transport_method(M),
    t0::Real=0.0,
    dt::Real=0.01,
    propagator::AbstractKFPropagator=EKFPropagator(M, f_tilde; B_M=B_M),
    updater::AbstractKFUpdater=EKFUpdater(M, M_obs, h; B_M=B_M, B_M_obs=B_M_obs),
    jacobian_w_f_tilde=default_jacobian_w_discrete(M, f_tilde; jacobian_basis=B_M),
    jacobian_w_h=default_jacobian_w_discrete(M_obs, h; jacobian_basis=B_M_obs),
    obs_inv_retr::AbstractInverseRetractionMethod=default_inverse_retraction_method(M_obs),
    state_retr::AbstractRetractionMethod=default_retraction_method(M),
    measurement_covariance_adapter::AbstractMeasurementCovarianceAdapter=ConstantMeasurementCovarianceAdapter(),
    process_covariance_adapter::AbstractProcessCovarianceAdapter=ConstantProcessCovarianceAdapter(),
    control_prototype=nothing,
    process_noise_dimensionality::Int=size(P0, 1),
)
    zero_noise = zeros(process_noise_dimensionality)
    zero_noise_obs = zeros(size(R, 1))
    instantiated_propagator = instantiate_propagator(M, propagator, p0)
    instantiated_updater =
        instantiate_updater(M, M_obs, updater, p0, h, control_prototype, zero_noise_obs, t0)
    initial_state = KalmanState(
        M,
        M_obs,
        B_M,
        B_M_obs,
        vt,
        instantiated_propagator,
        instantiated_updater,
        p0,
        t0,
        dt,
        obs_inv_retr,
        state_retr,
        measurement_covariance_adapter,
        process_covariance_adapter,
        P0,
        Q,
        R,
        f_tilde,
        h,
        zero_noise,
        zero_noise_obs,
        jacobian_w_f_tilde,
        jacobian_w_h,
    )
    return initial_state
end

function predict!(kalman::KalmanState, control)
    return predict!(kalman, kalman.propagator, control)
end

mutable struct EKFPropagator{TJacPF} <: AbstractKFPropagator
    jacobian_p_f_tilde::TJacPF
end

function EKFPropagator(
    M::AbstractManifold,
    f;
    B_M::AbstractBasis{ℝ}=DefaultOrthonormalBasis(),
)
    jacobian_p_f_tilde =
        default_jacobian_p_discrete(M, M, f; jacobian_basis_arg=B_M, jacobian_basis_val=B_M)
    return EKFPropagator(jacobian_p_f_tilde)
end

function predict!(kalman::KalmanState, prop::EKFPropagator, control)
    # mean propagation
    p_n_nm1 = kalman.f_tilde(kalman.p_n, control, kalman.zero_noise, kalman.t)
    # computing Jacobians
    F_n = prop.jacobian_p_f_tilde(kalman.p_n, control, kalman.zero_noise, kalman.t)
    L_n = kalman.jacobian_w_f_tilde(kalman.p_n, control, kalman.zero_noise, kalman.t)
    # covariance propagation
    P_n_nm1 = F_n * kalman.P_n * F_n' + L_n * kalman.Q * L_n'
    kalman.p_n = p_n_nm1
    kalman.P_n = P_n_nm1

    kalman.t += kalman.dt
    return kalman
end

function update!(kalman::KalmanState, control, measurement)
    return update!(kalman, kalman.updater, control, measurement)
end

abstract type UnscentedSigmaPoints end

"""



Source: E. A. Wan and R. Van Der Merwe, “The unscented Kalman filter for nonlinear
estimation,” in Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing,
Communications, and Control Symposium (Cat. No.00EX373), Lake Louise, Alta., Canada: IEEE,
2000, pp. 153–158. doi: 10.1109/ASSPCC.2000.882463.
"""
struct WanMerweSigmaPoints{TF<:Real} <: UnscentedSigmaPoints
    α::TF
    β::TF
    κ::TF
end

function WanMerweSigmaPoints(; α::T=1e-3, β::T=2.0, κ::T=0.0) where {T}
    return WanMerweSigmaPoints(α, β, κ)
end

function fill_weights!(
    mean_weights,
    cov_weights,
    sp::WanMerweSigmaPoints,
    M::AbstractManifold,
)
    L = manifold_dimension(M)
    λ = sp.α^2 * (L + sp.κ) - L
    fill!(mean_weights, 1 / (2 * (L + λ)))
    mean_weights[1] = λ / (L + λ)
    fill!(cov_weights, 1 / (2 * (L + λ)))
    return cov_weights[1] = λ / (L + λ) + (1 - sp.α^2 + sp.β)
end

function get_sigma_points!(
    sigma_points,
    sp::WanMerweSigmaPoints,
    M::AbstractManifold,
    p_n,
    P_n,
    B::AbstractBasis,
    retraction_method::AbstractRetractionMethod,
)

    # prepare sigma points and weights
    # note that our indices are shifted by 1 compared to Wan's paper.
    sigma_points[1] = p_n
    L = manifold_dimension(M)
    λ = sp.α^2 * (L + sp.κ) - L
    #sqrm = sqrt((L + λ) * P_n)
    regularization_parameter = 10 * eps(eltype(P_n))
    sqrm = cholesky(Symmetric((L + λ) * P_n) + regularization_parameter * I).L
    X = zero_vector(M, p_n)
    for i in 1:L
        Xc = view(sqrm, i, :)
        get_vector!(M, X, p_n, Xc, B)
        sigma_points[2 * i] = retract(M, p_n, X, retraction_method)
        sigma_points[2 * i + 1] = retract_fused(M, p_n, X, -1, retraction_method)
    end
    return sigma_points
end

"""



Source: E. A. Wan and R. Van Der Merwe, “The unscented Kalman filter for nonlinear
estimation,” in Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing,
Communications, and Control Symposium (Cat. No.00EX373), Lake Louise, Alta., Canada: IEEE,
2000, pp. 153–158. doi: 10.1109/ASSPCC.2000.882463.
"""
struct UnscentedPropagator{
    TSP<:UnscentedSigmaPoints,
    TIM<:AbstractInverseRetractionMethod,
} <: AbstractKFPropagator
    sp::TSP
    inverse_retraction_method::TIM
end
function UnscentedPropagator(;
    sigma_points::UnscentedSigmaPoints=WanMerweSigmaPoints(),
    inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
)
    return UnscentedPropagator{typeof(sigma_points),typeof(inverse_retraction_method)}(
        sigma_points,
        inverse_retraction_method,
    )
end

struct UnscentedPropagatorCache{
    TProp<:UnscentedPropagator,
    TMX<:AbstractMatrix,
    TSP<:AbstractVector,
    TW<:AbstractVector{<:Real},
    TXV,
} <: AbstractKFPropagator
    propagator::TProp
    Xcsr::TMX
    sigma_point_cache::TSP
    mean_weights::TW
    cov_weights::TW
    X::TXV
end

function instantiate_propagator(M::AbstractManifold, propagator::UnscentedPropagator, p0)
    N = manifold_dimension(M)
    num_sigma_points = 2 * N + 1
    sigma_point_cache = [allocate(M, p0) for _ in 1:num_sigma_points]
    Xcsr = Matrix{float(number_eltype(p0))}(undef, N, num_sigma_points)
    mean_weights = Vector{Float64}(undef, num_sigma_points)
    cov_weights = Vector{Float64}(undef, num_sigma_points)
    fill_weights!(mean_weights, cov_weights, propagator.sp, M)
    X = zero_vector(M, sigma_point_cache[1])
    return UnscentedPropagatorCache(
        propagator,
        Xcsr,
        sigma_point_cache,
        mean_weights,
        cov_weights,
        X,
    )
end

function predict!(kalman::KalmanState, propagator::UnscentedPropagatorCache, control)
    prop = propagator.propagator
    mean_weights = propagator.mean_weights
    cov_weights = propagator.cov_weights
    sigma_points = get_sigma_points!(
        propagator.sigma_point_cache,
        prop.sp,
        kalman.M,
        kalman.p_n,
        kalman.P_n,
        kalman.B_state,
        kalman.state_retr,
    )
    # compute function values

    fx = [kalman.f_tilde(p, control, kalman.zero_noise, kalman.t) for p in sigma_points]
    # compute new mean and covariance
    p_n = mean(kalman.M, fx, mean_weights)
    for i in 1:length(mean_weights)
        inverse_retract!(kalman.M, propagator.X, p_n, fx[i], prop.inverse_retraction_method)
        get_coordinates!(
            kalman.M,
            view(propagator.Xcsr, :, i),
            p_n,
            propagator.X,
            kalman.B_state,
        )
    end
    L_n = kalman.jacobian_w_f_tilde(kalman.p_n, control, kalman.zero_noise, kalman.t)

    N = manifold_dimension(kalman.M)

    P_n = similar(kalman.P_n)
    fill!(P_n, 0)
    for i in 1:length(sigma_points)
        xi = view(propagator.Xcsr, :, i)
        if any(isnan, xi)
            error("NaN encountered: ", xi)
        end
        P_n .+= cov_weights[i] .* xi * xi'
    end
    P_n += L_n * kalman.Q * L_n'

    # TODO: maybe make Q optional here?
    kalman.p_n = p_n
    kalman.P_n = P_n
    kalman.t += kalman.dt
    return kalman
end

mutable struct EKFUpdater{TJacPH} <: AbstractKFUpdater
    jacobian_p_h::TJacPH
end

function EKFUpdater(
    M::AbstractManifold,
    M_obs::AbstractManifold,
    h;
    B_M::AbstractBasis{ℝ}=DefaultOrthonormalBasis(),
    B_M_obs::AbstractBasis{ℝ}=DefaultOrthonormalBasis(),
)
    jacobian_p_h = default_jacobian_p_discrete(
        M,
        M_obs,
        h;
        jacobian_basis_arg=B_M,
        jacobian_basis_val=B_M_obs,
    )
    return EKFUpdater(jacobian_p_h)
end

function move_covariance!(kalman::KalmanState, p_n_new, P_n)
    P_n_e = eigen(Symmetric(P_n))
    Manopt.eigenvector_transport!(
        kalman.M,
        P_n_e,
        kalman.p_n,
        p_n_new,
        kalman.B_state,
        kalman.vt,
    )
    kalman.P_n .= P_n_e.vectors * Diagonal(P_n_e.values) * P_n_e.vectors'
    return nothing
end
function move_covariance!(
    kalman::KalmanState,
    p_n_new,
    P_n::AbstractMatrix{<:ForwardDiff.Dual},
)
    # don't calculate Jacobian of this correction for now
    # TODO: find a way to handle it
end

function update_from_kalman_gain!(
    kalman::KalmanState,
    y_expected,
    control,
    measurement,
    K_n,
    S_n,
    W_n,
    HPHT,
)
    # y_n is also called innovation
    y_n = inverse_retract(kalman.M_obs, y_expected, measurement, kalman.obs_inv_retr)
    Kyc = K_n * get_coordinates(kalman.M_obs, y_expected, y_n, kalman.B_obs)
    KyX = get_vector(kalman.M, kalman.p_n, Kyc, kalman.B_state)
    p_n_new = retract(kalman.M, kalman.p_n, KyX, kalman.state_retr)

    # adapt measurement covariance
    if !(kalman.measurement_covariance_adapter isa ConstantMeasurementCovarianceAdapter)
        hnew = kalman.h(p_n_new, control, kalman.zero_noise_obs, kalman.t)
        residual = inverse_retract(kalman.M_obs, hnew, measurement, kalman.obs_inv_retr)
        # println("innovation norm: ", norm(y_n))
        # println("residual norm: ", norm(residual))
        adapt_covariance!(
            kalman.R,
            kalman.measurement_covariance_adapter,
            W_n,
            HPHT,
            S_n,
            residual,
        )
    end

    # adapt process covariance
    if !(kalman.process_covariance_adapter isa ConstantProcessCovarianceAdapter)
        L_n = kalman.jacobian_w_f_tilde(kalman.p_n, control, kalman.zero_noise, kalman.t)
        adapt_covariance!(kalman.Q, kalman.process_covariance_adapter, Kyc, L_n)
    end

    kalman.P_n -= K_n * S_n * K_n'
    # move covariance to the new point and update Kalman filter state
    move_covariance!(kalman, p_n_new, kalman.P_n)
    kalman.p_n = p_n_new
    return kalman
end

function update!(kalman::KalmanState, upd::EKFUpdater, control, measurement)
    # compute discrepancy between expected and actual measurement
    y_expected = kalman.h(kalman.p_n, control, kalman.zero_noise_obs, kalman.t)
    # compute Kalman gain
    H_n = upd.jacobian_p_h(kalman.p_n, control, kalman.zero_noise_obs, kalman.t)
    W_n = kalman.jacobian_w_h(kalman.p_n, control, kalman.zero_noise_obs, kalman.t)
    HPHT = H_n * kalman.P_n * H_n'
    S_n = HPHT + W_n * kalman.R * W_n'
    K_n = kalman.P_n * H_n' / S_n
    update_from_kalman_gain!(kalman, y_expected, control, measurement, K_n, S_n, W_n, HPHT)
    return kalman
end

struct UnscentedUpdater{TSP<:UnscentedSigmaPoints,TIM<:AbstractInverseRetractionMethod} <:
       AbstractKFUpdater
    sp::TSP
    inverse_retraction_method::TIM
end
function UnscentedUpdater(;
    sigma_points::UnscentedSigmaPoints=WanMerweSigmaPoints(),
    inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
)
    return UnscentedUpdater{typeof(sigma_points),typeof(inverse_retraction_method)}(
        sigma_points,
        inverse_retraction_method,
    )
end

struct UnscentedUpdaterCache{
    TUpd<:UnscentedUpdater,
    TMX<:AbstractMatrix,
    TPxy<:AbstractMatrix,
    TXObs,
} <: AbstractKFUpdater
    updater::TUpd
    Hcsr::TMX
    Pxy::TPxy
    X_obs::TXObs
end

function instantiate_updater(
    M::AbstractManifold,
    M_obs::AbstractManifold,
    updater::UnscentedUpdater,
    p0,
    h,
    control_prototype,
    zero_noise_obs,
    t0::Real,
)
    N = manifold_dimension(M)
    N_obs = manifold_dimension(M_obs)
    num_sigma_points = 2 * N + 1
    X_obs = zero_vector(M_obs, h(p0, control_prototype, zero_noise_obs, t0))
    Hcsr = Matrix{float(eltype(X_obs))}(undef, manifold_dimension(M_obs), num_sigma_points)
    Pxy = similar(Hcsr, N, N_obs)
    return UnscentedUpdaterCache(updater, Hcsr, Pxy, X_obs)
end

function update!(kalman::KalmanState, upd::UnscentedUpdaterCache, control, measurement)
    mean_weights = kalman.propagator.mean_weights
    cov_weights = kalman.propagator.cov_weights
    sigma_points = get_sigma_points!(
        kalman.propagator.sigma_point_cache,
        upd.updater.sp,
        kalman.M,
        kalman.p_n,
        kalman.P_n,
        kalman.B_state,
        kalman.state_retr,
    )
    # compute measurement function values

    hx = [kalman.h(p, control, kalman.zero_noise_obs, kalman.t) for p in sigma_points]
    # compute new mean and covariance
    y_expected = mean(kalman.M_obs, hx, Weights(mean_weights, 1.0))

    for i in 1:length(mean_weights)
        inverse_retract!(
            kalman.M_obs,
            upd.X_obs,
            y_expected,
            hx[i],
            upd.updater.inverse_retraction_method,
        )
        get_coordinates!(
            kalman.M_obs,
            view(upd.Hcsr, :, i),
            y_expected,
            upd.X_obs,
            kalman.B_obs,
        )
    end
    N_obs = manifold_dimension(kalman.M_obs)
    S_n = similar(upd.Hcsr, N_obs, N_obs)
    fill!(S_n, 0)
    for i in 1:length(sigma_points)
        ci = view(upd.Hcsr, :, i)
        S_n .+= cov_weights[i] .* ci * ci'
    end
    HPHT = copy(S_n)
    # add measurement noise
    W_n = kalman.jacobian_w_h(kalman.p_n, control, kalman.zero_noise_obs, kalman.t)
    S_n += W_n * kalman.R * W_n'
    # cross-covariance
    fill!(kalman.updater.Pxy, 0)
    for i in 1:length(sigma_points)
        kalman.updater.Pxy .+=
            cov_weights[i] .* view(kalman.propagator.Xcsr, :, i) * view(upd.Hcsr, :, i)'
    end
    # Kalman gain
    K_n = kalman.updater.Pxy / S_n

    # final updates
    update_from_kalman_gain!(kalman, y_expected, control, measurement, K_n, S_n, W_n, HPHT)
    return kalman
end
