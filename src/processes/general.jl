
function default_discretization(
    M::AbstractManifold,
    f;
    dt::Real=0.01,
    retraction::AbstractRetractionMethod=ExponentialRetraction(),
)
    return function tilde_f(p, q, w, t::Real)
        return ManifoldsBase.retract_fused(M, p, f(p, q, w, t), dt, retraction)
    end
end

struct InvariantExponentialRetraction <: AbstractRetractionMethod end

function ManifoldsBase.retract(M::AbstractManifold, p, X, ::InvariantExponentialRetraction)
    return exp_inv(M, p, X)
end
function ManifoldsBase.retract!(
    M::AbstractManifold,
    q,
    p,
    X,
    ::InvariantExponentialRetraction,
)
    return exp_inv!(M, q, p, X)
end

function ManifoldsBase.retract(
    M::AbstractDecoratorManifold,
    p,
    X,
    ierm::InvariantExponentialRetraction,
)
    return invoke(
        retract,
        Tuple{AbstractManifold,Any,Any,InvariantExponentialRetraction},
        M,
        p,
        X,
        ierm,
    )
end

function ManifoldsBase.retract!(
    M::AbstractDecoratorManifold,
    q,
    p,
    X,
    ierm::InvariantExponentialRetraction,
)
    return invoke(
        retract!,
        Tuple{AbstractManifold,Any,Any,Any,InvariantExponentialRetraction},
        M,
        q,
        p,
        X,
        ierm,
    )
end

function gen_data(
    M::AbstractManifold,
    p0,
    fun_f,
    fun_h,
    fun_control,
    noise_f_distr,
    noise_h_distr;
    N::Int=100,
    dt::Real=0.01,
    f_kwargs=(;),
    retraction::AbstractRetractionMethod=InvariantExponentialRetraction(),
    print_intermediates::Bool=false,
)
    samples = [p0]
    times = collect(range(0.0; step=dt, length=N + 1))
    controls = []
    measurements = [fun_h(p0, fun_control(0.0), rand(noise_h_distr), 0.0)]
    p_i = p0
    for i in 1:N
        t = (i - 1) * dt
        noise_f = sqrt(dt) * rand(noise_f_distr)
        noise_h = rand(noise_h_distr)
        ut = fun_control(t)
        X = dt * fun_f(p_i, ut, noise_f, t; f_kwargs...)
        p_i = retract(M, p_i, X, retraction)
        push!(samples, p_i)
        push!(controls, ut)
        push!(measurements, fun_h(p_i, ut, noise_h, t))
        if print_intermediates
            println("p_i = ", p_i)
            println("noise_h = ", noise_h)
            println("noise_f = ", noise_f)
            println("meas = ", measurements[end])
        end
    end
    push!(controls, fun_control(N * dt))
    return times, samples, controls, measurements
end
