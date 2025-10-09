using Manifolds, GeometricKalman, Test, LowLevelParticleFilters

function gen_data_llpf()
    # Create a time series for filtering
    x = [zeros(50); 0:100]
    T = length(x)
    Y = x + randn(T)

    y = [[y] for y in Y] # create a vector of vectors for the KF
    u = fill([], T) # No inputs in this example :(

    # Define the model
    Ts = 1
    A = [1 Ts; 0 1]
    B = zeros(2, 0)
    C = [1 0]
    D = zeros(0, 0)
    R2 = [1;;]

    σw = 1.0e-2

    # The dynamics noise covariance matrix is σw*Bw*Bw' where Bw = [Ts^2/2; Ts]
    R1 = σw * [
        Ts^3 / 3 Ts^2 / 2
        Ts^2 / 2 Ts
    ]
    kf = KalmanFilter(A, B, C, D, R1, R2)
    yh = []
    yf = []
    measure = LowLevelParticleFilters.measurement(kf)
    for t in 1:T # Main filter loop
        kf(u[t], y[t]) # Performs both prediction and correction
        xh = state(kf)
        yht = measure(xh, u[t], nothing, t)
        push!(yh, yht)
        push!(yf, xh)
    end

    Yh = reduce(hcat, yh)
    return x, Y, u, yf
end

@testset "Basic comparison with LLPF" begin
    states, measurements, controls, llpf_states = gen_data_llpf()
    Mstate = Euclidean(2)
    Mobs = Euclidean(1)
    Ts = 1.0
    A = [1.0 Ts; 0.0 1.0]
    σw = 1.0e-2
    p0 = [0.0, 0.0]
    # LowLevelParticleFilters takes Q matrix as the default initial covariance
    P0 = σw * [Ts^3 / 3 Ts^2 / 2; Ts^2 / 2 Ts]
    Q = σw * [Ts^3 / 3 Ts^2 / 2; Ts^2 / 2 Ts]
    R = [1.0;;]
    kf = discrete_kalman_filter_manifold(
        Mstate,
        Mobs,
        p0,
        (p, c, n, t) -> A * p + n,
        (p, c, n, t) -> [p[1] + n[1]],
        P0,
        Q,
        R,
    )

    samples_kalman = []
    for i in eachindex(states)
        # LLPF does update first, then predict
        GeometricKalman.update!(kf, controls[i], measurements[i])
        predict!(kf, [controls[i]])

        push!(samples_kalman, kf.p_n)
        @test kf.p_n ≈ llpf_states[i]
    end
end
