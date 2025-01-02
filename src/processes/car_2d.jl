
car_control(t::Real) = sin(t / 2)

function car_f(p, q, noise, t::Real; vt=0.2)
    # based on "The invariant extended Kalman filter as a stable observer", Section 4.1 and 4.2
    pos, dir = p.x
    # noise entries: differential odometry noise, lateral odometry noise, transversal odometry noise
    X_dir = hat(SpecialOrthogonal(2), Identity(SpecialOrthogonal(2)), [q * vt + noise[1]])
    X_pos = dir * [vt + noise[2], noise[3]]
    return ArrayPartition(X_pos, X_dir)
end

car_h(p, q, noise, t::Real) = p.x[1] + noise

function gen_car_data(;
    N::Int=100,
    dt::Real=0.01,
    M::AbstractManifold=SpecialEuclidean(2),
    p0=identity_element(SpecialEuclidean(2)),
    noise_f_distr=MvNormal([0.0, 0.0, 0.0], diagm([0.001, 1e-10, 1.0])),
    noise_h_distr=MvNormal([0.0, 0.0], diagm([0.001, 0.001])),
    vt::Real=0.2,
)
    samples = [p0]
    controls = []
    measurements = [car_h(p0, car_control(0.0), rand(noise_h_distr), 0.0)]
    p_i = p0
    for i in 1:N
        t = (i - 1) * dt
        noise_f = sqrt(dt) * rand(noise_f_distr)
        noise_h = rand(noise_h_distr)
        ut = car_control(t)
        # println(noise_f)
        X = dt * car_f(p_i, ut, noise_f, t; vt=vt)
        p_i = exp_inv(M, p_i, X)
        push!(samples, p_i)
        push!(controls, ut)
        push!(measurements, car_h(p_i, ut, noise_h, t))
    end
    push!(controls, car_control(N * dt))
    return samples, controls, measurements
end
