
car_control(t::Real) = sin(t / 2)

function car_f(p, q, noise, t::Real; vt=0.2)
    # based on "The invariant extended Kalman filter as a stable observer", Section 4.1 and 4.2
    dir, pos = p.x
    # noise entries: differential odometry noise, lateral odometry noise, transversal odometry noise
    G = SpecialOrthogonalGroup(2)
    X_dir = hat(G, Identity(G), [q * vt + noise[1]])
    X_pos = dir * [vt + noise[2], noise[3]]
    return ArrayPartition(X_dir, X_pos)
end

car_h(p, q, noise, t::Real) = p.x[2] + noise

function gen_car_data(;
    N::Int=100,
    dt::Real=0.01,
    p0=identity_element(SpecialEuclideanGroup(2), ArrayPartition),
    noise_f_distr=MvNormal([0.0, 0.0, 0.0], diagm([0.001, 1e-10, 1.0])),
    noise_h_distr=MvNormal([0.0, 0.0], diagm([0.001, 0.001])),
    vt::Real=0.2,
)
    return gen_data(
        SpecialEuclideanGroup(2),
        p0,
        car_f,
        car_h,
        car_control,
        noise_f_distr,
        noise_h_distr;
        N=N,
        dt=dt,
        f_kwargs=(; vt=vt),
    )
end
