
function car_sphere_f(p, q, noise, t::Real; vt=0.2)
    S2 = Sphere(2)
    SO2 = SpecialOrthogonal(2)
    # spherical variant of car_f
    pos, dir = p.x
    # noise entries: differential odometry noise, lateral odometry noise, transversal odometry noise
    #X_dir = get_vector(S2, pos, [q * vt + noise[1], noise[2]])
    X_dir = cross(pos, [0.0, q * vt + noise[1], noise[2]])
    X_pos = vt * dir .+ get_vector(S2, pos, noise[3:4])
    return ArrayPartition(X_pos, X_dir)
end

function car_sphere_h(p, q, noise, t::Real)
    S2 = Sphere(2)
    return exp(S2, p.x[1], get_vector(S2, p.x[1], noise))
end

function gen_car_sphere_data(;
    N::Int=100,
    dt::Real=0.01,
    p0=ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    noise_f_distr=MvNormal([0.0, 0.0, 0.0, 0.0], diagm([0.001, 1e-10, 1e-10, 1.0])),
    noise_h_distr=MvNormal([0.0, 0.0], diagm([0.001, 0.001])),
    vt::Real=0.2,
    retraction::AbstractRetractionMethod=FiberBundleProductRetraction(),
)
    return gen_data(
        TangentBundle(Sphere(2)),
        p0,
        car_sphere_f,
        car_sphere_h,
        car_control,
        noise_f_distr,
        noise_h_distr;
        N=N,
        dt=dt,
        f_kwargs=(; vt=vt),
        retraction=retraction,
    )
end
