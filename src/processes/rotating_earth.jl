
earth_control(t::Real) = nothing

# for gen_rotating_earth_data
# state: SE(3) × T(3) × T(3) × T(3) × S(2)
# (position and orientation) × (joint orientation) × (velocity) × (angular velocity) × (acceleration) × (gyroscope bias A) × (gyroscope bias B) × (accelerometer bias A) × (accelerometer bias B)
const RotEarthManifold = ProductManifold(
    SpecialEuclidean(3),
    Sphere(2),
    Euclidean(3),
    Euclidean(3),
    Euclidean(3),
    Euclidean(3),
    Euclidean(3),
    Euclidean(3),
    Euclidean(3),
)

# observation manifold for rotating Earth example
# (gyroscope A reading) × (accelerometer A reading) × (gyroscope B reading) × (accelerometer B reading)
const RotEarthObsManifold =
    ProductManifold(Euclidean(3), Euclidean(3), Euclidean(3), Euclidean(3))

struct EarthModel{TMSO3<:SpecialOrthogonal,Te_SO3,TΩx,Tg<:AbstractVector}
    SO3::TMSO3
    e_SO3::Te_SO3
    Ωx::TΩx
    g::Tg
end

function EarthModel()
    SO3 = SpecialOrthogonal(3)
    e_SO3 = identity_element(SO3)

    latitude = 50.0
    Ω = 7.292e-5 * [cosd(latitude), 0, -sin(latitude)]
    Ωx = hat(em.SO3, em.e_SO3, Ω)
    g = [0, 0, -9.81]

    return EarthModel(SpecialOrthogonal(3), e_SO3, Ωx, g)
end

function (em::EarthModel)(p, q, noise, t::Real)
    # unpack state
    pos, R = p.x[1].x
    joint = p.x[2]
    vel = p.x[3]
    ω = p.x[4]
    acc = p.x[5]
    b_gyro_a = p.x[6]
    b_gyro_b = p.x[7]
    b_acc_a = p.x[8]
    b_acc_b = p.x[9]

    # compute tangents
    X_pos = vel
    X_R = -em.Ωx * R + R * hat(em.SO3, em.e_SO3, ω)
    X_joint = [0.0, 0.0, 0.0]
    X_vel = R * acc + em.g - 2 * em.Ωx * vel - em.Ωx^2 * pos
    X_ω = [0.0, 0.0, 0.0]
    X_acc = [0.0, 0.0, 0.0]
    X_b_gyro = [0.0, 0.0, 0.0]
    X_b_bias = [0.0, 0.0, 0.0]

    return ArrayPartition(
        ArrayPartition(X_pos, X_R),
        X_joint,
        X_vel,
        X_ω,
        X_acc,
        X_b_gyro,
        X_b_gyro,
        X_b_bias,
        X_b_bias,
    )
end

function earth_h(p, q, noise, t::Real)
    # IMU A is attached to the main body; IMU B is on the joint 
    return p.x[1] + noise
end

# the model from Section VI.A of http://arxiv.org/abs/2007.14097
# with added joint of Sphere(2) degrees of freedom
# see RotEarthManifold for state
function gen_rotating_earth_data(;
    N::Int=100,
    dt::Real=0.01,
    M::AbstractManifold=RotEarthManifold,
    p0=ArrayPartition(
        identity_element(SpecialEuclidean(3)),
        [0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ),
    noise_f_distr=MvNormal([0.0, 0.0, 0.0], diagm([0.001, 1e-10, 1.0])),
    noise_h_distr=MvNormal([0.0, 0.0], diagm([0.001, 0.001])),
)
    return gen_data(
        M,
        p0,
        earth_f,
        earth_h,
        earth_control,
        noise_f_distr,
        noise_h_distr;
        N=N,
        dt=dt,
    )
end
