
earth_control_zero(t::Real) = zeros(18)

function earth_control_stationary(t::Real)
    q = zeros(18)
    q[9] = 9.81 # counteracts gravity
    return q
end

Base.@kwdef struct EarthControlMovement
    X_pos_amplitude::Float64 = 1.0
    X_pos_angular_frequency::Float64 = 0.1
    X_b_gyro_amplitude::Float64 = 0.001
    X_b_gyro_angular_frequency::Float64 = 0.001
    X_b_acc_amplitude::Float64 = 0.001
    X_b_acc_angular_frequency::Float64 = 0.001
end

function (ecm::EarthControlMovement)(t::Real)
    X_joint = [0.0, 0.0, 0.0]
    X_ω = [ecm.X_pos_angular_frequency * sin(ecm.X_pos_angular_frequency * t), 0.0, 0.0]
    X_pos =
        ecm.X_pos_amplitude *
        [cos(ecm.X_pos_angular_frequency * t), sin(ecm.X_pos_angular_frequency * t), 0.0]
    # X_vel is 0 in axes X and Y because position changes are already handled by X_pos
    X_vel = [0.0, 0.0, 9.81]
    X_b_gyro = [ecm.X_b_gyro_amplitude * sin(ecm.X_b_gyro_angular_frequency * t), 0.0, 0.0]
    X_b_acc = [0.0, ecm.X_b_acc_amplitude * cos(ecm.X_b_acc_angular_frequency * t), 0.0]
    return vcat(X_joint, X_pos, X_vel, X_ω, X_b_gyro, X_b_acc)
end

const RotEarthRetraction = ProductRetraction(
    InvariantExponentialRetraction(),
    ntuple(i -> ExponentialRetraction(), Val(8))...,
)

const RotEarthProcessNoiseDimensionality = 6

# for gen_rotating_earth_data
# state: SE(3) × S(2) × T(3) × T(3) × T(3) × T(3) × T(3) × T(3)
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

function show_state(::typeof(RotEarthManifold), p)
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

    println("Position           : $pos")
    println("Rotation           : $R")
    println("Joint orientation  : $joint")
    println("Velocity:          : $vel")
    println("Rotational velocity: $ω")
    println("Acceleration       : $acc")
    println("Bias of gyro A     : $b_gyro_a")
    println("Bias of gyro B     : $b_gyro_b")
    println("Bias of acc A      : $b_acc_a")
    return println("Bias of acc B      : $b_acc_b")
end

# observation manifold for rotating Earth example
# (gyroscope A reading) × (accelerometer A reading) × (gyroscope B reading) × (accelerometer B reading) × (GPS position)
const RotEarthObsManifold =
    ProductManifold(Euclidean(3), Euclidean(3), Euclidean(3), Euclidean(3), Euclidean(3))

struct EarthModel{TMSO3<:SpecialOrthogonal,Te_SO3,TΩx,Tg<:AbstractVector}
    SO3::TMSO3
    e_SO3::Te_SO3
    Ωx::TΩx
    g::Tg
end

function joint_rotation_matrix(joint::AbstractVector)
    if joint[1] ≈ 1.0
        # Make it somewhat compatible with AD
        cosθ = 1.0
        sinθ = 0.0
        cosφ = joint[1]
        sinφ = 0.0

        return [
            (cosθ*cosφ) -(cosθ * sinφ) sinθ
            sinφ cosφ 0.0
            -(sinθ * cosφ) (sinθ*sinφ) cosθ
        ]
    end
    jy = atan(joint[3], joint[2])

    cosθ = cos(jy)
    sinθ = sin(jy)
    cosφ = joint[1]
    sinφ = sqrt(1 - cosφ^2)
    return [
        (cosθ*cosφ) -(cosθ * sinφ) sinθ
        sinφ cosφ 0.0
        -(sinθ * cosφ) (sinθ*sinφ) cosθ
    ]
end

function angles_to_joint_position(ry, rz)
    return [cos(rz), sin(rz) * cos(ry), sin(rz) * sin(ry)]
end

function joint_position_to_angles(p)
    srz = sqrt(1 - p[1]^2)
    if isapprox(srz, 0)
        a2 = 0.0
    else
        a2 = atan(p[3] / srz, p[2] / srz)
    end
    return (a2, acos(p[1]))
end

function EarthModel()
    SO3 = SpecialOrthogonal(3)
    e_SO3 = identity_element(SO3)

    latitude = 50.0
    Ω = 7.292e-5 * [cosd(latitude), 0, -sind(latitude)]
    Ωx = hat(SO3, e_SO3, Ω)
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
    X_pos = vel + noise[1:3] + q[4:6]
    X_R = -em.Ωx * R + R * hat(em.SO3, em.e_SO3, ω + noise[4:6])
    X_joint = q[1:3]
    X_vel = R * acc + em.g - 2 * em.Ωx * vel - em.Ωx^2 * pos + q[7:9]
    X_ω = q[10:12]
    X_acc = [0.0, 0.0, 0.0]
    X_b_gyro = q[13:15]
    X_b_acc = q[16:18]

    total_X = ArrayPartition(
        ArrayPartition(X_pos, X_R),
        X_joint,
        X_vel,
        X_ω,
        X_acc,
        X_b_gyro,
        X_b_gyro,
        X_b_acc,
        X_b_acc,
    )
    return total_X
end

function earth_h(p, q, noise, t::Real)
    # IMU A is attached to the main body; IMU B is on the joint 

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

    gyro_a_reading = ω + b_gyro_a + noise[1:3]
    acc_a_reading = acc + b_acc_a + noise[4:6]

    joint_R = joint_rotation_matrix(joint)
    gyro_b_reading = joint_R * ω + b_gyro_b + noise[7:9]
    acc_b_reading = joint_R * acc + b_acc_b + noise[10:12]

    gps_reading = pos + noise[13:15]

    return ArrayPartition(
        gyro_a_reading,
        acc_a_reading,
        gyro_b_reading,
        acc_b_reading,
        gps_reading,
    )
end

# the model from Section VI.A of http://arxiv.org/abs/2007.14097
# "Associating Uncertainty to Extended Poses for on Lie Group IMU Preintegration with Rotating Earth"
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
