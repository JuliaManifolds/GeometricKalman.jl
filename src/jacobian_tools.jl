
function default_jacobian_p_discrete(
    M_arg::AbstractManifold,
    M_val::AbstractManifold,
    f;
    jacobian_basis_arg::AbstractBasis=DefaultOrthonormalBasis(),
    jacobian_basis_val::AbstractBasis=DefaultOrthonormalBasis(),
)
    return function jacobian_p(p, q, w, t)
        f_val = f(p, q, w, t)
        return ForwardDiff.jacobian(
            _c -> get_coordinates(
                M_val,
                f_val,
                log(
                    M_val,
                    f_val,
                    f(exp(M_arg, p, get_vector(M_arg, p, _c, jacobian_basis_arg)), q, w, t),
                ),
                jacobian_basis_val,
            ),
            zeros(manifold_dimension(M_arg)),
        )
    end
end

function default_jacobian_w_discrete(
    M::AbstractManifold,
    f;
    jacobian_basis::AbstractBasis=DefaultOrthonormalBasis(),
)
    return function jacobian_w(p, q, w, t)
        fp = f(p, q, w, t)
        return ForwardDiff.jacobian(
            _w -> get_coordinates(M, fp, log(M, fp, f(p, q, _w, t)), jacobian_basis),
            w,
        )
    end
end
