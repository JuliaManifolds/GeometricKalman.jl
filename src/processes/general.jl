
function default_discretization(M::AbstractManifold, f; dt::Real=0.01)
    return function tilde_f(p, q, w, t::Real)
        return exp(M, p, f(p, q, w, t), dt)
    end
end
