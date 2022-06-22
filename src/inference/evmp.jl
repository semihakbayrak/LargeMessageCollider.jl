export approximateMarginal!
# Extended VMP (EVMP) by Akbayrak et al. This implementation does not contain Laplace solution for Gaussian case
# and only focus on importance sampling solution.

function approximateMarginal!(algo::F1, f::F2, out::T1, in::T2) where {F1<:EVMP, F2<:Union{Nothing,Function}, T1<:Distribution, T2<:Distribution}
    logp_nc = (z) -> logpdf(out,z)
    if F2 <: Function
        logp_nc = (z) -> logpdf(out,f(z))
    end
    q = collide(in, logp_nc, proposal=in, num_samples=algo.num_samples)[1]
    return q
end

# out is logpdf backward message
function approximateMarginal!(algo::F1, f::F2, out::T1, in::T2) where {F1<:EVMP, F2<:Union{Nothing,Function}, T1<:Function, T2<:Distribution}
    logp_nc = (z) -> out(z)
    if F2 <: Function
        logp_nc = (z) -> out(f(z))
    end
    q = collide(in, logp_nc, proposal=in, num_samples=algo.num_samples)[1]
    return q
end