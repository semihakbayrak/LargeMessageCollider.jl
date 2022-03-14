export ep!
# We use Expectation Propagation (EP) as an approximate inference engine rather than a set of closed form message calculations
# as in BP and VMP. The EP engine collects input and output messages at a deterministic function f and executes a chosen 
# approximate inference algorithm to find an approximate marginal at the input edge. If the marginal is a set of 
# weighted samples, the EP engine further applies moment matching to generate a compact marginal distribution. Once marginal
# is acquired, EP divides the marginal to the incoming input message to find an outgoing message towards input edge.
function ep!(algo::F1, f::Union{Nothing,Function}, out::T1, in::T2) where {F1<:Union{TSL,EVMP,CVI}, T1<:Distribution, T2<:Distribution}
    q = approximateMarginal!(algo,f,out,in)
    if isa(q,SampleList)
        q = momentMatch(T2,q)
    end
    m_back = q/in
    return m_back
end
