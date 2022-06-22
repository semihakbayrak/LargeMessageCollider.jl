export fegradient, approximateMarginal!

# Free Energy gradient estimation with Reparameterization and REINFORCE.

# Available reparameterization functions
scale_shift(a::Real, b::Real, ϵ::Real) = a + b*ϵ

# forward and backward are logpdf of forward and backward messages
function fegradient(q::Student, forward::Function, backward::Function)
    grad = [NaN, NaN, NaN]
    while any(isnan,grad)
        # Partial derivatives estimated with reparameterization trick
        η = [q.μ, q.σ]
        g = [0., 1/q.σ]
        ϵ = rand(TDist(q.ν))
        h(λ) = scale_shift(λ[1],λ[2],ϵ)
        forw(λ) = forward(h(λ))
        back(λ) = backward(h(λ))
        g .+= forw'(η) .+ back'(η)
        grad1 = -g
        
        # Partial derivative estimated with REINFORCE estimator
        logq(ν) = logpdf(Student(q.μ,q.σ,ν),h(η))
        grad2 = logq'(q.ν)*(logq(q.ν) - forward(h(η)) - backward(h(η)))

        grad = [grad1;grad2]
    end
    return grad

end

function fegradient(q::Normal, forward::Function, backward::Function)
    grad = [NaN, NaN, NaN]
    while any(isnan,grad)
        # Partial derivatives estimated with reparameterization trick
        η = [q.μ, q.σ]
        g = [0., 1/q.σ]
        ϵ = randn()
        h(λ) = scale_shift(λ[1],λ[2],ϵ)
        forw(λ) = forward(h(λ))
        back(λ) = backward(h(λ))
        g .+= forw'(η) .+ back'(η)
        grad = -g
    end
    return grad

end

# Approximate Marginal functions, where out is logpdf
function approximateMarginal!(algo::F1, q::Normal, out::F2, in::T) where {F1<:FEgradient, F2<:Function, T<:Distribution} 
    q_cand = deepcopy(q)
    logforw = (z) -> logpdf(in,z)
    η = [q_cand.μ, q_cand.σ]
    for i=1:algo.num_iterations
        ∇ = fegradient(q_cand,logforw,out)
        update!(algo.optimizer,η,∇)
        if η[2] < 0 η[2] = 0.01 end
        q_cand = Normal(η[1], η[2])
    end
    return q_cand
end   

function approximateMarginal!(algo::F1, q::Student, out::F2, in::T) where {F1<:FEgradient, F2<:Function, T<:Distribution} 
    q_cand = deepcopy(q)
    logforw = (z) -> logpdf(in,z)
    η = [q_cand.μ, q_cand.σ, q_cand.ν]
    for i=1:algo.num_iterations
        ∇ = fegradient(q_cand,logforw,out)
        update!(algo.optimizer,η,∇)
        if η[2] < 0 η[2] = 0.01 end
        if η[3] < 2 η[3] = 2.01 end
        q_cand = Student(η[1], η[2], η[3])
    end
    return q_cand
end  