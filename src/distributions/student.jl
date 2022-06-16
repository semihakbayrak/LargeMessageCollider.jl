export Student

# 1D Student's t distribution implemented. TDist exists in Distributions.jl. 
# Unfortunately their TDist is implemented for 0 location 1 scale case.
# In this implementation, we will include location and scale as parameters.

struct Student <: ContinuousUnivariateDistribution
    μ # location in real domain
    σ # scale in positive real domain
    ν # degrees of freedom
    function Student(μ, σ, ν)
        new(μ, σ, ν)
    end
end

Distributions.mean(p::Student) = if p.ν>1 p.μ else NaN end
Distributions.var(p::Student) = if p.ν>2 p.σ^2 * p.ν/(p.ν-2) else NaN end

Distributions.rand(rng::AbstractRNG, p::Student) = p.μ + p.σ*rand(TDist(p.ν))

function Distributions.pdf(p::Student, x::Real)
    gamma(p.ν/2+0.5)/gamma(p.ν/2) * 1/sqrt(pi*p.ν*p.σ) * (1+(x-p.μ)^2/(p.ν*p.σ^2))^(-p.ν/2-0.5)
end

function Distributions.logpdf(p::Student, x::Real)
    loggamma(p.ν/2+0.5) - loggamma(p.ν/2) - 0.5*log(pi*p.ν*p.σ) - (p.ν/2+0.5)*log(1+(x-p.μ)^2/(p.ν*p.σ^2))
end