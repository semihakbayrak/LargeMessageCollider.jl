export SampleList, momentMatch, centralmomentMatch

# SampleList doesn't have to be a univariate distribution or continuous valued.
# Inheritance directly from Distributions is not allowed. Therefore I arbitrarily choose
# ContinuousUnivariateDistribution. See https://juliastats.org/Distributions.jl/stable/extends/
# for more details.
struct SampleList <: ContinuousUnivariateDistribution
    samples
    weights
    num_samples::Int
    dimension::Int
    function SampleList(samples::Vector, weights::Vector)
        new(samples,Weights(weights./sum(weights)),length(weights),length(samples[1]))
    end
end

SampleList(samples::Vector) = SampleList(samples, ones(length(samples)))

Distributions.mean(p::SampleList) = sum(p.weights.*p.samples)
Distributions.var(p::SampleList) = (p.num_samples / (p.num_samples - 1.)) * sum(p.weights.*(p.samples.-mean(p)).^2)
function evaluate_cov(p::SampleList)
    V = zeros(p.dimension,p.dimension)
    for n=1:p.num_samples
        V .+= p.weights[n].*(p.samples[n].-mean(p))*(p.samples[n].-mean(p))'
    end
    return Matrix(Hermitian((p.num_samples / (p.num_samples - 1.)) .* V))
end
Distributions.cov(p::SampleList) = evaluate_cov(p)

# Resampling. For drawing n samples, this function only works in 1 dimension
# since SampleList is a ContinuousUnivariateDistribution type.
# For higher dimensions, directly use sample(p.samples,p.weights,n)
function Distributions.rand(rng::AbstractRNG, p::SampleList)
    sample(rng,p.samples,p.weights)
end

# Moment matching to convert SampleList to compact distribution form
function momentMatch(t::Type{F}, p::SampleList) where F<:Normal
    μ = [mean(p), squaremean(p)]
    η = invlink(Normal, μ)
    return convert(Normal,η)
end

function momentMatch(t::Type{F}, p::SampleList) where F<:MvNormal
    μ = [mean(p); vec(squaremean(p))]
    η = invlink(MvNormal, μ)
    return convert(MvNormal,η)
end

function momentMatch(t::Type{F}, p::SampleList) where F<:Poisson
    μ = [mean(p)]
    η = invlink(Poisson, μ)
    return convert(Poisson,η)
end

function momentMatch(t::Type{F}, p::SampleList) where F<:Bernoulli
    μ = [mean(p)]
    η = invlink(Bernoulli, μ)
    return convert(Bernoulli,η)
end

function momentMatch(t::Type{F}, p::SampleList) where F<:Categorical
    μ = mean(p)
    η = invlink(Categorical, μ)
    return convert(Categorical,η)
end

# Central moment matching to convert SampleList to compact distribution form
centralmomentMatch(t::Type{F}, p::SampleList) where {F<:Normal} = momentMatch(t,p)

centralmomentMatch(t::Type{F}, p::SampleList) where {F<:MvNormal} = momentMatch(t,p)

function centralmomentMatch(t::Type{F}, p::SampleList) where F<:Gamma
    m, v = mean(p), var(p)
    β = m/v
    α = β*m
    θ = 1/β
    Gamma(α,θ)
end

function centralmomentMatch(t::Type{F}, p::SampleList) where F<:InverseGamma
    m, v = mean(p), var(p)
    α = m/v + 2
    β = m*(α-1)
    InverseGamma(α,β)
end

centralmomentMatch(t::Type{F}, p::SampleList) where {F<:Poisson} = Poisson(mean(p))

function centralmomentMatch(t::Type{F}, p::SampleList) where F<:Beta
    b = mean(p)*(1-mean(p))^2/var(p) + mean(p) - 1
    a = mean(p)*b/(1-mean(p))
    Beta(a,b)
end

centralmomentMatch(t::Type{F}, p::SampleList) where {F<:Bernoulli} = Bernoulli(mean(p))

centralmomentMatch(t::Type{F}, p::SampleList) where {F<:Categorical} = Categorical(mean(p))