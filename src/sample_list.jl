export SampleList

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
