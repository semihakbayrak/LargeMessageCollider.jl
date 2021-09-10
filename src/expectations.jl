export logmean, logdetmean, squaremean
#Some expectation quantities that does not exist in Distributions.jl

#--------------------------
# univariate Normal distribution
#--------------------------
# Expectation of log base measure h(x)
baseE(p::Normal) = log(1/sqrt(2*pi))
# E[x^2]
squaremean(p::Normal) = mean(p)^2 + var(p)

#--------------------------
# multivariate Normal distribution
#--------------------------
# Expectation of log base measure h(x)
baseE(p::MvNormal) = log((2*pi)^(-length(mean(p))/2))
# E[xx']
squaremean(p::MvNormal) = mean(p)*mean(p)' + cov(p)

#--------------------------
# Gamma distribution
#--------------------------
# Expectation of log base measure h(x)
baseE(p::Gamma) = 0
# E[log(x)]
logmean(p::Gamma) = digamma(shape(p)) - log(rate(p))


#--------------------------
# Dirichlet distribution
#--------------------------
# Expectation of log base measure h(x), variant 2
baseE(p::Dirichlet) = 0
# E[log(x)]
logmean(p::Dirichlet) = digamma.(p.alpha) .- digamma(sum(p.alpha))

#--------------------------
# Categorical distribution
#--------------------------
baseE(p::Categorical) = 0

#--------------------------
# Wishart distribution
#--------------------------
# Expectation of log base measure h(x)
baseE(p::Wishart) = 0
#multivariate digamma function
function mvdigamma(ρ, α)
    lmg(α) = logmvgamma(ρ,α)
    return lmg'(α)
end
# E[log|x|]
function logdetmean(p::Wishart)
    V = p.S.mat
    n = p.df
    ρ = size(V)[1]
    return mvdigamma(ρ, n/2) + ρ*log(2) +logdet(V)
end

#--------------------------
# SampleList
#--------------------------
# E[x^2]
function squaremean(p::SampleList)
    if p.dimension == 1
        return sum(p.weights.*p.samples.*p.samples)
    else
        s = zeros(p.dimension,p.dimension)
        for n=1:p.num_samples
            s .+= p.weights[n] .* p.samples[n] * transpose(p.samples[n])
        end
        return s
    end
end

# E[logx]
function logmean(p::SampleList)
    sum(p.weights .* log.(p.samples))
end
