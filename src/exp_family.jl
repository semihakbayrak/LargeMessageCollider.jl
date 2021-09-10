export exp_family, Canonical

#https://en.wikipedia.org/wiki/Exponential_family
# p(x) = h(x) exp{η'T(x) - A(η)}

# Canonical parameterization of distributions. Useful to handle messages with improper dist.s
struct Canonical
    dist
    η
    function Canonical(t::Type{F}, c::Vector) where F<:Distribution
        new(t,c)
    end
end

exp_family(c::Canonical) = exp_family(c.dist, c.η)

#--------------------------
# univariate Normal distribution
#--------------------------
function exp_family(p::Normal)
    h(x::Number) = 1/sqrt(2*pi)
    T(x::Number) = [x,x^2]
    η = [mean(p)/var(p), -0.5/var(p)]
    A_eval = 0.5*mean(p)^2/var(p) + log(std(p))
    A(η::Array) = - η[1]^2/(4*η[2]) - 0.5*log(-2*η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

function exp_family(t::Type{F}, η::Vector) where F<:Normal
    v = -0.5/η[2]
    m = η[1]*v
    Normal(m,sqrt(v))
end

#--------------------------
# multivariate Normal distribution
#--------------------------
function exp_family(p::MvNormal)
    k = length(mean(p))
    W = inv(cov(p))

    h(x::Vector) = (2*pi)^(-k/2)
    T(x::Vector) = [x;vec(x*x')]
    η = [cov(p)\mean(p);vec(-0.5*W)]
    A_eval = 0.5*(mean(p)'*(cov(p)\mean(p)) + logdet(cov(p)))
    A(η::Array) = - 0.25*η[1:k]'*(reshape(η[k+1:end],(k,k))\η[1:k]) -0.5*logdet(-2*reshape(η[k+1:end],(k,k)))

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

# function exp_family(t::Type{F}, η::Vector) where F<:MvNormal
#     n = length(η)
#     Δ = 1 + 4*n
#     k = Int((-1 + sqrt(Δ))/2)
#     W = Matrix(Hermitian(-2*reshape(η[k+1:end],(k,k))))
#     V = Matrix(Hermitian(inv(W)))
#     m = W\η[1:k]
#     MvNormal(m,V)
# end

function exp_family(t::Type{F}, η) where F<:MvNormal
    n = length(η)
    Δ = 1 + 4*n
    k = Int((-1 + sqrt(Δ))/2)
    W = Matrix(Hermitian(-2*reshape(η[k+1:end],(k,k))))
    V = Matrix(Hermitian(inv(W)))
    m = W\η[1:k]
    MvNormal(m,V)
end

#--------------------------
# Gamma distribution
#--------------------------
function exp_family(p::Gamma)
    h(x::Number) = 1
    T(x::Number) = [log(x),x]
    η = [shape(p)-1, -rate(p)]
    A_eval = loggamma(shape(p)) - shape(p)*log(rate(p))
    A(η::Array) = loggamma(η[1]+1) - (η[1]+1)*log(-η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

function exp_family(t::Type{F}, η::Vector, check_args=true) where F<:Gamma
    α, θ = η[1] + 1, -1/η[2]
    Gamma(α,θ,check_args=check_args)
end

#--------------------------
# InverseGamma distribution
#--------------------------
function exp_family(p::InverseGamma)
    h(x::Number) = 1
    T(x::Number) = [log(x),1/x]
    η = [shape(p)-1, -scale(p)]
    A_eval = loggamma(shape(p)) - shape(p)*log(scale(p))
    A(η::Array) = loggamma(-η[1]-1) - (-η[1]-1)*log(-η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

function exp_family(t::Type{F}, η::Vector, check_args=true) where F<:InverseGamma
    α, θ = η[1] + 1, -η[2]
    InverseGamma(α,θ,check_args=check_args)
end

#--------------------------
# Poisson distribution
#--------------------------
function exp_family(p::Poisson)
    h(x::Number) = 1/factorial(x)
    T(x::Number) = [x]
    η = [log(rate(p))]
    A_eval = rate(p)
    A(η::Array) = exp(η[1])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

function exp_family(t::Type{F}, η::Vector) where F<:Poisson
    λ = exp(η[1])
    Poisson(λ)
end

#--------------------------
# Dirichlet distribution
#--------------------------
# We use variant 2 in https://en.wikipedia.org/wiki/Exponential_family
function exp_family(p::Dirichlet)
    h(x::Array) = 1
    T(x::Array) = log.(x)
    η = p.alpha .- 1
    A_eval = sum(loggamma.(p.alpha)) - loggamma(sum(p.alpha))
    A(η::Array) = sum(loggamma.(η .+ 1)) - loggamma(sum(η .+ 1))

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

function exp_family(t::Type{F}, η::Vector) where F<:Dirichlet
    α = η .+ 1
    Dirichlet(α)
end

#--------------------------
# Categorical distribution
#--------------------------
# We use variant 1 in https://en.wikipedia.org/wiki/Exponential_family
function exp_family(p::Categorical, normalize=false)
    if normalize p_vec = p.p ./ sum(p.p) else p_vec = p.p end
    h(x::Array) = 1
    T(x::Array) = x .== ones(length(p_vec))
    η = log.(p_vec)
    A_eval = 0
    A(η::Array) = 0

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

function exp_family(t::Type{F}, η::Vector, check_args=true, normalize=true) where F<:Categorical
    if normalize p_vec = exp.(η) ./ sum(exp.(η)) else p_vec = exp.(η) end
    Categorical(p_vec, check_args=check_args)
end

#--------------------------
# Wishart distribution
#--------------------------
function exp_family(p::Wishart)
    V = p.S.mat
    ρ = size(V)[1]
    n = p.df

    h(x::Matrix) = 1
    T(x::Matrix) = [vec(x);logdet(x)]
    η = [vec(-0.5*inv(V)); (n-ρ-1)/2]
    A_eval = p.logc0
    A(η::Array) = -(η[end] + (ρ+1)/2)*log(det(-reshape(η[1:end-1],(ρ,ρ)))) +  logmvgamma(ρ,η[end]+(ρ+1)/2)

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func, T_func, η, A_eval, A_func
end

function exp_family(t::Type{F}, η::Vector, check_args=true) where F<:Wishart
    ρ = Int(sqrt(length(η[1:end-1])))
    n = 2*η[end] + ρ + 1
    W = Matrix(Hermitian(-2*reshape(η[1:end-1],(ρ,ρ))))
    V = Matrix(Hermitian(inv(W)))
    Wishart(n,V,check_args)
end
