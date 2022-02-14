export Canonical

#https://en.wikipedia.org/wiki/Exponential_family
# p(x) = h(x) exp{η'T(x) - A(η)}

# Canonical parameterization of distributions.
struct Canonical
    dist # distribution
    h # base measure
    T # sufficient statistics
    η # natural (canonical) parameters
    A_eval # log partition evaluated
    A # log partition function
    function Canonical(t::Type{F}, h::Function, T::Function, η::AbstractVector, A_eval::Number, A::Function) where F<:Distribution
        new(t,h,T,η,A_eval,A)
    end
end

# logpdf Canonical form
logpdf(p::Canonical, x) = log(p.h(x)) + transpose(p.η)*p.T(x) - p.A_eval

# Convert methods between Canonical type and Distribution type
convert(::Type{F}, p::Canonical) where {F<:Distribution} = convert(p.dist, p.η)

#--------------------------
# univariate Normal distribution
#--------------------------
function convert(::Type{F}, p::Normal) where F<:Canonical
    h(x::Number) = 1/sqrt(2*pi)
    T(x::Number) = [x,x^2]
    η = [mean(p)/var(p), -0.5/var(p)]
    A_eval = 0.5*mean(p)^2/var(p) + log(std(p))
    A(η::Array) = - η[1]^2/(4*η[2]) - 0.5*log(-2*η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return Canonical(Normal, h_func, T_func, η, A_eval, A_func)
end

function convert(t::Type{F}, η::AbstractVector) where F<:Normal
    v = -0.5/η[2]
    m = η[1]*v
    Normal(m,sqrt(v))
end

#--------------------------
# multivariate Normal distribution
#--------------------------
function convert(::Type{F}, p::MvNormal) where F<:Canonical
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

    return Canonical(MvNormal, h_func, T_func, η, A_eval, A_func)
end

function convert(t::Type{F}, η::AbstractVector) where F<:MvNormal
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
function convert(::Type{F}, p::Gamma) where F<:Canonical
    h(x::Number) = 1
    T(x::Number) = [log(x),x]
    η = [shape(p)-1, -rate(p)]
    A_eval = loggamma(shape(p)) - shape(p)*log(rate(p))
    A(η::Array) = loggamma(η[1]+1) - (η[1]+1)*log(-η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return Canonical(Gamma, h_func, T_func, η, A_eval, A_func)
end

function convert(t::Type{F}, η::AbstractVector; check_args=true) where F<:Gamma
    α, θ = η[1] + 1, -1/η[2]
    Gamma(α,θ,check_args=check_args)
end

#--------------------------
# InverseGamma distribution
#--------------------------
function convert(::Type{F}, p::InverseGamma) where F<:Canonical
    h(x::Number) = 1
    T(x::Number) = [log(x),1/x]
    η = [-shape(p)-1, -scale(p)]
    A_eval = loggamma(shape(p)) - shape(p)*log(scale(p))
    A(η::Array) = loggamma(-η[1]-1) - (-η[1]-1)*log(-η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return Canonical(InverseGamma, h_func, T_func, η, A_eval, A_func)
end

function convert(t::Type{F}, η::AbstractVector; check_args=true) where F<:InverseGamma
    α, θ = -η[1] - 1, -η[2]
    InverseGamma(α,θ,check_args=check_args)
end

#--------------------------
# Poisson distribution
#--------------------------
function convert(::Type{F}, p::Poisson) where F<:Canonical
    h(x::Number) = 1/factorial(x)
    T(x::Number) = [x]
    η = [log(rate(p))]
    A_eval = rate(p)
    A(η::Array) = exp(η[1])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return Canonical(Poisson, h_func, T_func, η, A_eval, A_func)
end

function convert(t::Type{F}, η::AbstractVector) where F<:Poisson
    λ = exp(η[1])
    Poisson(λ)
end

#--------------------------
# Dirichlet distribution
#--------------------------
# We use variant 2 in https://en.wikipedia.org/wiki/Exponential_family
function convert(::Type{F}, p::Dirichlet) where F<:Canonical
    h(x::Array) = 1
    T(x::Array) = log.(x)
    η = p.alpha .- 1
    A_eval = sum(loggamma.(p.alpha)) - loggamma(sum(p.alpha))
    A(η::Array) = sum(loggamma.(η .+ 1)) - loggamma(sum(η .+ 1))

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return Canonical(Dirichlet, h_func, T_func, η, A_eval, A_func)
end

function convert(t::Type{F}, η::AbstractVector) where F<:Dirichlet
    α = η .+ 1
    Dirichlet(α)
end

#--------------------------
# Categorical distribution
#--------------------------
# We use variant 3 in https://en.wikipedia.org/wiki/Exponential_family
function convert(::Type{F}, p::Categorical) where F<:Canonical
    p_vec = p.p ./ sum(p.p) # make sure that mean parameters add up to 1
    h(x::Array) = 1
    T(x::Array) = x .== ones(length(p_vec))
    η = [log.(p_vec[1:end-1]./p_vec[end]);0]
    A_eval = -log(p_vec[end])
    A(η::Array) = log(sum(exp.(η)))

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return Canonical(Categorical, h_func, T_func, η, A_eval, A_func)
end

function convert(t::Type{F}, η::AbstractVector; check_args=true, normalize=true) where F<:Categorical
    if normalize p_vec = exp.(η) ./ sum(exp.(η)) else p_vec = exp.(η) end
    Categorical(p_vec, check_args=check_args)
end

#--------------------------
# Wishart distribution
#--------------------------
function convert(::Type{F}, p::Wishart) where F<:Canonical
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

function convert(t::Type{F}, η::AbstractVector; check_args=true) where F<:Wishart
    ρ = Int(sqrt(length(η[1:end-1])))
    n = 2*η[end] + ρ + 1
    W = Matrix(Hermitian(-2*reshape(η[1:end-1],(ρ,ρ))))
    V = Matrix(Hermitian(inv(W)))
    Wishart(n,V,check_args)
end
