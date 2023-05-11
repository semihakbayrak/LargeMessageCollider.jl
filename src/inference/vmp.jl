export normal, mvnormal, transit, normalmix, categorical, gammadist
#-------------------
# VMP rules (mean-field assumption) for standard distributions
#-------------------
# normal is parameterized with mean and precision
normal(algo::VMP, x::Real, μ::Distribution, τ::Nothing) = convert(Gamma,[0.5,x*mean(μ) - squaremean(μ)/2 - x^2/2])
normal(algo::VMP, x::Distribution, μ::Real, τ::Nothing) = convert(Gamma,[0.5,μ*mean(x) - squaremean(x)/2 - μ^2/2])
normal(algo::VMP, x::Distribution, μ::Distribution, τ::Nothing) = convert(Gamma,[0.5,mean(x)*mean(μ) - squaremean(μ)/2 - squaremean(x)/2])
normal(algo::VMP, x::Real, μ::Nothing, τ::Distribution) = convert(Normal,[x*mean(τ),-mean(τ)/2])
normal(algo::VMP, x::Distribution, μ::Nothing, τ::Distribution) = convert(Normal,[mean(x)*mean(τ),-mean(τ)/2])
normal(algo::VMP, x::Nothing, μ::Distribution, τ::Distribution) = convert(Normal,[mean(μ)*mean(τ),-mean(τ)/2])
function mvnormal(algo::VMP, x::Vector, μ::Distribution, τ::Nothing)
    k = length(x)
    V_inv = x*x' - x*mean(μ)' - mean(μ)*x' + squaremean(μ)
    η = [vec(-0.5*V_inv); 0.5]
    convert(Wishart,η,check_args=false)
end
function mvnormal(algo::VMP, x::Distribution, μ::Distribution, τ::Nothing)
    k = length(mean(x))
    V_inv = squaremean(x) - mean(x)*mean(μ)' - mean(μ)*mean(x)' + squaremean(μ)
    η = [vec(-0.5*V_inv); 0.5]
    convert(Wishart,η,check_args=false)
end
function mvnormal(algo::VMP, x::Vector, μ::Nothing, τ::Distribution)
    MvNormal(x,inv(mean(τ)))
end
function mvnormal(algo::VMP, x::Distribution, μ::Nothing, τ::Distribution)
    MvNormal(mean(x),inv(mean(τ)))
end
function mvnormal(algo::VMP, x::Nothing, μ::Distribution, τ::Distribution)
    MvNormal(mean(μ),inv(mean(τ)))
end
categorical(algo::VMP, x::Nothing, p::Dirichlet) = convert(Categorical, logmean(p))
categorical(algo::VMP, x::Categorical, p::Nothing) = Dirichlet(x.p .+ 1)
#gammadist(x::Real, α::Distribution, θ::Nothing) = InverseGamma(mean(α)-1,x,check_args=false)
gammadist(algo::VMP, x::Real, α::Distribution, θ::Nothing) = Canonical(InverseGamma,[-mean(α),-x])

#-------------------
# State transition nodes to calculate messages towards process noise node (for LDS) or stochastic matrix (for HMM)
#-------------------

# Structured VMP that find the forward message at the end of the transit node
transit(algo::VMP, m_f::Normal, a::Real, w::Gamma) = a*m_f + Normal(0,sqrt(1/mean(w)))
transit(algo::VMP, m_f::MvNormal, A::Matrix, W::Wishart) = A*m_f + MvNormal(zeros(size(A)[1]),Matrix(Hermitian(inv(mean(W)))))
# Below function is useful for reduced form explained further below
function transit(algo::VMP, m_f::MvNormal, A::Matrix, w::Gamma) 
    V = diagm(0=>1e-15*ones(size(A)[1]))
    V[1,1] = 1/mean(w)
    A*m_f + MvNormal(zeros(size(A)[1]),matrix_posdef_numeric_stable(V))
end

# p(x_{t+1}|x_{t}) = N(x_{t+1}; A*x_{t},W^{-1})
# m_f is filtered belief of x_{t}, m_s is smoothed belief of x_{t+1}
# Structured VMP that returns m_s(x_t), q(x_{t+1},x_t|y_{1:T})
transit(algo::VMP, m_f::Normal, m_s::Normal, a::Real, w::Gamma) = transit(m_f, m_s, a, mean(w))
transit(algo::VMP, m_f::MvNormal, m_s::MvNormal, A::Matrix, W::Wishart) = transit(m_f, m_s, A, mean(W))
# Below function is useful for reduced form explained further below
function transit(algo::VMP, m_f::MvNormal, m_s::MvNormal, A::Matrix, w::Gamma) 
    mean_W = diagm(0=>1e15*ones(size(A)[1]))
    mean_W[1,1] = mean(w)
    transit(m_f, m_s, A, matrix_posdef_numeric_stable(mean_W))
end

# Structured VMP message towards process noise precision
function transit(algo::VMP, q1::Normal, q2::Normal, q21::MvNormal, a::Real, w::Nothing)
    α = 1.5
    β = 0.5*(squaremean(q2) + a^2*squaremean(q1) - 2*a*squaremean(q21)[1,2])
    θ = 1/β
    return Gamma(α,θ,check_args=false)
end

function transit(algo::VMP, q1::MvNormal, q2::MvNormal, q21::MvNormal, A::Matrix, W::Nothing; reduced::Bool=false)
    k = length(mean(q1))
    V_inv = squaremean(q2) - (squaremean(q21)[1:k,k+1:2*k])*A' - A*(squaremean(q21)[k+1:2*k,1:k]) + A*squaremean(q1)*A'
    η = [vec(-0.5*V_inv); 0.5]
    if reduced
        # in reduced form, noise is injected only on the first state during state transition 
        # this is especially useful when the states are coppied from one state to the other
        α = 1.5
        β = 0.5*V_inv[1,1]
        θ = 1/β
        return Gamma(α,θ,check_args=false)
    else
        return Canonical(Wishart,η)
    end
end

#-------------------
# Gaussian Mixture Model likelihood related variational messages for mean-field assumption
#-------------------
# Scalar observation. Message for categorical choice variable. Mean-precision parameterization

function normalmix(algo::VMP, y_n::Number, z::Nothing, qm_::Array{Normal{Float64}}, qw_::Array{Gamma{Float64}})
    p_vec = exp.(-0.5*log(2pi) .+ 0.5*logmean.(qw_)
            -0.5*mean.(qw_).*(y_n^2 .- 2*y_n*mean.(qm_) + squaremean.(qm_)))
    return Categorical(p_vec, check_args=false)
end

function normalmix(algo::VMP, y_n::Number, z::Categorical, qm_::Nothing, qw_::Array{Gamma{Float64}})
    K = length(z.p)
    return Normal.(y_n*ones(K),sqrt.(1 ./ (mean.(qw_).*z.p)))
end

function normalmix(algo::VMP, y_n::Number, z::Categorical, qm_::Array{Normal{Float64}}, qw_::Nothing)
    K = length(z.p)
    return convert.(Gamma,[[0.5*z.p[k],z.p[k]*(mean(qm_[k])*y_n - y_n^2/2 - squaremean(qm_[k])/2)] for k=1:K])
end

function normalmix(algo::VMP, y_n::Nothing, z::Categorical, qm_::Array{Normal{Float64}}, qw_::Array{Gamma{Float64}})
    m = sum(z.p .* mean.(qm_))/sum(z.p)
    v = 1/sum(z.p .* mean.(qw_))
    return Normal(m,sqrt(v))
end

# Vector observation. Message for categorical choice variable. Mean-precision parameterization

function normalmix(algo::VMP, y_n::Vector, z::Nothing, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
    d = length(mean(qm_[1]))
    K = length(qm_)
    p_vec = zeros(K)
    for k=1:K
        p_vec[k] = exp(-0.5*d*log(2pi) + 0.5*logdetmean(qw_[k])
                    -0.5*(tr(mean(qw_[k])*y_n*y_n') -tr(mean(qw_[k])*mean(qm_[k])*y_n')
                    -tr(mean(qw_[k])*y_n*mean(qm_[k])') +tr(mean(qw_[k])*squaremean(qm_[k]))))
    end
    return Categorical(p_vec, check_args=false)
end

function normalmix(algo::VMP, y_n::Vector, z::Categorical, qm_::Nothing, qw_::Array{F}) where F<:Wishart
    K = length(z.p)
    #message = Array{MvNormal}(undef, K)
    message = Array{Canonical}(undef, K)
    for k=1:K
        #message[k] = MvNormal(y_n,inv(z.p[k]*mean(qw_[k])))
        message[k] = Canonical(MvNormal,[z.p[k]*mean(qw_[k])*y_n;vec(-0.5*z.p[k]*mean(qw_[k]))])
    end
    return message
end

function normalmix(algo::VMP, y_n::Vector, z::Categorical, qm_::Array{F}, qw_::Nothing) where F<:MvNormal
    K = length(z.p)
    return Canonical.(Wishart,[[vec(0.5*z.p[k]*(mean(qm_[k])*y_n' + y_n*mean(qm_[k])' - y_n*y_n' - squaremean(qm_[k]))); 0.5*z.p[k]] for k=1:K])
end

# Below normalmix rules are useful for Deep GMM
function normalmix(algo::VMP, y_n::Nothing, z::Categorical, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
    m = sum(z.p .* mean.(qm_))./sum(z.p)
    V = Matrix(Hermitian(inv(sum(z.p .* mean.(qw_)))))
    return MvNormal(m,V)
end

function normalmix(algo::VMP, y_n::MvNormal, z::Nothing, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
    d = length(mean(qm_[1]))
    K = length(qm_)
    p_vec = zeros(K)
    for k=1:K
        p_vec[k] = exp(-0.5*d*log(2pi) + 0.5*logdetmean(qw_[k])
                    -0.5*(tr(mean(qw_[k])*squaremean(y_n)) -tr(mean(qw_[k])*mean(qm_[k])*mean(y_n)')
                    -tr(mean(qw_[k])*mean(y_n)*mean(qm_[k])') +tr(mean(qw_[k])*squaremean(qm_[k]))))
    end
    return Categorical(p_vec, check_args=false)
end

function normalmix(algo::VMP, y_n::MvNormal, z::Categorical, qm_::Nothing, qw_::Array{F}) where F<:Wishart
    K = length(z.p)
    #message = Array{MvNormal}(undef, K)
    message = Array{Canonical}(undef, K)
    for k=1:K
        #message[k] = MvNormal(y_n,inv(z.p[k]*mean(qw_[k])))
        message[k] = Canonical(MvNormal,[z.p[k]*mean(qw_[k])*mean(y_n);vec(-0.5*z.p[k]*mean(qw_[k]))])
    end
    return message
end

function normalmix(algo::VMP, y_n::MvNormal, z::Categorical, qm_::Array{F}, qw_::Nothing) where F<:MvNormal
    K = length(z.p)
    return Canonical.(Wishart,[[vec(0.5*z.p[k]*(mean(qm_[k])*mean(y_n)' + mean(y_n)*mean(qm_[k])' - squaremean(y_n) - squaremean(qm_[k]))); 0.5*z.p[k]] for k=1:K])
end