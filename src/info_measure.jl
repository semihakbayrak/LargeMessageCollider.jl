export bregman_div, kl_div, cross_entropy, differential_entropy, normal_conditional_entropy, categorical_conditional_entropy

function bregman_div(x::Number, y::Number, f::Function)
    d_f = f'(y)
    f(x) - f(y) - (x-y)*d_f
end

function bregman_div(x::Array, y::Array, f::Function)
    grad_f = f'(y)
    f(x) - f(y) - transpose(x-y)*grad_f
end

# For details check "ENTROPIES AND CROSS-ENTROPIES OF EXPONENTIAL FAMILIES" by Nielsen and Nock
# KL[p||q] = E_p(x)[log(p(x)/q(x))]
function kl_div(p::F, q::F) where F<:Distribution
    p_canon, q_canon = convert(Canonical, p), convert(Canonical, q)
    return bregman_div(q_canon.η,p_canon.η,p_canon.A)
end

# https://en.wikipedia.org/wiki/Cross_entropy
# -E_q(x)[logp(x)]
cross_entropy(q::F, p::F) where {F<:Distribution} = entropy(q) + kl_div(q,p)
cross_entropy(q::MatrixDirichlet, p::MatrixDirichlet) = sum(cross_entropy.(matrix2list_dirichlet(q),matrix2list_dirichlet(p)))

# https://en.wikipedia.org/wiki/Conditional_entropy
# -∫q(x_{t},x_{t-1})*logq(x_{t}|x_{t-1}) dx_{t} dx_{t-1}
function normal_conditional_entropy(q1::MvNormal, q2::MvNormal, q21::MvNormal)
    k = length(mean(q1))
    Σ_22 = cov(q2)
    Σ_12 = cov(q21)[k+1:2*k,1:k]
    Σ_21 = cov(q21)[1:k,k+1:2*k]
    Σ_11 = cov(q1)
    μ_2, μ_1 = mean(q2), mean(q1)
    M = Σ_21*inv(Σ_11)
    N = Matrix(Hermitian(Σ_22 - M*Σ_12))
    invN = Matrix(Hermitian(inv(N)))
    res = k/2 * log(2*pi) + logdet(N)/2
    res2 = tr(invN*squaremean(q2)) - tr(invN*(μ_2*μ_2' + Σ_21*M')) - tr(invN*(μ_2*μ_2' + M*Σ_12))
    res2 += tr(invN*(μ_2*μ_2' + M*Σ_11*M'))
    return res + res2/2
end

# -∑_{z[t],z[t-1]} q(z[t],z[t-1])*logq(z[t]|z[t-1])
function categorical_conditional_entropy(q21::Matrix)
    K = size(q21)[1]
    γ_transpose = sum(q21, dims=1)
    logq_cond = log.(q21) - log.(ones(K)*γ_transpose)
    return -sum(q21.*logq_cond)
end

#----------------------------------------------
# Cross entropies -E_q[logp] for distinct distribution family p and q
#----------------------------------------------
cross_entropy(q::Dirichlet, p::Categorical) = - sum(p.p .* logmean(q))

function normal(x::Real, μ::Distribution, τ::Distribution)
    res = log(2*pi)/2 - logmean(τ)/2
    res += (mean(τ) * (x^2 - 2*x*mean(μ) + squaremean(μ)))/2
    return res
end

function normal(x::Distribution, μ::Distribution, τ::Distribution)
    res = log(2*pi)/2 - logmean(τ)/2
    res += (mean(τ) * (squaremean(x) - 2*mean(x)*mean(μ) + squaremean(μ)))/2
    return res
end

function mvnormal(x::Vector, μ::Distribution, τ::Distribution)
    k = length(mean(μ))
    res = k/2*log(2*pi) - logdetmean(τ)/2
    res += (tr(mean(τ)*(x*x' - x*mean(μ)' - mean(μ)*x' + squaremean(μ))))/2
    return res
end

function mvnormal(x::Distribution, μ::Distribution, τ::Distribution)
    k = length(mean(μ))
    res = k/2*log(2*pi) - logdetmean(τ)/2
    res += (tr(mean(τ)*(squaremean(x) - mean(x)*mean(μ)' - mean(μ)*mean(x)' + squaremean(μ))))/2
    return res
end

function normalmix(y_n::Number, z::Categorical, qm_::Array{Normal{Float64}}, qw_::Array{Gamma{Float64}})
    p_vec = -0.5*log(2pi) .+ 0.5*logmean.(qw_) -0.5*mean.(qw_).*(y_n^2 .- 2*y_n*mean.(qm_) + squaremean.(qm_))
    return sum(-z.p .* p_vec)
end

function normalmix(y_n::Vector, z::Categorical, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
    K = length(z.p)
    d = length(mean(qm_[1]))
    entropy = 0
    for k=1:K
        entropy -= z.p[k] * (-0.5*d*log(2pi) + 0.5*logdetmean(qw_[k])
                    -0.5*(y_n'*mean(qw_[k])*y_n - y_n'*mean(qw_[k])*mean(qm_[k]) - mean(qm_[k])'*mean(qw_[k])*y_n
                    + tr(mean(qw_[k])*squaremean(qm_[k]))))
    end
    return entropy
end

# Useful in DeepGMM
function normalmix(y_n::MvNormal, z::Categorical, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
    K = length(z.p)
    d = length(mean(qm_[1]))
    entropy = 0
    for k=1:K
        entropy -= z.p[k] * (-0.5*d*log(2pi) + 0.5*logdetmean(qw_[k])
                    -0.5*(tr(mean(qw_[k])*squaremean(y_n)) - mean(y_n)'*mean(qw_[k])*mean(qm_[k]) - mean(qm_[k])'*mean(qw_[k])*mean(y_n)
                    + tr(mean(qw_[k])*squaremean(qm_[k]))))
    end
    return entropy
end

# -∫q(W)q(x_{t},x_{t-1})*logp(x_{t}|x_{t-1},W) dW dx_{t} dx_{t-1}
function transit(q1::MvNormal, q2::MvNormal, q21::MvNormal, A::Matrix, qW::Wishart)
    k = length(mean(q1))
    V_inv = squaremean(q2) - (squaremean(q21)[1:k,k+1:2*k])*A' - A*(squaremean(q21)[k+1:2*k,1:k]) + A*squaremean(q1)*A'
    return k/2*log(2*pi) - logdetmean(qW)/2 + tr(V_inv*mean(qW))/2
end

# -∫q(A)∑_{z[t]}∑_{z[t-1]}q(z_{t},z_{t-1})*logp(z_{t}|z_{t-1},A) dA
transit(q21::Matrix, qA::MatrixDirichlet) = -sum(q21 .* logmean(qA))

# -∫q(B)∑_{y[t]}∑_{z[t]}q(y_{t})q(z_{t})*logp(y_{t}|z_{t},B) dB
emit(q_x::Categorical, q_z::Categorical, qB::MatrixDirichlet) = -sum((q_x.p * q_z.p') .* logmean(qB))
