export approximateMarginal!, forwardMessage, transit
# Unscented transofrmation through non-linear functions.
# For details, check Sarkka's "Bayesian Filtering and Smoothing"

function forwardMessage(algo::F, f::Function, in::Normal; α=0.001, β=2, κ=0) where F<:UT
    return forwardMessageUT(in, f, α, β, κ)
end

function forwardMessage(algo::F, f::Function, in::MvNormal; α=0.001, β=2, κ=0) where F<:UT
    return forwardMessageUT(in, f, α, β, κ)
end

function utHyperparams(α, β, κ, d)
    λ = α^2*(d + κ) - d
    return λ
end

function utWeights(d, λ, α, β)
    Wᵐ, Wᶜ = [λ/(d+λ); 1/(2*(d+λ))*ones(2*d)], [λ/(d+λ) + (1-α^2+β); 1/(2*(d+λ))*ones(2*d)]
    return Wᵐ, Wᶜ
end

function sigmaPoints(in::Normal, d, λ)
    μ, σ = mean(in), sqrt(var(in))
    s0 = μ
    s1 = μ + sqrt(d+λ)*σ
    s2 = μ - sqrt(d+λ)*σ
    sigma = [s0,s1,s2]
    return sigma
end

function sigmaPoints(in::MvNormal, d, λ)
    μ, L = mean(in), cholesky(cov(in)).L
    sigma = [μ]
    for i=1:d
        s1 = μ .+ sqrt(d+λ)*L[:,i]
        s2 = μ .- sqrt(d+λ)*L[:,i]
        push!(sigma,s1)
        push!(sigma,s2)
    end
    return sigma
end

function forwardMessageUT(in::Union{Normal,MvNormal}, f::Function, α, β, κ)
    μ = mean(in)
    d = length(μ)
    λ = utHyperparams(α, β, κ, d)

    # Sigma points
    sigma = sigmaPoints(in, d, λ)
    # Weights
    Wᵐ, Wᶜ = utWeights(d, λ, α, β)
    # Transformed Sigma points
    f_sigma = f.(sigma)

    f_μ = sum(Wᵐ .* f_sigma)
    f_d = length(f_μ)
    if f_d == 1
        f_v = sum(Wᶜ .* (f_sigma .- f_μ).^2)
        return Normal(f_μ, sqrt(f_v)), sigma, f_sigma
    else
        f_v = Wᶜ[1] * (f_sigma[1] .- f_μ) * (f_sigma[1] .- f_μ)'
        for i=2:2*d+1
            f_v .+= Wᶜ[i] * (f_sigma[i] .- f_μ) * (f_sigma[i] .- f_μ)'
        end
        return MvNormal(f_μ, Matrix(Hermitian(f_v+diagm(0=>1e-10*ones(f_d))))), sigma, f_sigma
    end
end

function approximateMarginal!(algo::F, f::Function, out::Normal, in::Normal; α=0.001, β=2, κ=0) where F<:UT
    p_f, sigma, f_sigma = forwardMessageUT(in, f, α, β, κ)
    λ = utHyperparams(α, β, κ, 1)
    Wᵐ, Wᶜ = utWeights(1, λ, α, β)
    s = var(p_f) + var(out)
    c = Wᶜ[1]*(sigma[1]-mean(in))*(f_sigma[1]-mean(out)) + Wᶜ[2]*(sigma[2]-mean(in))*(f_sigma[2]-mean(out)) + Wᶜ[3]*(sigma[3]-mean(in))*(f_sigma[3]-mean(out))
    v = mean(out) - mean(p_f)
    k = c/s
    m = mean(in) + k*v
    σ = sqrt(var(in) - s*k^2)
    q = Normal(m,σ)
    return q
end

function approximateMarginal!(algo::F, f::Function, out::Normal, in::MvNormal; α=0.001, β=2, κ=0) where F<:UT
    p_f, sigma, f_sigma = forwardMessageUT(in, f, α, β, κ)
    d = length(mean(in))
    λ = utHyperparams(α, β, κ, d)
    Wᵐ, Wᶜ = utWeights(d, λ, α, β)
    s = var(p_f) + var(out)
    c = Wᶜ[1]*(sigma[1]-mean(in))*(f_sigma[1]-mean(out))
    for i=2:2*d+1
        c .+= Wᶜ[i]*(sigma[i]-mean(in))*(f_sigma[i]-mean(out))
    end
    v = mean(out) - mean(p_f)
    k = c./s
    m = mean(in) + k*v
    V = Matrix(Hermitian(cov(in) - s*k*k'))
    q = MvNormal(m,V)
    return q
end

function approximateMarginal!(algo::F, f::Function, out::MvNormal, in::Normal; α=0.001, β=2, κ=0) where F<:UT
    p_f, sigma, f_sigma = forwardMessageUT(in, f, α, β, κ)
    λ = utHyperparams(α, β, κ, 1)
    Wᵐ, Wᶜ = utWeights(1, λ, α, β)
    S = Matrix(Hermitian(cov(p_f) + cov(out) + diagm(0=>1e-10*ones(length(mean(out))))))
    c = Wᶜ[1]*(sigma[1]-mean(in))*(f_sigma[1]-mean(out))'
    for i=2:3
        c .+= Wᶜ[i]*(sigma[i]-mean(in))*(f_sigma[i]-mean(out))'
    end
    v = mean(out) - mean(p_f)
    k = c*inv(S)
    m = mean(in) + k*v
    σ = sqrt(var(in) - k*S*k')
    q = Normal(m,σ)
    return q
end

function approximateMarginal!(algo::F, f::Function, out::MvNormal, in::MvNormal; α=0.001, β=2, κ=0) where F<:UT
    p_f, sigma, f_sigma = forwardMessageUT(in, f, α, β, κ)
    d = length(mean(in))
    λ = utHyperparams(α, β, κ, d)
    Wᵐ, Wᶜ = utWeights(d, λ, α, β)
    S = Matrix(Hermitian(cov(p_f) + cov(out) + diagm(0=>1e-10*ones(length(mean(out))))))
    C = Wᶜ[1]*(sigma[1]-mean(in))*(f_sigma[1]-mean(out))'
    for i=2:2*d+1
        C .+= Wᶜ[i]*(sigma[i]-mean(in))*(f_sigma[i]-mean(out))'
    end
    v = mean(out) - mean(p_f)
    K = C*inv(S)
    m = mean(in) + K*v
    V = Matrix(Hermitian(cov(in) - K*S*K'))
    q = MvNormal(m,V)
    return q
end

# transit function useful for Unscented Kalman Smoothing
# p(x_{t+1}|x_{t}) = N(x_{t+1}; f(x_{t}),W^{-1})
# m_f is filtered belief of x_{t}, m_s is smoothed belief of x_{t+1}
# return m_s(x_t), p(x_{t+1},x_t|y_{1:T})
function transit(m_f::Normal, m_s::Normal, func::Function, w::Real; α=0.001, β=2, κ=0)
    p_func, sigma, func_sigma = forwardMessageUT(m_f, func, α, β, κ)
    f, F = mean(m_f), var(m_f) # f_t, F_t
    k, K = mean(m_s), var(m_s) # k_{t+1}, K_{t+1}
    d = 1
    λ = utHyperparams(α, β, κ, d)
    Wᵐ, Wᶜ = utWeights(d, λ, α, β)
    P1 = F
    P21 = Wᶜ[1]*(sigma[1]-f)*(func_sigma[1]-mean(p_func)) + Wᶜ[2]*(sigma[2]-f)*(func_sigma[2]-mean(p_func)) + Wᶜ[3]*(sigma[3]-f)*(func_sigma[3]-mean(p_func))
    P12 = P21
    P2 = var(p_func) + 1/w
    P2_inv = 1/P2
    k_t = f - P12*P2_inv*(mean(p_func) - k)
    Pa = P12*P2_inv
    K_ = Pa*K
    K_t = K_*Pa + (P1 - Pa*P21)
    m = [k;k_t]
    V = zeros(2,2)
    V[1,1] = K
    V[1,2] = K_
    V[2,1] = K_
    V[2,2] = K_t
    return Normal(k_t, sqrt(K_t)), MvNormal(m,Matrix(Hermitian(V)))
end


function transit(m_f::MvNormal, m_s::MvNormal, func::Function, W::Matrix; α=0.001, β=2, κ=0)
    p_func, sigma, func_sigma = forwardMessageUT(m_f, func, α, β, κ)
    f, F = mean(m_f), cov(m_f) # f_t, F_t
    k, K = mean(m_s), cov(m_s) # k_{t+1}, K_{t+1}
    d = length(f)
    λ = utHyperparams(α, β, κ, d)
    Wᵐ, Wᶜ = utWeights(d, λ, α, β)
    P1 = F
    P21 = Wᶜ[1]*(sigma[1]-f)*(func_sigma[1]-mean(p_func))'
    for i=2:2*d+1
        P21 .+= Wᶜ[i]*(sigma[i]-f)*(func_sigma[i]-mean(p_func))'
    end
    P12 = P21'
    P2 = cov(p_func) + Matrix(Hermitian(inv(W)))
    P2_inv = Matrix(Hermitian(inv(P2)))
    k_t = f - P12*P2_inv*(mean(p_func) - k)
    Pa = P12*P2_inv
    Pb = Pa'
    K_ = Pa*K
    K_t = K_*Pb + (P1 - Pa*P21)
    m = [k;k_t]
    V = zeros(2*length(f), 2*length(f))
    V[1:length(f), 1:length(f)] = K
    V[1:length(f), length(f)+1:2*length(f)] = K_'
    V[length(f)+1:2*length(f), 1:length(f)] = K_
    V[length(f)+1:2*length(f), length(f)+1:2*length(f)] = K_t
    V = matrix_posdef_numeric_stable(V)
    return MvNormal(k_t, Matrix(Hermitian(K_t))), MvNormal(m,Matrix(Hermitian(V)))
end