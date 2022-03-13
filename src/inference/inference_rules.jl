# Belief Propagation and Variational Message Passing utilities
export VMP, collide, poisson, normal, mvnormal, transit, normalmix, categorical, gammadist

#-------------------
# Collision of beliefs
#-------------------
function collide(p::F...) where F<:Distribution
    q = convert(Canonical,p[1])
    η = q.η
    for dist in p[2:end]
        q = convert(Canonical,dist)
        η .+= q.η
    end
    return convert(F,η)
end

# For some reason Julia differentiates FullNormal and DiagNormal(diagonal covariance matrix)
# which restrain the generality of the above collider!
function collide(p::MvNormal, q::MvNormal)
    pc = convert(Canonical,p)
    qc = convert(Canonical,q)
    η = pc.η .+ qc.η
    return convert(MvNormal,η)
end

function collide(p::F, q::C; canonical=false) where F<:Distribution where C<:Canonical
    if F <: q.dist
        pc = convert(Canonical,p)
        η = pc.η .+ q.η
        res = Canonical(F,η)
        if canonical
            return res
        else
            return convert(F,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

collide(q::C, p::F; canonical=false) where {F<:Distribution, C<:Canonical} = collide(p,q,canonical=canonical)

function collide(p::C, q::C; canonical=true) where C<:Canonical
    if p.dist == q.dist
        res = Canonical(p.dist, p.η .+ q.η)
        if canonical
            return res
        else
            return convert(p.dist,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

function collide(p::F1, logq::Function; proposal::F2, num_samples::Int) where F1<:Distribution where F2<:Distribution
    samples = []
    unnorm_weights = []
    for n=1:num_samples
        sample = rand(proposal)
        push!(samples,sample)
        unnorm_weight = pdf(p,sample)*exp(logq(sample))/pdf(proposal,sample)
        push!(unnorm_weights,unnorm_weight)
    end
    return SampleList(samples, unnorm_weights), unnorm_weights
end

(*)(p::F...) where {F<:Distribution} = collide(p...)
(*)(p::MvNormal, q::MvNormal) = collide(p,q)
(*)(p::F, q::C) where {F<:Distribution, C<:Canonical} = collide(p,q)
(*)(q::C, p::F) where {F<:Distribution, C<:Canonical} = collide(p,q)
(*)(p::C, q::C) where C<:Canonical = collide(p,q)

#-------------------
# Division of beliefs
#-------------------
function divide(p::F, q::F; canonical=true) where F<:Distribution
    pc = convert(Canonical,p)
    qc = convert(Canonical,q)
    res = Canonical(F,pc.η.-qc.η)
    if canonical
        return res
    else
        return convert(F,res)
    end
end

# For some reason Julia differentiates FullNormal and DiagNormal(diagonal covariance matrix)
# which restrain the generality of the above collider!
function divide(p::MvNormal, q::MvNormal; canonical=true)
    pc = convert(Canonical,p)
    qc = convert(Canonical,q)
    res = Canonical(MvNormal,pc.η.-qc.η)
    if canonical
        return res
    else
        return convert(MvNormal,res)
    end
end

function divide(p::F, q::C; canonical=true) where F<:Distribution where C<:Canonical
    if F <: q.dist
        pc = convert(Canonical,p)
        η = pc.η .- q.η
        res = Canonical(F,η)
        if canonical
            return res
        else
            return convert(F,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

function divide(q::C, p::F; canonical=true) where F<:Distribution where C<:Canonical
    if F <: q.dist
        pc = convert(Canonical,p)
        η = q.η .- pc.η
        res = Canonical(F,η)
        if canonical
            return res
        else
            return convert(F,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

function divide(p::C, q::C; canonical=true) where C<:Canonical
    if p.dist == q.dist
        res = Canonical(p.dist, p.η .- q.η)
        if canonical
            return res
        else
            return convert(p.dist,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

(/)(p::F,q::F) where {F<:Distribution} = divide(p,q)
(/)(p::MvNormal, q::MvNormal) = divide(p,q)
(/)(p::F, q::C) where {F<:Distribution, C<:Canonical} = divide(p,q)
(/)(q::C, p::F) where {F<:Distribution, C<:Canonical} = divide(q,p)
(/)(p::C, q::C) where C<:Canonical = divide(p,q)

#-------------------
# Gaussian inference rules
#-------------------
(*)(p::Normal, a::Real) = Normal(a*mean(p),sqrt(a^2*var(p)))
(*)(a::Real, p::Normal) = Normal(a*mean(p),sqrt(a^2*var(p)))
(*)(p::Normal, a::Int) = Normal(a*mean(p),sqrt(a^2*var(p)))
(*)(a::Int, p::Normal) = Normal(a*mean(p),sqrt(a^2*var(p)))
(*)(A::Matrix, p::MvNormal) = if size(A)[1] != 1 MvNormal(A*mean(p),Matrix(Hermitian(A*cov(p)*A'))) else Normal((A*mean(p))[1],sqrt((A*cov(p)*A')[1])) end
(*)(A::Vector, p::Normal) = MvNormal(A*mean(p),Matrix(Hermitian(A*var(p)*A'+ 1e-15*diagm(0=>ones(length(A))))))
(+)(p::Normal, q::Normal) = Normal(mean(p)+mean(q),sqrt(var(p)+var(q)))
(+)(p::Normal, a::Real) = Normal(mean(p)+a,sqrt(var(p)))
(+)(p::Normal, a::Int) = Normal(mean(p)+a,sqrt(var(p)))
(+)(a::Real, p::Normal) = Normal(mean(p)+a,sqrt(var(p)))
(+)(a::Int, p::Normal) = Normal(mean(p)+a,sqrt(var(p)))
(+)(p::MvNormal, q::MvNormal) = MvNormal(mean(p)+mean(q),cov(p)+cov(q))
(+)(p::MvNormal, a::Vector) = MvNormal(mean(p)+a,cov(p))
(-)(p::Normal, q::Normal) = Normal(mean(p)-mean(q),sqrt(var(p)+var(q)))
(-)(p::Normal, a::Real) = Normal(mean(p)-a,sqrt(var(p)))
(-)(p::Normal, a::Int) = Normal(mean(p)-a,sqrt(var(p)))
(-)(a::Real, p::Normal) = Normal(a-mean(p),sqrt(var(p)))
(-)(a::Int, p::Normal) = Normal(a-mean(p),sqrt(var(p)))
(-)(p::MvNormal, q::MvNormal) = MvNormal(mean(p)-mean(q),cov(p)+cov(q))
(-)(p::MvNormal, a::Vector) = MvNormal(mean(p)-a,cov(p))
(-)(a::Vector, p::MvNormal) = MvNormal(a-mean(p),cov(p))
(\)(a::Real, p::Normal) = Normal(mean(p)/a,sqrt(var(p)/(a^2)))

function (\)(A::Matrix, p::MvNormal)
    Wy = inv(cov(p))
    if size(A)[1] >= size(A)[2]
        W = Matrix(Hermitian(A' * Wy * A))
        V = Matrix(Hermitian(inv(W)))
        m = V*A'*Wy*mean(p)
        return MvNormal(m,V)
    else
        W = A' * Wy * A
        ξ = A'*Wy*mean(p)
        return Canonical(MvNormal,[ξ;vec(-0.5*W)])
    end
end

function (\)(A::Matrix, p::Normal)
    Wy = 1/var(p)
    W = A' * Wy * A
    ξ = A'*Wy*mean(p)
    # It is weird that [ξ;vec(-0.5*W)] gives error altough it is exactly same with η
    η = vec([ξ;vec(-0.5*W)])
    return Canonical(MvNormal,η)
end

function (\)(A::Vector, p::MvNormal)
    Wy = inv(cov(p))
    W = (A' * Wy * A)[1]
    V = 1/W
    m = (V*A'*Wy*mean(p))[1]
    return Normal(m,V)
end


#-------------------
# Conditioning to observations
#-------------------
poisson(x::Int, λ::Nothing) = exp_family(Gamma,[x,-1])
# normal is parameterized with mean and precision
normal(x::Real, μ::Real, τ::Nothing) = exp_family(Gamma,[0.5,x*μ - μ^2/2 - x^2/2])
normal(x::Real, μ::Nothing, τ::Real) = exp_family(Normal,[x*τ,-τ/2])
categorical(x::Vector, p::Nothing) = Dirichlet(x .+ 1)

#-------------------
# VMP rules (mean-field assumption) for standard distributions
#-------------------
# normal is parameterized with mean and precision
normal(x::Real, μ::Distribution, τ::Nothing) = exp_family(Gamma,[0.5,x*mean(μ) - squaremean(μ)/2 - x^2/2])
normal(x::Distribution, μ::Distribution, τ::Nothing) = exp_family(Gamma,[0.5,mean(x)*mean(μ) - squaremean(μ)/2 - squaremean(x)/2])
normal(x::Real, μ::Nothing, τ::Distribution) = exp_family(Normal,[x*mean(τ),-mean(τ)/2])
normal(x::Distribution, μ::Nothing, τ::Distribution) = exp_family(Normal,[mean(x)*mean(τ),-mean(τ)/2])
normal(x::Nothing, μ::Distribution, τ::Distribution) = exp_family(Normal,[mean(μ)*mean(τ),-mean(τ)/2])
function mvnormal(x::Vector, μ::Distribution, τ::Nothing)
    k = length(x)
    V_inv = x*x' - x*mean(μ)' - mean(μ)*x' + squaremean(μ)
    η = [vec(-0.5*V_inv); 0.5]
    exp_family(Wishart,η,false)
end
function mvnormal(x::Distribution, μ::Distribution, τ::Nothing)
    k = length(mean(x))
    V_inv = squaremean(x) - mean(x)*mean(μ)' - mean(μ)*mean(x)' + squaremean(μ)
    η = [vec(-0.5*V_inv); 0.5]
    exp_family(Wishart,η,false)
end
function mvnormal(x::Vector, μ::Nothing, τ::Distribution)
    MvNormal(x,inv(mean(τ)))
end
function mvnormal(x::Distribution, μ::Nothing, τ::Distribution)
    MvNormal(mean(x),inv(mean(τ)))
end
function mvnormal(x::Nothing, μ::Distribution, τ::Distribution)
    MvNormal(mean(μ),inv(mean(τ)))
end
categorical(x::Nothing, p::Dirichlet) = exp_family(Categorical, logmean(p))
categorical(x::Categorical, p::Nothing) = Dirichlet(x.p .+ 1)
#gammadist(x::Real, α::Distribution, θ::Nothing) = InverseGamma(mean(α)-1,x,check_args=false)
gammadist(x::Real, α::Distribution, θ::Nothing) = Canonical(InverseGamma,[-mean(α),-x])

#-------------------
# State transition nodes to calculate joint distributions
# and messages towards process noise node (for LDS) or stochastic matrix (for HMM)
#-------------------
# p(x_{t+1}|x_{t}) = N(x_{t+1}; A*x_{t},W^{-1})
# m_f is filtered belief of x_{t}, m_s is smoothed belief of x_{t+1}
# return m_s(x_t), p(x_{t+1},x_t|y_{1:T})
function transit(m_f::Normal, m_s::Normal, a::Real, w::Real)
    f, F = mean(m_f), var(m_f) # f_t, F_t
    k, K = mean(m_s), var(m_s) # k_{t+1}, K_{t+1}
    P1, P21 = F, a*F
    P12, P2 = P21, a*P21 + 1/w
    P2_inv = 1/P2
    k_t = f - P12*P2_inv*(a*f - k)
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


function transit(m_f::MvNormal, m_s::MvNormal, A::Matrix, W::Matrix)
    f, F = mean(m_f), cov(m_f) # f_t, F_t
    k, K = mean(m_s), cov(m_s) # k_{t+1}, K_{t+1}
    P1, P21 = F, A*F
    P12, P2 = P21', P21*A' + Matrix(Hermitian(inv(W)))
    P2_inv = Matrix(Hermitian(inv(P2)))
    k_t = f - P12*P2_inv*(A*f - k)
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
    return MvNormal(k_t, Matrix(Hermitian(K_t))), MvNormal(m,Matrix(Hermitian(V)))
end

# Structured VMP message towards process noise precision
function transit(q1::Normal, q2::Normal, q21::MvNormal, a::Real, w::Nothing)
    α = 1.5
    β = 0.5*(squaremean(q2) + a^2*squaremean(q1) - 2*a*squaremean(q21)[1,2])
    θ = 1/β
    return Gamma(α,θ,check_args=false)
end

function transit(q1::MvNormal, q2::MvNormal, q21::MvNormal, A::Matrix, W::Nothing)
    k = length(mean(q1))
    V_inv = squaremean(q2) - (squaremean(q21)[1:k,k+1:2*k])*A' - A*(squaremean(q21)[k+1:2*k,1:k]) + A*squaremean(q1)*A'
    η = [vec(-0.5*V_inv); 0.5]
    return Canonical(Wishart,η)
end

#-------------------
# Gaussian Mixture Model likelihood related variational messages for mean-field assumption
#-------------------
# Scalar observation. Message for categorical choice variable. Mean-precision parameterization

function normalmix(y_n::Number, z::Nothing, qm_::Array{Normal{Float64}}, qw_::Array{Gamma{Float64}})
    p_vec = exp.(-0.5*log(2pi) .+ 0.5*logmean.(qw_)
            -0.5*mean.(qw_).*(y_n^2 .- 2*y_n*mean.(qm_) + squaremean.(qm_)))
    return Categorical(p_vec, check_args=false)
end

function normalmix(y_n::Number, z::Categorical, qm_::Nothing, qw_::Array{Gamma{Float64}})
    K = length(z.p)
    return Normal.(y_n*ones(K),sqrt.(1 ./ (mean.(qw_).*z.p)))
end

function normalmix(y_n::Number, z::Categorical, qm_::Array{Normal{Float64}}, qw_::Nothing)
    K = length(z.p)
    return exp_family.(Gamma,[[0.5*z.p[k],z.p[k]*(mean(qm_[k])*y_n - y_n^2/2 - squaremean(qm_[k])/2)] for k=1:K])
end

function normalmix(y_n::Nothing, z::Categorical, qm_::Array{Normal{Float64}}, qw_::Array{Gamma{Float64}})
    m = sum(z.p .* mean.(qm_))/sum(z.p)
    v = 1/sum(z.p .* mean.(qw_))
    return Normal(m,sqrt(v))
end

# Vector observation. Message for categorical choice variable. Mean-precision parameterization

function normalmix(y_n::Vector, z::Nothing, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
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

function normalmix(y_n::Vector, z::Categorical, qm_::Nothing, qw_::Array{F}) where F<:Wishart
    K = length(z.p)
    #message = Array{MvNormal}(undef, K)
    message = Array{Canonical}(undef, K)
    for k=1:K
        #message[k] = MvNormal(y_n,inv(z.p[k]*mean(qw_[k])))
        message[k] = Canonical(MvNormal,[z.p[k]*mean(qw_[k])*y_n;vec(-0.5*z.p[k]*mean(qw_[k]))])
    end
    return message
end

function normalmix(y_n::Vector, z::Categorical, qm_::Array{F}, qw_::Nothing) where F<:MvNormal
    K = length(z.p)
    return Canonical.(Wishart,[[vec(0.5*z.p[k]*(mean(qm_[k])*y_n' + y_n*mean(qm_[k])' - y_n*y_n' - squaremean(qm_[k]))); 0.5*z.p[k]] for k=1:K])
end

# Below normalmix rules are useful for Deep GMM
function normalmix(y_n::Nothing, z::Categorical, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
    m = sum(z.p .* mean.(qm_))./sum(z.p)
    V = Matrix(Hermitian(inv(sum(z.p .* mean.(qw_)))))
    return MvNormal(m,V)
end

function normalmix(y_n::MvNormal, z::Nothing, qm_::Array{F1}, qw_::Array{F2}) where F1<:MvNormal where F2<:Wishart
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

function normalmix(y_n::MvNormal, z::Categorical, qm_::Nothing, qw_::Array{F}) where F<:Wishart
    K = length(z.p)
    #message = Array{MvNormal}(undef, K)
    message = Array{Canonical}(undef, K)
    for k=1:K
        #message[k] = MvNormal(y_n,inv(z.p[k]*mean(qw_[k])))
        message[k] = Canonical(MvNormal,[z.p[k]*mean(qw_[k])*mean(y_n);vec(-0.5*z.p[k]*mean(qw_[k]))])
    end
    return message
end

function normalmix(y_n::MvNormal, z::Categorical, qm_::Array{F}, qw_::Nothing) where F<:MvNormal
    K = length(z.p)
    return Canonical.(Wishart,[[vec(0.5*z.p[k]*(mean(qm_[k])*mean(y_n)' + mean(y_n)*mean(qm_[k])' - squaremean(y_n) - squaremean(qm_[k]))); 0.5*z.p[k]] for k=1:K])
end
