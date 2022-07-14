export poisson, normal, mvnormal, categorical, bernoulli, transit
#-------------------
# Gaussian BP rules
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

function (+)(p::Canonical, q::MvNormal)
    if p.dist != MvNormal
        error("Canonical distribution does not belong to MvNormal")
    else
        m_q = mean(q)
        V_q = cov(q)
        W_q = inv(V_q)
        ξ_q = W_q*m_q
        d = length(m_q)

        ξ_p = p.η[1:d]
        W_p = -2 * reshape(p.η[d+1:end],(d,d)) # Highly probable that p is improper dist so no Matrix(Hermitian())
        
        S = inv(W_p+W_q)
        W = Matrix(Hermitian(W_p*S*W_q))
        ξ = W_p*S*ξ_q + W_q*S*ξ_p
        η = [ξ;vec(-0.5*W)]
        return Canonical(MvNormal, η)
    end
end

(+)(p::MvNormal, q::Canonical) = q + p

function (-)(p::Canonical, q::MvNormal)
    if p.dist != MvNormal
        error("Canonical distribution does not belong to MvNormal")
    else
        m_q = mean(q)
        V_q = cov(q)
        W_q = inv(V_q)
        ξ_q = W_q*m_q
        d = length(m_q)

        ξ_p = p.η[1:d]
        W_p = -2 * reshape(p.η[d+1:end],(d,d)) # Highly probable that p is improper dist so no Matrix(Hermitian())
        
        S = inv(W_p+W_q)
        W = Matrix(Hermitian(W_p*S*W_q))
        ξ = W_q*S*ξ_p - W_p*S*ξ_q
        η = [ξ;vec(-0.5*W)]
        return Canonical(MvNormal, η)
    end
end

function (-)(q::MvNormal, p::Canonical)
    if p.dist != MvNormal
        error("Canonical distribution does not belong to MvNormal")
    else
        m_q = mean(q)
        V_q = cov(q)
        W_q = inv(V_q)
        ξ_q = W_q*m_q
        d = length(m_q)

        ξ_p = p.η[1:d]
        W_p = -2 * reshape(p.η[d+1:end],(d,d)) # Highly probable that p is improper dist so no Matrix(Hermitian())
        
        S = inv(W_p+W_q)
        W = Matrix(Hermitian(W_p*S*W_q))
        ξ = W_p*S*ξ_q - W_q*S*ξ_p
        η = [ξ;vec(-0.5*W)]
        return Canonical(MvNormal, η)
    end
end

# BP rule on Gaussian node with Gamma prec, clamped mean or out
normal(x::Nothing, μ::Real, τ::Gamma) = Student(μ, sqrt(mean(τ)), 2*shape(τ))
normal(x::Real, μ::Nothing, τ::Gamma) = Student(x, sqrt(mean(τ)), 2*shape(τ))


#-------------------
# Conditioning to observations
#-------------------
poisson(x::Int, λ::Nothing) = convert(Gamma,[x,-1.],check_args=false)
# normal is parameterized with mean and precision
normal(x::Real, μ::Real, τ::Nothing) = convert(Gamma,[0.5,x*μ - μ^2/2 - x^2/2],check_args=false)
normal(x::Real, μ::Nothing, τ::Real) = convert(Normal,[x*τ,-τ/2])
categorical(x::Vector, p::Nothing) = Dirichlet(x .+ 1)
bernoulli(x::Real, p::Nothing) = Beta(x+1,2-x)


#-------------------
# State transition nodes to calculate smoothing and joint distributions
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