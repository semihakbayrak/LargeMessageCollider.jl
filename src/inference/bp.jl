export poisson, normal, mvnormal, categorical, bernoulli
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


#-------------------
# Conditioning to observations
#-------------------
poisson(x::Int, λ::Nothing) = convert(Gamma,[x,-1.],check_args=false)
# normal is parameterized with mean and precision
normal(x::Real, μ::Real, τ::Nothing) = convert(Gamma,[0.5,x*μ - μ^2/2 - x^2/2],check_args=false)
normal(x::Real, μ::Nothing, τ::Real) = convert(Normal,[x*τ,-τ/2])
categorical(x::Vector, p::Nothing) = Dirichlet(x .+ 1)
bernoulli(x::Real, p::Nothing) = Beta(x+1,2-x)