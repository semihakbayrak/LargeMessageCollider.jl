export approximateMarginal!
# Conjugate-computation Variational Inference by Khan and Lin for nonconjugate components
# https://arxiv.org/pdf/1703.04265.pdf

# Univariate Normal incoming message
function approximateMarginal!(algo::F1, f::F2, out::T, in::Normal) where {F1<:CVI, F2<:Union{Nothing,Function}, T<:Distribution}
    η = convert(Canonical,in).η
    λ = deepcopy(η)
    q = convert(Normal,λ)

    logp_nc = (z) -> logpdf(out,z)
    if F2 <: Function
        logp_nc = (z) -> logpdf(out,f(z))
    end
    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.derivative(df_m,z)

    for i=1:algo.num_iterations
        z_s = rand(q)
        df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(q)
        df_μ2 = df_v(z_s)
        ∇f = [df_μ1, df_μ2]
        λ_old = deepcopy(λ)
        ∇ = λ .- η .- ∇f
        update!(algo.optimizer,λ,∇)
        try
            q = convert(Normal,λ)
        catch
            q = convert(Normal,λ_old)
        end
    end
    return q
end

# Multivariate Normal incoming message
function approximateMarginal!(algo::F1, f::F2, out::T, in::MvNormal) where {F1<:CVI, F2<:Union{Nothing,Function}, T<:Distribution}
    η = convert(Canonical,in).η
    λ = deepcopy(η)
    q = convert(MvNormal,λ)

    logp_nc = (z) -> logpdf(out,z)
    if F2 <: Function
        logp_nc = (z) -> logpdf(out,f(z))
    end
    df_m(z) = ForwardDiff.gradient(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.jacobian(df_m,z)

    for i=1:algo.num_iterations
        z_s = rand(q)
        df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(q)
        df_μ2 = df_v(z_s)
        ∇f = [df_μ1; vec(df_μ2)]
        λ_old = deepcopy(λ)
        ∇ = λ .- η .- ∇f
        update!(algo.optimizer,λ,∇)
        try
            q = convert(MvNormal,λ)
        catch
            q = convert(MvNormal,λ_old)
        end
    end
    return q
end

# Exponential Family of distributions in general
function approximateMarginal!(algo::F1, f::F2, out::T1, in::T2) where {F1<:CVI, F2<:Union{Nothing,Function}, T1<:Distribution, T2<:Union{Gamma, InverseGamma, Beta, Bernoulli, Dirichlet, Categorical, Wishart}}
    canonical_in = convert(Canonical,in)
    η = canonical_in.η
    λ = deepcopy(η)
    q = convert(T2,λ)

    logp_nc = (z) -> logpdf(out,z)
    if F2 <: Function
        logp_nc = (z) -> logpdf(out,f(z))
    end
    A(η) = canonical_in.A(η)
    gradA(η) = A'(η)
    Fisher(η) = ForwardDiff.jacobian(gradA,η)

    for i=1:algo.num_iterations
        z_s = rand(q)
        q_canonical = convert(Canonical,q)
        logq(λ) = log(q_canonical.h(z_s)) + transpose(λ)*q_canonical.T(z_s) - q_canonical.A(λ)
        ∇logq = logq'(λ)
        ∇f = Fisher(λ)\(logp_nc(z_s).*∇logq)
        λ_old = deepcopy(λ)
        ∇ = λ .- η .- ∇f
        update!(algo.optimizer,λ,∇)
        try
            q = convert(T2,λ)
        catch
            q = convert(T2,λ_old)
        end
    end
    return q
end