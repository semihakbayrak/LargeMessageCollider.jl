# Conjugate-computation Variational Inference by Khan and Lin for nonconjugate components
# Estimate the gradients in mean-parameter space
# https://arxiv.org/pdf/1703.04265.pdf

export cvi, grad_cvi

function cvi(logp::Function, q::Normal, num_samples::Int = 1)
    dlogp_z(z) = logp'(z)
    d2logp_z2(z) = 0.5*dlogp_z'(z)
    z_s = rand(q, num_samples)
    dlogp_m = sum(dlogp_z.(z_s))/num_samples
    dlogp_v = sum(d2logp_z2.(z_s))/num_samples
    dlogp_μ1 = dlogp_m - 2*dlogp_v*mean(q)
    dlogp_μ2 = dlogp_v
    return [dlogp_μ1, dlogp_μ2]
end

function cvi(logp::Function, q::MvNormal, num_samples::Int = 1)
    d = length(mean(q))
    #dlogp_z(z) = logp'(z)
    dlogp_z(z) = ForwardDiff.gradient(logp,z)
    d2logp_z2(z) = 0.5*ForwardDiff.hessian(logp,z)
    dlogp_m, dlogp_v = zeros(d), zeros(d,d)
    for n=1:num_samples
        z_s = rand(q)
        dlogp_m += dlogp_z(z_s)
        dlogp_v += d2logp_z2(z_s)
    end
    dlogp_m = dlogp_m/num_samples
    dlogp_v = dlogp_v/num_samples
    dlogp_μ1 = dlogp_m - 2*dlogp_v*mean(q)
    dlogp_μ2 = dlogp_v
    return [dlogp_μ1; vec(dlogp_μ2)]
end

function grad_cvi(logp_nc::Function, p_c::Normal, q::Normal, num_samples::Int = 1)
    d = length(mean(q))
    samples = []
    # calculate expected gradient for logp_nc
    dlogp_z(z) = logp_nc'(z)
    d2logp_z2(z) = 0.5*dlogp_z'(z)
    z_s = rand(q, num_samples)
    dlogp_m = sum(dlogp_z.(z_s))/num_samples
    dlogp_v = sum(d2logp_z2.(z_s))/num_samples
    dlogp_μ1 = dlogp_m - 2*dlogp_v*mean(q)
    dlogp_μ2 = dlogp_v
    ∇_μ_logp_nc = [dlogp_μ1, dlogp_μ2]
    _, _, λ_old, _, _ = exp_family(q)
    _, _, η, _, _ = exp_family(p_c)
    ∇ = λ_old .- η .- ∇_μ_logp_nc
    return samples, ∇
end

function grad_cvi(logp_nc::Function, p_c::MvNormal, q::MvNormal, num_samples::Int = 1)
    d = length(mean(q))
    samples = []
    # calculate expected gradient for logp_nc
    dlogp_z(z) = ForwardDiff.gradient(logp_nc,z)
    d2logp_z2(z) = 0.5*ForwardDiff.hessian(logp_nc,z)
    dlogp_m, dlogp_v = zeros(d), zeros(d,d)
    for n=1:num_samples
        z_s = rand(q)
        push!(samples,z_s)
        dlogp_m += dlogp_z(z_s)
        dlogp_v += d2logp_z2(z_s)
    end
    dlogp_m = dlogp_m/num_samples
    dlogp_v = dlogp_v/num_samples
    dlogp_μ1 = dlogp_m - 2*dlogp_v*mean(q)
    dlogp_μ2 = dlogp_v
    ∇_μ_logp_nc = [dlogp_μ1; vec(dlogp_μ2)]
    _, _, λ_old, _, _ = exp_family(q)
    _, _, η, _, _ = exp_family(p_c)
    ∇ = λ_old .- η .- ∇_μ_logp_nc
    return samples, ∇
end
