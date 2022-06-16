export approximateMarginal!, forwardMessage
# Linearization of non-linear functions using Taylor series expansion.
# For details, check Sarkka's "Bayesian Filtering and Smoothing"

function forwardMessage(algo::F, f::Function, in::Normal) where F<:TSL
    m = f(mean(in))
    h = ForwardDiff.derivative(f,mean(in))
    if length(m) == 1
        return Normal(m, sqrt(h^2*var(in)))
    else
        return MvNormal(m, Matrix(Hermitian(var(in)*h*h'+diagm(0=>1e-10*ones(length(m))))))
    end
end

function forwardMessage(algo::F, f::Function, in::MvNormal) where F<:TSL
    m = f(mean(in))
    if length(m) == 1
        h = transpose(ForwardDiff.gradient(f,mean(in)))
        return Normal(m,sqrt(h*cov(in)*h'))
    else
        h = ForwardDiff.jacobian(f,mean(in))
        return MvNormal(m, Matrix(Hermitian(h*cov(in)*h'+diagm(0=>1e-10*ones(length(m))))))
    end
end

function approximateMarginal!(algo::F, f::Function, out::Normal, in::Normal) where F<:TSL
    p, r = var(in), var(out)
    h = ForwardDiff.derivative(f,mean(in))
    v = mean(out) - f(mean(in))
    s = h^2*p + r
    k = p*h/s
    m = mean(in) + k*v
    σ = sqrt(p - s*k^2)
    q = Normal(m,σ)
    return q
end

function approximateMarginal!(algo::F, f::Function, out::Normal, in::MvNormal) where F<:TSL
    P, r = cov(in), var(out)
    h = transpose(ForwardDiff.gradient(f,mean(in)))
    v = mean(out) - f(mean(in))
    s = h*P*h' + r
    k = P*h'./s
    m = mean(in) + k*v
    V = P - s*k*k'
    q = MvNormal(m,V)
    return q
end

function approximateMarginal!(algo::F, f::Function, out::MvNormal, in::Normal) where F<:TSL
    p, R = var(in), cov(out)
    h = ForwardDiff.derivative(f,mean(in))
    v = mean(out) - f(mean(in))
    S = p*h*h' + R
    k = p*h'*inv(S)
    m = mean(in) + k*v
    σ = sqrt(p - k*S*k')
    q = Normal(m,σ)
    return q
end

function approximateMarginal!(algo::F, f::Function, out::MvNormal, in::MvNormal) where F<:TSL
    P, R = cov(in), cov(out)
    H = ForwardDiff.jacobian(f,mean(in))
    v = mean(out) - f(mean(in))
    S = Matrix(Hermitian(H*P*H' + R)+diagm(0=>1e-10*ones(length(mean(out)))))
    K = P*H'*inv(S)
    m = mean(in) + K*v
    V = P - K*S*K'
    q = MvNormal(m,V)
    return q
end