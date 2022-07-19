export approximateMarginal!, forwardMessage, transit
# Linearization of non-linear functions using Taylor series expansion.
# For details, check Sarkka's "Bayesian Filtering and Smoothing"

function forwardMessage(algo::F, f::Function, in::Normal) where F<:TSL
    m = f(mean(in))
    h = ForwardDiff.derivative(f,mean(in))
    if length(m) == 1
        return Normal(m, sqrt(h^2*var(in))), h
    else
        return MvNormal(m, Matrix(Hermitian(var(in)*h*h'+diagm(0=>1e-10*ones(length(m)))))), h
    end
end

function forwardMessage(algo::F, f::Function, in::MvNormal) where F<:TSL
    m = f(mean(in))
    if length(m) == 1
        h = transpose(ForwardDiff.gradient(f,mean(in)))
        return Normal(m,sqrt(h*cov(in)*h')), h
    else
        h = ForwardDiff.jacobian(f,mean(in))
        return MvNormal(m, Matrix(Hermitian(h*cov(in)*h'+diagm(0=>1e-10*ones(length(m)))))), h
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
    V = Matrix(Hermitian(P - s*k*k'))
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
    V = Matrix(Hermitian(P - K*S*K'))
    q = MvNormal(m,V)
    return q
end

# transit function useful for Extended Kalman Smoothing
# p(x_{t+1}|x_{t}) = N(x_{t+1}; f(x_{t}),W^{-1})
# m_f is filtered belief of x_{t}, m_s is smoothed belief of x_{t+1}
# return m_s(x_t), p(x_{t+1},x_t|y_{1:T})
function transit(m_f::Normal, m_s::Normal, func::Function, a::Real, w::Real)
    f, F = mean(m_f), var(m_f) # f_t, F_t
    k, K = mean(m_s), var(m_s) # k_{t+1}, K_{t+1}
    P1, P21 = F, a*F
    P12, P2 = P21, a*P21 + 1/w
    P2_inv = 1/P2
    k_t = f - P12*P2_inv*(func(f) - k)
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


function transit(m_f::MvNormal, m_s::MvNormal, func::Function, A::Matrix, W::Matrix)
    f, F = mean(m_f), cov(m_f) # f_t, F_t
    k, K = mean(m_s), cov(m_s) # k_{t+1}, K_{t+1}
    P1, P21 = F, A*F
    P12, P2 = P21', P21*A' + Matrix(Hermitian(inv(W)))
    P2_inv = Matrix(Hermitian(inv(P2)))
    k_t = f - P12*P2_inv*(func(f) - k)
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