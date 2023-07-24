export inner_product, invdigamma, matrix_posdef_numeric_stable

inner_product(a::Vector, b::Vector) = a'*b
inner_product(A::Array, B::Array) = tr(B*A)

# approximation to inverse of digamma function https://math.stackexchange.com/questions/3164682/digamma-function-inverse-and-special-value
invdigamma(x) = 1/log(1+exp(-x))

function matrix_posdef_numeric_stable(X)
    d = size(X)[1]
    ϵ = 1e-15
    X_new = Matrix(Hermitian(X + diagm(0=>ϵ*ones(d))))
    while isposdef(X_new) == false
        ϵ = 10*ϵ
        X_new = Matrix(Hermitian(X + diagm(0=>ϵ*ones(d))))
        if ϵ >= 1e-1
             error("Matrix can't be transformed to a positive definite matrix")
        end
    end
    return X_new
end

# Adapted from https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
function logsumexp(x::Array)
    c = maximum(x)
    return c + log(sum(exp.(x .- c)))
end

normalize_prob_array(p::Array) = p./sum(p)
normalize_logprob_array(logp::Array) = exp.(logp.-logsumexp(logp)) # return normalized p

# MatrixDirichlet to Vector of Dirichlet dists
function matrix2list_dirichlet(p::MatrixDirichlet)
    α_list = [p.alpha[:,i] for i in 1:size(p.alpha,2)] # N-elements of vector comprising K-elements of vectors
    p_list = Dirichlet.(α_list) # N-elements vector of Dirichlet dists
    return p_list
end