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
