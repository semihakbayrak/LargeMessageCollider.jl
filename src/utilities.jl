export inner_product, invdigamma

inner_product(a::Vector, b::Vector) = a'*b
inner_product(A::Array, B::Array) = tr(B*A)

# approximation to inverse of digamma function https://math.stackexchange.com/questions/3164682/digamma-function-inverse-and-special-value
invdigamma(x) = 1/log(1+exp(-x))
