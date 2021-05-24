export inner_product

inner_product(a::Vector, b::Vector) = a'*b
inner_product(A::Array, B::Array) = tr(B*A)
