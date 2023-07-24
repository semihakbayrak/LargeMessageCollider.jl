export MatrixDirichlet

# p(X) = ∏_{n=1:N} p(x_n), where p(x_i) = Dir(α_i), dim(α_i) = K
# X is K by N matrix.
# Parameter of MatrixDirichlet is K by N dimensional matrix α
# In other words, p(X[:,i]) = p(x_i) = Dir(α_i) = Dir(α[:,i])

struct MatrixDirichlet <: ContinuousMatrixDistribution
    alpha # K by N dim concentration parameter. In other words, N concatenated K dimensional concentrations
    function MatrixDirichlet(α)
        new(α)
    end
end

function Distributions.mean(p::MatrixDirichlet)
    p_list = matrix2list_dirichlet(p) # N-elements vector of Dirichlet dists
    #mean_list = map(x -> mean(x), p_list) # find means for each Dirichlet dist. N-elements of vector comprising K-elements of vectors
    mean_list = mean.(p_list) # find means for each Dirichlet dist. N-elements of vector comprising K-elements of vectors
    return hcat(mean_list...) # Concatenation along dim 2 to create K by N matrix
end

function Distributions.var(p::MatrixDirichlet)
    p_list = matrix2list_dirichlet(p) # N-elements vector of Dirichlet dists
    #var_list = map(x -> var(x), p_list) # find vars for each Dirichlet dist. N-elements of vector comprising K-elements of vectors
    var_list = var.(p_list) # find vars for each Dirichlet dist. N-elements of vector comprising K-elements of vectors
    return hcat(var_list...) # Concatenation along dim 2 to create K by N matrix
end

function Distributions._rand!(rng::AbstractRNG, p::MatrixDirichlet, A::AbstractMatrix)
    p_list = matrix2list_dirichlet(p) # N-elements vector of Dirichlet dists
    sample_list = rand.(p_list) # draw samples from each Dirichlet dist. N-elements of vector comprising K-elements of vectors
    A .= hcat(sample_list...) # Concatenation along dim 2 to create K by N matrix
end

function Distributions._logpdf(p::MatrixDirichlet, X::AbstractMatrix{<:Real})
    p_list = matrix2list_dirichlet(p) # N-elements vector of Dirichlet dists
    x_list = [X[:,i] for i in axes(X,2)] # N-elements of vector comprising K-elements of vectors
    logpdf_list = logpdf.(p_list, x_list)
    reduce(+, logpdf_list)
end

function Distributions.entropy(p::MatrixDirichlet)
    p_list = matrix2list_dirichlet(p) # N-elements vector of Dirichlet dists
    entropy_list = entropy.(p_list)
    reduce(+, entropy_list)
end

function Distributions.size(p::MatrixDirichlet)
    return size(p.alpha)
end
