# Collision and division of distributions
export collide, divide

#-------------------
# Collision of beliefs
#-------------------
function collide(p::F...) where F<:Distribution
    q = convert(Canonical,p[1])
    η = q.η
    for dist in p[2:end]
        q = convert(Canonical,dist)
        η .+= q.η
    end
    if F<:Categorical return convert(Categorical,η) elseif F<:MatrixDirichlet return convert(F,η,size(p[1])) else return convert(F,η) end # * Check below
end
# *: Categorical is encoded as DiscreteNonParametric in Distributions.jl, which has a different convert method and causes problems
# That is why convert(Categorical,η) is handled as a special case.
# MatrixDirichlet requires size so that the concentration matrix can be recovered with the proper shape.

# For some reason Julia differentiates FullNormal and DiagNormal(diagonal covariance matrix)
# which restrain the generality of the above collider!
function collide(p::MvNormal, q::MvNormal)
    pc = convert(Canonical,p)
    qc = convert(Canonical,q)
    η = pc.η .+ qc.η
    return convert(MvNormal,η)
end

function collide(p::F, q::C; canonical=false) where F<:Distribution where C<:Canonical
    if F <: q.dist
        pc = convert(Canonical,p)
        η = pc.η .+ q.η
        res = Canonical(F,η)
        if canonical
            return res
        else
            return convert(F,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

collide(q::C, p::F; canonical=false) where {F<:Distribution, C<:Canonical} = collide(p,q,canonical=canonical)

function collide(p::C, q::C; canonical=true) where C<:Canonical
    if (p.dist <: q.dist) || (q.dist <: p.dist)
        res = Canonical(p.dist, p.η .+ q.η)
        if canonical
            return res
        else
            return convert(p.dist,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

function collide(p::F1, logq::Function; proposal::F2, num_samples::Int) where F1<:Distribution where F2<:Distribution
    samples = []
    unnorm_weights = []
    for n=1:num_samples
        sample = rand(proposal)
        push!(samples,sample)
        unnorm_weight = pdf(p,sample)*exp(logq(sample))/pdf(proposal,sample)
        push!(unnorm_weights,unnorm_weight)
    end
    return SampleList(samples, unnorm_weights), unnorm_weights
end

(*)(p::F...) where {F<:Distribution} = collide(p...)
(*)(p::MvNormal, q::MvNormal) = collide(p,q)
(*)(p::F, q::C) where {F<:Distribution, C<:Canonical} = collide(p,q)
(*)(q::C, p::F) where {F<:Distribution, C<:Canonical} = collide(p,q)
(*)(p::C, q::C) where C<:Canonical = collide(p,q)

#-------------------
# Division of beliefs
#-------------------
function divide(p::F, q::F; canonical=true) where F<:Distribution
    pc = convert(Canonical,p)
    qc = convert(Canonical,q)
    res = Canonical(F,pc.η.-qc.η)
    if canonical
        return res
    else
        return convert(F,res)
    end
end

# For some reason Julia differentiates FullNormal and DiagNormal(diagonal covariance matrix)
# which restrain the generality of the above collider!
function divide(p::MvNormal, q::MvNormal; canonical=true)
    pc = convert(Canonical,p)
    qc = convert(Canonical,q)
    res = Canonical(MvNormal,pc.η.-qc.η)
    if canonical
        return res
    else
        return convert(MvNormal,res)
    end
end

function divide(p::F, q::C; canonical=true) where F<:Distribution where C<:Canonical
    if F <: q.dist
        pc = convert(Canonical,p)
        η = pc.η .- q.η
        res = Canonical(F,η)
        if canonical
            return res
        else
            return convert(F,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

function divide(q::C, p::F; canonical=true) where F<:Distribution where C<:Canonical
    if F <: q.dist
        pc = convert(Canonical,p)
        η = q.η .- pc.η
        res = Canonical(F,η)
        if canonical
            return res
        else
            return convert(F,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

function divide(p::C, q::C; canonical=true) where C<:Canonical
    if p.dist == q.dist
        res = Canonical(p.dist, p.η .- q.η)
        if canonical
            return res
        else
            return convert(p.dist,res)
        end
    else
        error("Nonconjugacy detected")
    end
end

(/)(p::F,q::F) where {F<:Distribution} = divide(p,q)
(/)(p::MvNormal, q::MvNormal) = divide(p,q)
(/)(p::F, q::C) where {F<:Distribution, C<:Canonical} = divide(p,q)
(/)(q::C, p::F) where {F<:Distribution, C<:Canonical} = divide(q,p)
(/)(p::C, q::C) where C<:Canonical = divide(p,q)
