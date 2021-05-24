export Adam, ConvexAdam, opt_step!

mutable struct Adam
    η
    G
    dim
    function Adam(η::Float64, dim::Int)
        if dim == 1
            G = 0
        else
            G = zeros(dim,dim)
        end
        new(η,G,dim)
    end
end

mutable struct ConvexAdam
    η
    G
    dim
    function ConvexAdam(η::Float64, dim::Int)
        if dim == 1
            G = 0
        else
            G = zeros(dim,dim)
        end
        new(η,G,dim)
    end
end

function opt_step!(optimizer::Adam, param::Real, grad::Real)
    optimizer.G = optimizer.G + grad^2
    step_size = optimizer.η / (sqrt(optimizer.G)+1e-20)
    # update parameter with stochastic gradient descent
    new_param = param - step_size*grad
end

function opt_step!(optimizer::Adam, param::Vector, grad::Vector)
    optimizer.G = optimizer.G + diagm(0=>diag(grad*grad'))
    step_size = optimizer.η .* 1 ./ (diag(sqrt.(optimizer.G)) + 1e-20*ones(length(grad)))
    # update parameter with stochastic gradient descent
    new_param = param .- step_size .* grad
end

function opt_step!(optimizer::ConvexAdam, param::Real, grad::Real)
    optimizer.G = optimizer.G + grad^2
    step_size = optimizer.η / (sqrt(optimizer.G)+1e-20)
    # update parameter as convex combination
    new_param = (1-step_size)*param + step_size*grad
end

function opt_step!(optimizer::ConvexAdam, param::Vector, grad::Vector)
    optimizer.G = optimizer.G + diagm(0=>diag(grad*grad'))
    step_size = optimizer.η .* 1 ./ (diag(sqrt.(optimizer.G)) + 1e-20*ones(length(grad)))
    # update parameter as convex combination
    new_param = (1 .- step_size).*param .+ step_size .* grad
end
