module LargeMessageCollider

using Random
using LinearAlgebra
using Distributions
using Zygote
using ForwardDiff
using SpecialFunctions
using StatsFuns
using StatsBase

import Base.*
import Base.+
import Base.-
import LinearAlgebra.\

include("utilities.jl")
include("info_measure.jl")
include("exp_family.jl")
include("sample_list.jl")
include("expectations.jl")
include("inference_rules.jl")
include("optimizer.jl")
include("cvi.jl")


end
