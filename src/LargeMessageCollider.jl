module LargeMessageCollider

using Random
using LinearAlgebra
using Distributions
using Zygote
using ForwardDiff
using SpecialFunctions
using StatsFuns
using StatsBase
using Flux.Optimise

import Base.*
import Base.+
import Base.-
import Base./
import Base.convert
import LinearAlgebra.\
import Distributions.logpdf
import Distributions.pdf

include("utilities.jl")
include("info_measure.jl")
include("distributions/canonical.jl")
include("distributions/sample_list.jl")
include("distributions/expectations.jl")
include("inference/inference_algorithms.jl")
include("inference/inference_rules.jl")
include("inference/bp.jl")
include("inference/vmp.jl")
include("inference/evmp.jl")
include("inference/cvi.jl")
include("inference/ep.jl")

end
