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
include("distributions/matrix_dirichlet.jl")
include("info_measure.jl")
include("distributions/canonical.jl")
include("distributions/sample_list.jl")
include("distributions/student.jl")
include("distributions/expectations.jl")
include("inference/inference_algorithms.jl")
include("inference/inference_rules.jl")
include("inference/bp.jl")
include("inference/vmp.jl")
include("inference/tsl.jl")
include("inference/ut.jl")
include("inference/evmp.jl")
include("inference/cvi.jl")
include("inference/fegradient.jl")
include("inference/ep.jl")

end
