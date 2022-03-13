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
include("optimizer.jl")
include("inference/cvi.jl")


end
