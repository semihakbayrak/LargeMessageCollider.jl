export InferenceAlgorithm, BP, VMP, EP, TSL, EVMP, CVI
# Inference algortihms include 
# Implemented:
# Belief Propagation (BP) 
# Variational Message Passing (VMP) 
# Expectation Propagation (EP) 
# Taylor Series Linearization (TSL)
# Conjugate-computation Variational Inference (CVI)
# Extended Variational Message Passing (EVMP)
# To be implemented:
# Unscented Transform (UT)
# Black Box Variational Inference (BBVI)
# Reparameterization Gradient Message Passing (RGMP)
# Stochastic Variational Inference (SVI)
# Sequential Monte Carlo (SMC)
# Markov Chain Monte Carlo with Gibbs and Metropolis-Hastings (MCMC)
# Expectation Maximization (EM)

abstract type InferenceAlgorithm end

abstract type BP <: InferenceAlgorithm end
abstract type VMP <: InferenceAlgorithm end
abstract type EP <: InferenceAlgorithm end

struct TSL <: InferenceAlgorithm end

struct EVMP <: VMP
    num_samples
end

mutable struct CVI <: InferenceAlgorithm
    optimizer
    num_iterations
end