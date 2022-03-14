export InferenceAlgorithm, BP, VMP, EP, CVI, EVMP
# Inference algortihms include 
# Implemented:
# Belief Propagation (BP) 
# Variational Message Passing (VMP) 
# Expectation Propagation (EP) 
# Conjugate-computation Variational Inference (CVI)
# To be implemented:
# Unscented Transform (UT)
# Taylor Series Linearization (TSL)
# Black Box Variational Inference (BBVI)
# Reparameterization Gradient Message Passing (RGMP)
# Extended Variational Message Passing (EVMP)
# Stochastic Variational Inference (SVI)
# Sequential Monte Carlo (SMC)
# Markov Chain Monte Carlo with Gibbs and Metropolis-Hastings (MCMC)
# Expectation Maximization (EM)

abstract type InferenceAlgorithm end

abstract type BP <: InferenceAlgorithm end
abstract type VMP <: InferenceAlgorithm end
abstract type EP <: InferenceAlgorithm end

mutable struct CVI <: InferenceAlgorithm
    optimizer
    num_iterations
end

struct EVMP <: VMP
    num_samples
end