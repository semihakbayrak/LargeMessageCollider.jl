export InferenceAlgorithm, BP, VMP, EP, TSL, UT, EVMP, CVI, FEgradient
# Inference algortihms include 
# Implemented:
# Belief Propagation (BP) 
# Variational Message Passing (VMP) 
# Expectation Propagation (EP) 
# Taylor Series Linearization (TSL)
# Unscented Transform (UT)
# Conjugate-computation Variational Inference (CVI) partially
# Extended Variational Message Passing (EVMP)
# Black Box Variational Inference (BBVI) partially
# Reparameterization Gradient Message Passing (RGMP) partially
# To be implemented:
# Stochastic Variational Inference (SVI)
# Laplace Approximation
# Sequential Monte Carlo (SMC)
# Markov Chain Monte Carlo with Gibbs and Metropolis-Hastings (MCMC)
# Stochastic Gradient Langevin Dynamics (SGLD)
# Expectation Maximization (EM)

abstract type InferenceAlgorithm end

abstract type BP <: InferenceAlgorithm end
abstract type VMP <: InferenceAlgorithm end
abstract type EP <: InferenceAlgorithm end

struct TSL <: InferenceAlgorithm end

struct UT <: InferenceAlgorithm end

struct EVMP <: VMP
    num_samples
end

mutable struct CVI <: InferenceAlgorithm
    optimizer
    num_iterations
end

# FEgradient algorithm enables us to use noisy free energy gradients calculated with RGMP and BBVI.
# It is possible estimate gradients (as it is a collection of partial derivatives) by the combination of RGMP and BBVI.
mutable struct FEgradient <: InferenceAlgorithm
    optimizer
    num_iterations
end