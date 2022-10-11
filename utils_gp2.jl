using ReactiveMP, GraphPPL, LinearAlgebra, Random, KernelFunctions
using DomainSets
import ReactiveMP: cholinv, logdet
import KernelFunctions: Kernel

"""
This file defines another GP node design for ReactiveMP. Everything in this file is inspired by Ismail's code 
"""


### GaussianProcess structure and node 
struct GaussianProcess 
    meanfunction
    kernelfunction
    finitemarginal
    testinput
    traininput 
    invKff  
end

@node GaussianProcess Stochastic [out, meanfunc, kernelfunc, θ]

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::PointMass, q_θ::UnivariateGaussianDistributionsFamily) = begin 
    kernelfunc = with_lengthscale(q_kernelfunc.point, mean(q_θ)) 
    return GaussianProcess(q_meanfunc.point,kernelfunc,nothing,nothing,nothing,nothing)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::PointMass, m_θ::UnivariateGaussianDistributionsFamily) = begin 
    kernelfunc = with_lengthscale(q_kernelfunc.point, mean(m_θ)) 
    return GaussianProcess(q_meanfunc.point,kernelfunc,nothing,nothing,nothing,nothing)
end

## θ here is 1-D, so it's drawn from univariate distribution 
@rule GaussianProcess(:θ, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass) = begin
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernfunc(x) = with_lengthscale(q_kernelfunc.point, x)
    y, Σ = mean_cov(q_out.finitemarginal)
    log_llh(x) = -1/2 * (y - meanf.(test))' * cholinv(kernelmatrix(kernfunc(x),test,test) + Diagonal(Σ) + 1e-3*diageye(length(test))) * (y- meanf.(test)) - 1/2 * logdet(kernelmatrix(kernfunc(x),test,test) + Diagonal(Σ) + 1e-3*diageye(length(test)))

    return ContinuousUnivariateLogPdf(log_llh)
end

####### Meta structure 
struct ProcessMeta
    index
    Kxx
    Kff
    Kfx
end

##### additional rule for NormalMeanPrecision 
@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Any, q_μ::GaussianProcess,meta::ProcessMeta) = begin
    m_right, cov_right = mean_cov(q_μ.finitemarginal)
    kernelf = q_μ.kernelfunction
    meanf   = q_μ.meanfunction
    test    = q_μ.testinput
    train   = q_μ.traininput
    mμ, vμ = predMVN_fast(q_μ,test,[train[meta.index]],m_right) #changed here
    vμ = clamp(vμ[1],1e-8,huge)
    θ = 2 / (var(q_out) + vμ[1] + abs2(mean(q_out) - mμ[1]))
    α = convert(typeof(θ), 1.5)
    return Gamma(α, θ)
end

###### important function 
function ReactiveMP.constvar(name::Symbol, fn::Function) 
    return ReactiveMP.ConstVariable(name, ReactiveMP.VariableIndividual(), PointMass(fn), of(Message(PointMass(fn), true, false)), 0)
end
function ReactiveMP.constvar(name::Symbol, kernel::Kernel ) 
    return ReactiveMP.ConstVariable(name, ReactiveMP.VariableIndividual(), PointMass(kernel), of(Message(PointMass(kernel), true, false)), 0)
end
#################
#### Some useful functions 
function make_multivariate_message(messages) ## function for concatinating messages
    m = mean.(messages) 
    v = Diagonal(var.(messages))
    return m,v
end

function predMVN_fast(gp::GaussianProcess,xtest,xtrain,y)      #function for computing FE and message on edge τ
    kernelfunc         = gp.kernelfunction 
    meanfunc           = gp.meanfunction
    Kxx                = kernelmatrix(kernelfunc,xtrain,xtrain)
    Kxf                = kernelmatrix(kernelfunc,xtrain,xtest)
    invKff             = gp.invKff 
    K                  = Kxx - Kxf*invKff*Kxf'
    m                  = meanfunc.(xtrain) + Kxf*invKff*(y-meanfunc.(xtest))
    
    return m,K
end

function predMVN(kernelfunc,meanfunc,xtrain,xtest,y,Σy)  # function for computing GP marginal 
    Kxx                = kernelmatrix(kernelfunc,xtrain,xtrain)
    Kfx                = kernelmatrix(kernelfunc,xtest,xtrain)
    Kff                = kernelmatrix(kernelfunc,xtest,xtest)
    K                  = Kff - Kfx*ReactiveMP.cholinv(Kxx+Σy)*Kfx'
    m                  = meanfunc.(xtest) + Kfx*ReactiveMP.cholinv(Kxx+Σy)*(y-meanfunc.(xtrain))
    
    return m,K
end
