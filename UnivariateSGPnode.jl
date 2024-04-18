include("gphelper.jl")
include("ut_approx.jl")
using ReactiveMP,RxInfer,GraphPPL
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using Zygote, Optim, ForwardDiff
using KernelFunctions, LoopVectorization
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: WishartFast, approximate_kernel_expectation, approximate_meancov

function ReactiveMP.approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::Real, P::Real)

    weights = ReactiveMP.getweights(method, m, P)
    points  = ReactiveMP.getpoints(method, m, P)

    gbar = g(m) - g(m)
    foreach(zip(weights, points)) do (weight, point)
        gbar += weight * g(point)
    end

    return gbar
end

function ReactiveMP.approximate_kernel_expectation(method::GenUnscented, g::Function, q::D) where {D}
    return approximate_expectation(method ,q, g)
end

function ReactiveMP.prod(::GenericProd, left::UnivariateGaussianDistributionsFamily, right::ContinuousUnivariateLogPdf) 
    m,v = approximate_meancov(ghcubature(21),(x) -> exp(right.logpdf(x)),left)
    if isnan(m) || isnan(v) || isinf(entropy(NormalMeanVariance(m,v)))
        return left 
    else
        return NormalMeanVariance(m,v)
    end
end
function ReactiveMP.prod(::GenericProd, left::ContinuousUnivariateLogPdf, right::UnivariateGaussianDistributionsFamily) 
    m,v = approximate_meancov(ghcubature(21),(x) -> exp(left.logpdf(x)),right)
    if isnan(m) || isnan(v)
        return right
    else
        return NormalMeanVariance(m,v)
    end
end

## specify GP node 
struct GPTransition end 

@node GPTransition Stochastic [ out, in, v , w] ## out: x_t , in: x_{t-1},  v: transformed-inducing points , w: precision of process noise 

## out rule 
@rule GPTransition(:out, Marginalisation) (q_in::UnivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::GPTransitionMeta,) = begin
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    μ_v = mean(q_v)
    Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel, [x], Xu) , q_in)
    return NormalMeanPrecision((Ψ1 * μ_v)[1], mean(q_w))
end
@rule GPTransition(:out, Marginalisation) (q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::GPTransitionMeta,) = begin
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    μ_v = mean(q_v)
    μ_in = mean(q_in)
    Ψ1 = kernelmatrix(kernel, [μ_in], Xu) 
    return NormalMeanPrecision((Ψ1 * μ_v)[1], mean(q_w))
end

## in rule 
@rule GPTransition(:in, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::GPTransitionMeta) = begin
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    A = (x) -> kernelmatrix(kernel,[x],[x]) .- kernelmatrix(kernel,[x],Xu) * Kuu_inverse * kernelmatrix(kernel,Xu,[x])
    B = (x) -> kernelmatrix(kernel, [x], Xu)
    
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)

    log_backwardmess = (x) -> -0.5 * A(x)[1] * w_bar + w_bar * μ_y * (B(x) * μ_v)[1] - 0.5 * w_bar * tr((μ_v * μ_v' + Σ_v) * B(x)' * B(x))
    return ContinuousUnivariateLogPdf(log_backwardmess)
end

@rule GPTransition(:in, Marginalisation) (q_out::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::GPTransitionMeta) = begin
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    A = (x) -> kernelmatrix(kernel,[x],[x]) .- kernelmatrix(kernel,[x],Xu) * Kuu_inverse * kernelmatrix(kernel,Xu,[x])
    B = (x) -> kernelmatrix(kernel, [x], Xu)
    
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)

    log_backwardmess = (x) -> -0.5 * A(x)[1] * w_bar + w_bar * μ_y * (B(x) * μ_v)[1] - 0.5 * w_bar * tr((μ_v * μ_v' + Σ_v) * B(x)' * B(x))
    return ContinuousUnivariateLogPdf(log_backwardmess)
end

## v rule 
@rule GPTransition(:v, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateNormalDistributionsFamily, q_w::Any,meta::GPTransitionMeta) = begin
    w = mean(q_w)
    μ_y = mean(q_out)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)

    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) + 1e-7*I 

    ξ_v = μ_y * w * Ψ1' #weighted-mean
    W_v = w * Ψ2  #precision 
    return MvNormalWeightedMeanPrecision(vcat(ξ_v...), W_v)
end

@rule GPTransition(:v, Marginalisation) (q_out::PointMass, q_in::PointMass, q_w::Any,meta::GPTransitionMeta) = begin
    w = mean(q_w)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)

    Ψ1 = kernelmatrix(kernel, [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel, Xu, [μ_in]) * kernelmatrix(kernel, [μ_in], Xu) + 1e-7*I 

    ξ_v = μ_y * w * Ψ1' #weighted-mean
    W_v = w * Ψ2  #precision 
    return MvNormalWeightedMeanPrecision(vcat(ξ_v...), W_v)
end

@rule GPTransition(:v, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_w::GammaShapeRate, meta::GPTransitionMeta) = begin 
    w = mean(q_w)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)

    Ψ1 = kernelmatrix(kernel, [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel, Xu, [μ_in]) * kernelmatrix(kernel, [μ_in], Xu)

    ξ_v = μ_y * w * Ψ1' #weighted-mean
    W_v = w * Ψ2  #precision 
    return MvNormalWeightedMeanPrecision(vcat(ξ_v...), W_v)
end

## w rule 
@rule GPTransition(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily,q_v::MultivariateNormalDistributionsFamily,meta::GPTransitionMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], [x]),q_in)[]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) + 1e-7*I 
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end

@rule GPTransition(:w, Marginalisation) (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,meta::GPTransitionMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)
    Ψ0 =  getindex(kernelmatrix(kernel, [μ_in], [μ_in]),1)
    Ψ1 =  kernelmatrix(kernel, [μ_in], Xu)
    Ψ2 =  kernelmatrix(kernel, Xu, [μ_in]) * kernelmatrix(kernel, [μ_in], Xu)
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end

@rule GPTransition(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass,q_v::MultivariateNormalDistributionsFamily,meta::GPTransitionMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)
    Ψ0 = getindex(kernelmatrix(kernel, [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel, [μ_in], Xu)
    Ψ2 =  kernelmatrix(kernel, Xu, [μ_in]) * kernelmatrix(kernel, [μ_in], Xu)
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end

## average energy 
@average_energy GPTransition (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate, meta::GPTransitionMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], [x]),q_in)[]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) + 1e-7*I  
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy GPTransition (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate, meta::GPTransitionMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)
    Ψ0 = getindex(kernelmatrix(kernel, [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel, [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel, Xu, [μ_in]) * kernelmatrix(kernel, [μ_in], Xu)

    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    Ax = getindex(kernelmatrix(kernel, [μ_in], [μ_in]) - kernelmatrix(kernel, [μ_in], Xu) * Kuu_inverse * kernelmatrix(kernel, Xu, [μ_in]),1)
    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) #- 0.5 * Ax
end

@average_energy GPTransition (q_out::UnivariateNormalDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate, meta::GPTransitionMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)
    Ψ0 = getindex(kernelmatrix(kernel, [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel, [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel, Xu, [μ_in]) * kernelmatrix(kernel, [μ_in], Xu)

    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy GPTransition (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass, meta::GPTransitionMeta,) = begin
    w_bar = mean(q_w)
    E_logw = log(w_bar)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], [x]),q_in)[]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) + 1e-7*I 
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end