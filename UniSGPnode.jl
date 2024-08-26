include("ut_approx.jl")
include("gp_helperfunction.jl")
using ReactiveMP,RxInfer,GraphPPL
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using Zygote, Optim, ForwardDiff
using KernelFunctions, LoopVectorization
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: WishartFast,approximate_kernel_expectation,  approximate_meancov

function ReactiveMP.approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::Real, P::Real)

    weights = ReactiveMP.getweights(method, m, P)
    points  = ReactiveMP.getpoints(method, m, P)

    gbar = g(m) - g(m)
    foreach(zip(weights, points)) do (weight, point)
        gbar += weight * g(point)
    end

    return gbar
end

function ReactiveMP.approximate_kernel_expectation(method::GenUnscented, g::Function, q::D) where {D <: UnivariateDistribution}
    return approximate_expectation(method ,q, g)
end

function ReactiveMP.prod(::GenericProd, left::UnivariateGaussianDistributionsFamily, right::ContinuousUnivariateLogPdf) 
    m,v = approximate_meancov(ghcubature(21),(x) -> exp(right.logpdf(x)),left)
    if isnan(m) || isnan(v)
        return left 
    else
        return NormalMeanVariance(m,clamp(v,1e-12,1e6))
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
;

struct UniSGP end 

@node UniSGP Stochastic [ out, in, v , w, θ] 

## out: x_t , in: x_{t-1},  
## v: transformed-inducing points , w: precision of process noise 
## θ: kernel hyper-parameters

## out rule 
@rule UniSGP(:out, Marginalisation) (q_in::UnivariateNormalDistributionsFamily, 
        q_v::MultivariateNormalDistributionsFamily,q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ) #get kernel hyperparameters => kernel(exp(θ))
    μ_v = mean(q_v)
    Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel(θ), [x], Xu) , q_in)
    return NormalMeanPrecision((Ψ1 * μ_v)[1], mean(q_w))
end

@rule UniSGP(:out, Marginalisation) (q_in::PointMass, 
        q_v::MultivariateNormalDistributionsFamily,q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ) #get kernel hyperparameters => kernel(θ)
    μ_v = mean(q_v)
    Ψ1 = kernelmatrix(kernel(θ), [mean(q_in)], Xu)
    return NormalMeanPrecision((Ψ1 * μ_v)[1], mean(q_w))
end

## in rule 
@rule UniSGP(:in, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, 
    q_v::MultivariateNormalDistributionsFamily,q_w::Any,q_θ::PointMass, meta::UniSGPMeta) = begin
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    θ = mean(q_θ)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    A = (x) -> kernelmatrix(kernel(θ),[x],[x]) .- kernelmatrix(kernel(θ),[x],Xu) * Kuu_inverse * kernelmatrix(kernel(θ),Xu,[x])
    B = (x) -> kernelmatrix(kernel(θ), [x], Xu)
    
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)

    log_backwardmess = (x) -> -0.5 * A(x)[1] * w_bar + w_bar * μ_y * (B(x) * μ_v)[1] - 0.5 * w_bar * tr((μ_v * μ_v' + Σ_v) * B(x)' * B(x))
    return ContinuousUnivariateLogPdf(log_backwardmess)
end

## v rule 
@rule UniSGP(:v, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, 
    q_in::UnivariateNormalDistributionsFamily, q_w::Any,q_θ::PointMass, meta::UniSGPMeta) = begin
    w = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)

    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 

    ξ_v = μ_y * w * Ψ1' #weighted-mean
    W_v = w * Ψ2  #precision 
    return MvNormalWeightedMeanPrecision(vcat(ξ_v...), W_v)
end

@rule UniSGP(:v, Marginalisation) (q_out::PointMass, q_in::PointMass, 
        q_w::Any,q_θ::PointMass,meta::UniSGPMeta) = begin
    w = mean(q_w)
    θ = mean(q_θ)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)

    Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7*I 

    ξ_v = μ_y * w * Ψ1' #weighted-mean
    W_v = w * Ψ2  #precision 
    return MvNormalWeightedMeanPrecision(vcat(ξ_v...), W_v)
end

@rule UniSGP(:v, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_w::GammaShapeRate, q_θ::PointMass, meta::UniSGPMeta) = begin 
    w = mean(q_w)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)

    Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7*I

    ξ_v = μ_y * w * Ψ1' #weighted-mean
    W_v = w * Ψ2  #precision 
    return MvNormalWeightedMeanPrecision(vcat(ξ_v...), W_v)
end

## w rule 
@rule UniSGP(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily,
            q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass,meta::UniSGPMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 

    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end

@rule UniSGP(:w, Marginalisation) (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    Ψ0 =  getindex(kernelmatrix(kernel(θ), [μ_in], [μ_in]),1)
    Ψ1 =  kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ2 =  kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7 * I
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end

@rule UniSGP(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    Ψ0 = getindex(kernelmatrix(kernel(θ), [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ2 =  kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7 * I
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end

## θ rule 
@rule UniSGP(:θ, Marginalisation) (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::GammaShapeRate, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
    Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7*I 
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) - tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule UniSGP(:θ, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::GammaShapeRate, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
    Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7*I 
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) - tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule UniSGP(:θ, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateNormalDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::GammaShapeRate, meta::UniSGPMeta) = begin 
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ_1 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ_2 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) - tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

## average_energy 
@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, 
        q_w::GammaShapeRate,q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy UniSGP (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate,
        q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    Ψ0 = getindex(kernelmatrix(kernel(θ), [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7*I

    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate, 
       q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    Ψ0 = getindex(kernelmatrix(kernel(θ), [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu)
    Ψ2 = kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-7*I

    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass, 
        q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = log(w_bar)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end