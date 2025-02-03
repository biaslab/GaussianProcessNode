include("../helper_functions/ut_approx.jl")
include("../helper_functions/gp_helperfunction.jl")
using ReactiveMP,RxInfer,GraphPPL
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using Zygote, Optim, ForwardDiff
using KernelFunctions, LoopVectorization
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: WishartFast,approximate_kernel_expectation,  approximate_meancov
import LinearAlgebra: mul!

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, m::Real, P::Real) where {K <: Array}
    weights = ReactiveMP.getweights(method, m, P)
    points  = ReactiveMP.getpoints(method, m, P)
    gbar .= 0
    foreach(zip(weights, points)) do (weight, point)
        gbar .+= weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, distribution::D) where {K <: Array, D <: UnivariateDistribution}
    return approximate_kernel_expectation!(gbar, method, g, mean(distribution), var(distribution))
end

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
        return NormalMeanVariance(m,v + 1e-6)
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

mutable struct BufferUniSGP{D,M} 
    qv      :: D 
    meta    :: M
end 


function ReactiveMP.prod(::GenericProd, left::NormalDistributionsFamily, right::BufferUniSGP)
    marginal_v = ReactiveMP.prod(GenericProd(),left,right.qv)
    right.meta.counter += 1
    if right.meta.counter == right.meta.N 
        μ_v, Σ_v = mean_cov(marginal_v)
        mul!(Σ_v,μ_v,μ_v',1,1) 
        Uv = fastcholesky!(Σ_v).U
        right.meta.Uv = Uv
        right.meta.counter = 0
    end
    return marginal_v
end


struct UniSGP end 

@node UniSGP Stochastic [ out, in, v , w, θ] 

## out: x_t , in: x_{t-1},  
## v: transformed-inducing points , w: precision of process noise 
## θ: kernel hyper-parameters

## out rule 
@rule UniSGP(:out, Marginalisation) (q_in::UnivariateNormalDistributionsFamily, 
        q_v::MultivariateNormalDistributionsFamily,q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    kernel = getKernel(meta)
    θ = mean(q_θ) 
    μ_v = mean(q_v)
    # Ψ1_trans = similar(meta.Ψ1_trans)
    Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu,[x]) , q_in)
    return NormalMeanPrecision(jdotavx(Ψ1_trans ,μ_v), mean(q_w))
end

#change here 
@rule UniSGP(:out, Marginalisation) (q_in::PointMass, 
        q_v::MultivariateNormalDistributionsFamily,q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    kernel = getKernel(meta)
    θ = mean(q_θ) #get kernel hyperparameters => kernel(θ)
    μ_v = mean(q_v)
    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(meta.Ψ1_trans,kernel(θ),meta.Xu, [mean(q_in)])
    return NormalMeanPrecision(jdotavx(meta.Ψ1_trans, μ_v), mean(q_w))
end

## in rule 
@rule UniSGP(:in, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, 
    q_v::MultivariateNormalDistributionsFamily,q_w::Any,q_θ::PointMass, meta::UniSGPMeta) = begin
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    B_trans = (x) -> kernelmatrix(kernel(θ), meta.Xu,[x])
    α = (x) -> meta.KuuL \ B_trans(x)
    A = (x) -> kernelmatrix(kernel(θ),[x]) .- dot(α(x),α(x))
    β = (x) -> meta.Uv * B_trans(x)
    
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)

    log_backwardmess = (x) -> -0.5 * A(x)[1] * w_bar + w_bar * μ_y * dot(B_trans(x),μ_v) - 0.5 * w_bar * dot(β(x), β(x))  #here
    return ContinuousUnivariateLogPdf(log_backwardmess)
end

## v rule 
@rule UniSGP(:v, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, 
    q_in::UnivariateNormalDistributionsFamily, q_w::Any,q_θ::PointMass, meta::UniSGPMeta) = begin
    w = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    
    # Ψ1_trans = similar(meta.Ψ1_trans)
    # Ψ2 = similar(meta.Ψ2)
    Ψ1_trans = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ),meta.Xu, [x]),q_in) 
    Ψ2 =  approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I
    
    Ψ1_trans .*= μ_y * w #weighted-mean μ_y * w * Ψ1_transpose
    Ψ2 .*= w  #precision W_v = w * Ψ2
    return BufferUniSGP(MvNormalWeightedMeanPrecision(vec(Ψ1_trans), Ψ2),meta)
end

####### make faster
### regression case #####
@rule UniSGP(:v, Marginalisation) (q_out::PointMass, q_in::PointMass, 
        q_w::Any,q_θ::PointMass,meta::UniSGPMeta) = begin
    w = mean(q_w)
    θ = mean(q_θ)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    kernel = getKernel(meta)

    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    mul!(meta.Ψ2,Ψ1_trans,Ψ1_trans',w,0) #W = w * Ψ1_trans * Ψ1_trans'
    Ψ1_trans .*= μ_y * w
    return BufferUniSGP(MvNormalWeightedMeanPrecision(vec(Ψ1_trans), meta.Ψ2),meta)
end

## classification case ###
@rule UniSGP(:v, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_w::Any, q_θ::PointMass, meta::UniSGPMeta) = begin 
    w = mean(q_w)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)

    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])
    mul!(meta.Ψ2,Ψ1_trans,Ψ1_trans',w,0) #W = w * Ψ1_trans * Ψ1_trans'
    Ψ1_trans .*= μ_y * w
    return BufferUniSGP(MvNormalWeightedMeanPrecision(vec(Ψ1_trans), meta.Ψ2),meta)
end
########

## w rule 
@rule UniSGP(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily,
            q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass,meta::UniSGPMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], meta.Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I 

    I1 = clamp(Ψ0 - tr(meta.KuuL' \ (meta.KuuL \ Ψ2)),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * dot(Ψ1, μ_v) + tr(meta.Uv' * meta.Uv * Ψ2 ),1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end

#### make faster ####
### regression 
@rule UniSGP(:w, Marginalisation) (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    μ_v = mean(q_v)

    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans) 
    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α) #I1

    I2 = μ_y^2 - 2*μ_y*jdotavx(Ψ1_trans,μ_v)
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 

    return GammaShapeRate(1.5, 0.5*(Ψ0[1] + I2))
end


@rule UniSGP(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)

    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α) #I1

    I2 = μ_y^2 + v_y - 2*μ_y*jdotavx(Ψ1_trans,μ_v) 
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return GammaShapeRate(1.5, 0.5*(Ψ0[1] + I2))
end
######

## θ rule 
@rule UniSGP(:θ, Marginalisation) (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu) )
    Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
    Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu) 
    Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu)
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) + tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule UniSGP(:θ, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> inv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
    Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu) 
    Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) 
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) + tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule UniSGP(:θ, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateNormalDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::Any, meta::UniSGPMeta) = begin 
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> inv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ_1 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in) 
    Ψ_2 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) 
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) + tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

## average_energy 
@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, 
        q_w::GammaShapeRate,q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)
    Ψ2 = similar(meta.Ψ2)
    # approximate_kernel_expectation!(Ψ0,getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)
    # approximate_kernel_expectation!(Ψ1_trans,getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]),q_in)
    # approximate_kernel_expectation!(Ψ2,getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)
    Ψ1_trans = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-8*I
    I1 = clamp(Ψ0[1] - tr(meta.KuuL' \ (meta.KuuL \ Ψ2)),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * dot(Ψ1_trans, μ_v) + tr(meta.Uv' * meta.Uv * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

# #####
# @average_energy UniSGP (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate,
#         q_θ::PointMass, meta::UniSGPMeta,) = begin
#     w_bar = mean(q_w)
#     E_logw = mean(log,q_w)
#     μ_y = mean(q_out)
#     μ_v, Σ_v = mean_cov(q_v)
#     μ_in = mean(q_in)
#     θ = mean(q_θ)
#     kernel = getKernel(meta)
#     Xu = getInducingInput(meta)
#     Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu) + 1e-8*I)
    
#     Ψ0 = getindex(kernelmatrix(kernel(θ), [μ_in], [μ_in]),1)
#     Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu)
#     Ψ2 = kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) + 1e-8*I

#     I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
#     I2 = clamp(μ_y^2 -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)
#     return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
# end
## regression 
@average_energy UniSGP (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate,
        q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y = mean(q_out)
    μ_v = mean(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans) 

    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    α = meta.KuuL \ Ψ1_trans 
    Ψ0 .-= jdotavx(α,α)

    I2 = μ_y^2 - 2*μ_y*jdotavx(Ψ1_trans,μ_v) 
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return 0.5*(Ψ0[1]*w_bar - E_logw + log(2π) + I2 * w_bar) 
end
#####

## classification
@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate, 
       q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)

    μ_v = mean(q_v)
    μ_in = mean(q_in)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)

    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])


    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α)
    
    I2 = μ_y^2 + v_y - 2*μ_y*jdotavx(Ψ1_trans,μ_v)
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return 0.5*(Ψ0[1]*w_bar - E_logw + log(2π) + I2 * w_bar) 
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
    Kuu_inverse = inv(kernelmatrix(kernel(θ),Xu) .+ 1e-8)
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in) .+ 1e-8
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) .+ 1e-8 
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy UniSGP (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,
        q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = log(w_bar)
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)

    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])


    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α)

    mul!(Σ_v,μ_v,μ_v',1,1)  # Σ_v = Σ_v + μ_v * μ_v'
    Lu = cholesky!(Σ_v).U
    I2 = μ_y^2 - 2*μ_y*jdotavx(Ψ1_trans,μ_v)
    mul!(Ψ1_trans,Lu,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return 0.5*(Ψ0[1]*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass, 
       q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = log(w_bar)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Kuu_inverse = inv(kernelmatrix(kernel(θ),Xu) .+ 1e-8)
    
    Ψ0 = getindex(kernelmatrix(kernel(θ), [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu) .+ 1e-8
    Ψ2 = kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) .+ 1e-8

    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end