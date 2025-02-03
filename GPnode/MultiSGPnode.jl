include("../helper_functions/gp_helperfunction.jl")
include("../helper_functions//derivative_helper.jl")
using ReactiveMP,RxInfer,GraphPPL
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using Zygote, Optim, ForwardDiff
using KernelFunctions, LoopVectorization

import KernelFunctions: with_lengthscale, Kernel, kernelmatrix 
import ReactiveMP: approximate_meancov, approximate_kernel_expectation, WishartFast, logdet

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, distribution::D) where {K <: Array, D <: MultivariateDistribution}
    return approximate_kernel_expectation!(gbar, method, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, m::AbstractVector{T}, P::AbstractMatrix{T}) where {K <: Array, T <: Real}
    weights = ReactiveMP.getweights(method, m, P)
    points  = ReactiveMP.getpoints(method, m, P)

    gbar .= 0
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::AbstractVector{T}, P::AbstractMatrix{T}) where {T <: Real}
    weights = ReactiveMP.getweights(method, m, P)
    points  = ReactiveMP.getpoints(method, m, P)

    gbar = g(m) .* 0.0
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end
    return gbar
end

function ReactiveMP.prod(::GenericProd, left::MultivariateGaussianDistributionsFamily, right::ContinuousMultivariateLogPdf) 
    m,v = approximate_meancov(srcubature(),(x) -> exp(right.logpdf(x)),left)
    if isnan(m[1])
        return left 
    else
        return MvNormalMeanCovariance(m,v)
    end
end

## Multivariate GP node 
struct MultiSGP end 

@node MultiSGP Stochastic [ out, in, v , w, θ] ## out: x_t , in: x_{t-1},  v: transformed-inducing points Kuu_inv * u , w: precision of process noise 

#--------------- out ------------------# 
##### original code
# @rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
#     μ_v = mean(q_v)
#     W = mean(q_w) 
#     θ = mean(q_θ)
#     Xu = getInducingInput(meta)
#     kernel = getKernel(meta)
#     C = getCoregionalizationMatrix(meta)
#     M = length(Xu) #number of inducing points
#     D = size(C,1)
#     @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
#     μ_y = getcache(getGPCache(meta), (:μ_y, D))
#     Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel(θ), [x], Xu), q_in)
#     @inbounds for (i,μ_v_entry) in enumerate(μ_v)
#         μ_y[i] = jdotavx(Ψ1,μ_v_entry)
#     end
#     return MvNormalMeanPrecision(μ_y, W)
# end

# @rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass,meta::MultiSGPMeta,) = begin
#     μ_v = mean(q_v)
#     W = mean(q_w) 
#     θ = mean(q_θ)
#     Xu = getInducingInput(meta)
#     kernel = getKernel(meta)
#     C = getCoregionalizationMatrix(meta)
#     M = length(Xu) #number of inducing points
#     D = size(C,1)
#     @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
#     μ_y = getcache(getGPCache(meta), (:μ_y, D))
#     Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel(θ), [x], Xu), q_in)
#     @inbounds for (i,μ_v_entry) in enumerate(μ_v)
#         μ_y[i] = jdotavx(Ψ1,μ_v_entry)
#     end
#     return MvNormalMeanPrecision(μ_y, W)
# end

### faster code #####
@rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    cache = getGPCache(meta)
    Ψ1_trans = getΨ1_trans(meta)
    M = length(Xu) #number of inducing points
    D = size(W,1)
    method = getmethod(meta)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    approximate_kernel_expectation!(Ψ1_trans,method, (x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]), q_in)
    return MvNormalMeanPrecision(map!(yi -> jdotavx(Ψ1_trans, yi),getcache(cache, (:μ_y, D)), μ_v), W)
end

@rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass,meta::MultiSGPMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    cache = getGPCache(meta)
    Ψ1_trans = getΨ1_trans(meta)
    M = length(Xu) #number of inducing points
    D = size(W,1)
    method = getmethod(meta)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    approximate_kernel_expectation!(Ψ1_trans,method, (x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ),Xu, [x]), q_in) 
    return MvNormalMeanPrecision(map!(yi -> jdotavx(Ψ1_trans, yi),getcache(cache, (:μ_y, D)), μ_v), W)
end
#--------------- in rule ------------------# 
# @rule MultiSGP(:in, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta) = begin
#     μ_y, Σ_y = mean_cov(q_out)
#     μ_v, Σ_v = mean_cov(q_v)
#     W = mean(q_w) 
#     θ = mean(q_θ)
#     kernel = getKernel(meta)
#     Xu = getInducingInput(meta)
#     Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
#     D = length(μ_y) #output dimension

#     A = (x) -> kron(C,kernelmatrix(kernel(θ),[x],[x]) - kernelmatrix(kernel(θ),[x],Xu)*Kuu_inverse*kernelmatrix(kernel(θ),Xu,[x]))
#     B = (x) -> kron(C,kernelmatrix(kernel(θ), [x], Xu))
    
#     log_backwardmess = (x) -> -0.5  * tr(W*(A(x) + B(x)*(Σ_v + μ_v*μ_v')*B(x)')) + μ_y' * W * B(x) * μ_v
 
#     return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
# end

# @rule MultiSGP(:in, Marginalisation) (q_out::PointMass,q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta) = begin 
#     μ_y = mean(q_out)
#     μ_v, Σ_v = mean_cov(q_v)
#     W = mean(q_w) 
#     θ = mean(q_θ)
#     kernel = getKernel(meta)
#     Xu = getInducingInput(meta)
#     Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))

#     A = (x) -> kron(C,kernelmatrix(kernel(θ),[x],[x]) - kernelmatrix(kernel(θ),[x],Xu)*Kuu_inverse*kernelmatrix(kernel(θ),Xu,[x]))
#     B = (x) -> kron(C,kernelmatrix(kernel(θ), [x], Xu))

#     neg_log_backwardmess = (x) -> -(-0.5 * tr(W*A(x)) + μ_y' * W * B(x) * μ_v - 0.5 * tr((μ_v * μ_v' + Σ_v)*B(x)' * W * B(x)))
#     res = optimize(neg_log_backwardmess,mean(q_in),Optim.Options(iterations=20))
#     m_z = res.minimizer
#     W_z = Zygote.hessian(neg_log_backwardmess, m_z) 
    
#     return MvNormalWeightedMeanPrecision(W_z * m_z, W_z)
# end

## faster 
@rule MultiSGP(:in, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getKuuInverse(meta)
    cache = getGPCache(meta)
    M = length(Xu)
    D = length(μ_y)

    Ψ0 = (x) -> kernelmatrix(kernel(θ),[x])[1]
    Ψ1_trans = (x) -> kernelmatrix(kernel(θ),Xu,[x])
    Ψ2 = (x) -> Ψ1_trans(x) * Ψ1_trans(x)'
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W)

    log_backwardmess = (x) -> -0.5 * tr(W) * (Ψ0(x) - sum(Kuu_inverse .* Ψ2(x))) + sum(sumdiagV .* Ψ1_trans(x)) - 0.5 * sum(Ψ2(x) .* sumRvblk_W)
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule MultiSGP(:in, Marginalisation) (q_out::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    logdetW = log(det(W))
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getKuuInverse(meta)
    cache = getGPCache(meta)
    M = length(Xu)
    D = length(μ_y)

    Ψ0 = (x) -> kernelmatrix(kernel(θ),[x])[1]
    Ψ1_trans = (x) -> kernelmatrix(kernel(θ),Xu,[x])
    Ψ2 = (x) -> Ψ1_trans(x) * Ψ1_trans(x)'
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W)
    log_backwardmess = (x) -> -0.5 * tr(W) * (Ψ0(x) - sum(Kuu_inverse .* Ψ2(x))) + sum(sumdiagV .* Ψ1_trans(x)) - 0.5 * sum(Ψ2(x) .* sumRvblk_W) #- D/2 * log(2π) + 0.5*logdetW - 0.5*μ_y'*W*μ_y
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule MultiSGP(:in, Marginalisation) (q_out::PointMass,q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getKuuInverse(meta)
    M = length(Xu)
    D = length(μ_y)
    
    Ψ0 = (x) -> kernelmatrix(kernel(θ),[x])[1]
    Ψ1_trans = (x) -> kernelmatrix(kernel(θ),Xu,[x])
    Ψ2 = (x) -> Ψ1_trans(x) * Ψ1_trans(x)'
    Rv = Σ_v + μ_v * μ_v'
    V = μ_v * μ_y' * W
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Rv,D,M) .* W)

    neg_log_backwardmess = (x) -> -(-0.5 * tr(W) * (Ψ0(x) - sum(Kuu_inverse .* Ψ2(x))) + sum(sumdiagV .* Ψ1_trans(x)) - 0.5 * sum(Ψ2(x) .* sumRvblk_W))
    grad_func! = (G,x) -> ForwardDiff.gradient!(G,neg_log_backwardmess,x)
    res = optimize(neg_log_backwardmess,grad_func!, mean(q_in),LBFGS(),Optim.Options(iterations=20);inplace=true)#,Optim.Options(iterations=20))
    m_z = res.minimizer
    W_z = Zygote.hessian(neg_log_backwardmess, m_z) 
    
    return MvNormalWeightedMeanPrecision(W_z * m_z, W_z)
end

#---------- v rule ------------------#
# @rule MultiSGP(:v, Marginalisation) (q_out::MultivariateGaussianDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta) = begin 
#     W = mean(q_w)
#     μ_y = mean(q_out)
#     θ = mean(q_θ)
#     C = getCoregionalizationMatrix(meta)
#     cache = getGPCache(meta)
#     Xu = getInducingInput(meta)
#     kernel = getKernel(meta)
#     M = length(Xu) #number of inducing points
#     D = length(μ_y) #dimension

#     Ψ1 = getcache(cache,(:Ψ1,M))
#     Ψ2 = getcache(cache,(:Ψ2, (M,M)))
#     Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
#     Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I
#     ξ_v = Float64[]
#     μ_y_transformed = W * μ_y 
#     for mu_y in μ_y_transformed
#         append!(ξ_v, mu_y .* Ψ1)
#     end
#     W_v = getcache(cache, (:W_v, (D*M,D*M)))
#     kron!(W_v,W, Ψ2)
#     return MvNormalWeightedMeanPrecision(ξ_v, W_v)
# end

# @rule MultiSGP(:v, Marginalisation) (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta) = begin 
#     W = mean(q_w)
#     μ_y = mean(q_out)
#     θ = mean(q_θ)
#     C = getCoregionalizationMatrix(meta)
#     cache = getGPCache(meta)
#     Xu = getInducingInput(meta)
#     kernel = getKernel(meta)
#     M = length(Xu) #number of inducing points
#     D = length(μ_y) #dimension

#     Ψ1 = getcache(cache,(:Ψ1,M))
#     Ψ2 = getcache(cache,(:Ψ2, (M,M)))
#     Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
#     Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I
#     ξ_v = Float64[]
#     μ_y_transformed = W * μ_y 
#     for mu_y in μ_y_transformed
#         append!(ξ_v, mu_y .* Ψ1)
#     end
#     W_v = getcache(cache, (:W_v, (D*M,D*M)))
#     kron!(W_v,W, Ψ2)
#     return MvNormalWeightedMeanPrecision(ξ_v, W_v)
# end

##faster 
@rule MultiSGP(:v, Marginalisation) (q_out::MultivariateGaussianDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    method = getmethod(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ),Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2) # precision matrix 
    return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
end

@rule MultiSGP(:v, Marginalisation) (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    method = getmethod(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu,[x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2)
    return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
end

#--------------------- w rule ---------------------#
# @rule MultiSGP(:w, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::MultiSGPMeta,) = begin
#     μ_y, Σ_y = mean_cov(q_out)
#     μ_v, Σ_v = mean_cov(q_v)
#     θ = mean(q_θ)
#     Xu = getInducingInput(meta)
#     kernel = getKernel(meta)
#     C = getCoregionalizationMatrix(meta)
#     cache = getGPCache(meta)
#     M = length(Xu) #number of inducing points
#     D = length(μ_y) #dimension
#     μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
#     Σ_v = create_blockmatrix(Σ_v,D,M)
#     Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))

#     Ψ1 = getcache(cache,(:Ψ1, M))
#     Ψ2 = getcache(cache,(:Ψ2, (M,M)))

#     Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
#     Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
#     Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I

#     trace_A = mul_A_B!(cache,Kuu_inverse,Ψ2,M) |> tr #trA =tr(Kuu_inverse * Ψ2)
#     I1 = (Ψ0 - trace_A) .* C # kron(C, getindex(Ψ0,1) - trace_A)

#     E = [getindex(Ψ1 * i,1) for i in μ_v]
#     Ψ_5 = getcache(cache,(:Ψ_5,(D,D)))
#     for j=1:D
#         for i=1:D
#             Ψ_5[i,j] = tr((μ_v[i]*μ_v[j]' + Σ_v[i,j]) * Ψ2)
#         end
#     end
#     I2 = μ_y * μ_y' + Σ_y - μ_y * E' - E * μ_y' + Ψ_5 
#     return WishartFast(D+2, I1 + I2)
# end

#### faster 
@rule MultiSGP(:w, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    cache = getGPCache(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    C = diageye(D)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D)
    R_v = create_blockmatrix(Σ_v,D,M)
    μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu,[x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in) 

    Ψ0 .-= tr(Kuu_inverse * Ψ2)
    I1 = map(x -> Ψ0[1] * x, C) # kron(C, getindex(Ψ0,1) - trace_A)

    E = getcache(cache,(:E,D))
    map!(yi -> jdotavx(Ψ1_trans, yi),E, μ_v)
    Ψ_4 = getcache(cache,(:Ψ_4,(D,D)))
    map!(Rv_i -> sum(Rv_i .* Ψ2'),Ψ_4, R_v)
    tmp = mul_A_B!(cache,μ_y, E',D)
    tmp += tmp'
    Σ_y += mul_A_B!(cache, μ_y, μ_y',D)
    Ψ_4 += Σ_y
    Ψ_4 -= tmp #this is I2
    Ψ_4 += I1
    return WishartFast(D+2, Ψ_4)
end

@rule MultiSGP(:w, Marginalisation) (q_out::PointMass, q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    cache = getGPCache(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    C = diageye(D)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D)
    R_v = create_blockmatrix(Σ_v,D,M)
    μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu,[x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in) 

    Ψ0 .-= tr(Kuu_inverse * Ψ2)
    I1 = map(x -> Ψ0[1] * x, C) # kron(C, getindex(Ψ0,1) - trace_A)

    E = getcache(cache,(:E,D))
    map!(yi -> jdotavx(Ψ1_trans, yi),E, μ_v)
    Ψ_4 = getcache(cache,(:Ψ_4,(D,D)))
    map!(Rv_i -> sum(Rv_i .* Ψ2'),Ψ_4, R_v)
    tmp = mul_A_B!(cache,μ_y, E',D)
    tmp += tmp'
    Ψ_4 += mul_A_B!(cache, μ_y, μ_y',D)
    Ψ_4 -= tmp #this is I2
    Ψ_4 += I1
    return WishartFast(D+2, Ψ_4)
end

#-------------- θ rule ----------------#
@rule MultiSGP(:θ, Marginalisation) (q_out::Any, q_in::MultivariateGaussianDistributionsFamily,q_v::MultivariateGaussianDistributionsFamily, q_w::Any, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    R_v = Σ_v + μ_v * μ_v'
    W_bar = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    C = diageye(length(μ_y))

    Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ_1 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ_2 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 

    I1 = (θ) -> kron(C, Ψ_0(θ) - tr(Kuu_inverse(θ) * Ψ_2(θ)))
    Ψ_1_tilde = (θ) -> kron(C, Ψ_1(θ))
    Ψ_3 = (θ) -> kron(W_bar, Ψ_2(θ))
    log_backwardmess = (θ) -> -0.5 * tr(W_bar * I1(θ)) + μ_y' * W_bar * Ψ_1_tilde(θ) * μ_v - 0.5 * tr(Ψ_3(θ) * R_v)
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

#------------ average energy ------------# 
# @average_energy MultiSGP (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
#     μ_y, Σ_y = mean_cov(q_out)
#     μ_v, Σ_v = mean_cov(q_v)
#     W_bar = mean(q_w) 
#     θ = mean(q_θ)
#     E_logW = mean(logdet,q_w)
#     Xu = getInducingInput(meta)
#     cache = getGPCache(meta)
#     kernel = getKernel(meta)
#     C = getCoregionalizationMatrix(meta)
#     Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
#     D = length(μ_y)
#     M = length(Xu) #number of inducing points
    
#     Ψ1 = getcache(cache,(:Ψ1, M))
#     Ψ2 = getcache(cache,(:Ψ2, (M,M)))
#     Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
#     Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
#     Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I

#     μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
#     Σ_v = create_blockmatrix(Σ_v,D,M)
#     trace_A = mul_A_B!(cache,Kuu_inverse,Ψ2,M) |> tr #trA =tr(Kuu_inverse * Ψ2)
#     I1 = (Ψ0 - trace_A) .* C # kron(C, getindex(Ψ0,1) - trace_A)

#     E = [getindex(Ψ1 * i,1) for i in μ_v]
#     Ψ_5 = getcache(cache,(:Ψ_5,(D,D)))
#     for j=1:D
#         for i=1:D
#             Ψ_5[i,j] = tr((μ_v[i]*μ_v[j]' + Σ_v[i,j]) * Ψ2)
#         end
#     end
#     I2 = μ_y * μ_y' + Σ_y - μ_y * E' - E * μ_y' + Ψ_5 

#     return 0.5*tr(W_bar*I1) + 0.5*D*log(2π) - 0.5*E_logW + 0.5 * tr(W_bar * I2)
# end


# @average_energy MultiSGP (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta,) = begin
#     μ_y = mean(q_out)
#     μ_v, Σ_v = mean_cov(q_v)
#     θ = mean(q_θ)
#     W_bar = mean(q_w) 
#     E_logW = log(det(W_bar))
#     Xu = getInducingInput(meta)
#     C = getCoregionalizationMatrix(meta)
#     kernel = getKernel(meta)
#     Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
#     cache = getGPCache(meta)

#     D = length(μ_y)
#     M = length(Xu) #number of inducing points
    
#     Ψ1 = getcache(cache,(:Ψ1, M))
#     Ψ2 = getcache(cache,(:Ψ2, (M,M)))
#     Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
#     Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
#     Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I
#     μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
#     Σ_v = create_blockmatrix(Σ_v,D,M)
#     trace_A = mul_A_B!(cache,Kuu_inverse,Ψ2,M) |> tr #trA =tr(Kuu_inverse * Ψ2)
#     I1 = (Ψ0 - trace_A) .* C # kron(C, getindex(Ψ0,1) - trace_A)

#     E = [getindex(Ψ1 * i,1) for i in μ_v]
#     Ψ_5 = getcache(cache,(:Ψ_5,(D,D)))
#     for j=1:D
#         for i=1:D
#             Ψ_5[i,j] = tr((μ_v[i]*μ_v[j]' + Σ_v[i,j]) * Ψ2)
#         end
#     end
#     I2 = μ_y * μ_y' - μ_y * E' - E * μ_y' + Ψ_5 

#     return 0.5*tr(W_bar*I1) + 0.5*D*log(2π) - 0.5*E_logW + 0.5 * tr(W_bar * I2)
# end
###### faster 
@average_energy MultiSGP (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_bar = mean(q_w) 
    θ = mean(q_θ)
    E_logW = mean(logdet,q_w)
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W_bar,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W_bar)
    Ry = Σ_y + μ_y * μ_y'

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 
    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)[1]
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    return  0.5*D*log(2π) - 0.5*E_logW + 0.5*tr(W_bar*Ry)+ 0.5 * tr(W_bar) * (Ψ0[1] - sum(Kuu_inverse .* Ψ2)) - sum(sumdiagV .* Ψ1_trans) + 0.5 * sum(Ψ2 .* sumRvblk_W)
end


@average_energy MultiSGP (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    W_bar = mean(q_w) 
    E_logW = log(det(W_bar))
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W_bar,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W_bar)
    Ry = μ_y * μ_y'

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)[1]
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    return  0.5*D*log(2π) - 0.5*E_logW + 0.5*tr(W_bar*Ry)+ 0.5 * tr(W_bar) * (Ψ0[1] - sum(Kuu_inverse .* Ψ2)) - sum(sumdiagV .* Ψ1_trans) + 0.5 * sum(Ψ2 .* sumRvblk_W)
end

@average_energy MultiSGP (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    W_bar = mean(q_w) 
    E_logW = mean(logdet,q_w)
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W_bar,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W_bar)
    Ry = μ_y * μ_y'

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)[1]
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    return  0.5*D*log(2π) - 0.5*E_logW + 0.5*tr(W_bar*Ry)+ 0.5 * tr(W_bar) * (Ψ0[1] - sum(Kuu_inverse .* Ψ2)) - sum(sumdiagV .* Ψ1_trans) + 0.5 * sum(Ψ2 .* sumRvblk_W)
end