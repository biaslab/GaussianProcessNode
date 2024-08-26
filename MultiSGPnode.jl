include("gp_helperfunction.jl")
using ReactiveMP,RxInfer,GraphPPL
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using Zygote, Optim, ForwardDiff
using KernelFunctions, LoopVectorization

import KernelFunctions: with_lengthscale, Kernel, kernelmatrix 
import ReactiveMP: approximate_meancov, approximate_kernel_expectation, WishartFast, logdet

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::AbstractVector{T}, P::AbstractMatrix{T}) where {T <: Real}
    weights = ReactiveMP.getweights(method, m, P)
    points  = ReactiveMP.getpoints(method, m, P)

    gbar = g(m) - g(m)
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

#out
@rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    C = getCoregionalizationMatrix(meta)
    M = length(Xu) #number of inducing points
    D = size(C,1)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    μ_y = getcache(getGPCache(meta), (:μ_y, D))
    Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel(θ), [x], Xu), q_in)
    for (i,μ_v_entry) in enumerate(μ_v)
        @inbounds μ_y[i] = jdotavx(Ψ1,μ_v_entry)
    end
    return MvNormalMeanPrecision(μ_y, W)
end

@rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass,meta::MultiSGPMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    C = getCoregionalizationMatrix(meta)
    M = length(Xu) #number of inducing points
    D = size(C,1)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    μ_y = getcache(getGPCache(meta), (:μ_y, D))
    Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel(θ), [x], Xu), q_in)
    for (i,μ_v_entry) in enumerate(μ_v)
        @inbounds μ_y[i] = jdotavx(Ψ1,μ_v_entry)
    end
    return MvNormalMeanPrecision(μ_y, W)
end

#in rule 
@rule MultiSGP(:in, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    
    D = length(μ_y) #output dimension

    A = (x) -> kron(C,kernelmatrix(kernel(θ),[x],[x]) - kernelmatrix(kernel(θ),[x],Xu)*Kuu_inverse*kernelmatrix(kernel(θ),Xu,[x]))
    B = (x) -> kron(C,kernelmatrix(kernel(θ), [x], Xu))
    
    log_backwardmess = (x) -> -0.5  * tr(W*(A(x) + B(x)*(Σ_v + μ_v*μ_v')*B(x)')) + μ_y' * W * B(x) * μ_v
 
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule MultiSGP(:in, Marginalisation) (q_out::PointMass,q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))

    A = (x) -> kron(C,kernelmatrix(kernel(θ),[x],[x]) - kernelmatrix(kernel(θ),[x],Xu)*Kuu_inverse*kernelmatrix(kernel(θ),Xu,[x]))
    B = (x) -> kron(C,kernelmatrix(kernel(θ), [x], Xu))

    neg_log_backwardmess = (x) -> -(-0.5 * tr(W*A(x)) + μ_y' * W * B(x) * μ_v - 0.5 * tr((μ_v * μ_v' + Σ_v)*B(x)' * W * B(x)))
    res = optimize(neg_log_backwardmess,mean(q_in),Optim.Options(iterations=20))
    m_z = res.minimizer
    W_z = Zygote.hessian(neg_log_backwardmess, m_z) 
    
    return MvNormalWeightedMeanPrecision(W_z * m_z, W_z)
end

#v rule
@rule MultiSGP(:v, Marginalisation) (q_out::MultivariateGaussianDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    C = getCoregionalizationMatrix(meta)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension

    Ψ1 = getcache(cache,(:Ψ1,M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I
    ξ_v = Float64[]
    μ_y_transformed = W * μ_y 
    for mu_y in μ_y_transformed
        append!(ξ_v, mu_y .* Ψ1)
    end
    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2)
    return MvNormalWeightedMeanPrecision(ξ_v, W_v)
end

@rule MultiSGP(:v, Marginalisation) (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    C = getCoregionalizationMatrix(meta)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension

    Ψ1 = getcache(cache,(:Ψ1,M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I
    ξ_v = Float64[]
    μ_y_transformed = W * μ_y 
    for mu_y in μ_y_transformed
        append!(ξ_v, mu_y .* Ψ1)
    end
    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2)
    return MvNormalWeightedMeanPrecision(ξ_v, W_v)
end

# w rule 
@rule MultiSGP(:w, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    C = getCoregionalizationMatrix(meta)
    cache = getGPCache(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
    Σ_v = create_blockmatrix(Σ_v,D,M)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))

    Ψ1 = getcache(cache,(:Ψ1, M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))

    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I

    trace_A = mul_A_B!(cache,Kuu_inverse,Ψ2,M) |> tr #trA =tr(Kuu_inverse * Ψ2)
    I1 = (Ψ0 - trace_A) .* C # kron(C, getindex(Ψ0,1) - trace_A)

    E = [getindex(Ψ1 * i,1) for i in μ_v]
    Ψ_5 = getcache(cache,(:Ψ_5,(D,D)))
    for j=1:D
        for i=1:D
            Ψ_5[i,j] = tr((μ_v[i]*μ_v[j]' + Σ_v[i,j]) * Ψ2)
        end
    end
    I2 = μ_y * μ_y' + Σ_y - μ_y * E' - E * μ_y' + Ψ_5 
    return WishartFast(D+2, I1 + I2)
end

# θ rule 
@rule MultiSGP(:θ, Marginalisation) (q_out::Any, q_in::MultivariateGaussianDistributionsFamily,q_v::MultivariateGaussianDistributionsFamily, q_w::Any, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    R_v = Σ_v + μ_v * μ_v'
    W_bar = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    C = getCoregionalizationMatrix(meta)

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
#average energy 
@average_energy MultiSGP (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_bar = mean(q_w) 
    θ = mean(q_θ)
    E_logW = mean(logdet,q_w)
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    C = getCoregionalizationMatrix(meta)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    
    Ψ1 = getcache(cache,(:Ψ1, M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I

    μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
    Σ_v = create_blockmatrix(Σ_v,D,M)
    trace_A = mul_A_B!(cache,Kuu_inverse,Ψ2,M) |> tr #trA =tr(Kuu_inverse * Ψ2)
    I1 = (Ψ0 - trace_A) .* C # kron(C, getindex(Ψ0,1) - trace_A)

    E = [getindex(Ψ1 * i,1) for i in μ_v]
    Ψ_5 = getcache(cache,(:Ψ_5,(D,D)))
    for j=1:D
        for i=1:D
            Ψ_5[i,j] = tr((μ_v[i]*μ_v[j]' + Σ_v[i,j]) * Ψ2)
        end
    end
    I2 = μ_y * μ_y' + Σ_y - μ_y * E' - E * μ_y' + Ψ_5 

    return 0.5*tr(W_bar*I1) + 0.5*D*log(2π) - 0.5*E_logW + 0.5 * tr(W_bar * I2)
end


@average_energy MultiSGP (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    W_bar = mean(q_w) 
    E_logW = log(det(W_bar))
    Xu = getInducingInput(meta)
    C = getCoregionalizationMatrix(meta)
    kernel = getKernel(meta)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ),Xu))
    cache = getGPCache(meta)

    D = length(μ_y)
    M = length(Xu) #number of inducing points
    
    Ψ1 = getcache(cache,(:Ψ1, M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I
    μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
    Σ_v = create_blockmatrix(Σ_v,D,M)
    trace_A = mul_A_B!(cache,Kuu_inverse,Ψ2,M) |> tr #trA =tr(Kuu_inverse * Ψ2)
    I1 = (Ψ0 - trace_A) .* C # kron(C, getindex(Ψ0,1) - trace_A)

    E = [getindex(Ψ1 * i,1) for i in μ_v]
    Ψ_5 = getcache(cache,(:Ψ_5,(D,D)))
    for j=1:D
        for i=1:D
            Ψ_5[i,j] = tr((μ_v[i]*μ_v[j]' + Σ_v[i,j]) * Ψ2)
        end
    end
    I2 = μ_y * μ_y' - μ_y * E' - E * μ_y' + Ψ_5 

    return 0.5*tr(W_bar*I1) + 0.5*D*log(2π) - 0.5*E_logW + 0.5 * tr(W_bar * I2)
end

