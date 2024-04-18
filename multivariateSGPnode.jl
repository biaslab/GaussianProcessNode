include("gphelper.jl")
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
struct GPTransition end 

@node GPTransition Stochastic [ out, in, v , w] ## out: x_t , in: x_{t-1},  v: transformed-inducing points Kuu_inv * u , w: precision of process noise 

#out
@rule GPTransition(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,meta::GPTransitionMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    C = getCoregionalizationMatrix(meta)
    M = length(Xu) #number of inducing points
    D = size(C,1)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    μ_y = getcache(getGPCache(meta), (:μ_y, D))
    Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel, [x], Xu), q_in)
    for (i,μ_v_entry) in enumerate(μ_v)
        @inbounds μ_y[i] = jdotavx(Ψ1,μ_v_entry)
    end
    return MvNormalMeanPrecision(μ_y, W)
end

@rule GPTransition(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,meta::GPTransitionMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    C = getCoregionalizationMatrix(meta)
    M = length(Xu) #number of inducing points
    D = size(C,1)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    μ_y = getcache(getGPCache(meta), (:μ_y, D))
    Ψ1 = approximate_kernel_expectation(getmethod(meta), (x) -> kernelmatrix(kernel, [x], Xu), q_in)
    for (i,μ_v_entry) in enumerate(μ_v)
        @inbounds μ_y[i] = jdotavx(Ψ1,μ_v_entry)
    end
    return MvNormalMeanPrecision(μ_y, W)
end

#in rule 
@rule GPTransition(:in, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Wishart, meta::GPTransitionMeta) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    kernel = getKernel(meta)
    Kuu_inverse = getinverseKuu(meta)
    Xu = getInducingInput(meta)
    D = length(μ_y) #output dimension

    kernel = getKernel(meta)
    A = (x) -> kron(C,kernelmatrix(kernel,[x]) - kernelmatrix(kernel,[x],Xu)*Kuu_inverse*kernelmatrix(kernel,Xu,[x]))
    B = (x) -> kron(C,kernelmatrix(kernel, [x], Xu))
    
    log_backwardmess = (x) -> -(0.5  * (tr(W*(A(x) + Σ_y + B(x)*Σ_v*B(x)')) + (μ_y - B(x)*μ_v)' * W * (μ_y - B(x)*μ_v)))
 
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule GPTransition(:in, Marginalisation) (q_out::PointMass,q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::PointMass, meta::GPTransitionMeta) = begin 
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    kernel = getKernel(meta)
    Kuu_inverse = getinverseKuu(meta)
    Xu = getInducingInput(meta)

    kernel = getKernel(meta)
    A = (x) -> kron(C,kernelmatrix(kernel,[x]) - kernelmatrix(kernel,[x],Xu)*Kuu_inverse*kernelmatrix(kernel,Xu,[x]))
    B = (x) -> kron(C,kernelmatrix(kernel, [x], Xu))

    neg_log_backwardmess = (x) -> -(-0.5 * tr(W*A(x)) + μ_y' * W * B(x) * μ_v - 0.5 * tr((μ_v * μ_v' + Σ_v)*B(x)' * W * B(x)))
    res = optimize(neg_log_backwardmess,mean(q_in))
    m_z = res.minimizer
    W_z = Zygote.hessian(neg_log_backwardmess, m_z) 
    
    return MvNormalWeightedMeanPrecision(W_z * m_z, W_z)
end

#v rule
@rule GPTransition(:v, Marginalisation) (q_out::MultivariateGaussianDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_w::Wishart, meta::GPTransitionMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    C = getCoregionalizationMatrix(meta)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension

    Ψ1 = getcache(cache,(:Ψ1,M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) 
    ξ_v = Float64[]
    μ_y_transformed = W * μ_y 
    for mu_y in μ_y_transformed
        append!(ξ_v, mu_y .* Ψ1)
    end
    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2)
    return MvNormalWeightedMeanPrecision(ξ_v, W_v)
end

@rule GPTransition(:v, Marginalisation) (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_w::PointMass, meta::GPTransitionMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    C = getCoregionalizationMatrix(meta)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension

    Ψ1 = getcache(cache,(:Ψ1,M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) 
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
@rule GPTransition(:w, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, meta::GPTransitionMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    C = getCoregionalizationMatrix(meta)
    cache = getGPCache(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
    Σ_v = create_blockmatrix(Σ_v,D,M)
    Kuu_inverse = getinverseKuu(meta)
    kernel = getKernel(meta)

    Ψ1 = getcache(cache,(:Ψ1, M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], [x]),q_in)[]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) 

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
#average energy 
@average_energy GPTransition (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart, meta::GPTransitionMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_bar = mean(q_w) 
    E_logW = mean(logdet,q_w)
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    
    Ψ1 = getcache(cache,(:Ψ1, M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], [x]),q_in)[]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) 

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


@average_energy GPTransition (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass, meta::GPTransitionMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_bar = mean(q_w) 
    E_logW = log(det(W_bar))
    Xu = getInducingInput(meta)
    Kuu_inverse = getinverseKuu(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    
    Ψ1 = getcache(cache,(:Ψ1, M))
    Ψ2 = getcache(cache,(:Ψ2, (M,M)))
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], [x]),q_in)[]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, [x], Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_in) 
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

