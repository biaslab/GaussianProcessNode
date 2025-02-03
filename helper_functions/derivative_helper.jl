using KernelFunctions
using ForwardDiff
using LinearSolve, LinearAlgebra, PDMats
import RxInfer: skipindex
import ForwardDiff: Chunk, GradientConfig
include("gp_helperfunction.jl")
include("../GPnode/UniSGPnode.jl")
# include("MultiSGPnode.jl")
#new approach
"""
this function computes the log marginal likelihood w.r.t theta. This is the sum of logs 
    y_data, x_data: output and input, given in array
    v, Lv: mean of qv, Lv is the cholesky lower triangular matrix of Rv = Σv + vv'
    w: mean of qw
    kernel: kernel function, this is a function of θ
    Xu: inducing inputs    

    -0.5 * w * kxx(θ) + 0.5 * w * kxu(θ)*Kuu_invese(θ)*kux(θ) - 0.5 * w * kxu(θ) * Rv * kux(θ)  + w*μ_y *kxu(θ)*μ_v
"""


#for regression/classification with known input 
function neg_log_backwardmess_fast(θ; y_data, x_data, v, Uv, w, kernel, Xu)
    Kuu = kernelmatrix(kernel(θ), Xu)
    Lu = fastcholesky(Kuu).L
    kxx = kernelmatrix_diag(kernel(θ), x_data)
    Kux = kernelmatrix(kernel(θ), Xu, x_data)
    
    llh = 0.0
    α = Lu \ view(Kux, :, 1)
    β = Uv * view(Kux, :, 1)
    llh += -0.5 * w * view(kxx, 1) .+ 0.5 * w * dot(α, α) .- 0.5 * w * dot(β, β) .+ w * view(y_data,1) * dot(v, view(Kux, :, 1))
    @inbounds @simd for i in 2:size(y_data,1)
        ldiv!(α, Lu , view(Kux, :, i))
        mul!(β,Uv, view(Kux, :, i))
        llh += -0.5 * w * view(kxx, i) .+ 0.5 * w * dot(α, α) .- 0.5 * w * dot(β, β) .+ w * view(y_data,i) * dot(v, view(Kux, :, i))
    end
    return -llh
end

#case that input is random variable 
function neg_log_backwardmess_uncertain(θ; y_data, qx,v, Uv,w,kernel,Xu,method)
    Kuu_inverse = inv(kernelmatrix(kernel(θ), Xu) + 1e-12*I) 
    llh = 0.0
    @inbounds for i in eachindex(y_data)
        Ψ0 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),qx[i])[1]
        Ψ1_trans = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu,[x]),qx[i]) 
        Ψ2 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), qx[i])
        llh += w * y_data[i] * dot(Ψ1_trans, v) - 0.5 * w * (Ψ0 + tr(Ψ2*(Uv'*Uv - Kuu_inverse)))
    end
    return -llh
end

#test with neg_log_backwardmess_fast
function grad_llh_new_default!(grad,θ; y_data,x_data,v,Uv,w,kernel,Xu)
    return ForwardDiff.gradient!(grad, (x) -> neg_log_backwardmess_fast(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu), θ)
end

function grad_llh_new!(grad, θ; y_data,x_data,v,Uv,w,kernel,Xu,chunk_size)
    newfunc = (x) -> neg_log_backwardmess_fast(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu)
    cfg = GradientConfig(newfunc, θ, Chunk{chunk_size}())
    return ForwardDiff.gradient!(grad, newfunc, θ,cfg)
end

function grad_llh_uncertain!(grad,θ; y_data, qx,v, Uv,w,kernel,Xu,method)
    return ForwardDiff.gradient!(grad, (x) -> neg_log_backwardmess_uncertain(x; y_data=y_data, qx=qx,v=v, Uv=Uv,w=w,kernel=kernel,Xu=Xu,method=method), θ)
end

# ## faster (not that fast)
# function grad_llh_new_faster(θ; y_data,x_data,v,Uv,w,kernel,Xu)
#     return ForwardDiff.gradient((x) -> neg_log_backwardmess_fast_optimized(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu), θ)
# end

# function grad_llh_new_faster!(grad, θ; y_data,x_data,v,Uv,w,kernel,Xu,chunk_size)
#     func_temp = (x) -> neg_log_backwardmess_fast_optimized(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu)
#     cfg = GradientConfig(func_temp, θ, Chunk{chunk_size}())
#     return ForwardDiff.gradient!(grad, func_temp, θ,cfg)
# end

"""
this function is only for C = I
    y_data: Array of mean.(qy), or array of output 
    qx : Array of distributions 
    sumRv_Wbar : Real, sum(Rv_blk .* W)
    v : mean(qv)
    W : mean(qW)
    Tr_W : trace(W)
    kernel : kernel of gp 
    Xu : inducing points 
    method: method to approximate kernel expectation 
"""
function neg_log_backwardmess_multi(θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
        Kuu_inverse = inv(kernelmatrix(kernel(θ), Xu) + 1e-12*I)        
        llh = 0.0
        M = size(Xu,1)
        @inbounds for i in eachindex(qx)
            @inbounds V = v * y_data[i]' * W
            sumdiagV = sum_diagonal_M(V,M)
            @inbounds Ψ_0 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),qx[i])[1] # kxx
            @inbounds Ψ_1_trans = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu,[x]),qx[i]) # kux
            @inbounds Ψ_2 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), qx[i]) # kux*kxu 

            llh += -0.5 * tr_W * (Ψ_0 - sum(Kuu_inverse .* Ψ_2)) + sum(sumdiagV .* Ψ_1_trans) - 0.5 * sum(sumRv_Wbar .* Ψ_2)
        end
    return -llh
end

function grad_llh_multi(θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
    return ForwardDiff.gradient((x) -> neg_log_backwardmess_multi(x;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method), θ)
end

function grad_llh_multi!(grad,θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
    return ForwardDiff.gradient!(grad,(x) -> neg_log_backwardmess_multi(x;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method), θ)
end


##helper function 
#this function computes ΣV
function sum_diagonal_M(V, M)
    @inbounds sumΣV = sum(view(V,M*(i-1)+1:i*M,i)  for i=1:size(V,2))
    return sumΣV
end

#compute trace of block matrix 
function trace_blkmatrix(Rv,D,M)
    return [tr(view(Rv,i:i+M-1,j:j+M-1)) for i=1:M:M*D, j=1:M:M*D]
end

#ground truth
# function check_llh(θ; ydata,xdata,v,Rv,w,kernel,Xu)
#     Kuu_inverse = inv(kernelmatrix(kernel(θ),Xu))
#     llh = 0.0
#     for i=1:length(xdata)
#         kxx = kernelmatrix(kernel(θ),[xdata[i]])
#         kxu = kernelmatrix(kernel(θ),[xdata[i]],Xu)
#         llh += -0.5 * w * kxx[1] + 0.5*w*getindex(kxu * Kuu_inverse * kxu',1) - 0.5 * w * getindex(kxu * Rv * kxu',1) + w*ydata[i]*dot(kxu, v)
#     end
#     return -llh
# end
# function neg_log_backwardmess_forward(θ; y_data,x_data,v,Uv,w,kernel,Xu)
#     Kuu = Symmetric(kernelmatrix(kernel(θ),Xu))
#     Lu = cholesky(Kuu).L 
#     N = length(y_data)
#     kxx = kernelmatrix(kernel(θ),[x_data[1]])
#     kux = kernelmatrix(kernel(θ),Xu,[x_data[1]])
#     α = Lu \ kux
#     β = Uv * kux
#     llh = -0.5 * w * kxx[1] + 0.5 * w * dot(α,α) - 0.5 * w * dot(β,β) + w * y_data[1] * dot(v,kux)

#     @inbounds for i=2:N
#         kxx = kernelmatrix(kernel(θ),[x_data[i]])
#         kux = kernelmatrix(kernel(θ),Xu,[x_data[i]])
#         α = Lu \ kux
#         β = Uv * kux
#         llh += -0.5 * w * kxx[1] + 0.5 * w * dot(α, α) - 0.5 * w * dot(β, β) + w * y_data[i] * dot(v, kux)
#     end
#     return -llh
# end

# function neg_log_backwardmess_fast_optimized(θ; y_data, x_data, v, Uv, w, kernel, Xu)
#     Kuu = kernelmatrix(kernel(θ), Xu) + 1e-8 * I
#     Lu = cholesky(Kuu).L
#     kxx = kernelmatrix_diag(kernel(θ), x_data)
#     Kux = kernelmatrix(kernel(θ), Xu, x_data)

#     # Batch computations
#     α_matrix = Lu \ Kux
#     β_matrix = Uv * Kux

#     llh = mapreduce(i -> 
#         -0.5 * w * view(kxx, i) .+
#         0.5 * w * dot(view(α_matrix, :, i), view(α_matrix, :, i)) .-
#         0.5 * w * dot(view(β_matrix, :, i), view(β_matrix, :, i)) .+
#         w * view(y_data, i) * dot(v, view(Kux, :, i)),
#         +, 1:size(y_data, 1)
#     )

#     return -llh
# end


#this seems fast (but unstable)
# function neg_log_backwardmess_fast(θ; y_data,x_data,v,Lv,w,kernel,Xu)
#     Kuu = kernelmatrix(kernel(θ),Xu) + 1e-8*I
#     Lu = cholesky(Kuu).L 
#     kxx = kernelmatrix(kernel(θ),x_data)
#     Kux = kernelmatrix(kernel(θ),Xu,x_data)
#     @inbounds α = [Lu \ view(Kux,:,i) for i in eachindex(y_data)]
#     @inbounds β = [Lv * view(Kux,:,i) for i in eachindex(y_data)]
#     return -sum(@inbounds  begin
#         [
#             -0.5 * w * view(kxx,i,i) .+ 0.5 * w * dot(view(α,i), view(α,i)).- 0.5 * w * dot(view(β,i), view(β,i)) .+ w * view(y_data,i) * dot(v, view(Kux,:,i)) 
#             for i in eachindex(y_data)
#         ] 
#     end)
# end



# function grad_llh(θ; y_data,x_data,v,Uv,w,kernel,Xu)
#     return ForwardDiff.gradient((x) -> neg_log_backwardmess_forward(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu), θ)
# end


# function grad_llh!(grad, θ; y_data,x_data,v,Uv,w,kernel,Xu)
#     return ForwardDiff.gradient!(grad,(x) -> neg_log_backwardmess_forward(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu), θ,)
# end
