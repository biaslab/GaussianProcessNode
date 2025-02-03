using LoopVectorization, LinearAlgebra
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: AbstractApproximationMethod


## create UniSGP meta   (old)
# struct UniSGPMeta{I,K}
#     method      :: Union{Nothing,AbstractApproximationMethod}
#     Xu          :: I    # inducing inputs
#     kernel      :: K
# end
# getmethod(meta::UniSGPMeta) = meta.method
# getInducingInput(meta::UniSGPMeta) = meta.Xu
# getKernel(meta::UniSGPMeta) = meta.kernel

struct GPCache
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
    cache_LowerTriangular::Dict{Tuple{Symbol,Int}, LowerTriangular{Float64, Matrix{Float64}}}
end

#use for supervised learning with big data
mutable struct UniSGPBigDataMeta{T}
    I1          :: Float64
    Ψ1_trans    :: T
    Ψ2          :: Matrix{Float64}
    Uv          :: AbstractArray
    ncounter    :: Int 
    N           :: Int
end

## create UniSGP meta  
mutable struct UniSGPMeta{I,K}
    method      :: Union{Nothing,AbstractApproximationMethod}
    Xu          :: I    # inducing inputs
    Ψ0          :: Matrix{Float64}
    Ψ1_trans    :: Matrix{Float64}
    Ψ2          :: Matrix{Float64}
    KuuL        :: AbstractArray
    kernel      :: K
    Uv          :: AbstractArray
    counter     :: Int
    N           :: Int
end
getmethod(meta::UniSGPMeta) = meta.method
getInducingInput(meta::UniSGPMeta) = meta.Xu
getKernel(meta::UniSGPMeta) = meta.kernel
getΨ0(meta::UniSGPMeta) = meta.Ψ0 
getΨ1_trans(meta::UniSGPMeta) = meta.Ψ1_trans
getΨ2(meta::UniSGPMeta) = meta.Ψ2
getUv(meta::UniSGPMeta) = meta.Uv # Cholesky upper triangular of Rv = μ_v * μ_v' + Σ_v
getKuuInverse(meta::UniSGPMeta) = meta.Kuu_inverse

## create MultiSGP meta 
mutable struct MultiSGPMeta{I,K}
    method      :: Union{Nothing,AbstractApproximationMethod}
    Xu          :: I    # inducing inputs
    Ψ0          :: Matrix{Float64}
    Ψ1_trans    :: Matrix{Float64}
    Ψ2          :: Matrix{Float64}
    Kuu_inverse :: Matrix{Float64}
    kernel      :: K
    GPCache     :: Union{Nothing,GPCache}
end
getInducingInput(meta::MultiSGPMeta) = meta.Xu
# getCoregionalizationMatrix(meta::MultiSGPMeta) = meta.C
getΨ0(meta::MultiSGPMeta) = meta.Ψ0 
getΨ1_trans(meta::MultiSGPMeta) = meta.Ψ1_trans
getΨ2(meta::MultiSGPMeta) = meta.Ψ2
getKuuInverse(meta::MultiSGPMeta) = meta.Kuu_inverse
getKernel(meta::MultiSGPMeta) = meta.kernel
getGPCache(meta::MultiSGPMeta) = meta.GPCache
getmethod(meta::MultiSGPMeta) = meta.method




GPCache() = GPCache(Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}(), Dict{Tuple{Symbol,Int}, LowerTriangular{Float64, Matrix{Float64}}}())

function getcache(cache::GPCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::GPCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end

function getcache_lowermatrix(cache::GPCache, label::Tuple{Symbol, Int})
    return get!(() -> LowerTriangular{Float64}(rand(label[2],label[2])), cache.cache_LowerTriangular, label)
end

function mul_A_B!(cache::GPCache, A::AbstractArray, B, sizeA1::Int, sizeB2::Int)
    AB = getcache(cache, (:ABdiff, (sizeA1,sizeB2)))
    return mul!(AB, A, B)
end

function mul_A_B!(cache::GPCache, A::Array, B, size1::Int)
    #multiply 2 matrices with the same size
    AB = getcache(cache, (:AB, (size1, size1)))
    return mul!(AB, A, B)
end

function mul_A_B_A!(cache::GPCache, A::Matrix, B::Matrix, size1::Int)
    #A, B are square matrices with a same size
    AB = getcache(cache, (:AB, (size1, size1)))
    ABA = getcache(cache, (:ABA, (size1, size1)))
    mul!(AB, A, B)

    return mul!(ABA, AB, A)
end

function mul_A_B_At!(cache::GPCache, A::Matrix, B::Matrix, sizeA1::Int, sizeB1::Int)
    # A: matrix with size (sizeA1,sizeB1)
    # B: matrix with size (sizeB1,sizeB1)
    AB = getcache(cache, (:AB, (sizeA1, sizeB1)))
    ABAt = getcache(cache, (:ABA, (sizeA1, sizeA1)))
    mul!(AB, A, B)
    return mul!(ABAt, AB, A')
end
function mul_A_v!(cache::GPCache, A::Matrix, v::Vector, sizeA1::Int)
    Av = getcache(cache, (:Av, sizeA1))
    return mul!(Av, A, v)
end

function jdotavx(a::T, b::F) where {T<:AbstractArray, F <: AbstractArray}
    s = zero(eltype(a))
    @turbo for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end

function create_blockmatrix(A,d,M)
    return [view(A,i:i+M-1,j:j+M-1) for i=1:M:M*d, j=1:M:M*d]
end

function split2batch(data::Tuple,batch_size)
    x, y = data
    x_batch = [x[i:min(i + batch_size - 1, end)] for i in 1:batch_size:length(x)]
    y_batch = [y[i:min(i + batch_size - 1, end)] for i in 1:batch_size:length(y)]
    return x_batch, y_batch
end

# standardized mean squared error (for regression)
function SMSE(y_true, y_approx)
    N = length(y_true)
    mse = norm(y_true - y_approx)^2 / N 
    return mse / var(y_true)
end

# count number of errors (for classification)
function num_error(ytrue, y)
    return sum(abs.(y - ytrue))
end

function error_rate(ytrue, y)
    return num_error(ytrue,y) / length(ytrue)
end