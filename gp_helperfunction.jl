using LoopVectorization
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: AbstractApproximationMethod


## create UniSGP meta  
struct UniSGPMeta{I,K}
    method      :: Union{Nothing,AbstractApproximationMethod}
    Xu          :: I    # inducing inputs
    kernel      :: K
end
getmethod(meta::UniSGPMeta) = meta.method
getInducingInput(meta::UniSGPMeta) = meta.Xu
getKernel(meta::UniSGPMeta) = meta.kernel

struct GPCache
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
end

## create MultiSGP meta 
struct MultiSGPMeta{I,T,K}
    method      :: Union{Nothing,AbstractApproximationMethod}
    Xu          :: I    # inducing inputs
    C           :: Union{Nothing,T}    # co-regionalization matrix 
    kernel      :: K
    GPCache     :: Union{Nothing,GPCache}
end
getInducingInput(meta::MultiSGPMeta) = meta.Xu
getCoregionalizationMatrix(meta::MultiSGPMeta) = meta.C
getKernel(meta::MultiSGPMeta) = meta.kernel
getGPCache(meta::MultiSGPMeta) = meta.GPCache
getmethod(meta::MultiSGPMeta) = meta.method




GPCache() = GPCache(Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}())

function getcache(cache::GPCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::GPCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end

function mul_A_B!(cache::GPCache, A::Matrix, B, sizeA1::Int, sizeB2::Int)
    AB = getcache(cache, (:ABdiff, (sizeA1,sizeB2)))
    return mul!(AB, A, B)
end

function mul_A_B!(cache::GPCache, A::Matrix, B, size1::Int)
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

function jdotavx(a::T, b::F) where {T,F <: AbstractArray}
    s = zero(eltype(a))
    @turbo for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end

function create_blockmatrix(A,d,M)
    return [view(A,i:i+M-1,j:j+M-1) for j=1:M:M*d, i=1:M:M*d]
end