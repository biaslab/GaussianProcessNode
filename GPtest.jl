using Test 
using ReactiveMP,RxInfer
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using KernelFunctions, LoopVectorization
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: WishartFast, approximate_kernel_expectation, approximate_meancov
using Revise 

include("UnivariateSGPnode.jl")
include("multivariateSGPnode.jl")


method = ghcubature(21)
Xu = collect(1:10)
kernel = SEKernel()
Kuu_inverse = inv(kernelmatrix(kernel,Xu))
C = [1. 0.;0. 1.]
gpcache = GPCache()
gpmeta = GPTransitionMeta(method,Xu,Kuu_inverse,C,kernel,gpcache)

A = rand(4,4)
blk_A = [A[1:2,1:2], A[1:2,3:4], A[3:4,1:2],A[3:4,3:4]]
B = rand(3,4)
D = rand(4,4)
a = rand(4)
b = rand(4)

A_x = (x) -> kernelmatrix(kernel,[x],[x]) .- kernelmatrix(kernel,[x],Xu) * Kuu_inverse * kernelmatrix(kernel,Xu,[x])
B_x = (x) -> kernelmatrix(kernel, [x], Xu)

@testset "Test GPCache" begin
    @test typeof(gpcache.cache_matrices) <: Dict
    @test typeof(gpcache.cache_vectors) <: Dict 
    @test typeof(getcache(gpcache,(:A, (3,3)))) <: Matrix
    @test typeof(getcache(gpcache,(:a,3))) <: Vector
    @test mul_A_B!(gpcache, B, A, size(B,1),size(A,2)) == B * A 
    @test mul_A_B!(gpcache,A,D,size(A,1)) == A * D
    @test mul_A_B_A!(gpcache,A,D,size(A,1)) == A * D * A
    @test mul_A_B_At!(gpcache,B,A,size(B,1),size(A,1)) == B * A * B'
    @test mul_A_v!(gpcache,A,a,size(A,1)) == A * a
end

@testset "Test GPMeta" begin
    @test getInducingInput(gpmeta) == Xu
    @test getKernel(gpmeta) == kernel
    @test getinverseKuu(gpmeta) == Kuu_inverse
    @test getGPCache(gpmeta) == gpcache
    @test getmethod(gpmeta) == method
    @test getCoregionalizationMatrix(gpmeta) == C
end

@testset "Test helper functions" begin
    @test jdotavx(a,b) == dot(a,b)
    blk_matrix = create_blockmatrix(A,2,2)
    for i in eachindex(blk_matrix)
        @test blk_matrix[i] == blk_A[i] 
    end
end

@testset "Test Univariate SGP" begin
    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(4) |> (x) -> sin.(x), diageye(4))
    q_x = Normal(0,1)
    sample_x = rand(q_x,3000)
    Ψ2_func = (x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu)
    Ψ0_func = (x) -> getindex(kernelmatrix(kernel,[x]),1)

    Ψ0_gt = mean(Ψ0_func.(sample_x))
    Ψ1_gt = mean(B_x.(sample_x))
    Ψ2_gt = mean(Ψ2_func.(sample_x))

    Ψ0_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel, [x], [x]),q_x)[]
    Ψ1_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel, [x], Xu),q_x)
    Ψ2_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel, Xu, [x]) * kernelmatrix(kernel, [x], Xu), q_x)

    @test isapprox(Ψ0_gt, Ψ0_approx ;atol = 1e-4)
    @test isapprox(Ψ1_gt, Ψ1_approx ;atol = 0.05)
    @test isapprox(Ψ2_gt, Ψ2_approx ;atol = 0.05)
    
end

# @testset "Test Multivariate SGP" begin
    
# end
