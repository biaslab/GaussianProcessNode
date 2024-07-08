using Test 
using ReactiveMP,RxInfer
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using KernelFunctions, LoopVectorization
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: WishartFast, approximate_kernel_expectation, approximate_meancov
using Revise 

include("UniSGPnode.jl")
include("MultiSGPnode.jl")


method = ghcubature(21)
θ_val = [1,1]
Xu = collect(1:10)
kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
C = [1. 0.;0. 1.]
gpcache = GPCache()
Unimeta = UniSGPMeta(method,Xu,kernel)
Multimeta = MultiSGPMeta(method, Xu, C, kernel, gpcache)

A = rand(4,4)
blk_A = [A[1:2,1:2], A[1:2,3:4], A[3:4,1:2],A[3:4,3:4]]
B = rand(3,4)
D = rand(4,4)
a = rand(4)
b = rand(4)

A_xθ = (x,θ) -> kernelmatrix(kernel(θ),[x],[x]) .- kernelmatrix(kernel(θ),[x],Xu) * inv(kernelmatrix(kernel(θ),Xu)) * kernelmatrix(kernel(θ),Xu,[x])
B_xθ = (x,θ) -> kernelmatrix(kernel(θ), [x], Xu)

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
    # Univariate GP 
    @test getInducingInput(Unimeta) == Xu
    @test getKernel(Unimeta) == kernel
    @test typeof(getKernel(Unimeta)) <: Function
    @test getmethod(Unimeta) == method

    # Multivariate GP 
    @test getInducingInput(Multimeta) == Xu
    @test getKernel(Multimeta) == kernel
    @test typeof(getKernel(Multimeta)) <: Function
    @test getmethod(Multimeta) == method
    @test getGPCache(Multimeta) == gpcache
    @test getCoregionalizationMatrix(Multimeta) == C
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
    q_θ = PointMass(θ_val)
    sample_x = rand(q_x,5000)
    Ψ2_func = (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu)
    Ψ0_func = (x) -> getindex(kernelmatrix(kernel(θ_val),[x]),1)
    B_x = (x) -> B_xθ(x,θ_val)

    Ψ0_gt = mean(Ψ0_func.(sample_x))
    Ψ1_gt = mean(B_x.(sample_x))
    Ψ2_gt = mean(Ψ2_func.(sample_x))

    Ψ0_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_x)[]
    Ψ1_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ_val), [x], Xu),q_x)
    Ψ2_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_x)

    @test isapprox(Ψ0_gt, Ψ0_approx ;atol = 1e-4)
    @test isapprox(Ψ1_gt, Ψ1_approx ;atol = 0.05)
    @test isapprox(Ψ2_gt, Ψ2_approx ;atol = 0.05)

    #test rules for out message 
    gt_mean_y =  getindex(Ψ1_gt * mean(q_v),1)
    gt_var_y = inv(mean(q_w))
    ν_y =  @call_rule UniSGP(:out, Marginalisation) (q_in=q_x, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    @test mean_var(ν_y) == (gt_mean_y)

    #test rules for in message 
    # ν_x = @call_rule UniSGP(:in, Marginalisation) ()
end

# @testset "Test Multivariate SGP" begin
    
# end
