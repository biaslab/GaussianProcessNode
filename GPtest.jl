using Test 
using ReactiveMP,RxInfer
using Zygote, Optim, ForwardDiff
using Random, Distributions, LinearAlgebra, SpecialFunctions 
using KernelFunctions, LoopVectorization
import KernelFunctions: with_lengthscale, kernelmatrix 
import ReactiveMP: WishartFast, approximate_kernel_expectation, approximate_meancov
using Revise 

include("GPnode/UniSGPnode.jl")
include("GPnode/MultiSGPnode.jl")
include("helper_functions/derivative_helper.jl")

method_uni = ghcubature(21)
method_multi = srcubature()
θ_val = [1.,1.]
Nu = 10
Nu_2d = 25
Xu = collect(1:Nu) #inducing points for univariate case
Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d) #inducing points for multivariate case
kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
C = [1. 0.;0. 1.]
Kuu_inverse = cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I)
gpcache = GPCache()
Ψ0 = [1.0;;]
Ψ1_trans = kernelmatrix(kernel(θ_val),Xu,[1.])
Ψ2 = kernelmatrix(kernel(θ_val),Xu,[1.]) * kernelmatrix(kernel(θ_val),[1.],Xu);
Kuu = kernelmatrix(kernel(θ_val), Xu) + 1e-8 * I
Uv = cholesky(Kuu).U;
KuuL = fastcholesky(Kuu).L
Unimeta = UniSGPMeta(method_uni,Xu,Ψ0,Ψ1_trans,Ψ2,KuuL,kernel,Uv,0,Nu)

Ψ1_trans_2d = kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]])
Ψ2_2d = kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d);
Multimeta = MultiSGPMeta(method_multi, Xu_2d,Ψ0,Ψ1_trans_2d,Ψ2_2d,Kuu_inverse, kernel, gpcache)

A = rand(4,4)
blk_A = [A[1:2,1:2], A[3:4,1:2], A[1:2,3:4],A[3:4,3:4]]
B = rand(3,4)
D = rand(4,4)
a = rand(4)
b = rand(4)

A_xθ = (x,θ) -> kernelmatrix(kernel(θ),[x],[x]) .- kernelmatrix(kernel(θ),[x],Xu) * inv(kernelmatrix(kernel(θ),Xu)) * kernelmatrix(kernel(θ),Xu,[x])
B_xθ = (x,θ) -> kernelmatrix(kernel(θ), [x], Xu)

A_2d_xθ = (x,θ) -> kernelmatrix(kernel(θ),[x],[x]) - kernelmatrix(kernel(θ),[x],Xu_2d) * inv(kernelmatrix(kernel(θ),Xu_2d)) * kernelmatrix(kernel(θ),Xu_2d,[x])
B_2d_xθ = (x,θ) -> kernelmatrix(kernel(θ), [x], Xu_2d)

@testset "Test derivative_helper" begin
    xdata = collect(-5:1:5)
    ydata = sin.(xdata.^2 .- 1) .+ cos.(xdata)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_w = GammaShapeRate(1,1)
    θ_val = [1., 1.];

    μ_v, Σ_v = mean_cov(q_v)
    R_v = Σ_v + μ_v * μ_v'
    w = mean(q_w)

    Ψ0 = (x) -> kernelmatrix(kernel(θ_val), [x], [x])[1]
    Ψ1 = (x) -> kernelmatrix(kernel(θ_val), [x], Xu)
    Ψ2 = (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu)
    Kuu_inverse = inv(kernelmatrix(kernel(θ_val),Xu))
    gt_logbackwardmess = (x,y) -> -0.5 * w * (Ψ0(x) + tr(Ψ2(x) * (R_v - Kuu_inverse)) ) + w * y * dot(Ψ1(x), μ_v)
    gt_negllh = 0.0
    for i=1:length(xdata)
        gt_negllh += gt_logbackwardmess(xdata[i],ydata[i])
    end
    Uv = cholesky(R_v).U
    approx_negllh = neg_log_backwardmess_fast(θ_val; y_data=ydata, x_data=xdata, v=μ_v, Uv=Uv, w=w, kernel=kernel, Xu=Xu)
    # approx_negllh_2 = neg_log_backwardmess_forward(θ_val; y_data=ydata, x_data=xdata, v=μ_v, Uv=Uv, w=w, kernel=kernel, Xu=Xu)
    @test isapprox(-gt_negllh,approx_negllh;atol=1e-6)
    # @test isapprox(-gt_negllh,approx_negllh_2;atol=1e-6)
end

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
    @test getmethod(Unimeta) == method_uni

    # Multivariate GP 
    @test getInducingInput(Multimeta) == Xu_2d
    @test getKernel(Multimeta) == kernel
    @test typeof(getKernel(Multimeta)) <: Function
    @test getmethod(Multimeta) == method_multi
    @test getGPCache(Multimeta) == gpcache
    # @test getCoregionalizationMatrix(Multimeta) == C
    @test getKuuInverse(Multimeta) == Kuu_inverse
end

@testset "Test helper functions" begin
    @test jdotavx(a,b) ≈ dot(a,b)
    blk_matrix = create_blockmatrix(A,2,2)
    for i in eachindex(blk_matrix)
        @test blk_matrix[i] == blk_A[i] 
    end
end

@testset "Test Univariate SGP" begin
    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_x = Normal(0,1)
    q_θ = PointMass(θ_val)

    μ_y = mean(q_out)
    μ_v = mean(q_v)
    R_v = μ_v * μ_v' + cov(q_v)
    E_logw = mean(log,q_w)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ_val),Xu))

    sample_x = rand(q_x,5000)
    Ψ2_func = (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu)
    Ψ0_func = (x) -> getindex(kernelmatrix(kernel(θ_val),[x]),1)
    A_x = (x) -> A_xθ(x,θ_val)
    B_x = (x) -> B_xθ(x,θ_val)

    Ψ0_gt = mean(Ψ0_func.(sample_x))
    Ψ1_gt = mean(B_x.(sample_x))
    Ψ2_gt = mean(Ψ2_func.(sample_x)) + 1e-7*I

    Ψ0_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_x)[]
    Ψ1_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ_val), [x], Xu),q_x)
    Ψ2_approx = approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_x)

    @test isapprox(Ψ0_gt, Ψ0_approx ;atol = 1e-4)
    @test isapprox(Ψ1_gt, Ψ1_approx ;atol = 0.05)
    @test isapprox(Ψ2_gt, Ψ2_approx ;atol = 0.05)

    Kuu = kernelmatrix(kernel(θ_val), Xu)
    KuuL = fastcholesky(Kuu).L
    Uv = cholesky(R_v).U
    Ψ0 = [1.0;;]
    Ψ1_trans = kernelmatrix(kernel(θ_val),Xu,[1.])
    Ψ2 = kernelmatrix(kernel(θ_val),Xu,[1.]) * kernelmatrix(kernel(θ_val),[1.],Xu);
    Unimeta_new = UniSGPMeta(method_uni,Xu,Ψ0,Ψ1_trans,Ψ2,KuuL,kernel,Uv,0,1)
    ### test rules for "out" message 
    @testset "Rules for out" begin
    gt_mean_y =  getindex(Ψ1_approx * mean(q_v),1)
    gt_var_y = inv(mean(q_w))
        @testset "q_in::Normal" begin
            ν_y_1 =  @call_rule UniSGP(:out, Marginalisation) (q_in = q_x, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta_new)
            @test typeof(ν_y_1) <: UnivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_y_1), gt_mean_y ; atol=1e-7)
            @test isapprox(var(ν_y_1), gt_var_y)
        end

        @testset "q_in::PointMass" begin 
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            ν_y_2 = @call_rule UniSGP(:out, Marginalisation) (q_in = PointMass(1.0), q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta_new)
            @test typeof(ν_y_2) <: UnivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_y_2), getindex(Ψ1 * mean(q_v),1))
            @test isapprox(var(ν_y_2), gt_var_y)
        end
    end

    ### test rules for "in" message 
    @testset "Rules for in" begin
        gt_logbackwardmess_x = (x) -> getindex(-0.5 * mean(q_w) * (A_x(x) + B_x(x) * R_v * B_x(x)' - 2* μ_y * B_x(x)*μ_v),1)
        ν_x = @call_rule UniSGP(:in, Marginalisation) (q_out = q_out, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta_new)
        @test typeof(ν_x) <: ContinuousUnivariateLogPdf
        @test isapprox(logpdf(ν_x,1.0), gt_logbackwardmess_x(1.0))
        @test isapprox(logpdf(ν_x,sqrt(2)), gt_logbackwardmess_x(sqrt(2)))
        @test isapprox(logpdf(ν_x,4.2), gt_logbackwardmess_x(4.2))
    end

    ### test rules for "v" message 
    @testset "Rules for v" begin
        @testset "q_out::Normal, q_in::Normal" begin
            ν_v_1 = @call_rule UniSGP(:v, Marginalisation) (q_out = q_out, q_in = q_x, q_w = q_w, q_θ = q_θ, meta = Unimeta_new)
            gt_mean_v_1 = vcat(inv(Ψ2_approx + 1e-8*I) * Ψ1_approx' * μ_y...) 
            gt_cov_v_1 = inv(mean(q_w) * (Ψ2_approx + 1e-8*I))
            @test typeof(ν_v_1) <: BufferUniSGP
            @test typeof(ν_v_1.qv) <: MultivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_v_1.qv), gt_mean_v_1)
            @test isapprox(cov(ν_v_1.qv), gt_cov_v_1)
        end

        @testset "q_out, q_in::PointMass" begin 
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) 
            ν_v_2 = @call_rule UniSGP(:v, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_w = q_w, q_θ = q_θ, meta = Unimeta_new)
            gt_mean_v_2 = vcat(cholinv(Ψ2) * Ψ1' * 2...) 
            gt_cov_v_2 = cholinv(mean(q_w) * Ψ2)
            @test typeof(ν_v_2) <: BufferUniSGP
            @test typeof(ν_v_2.qv) <: MultivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_v_2.qv), gt_mean_v_2)
            @test isapprox(cov(ν_v_2.qv), gt_cov_v_2)
        end

        @testset "q_out::Normal, q_in::PointMass" begin
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu)
            ν_v_3 = @call_rule UniSGP(:v, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_w = q_w, q_θ = q_θ, meta = Unimeta_new)
            gt_mean_v_3 = vcat(cholinv(Ψ2) * Ψ1' * μ_y...) 
            gt_cov_v_3 = cholinv(mean(q_w) * Ψ2)
            @test typeof(ν_v_3) <: BufferUniSGP
            @test typeof(ν_v_3.qv) <: MultivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_v_3.qv), gt_mean_v_3)
            @test isapprox(cov(ν_v_3.qv), gt_cov_v_3);
        end
    end

    ### test rule for "w" message 
    @testset "Rules for w" begin
        @testset "q_out, q_in::Normal" begin
            I1 = Ψ0_approx - tr(Kuu_inverse * Ψ2_approx)
            I2 = mean(q_out)^2 + var(q_out) - 2*mean(q_out)*getindex(Ψ1_approx*mean(q_v),1) + tr(R_v*Ψ2_approx)
            rate_gt = 0.5 * (I1 + I2)
            ν_w_1 = @call_rule UniSGP(:w, Marginalisation) (q_out = q_out, q_in = q_x, q_v = q_v, q_θ = q_θ, meta = Unimeta_new)
            @test typeof(ν_w_1) <: GammaDistributionsFamily
            @test shape(ν_w_1) == 1.5
            @test isapprox(rate(ν_w_1), rate_gt; atol=1e-5)
        end

        @testset "q_out, q_in::PointMass" begin
            Ψ0 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-7*I 
            I1 = Ψ0 - tr(Kuu_inverse * Ψ2)
            I2 = 2.0^2 - 2*2.0*getindex(Ψ1 * mean(q_v),1) + tr(R_v * Ψ2)
            ν_w_2 = @call_rule UniSGP(:w, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_v = q_v, q_θ = q_θ, meta = Unimeta_new)
            @test typeof(ν_w_2) <: GammaDistributionsFamily
            @test shape(ν_w_2) == 1.5
            @test isapprox(rate(ν_w_2),0.5 * (I1 + I2); atol=1e-5)
        end

        @testset "q_out::Normal, q_in::PointMass" begin 
            Ψ0 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-7*I 
            I1 = Ψ0 - tr(Kuu_inverse * Ψ2)
            I2 = mean(q_out)^2 + var(q_out) - 2*mean(q_out)*getindex(Ψ1*mean(q_v),1) + tr(R_v*Ψ2)
            ν_w_3 = @call_rule UniSGP(:w, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_v = q_v, q_θ = q_θ, meta = Unimeta_new)
            @test typeof(ν_w_3) <: GammaDistributionsFamily
            @test shape(ν_w_3) == 1.5
            @test isapprox(rate(ν_w_3),0.5 * (I1 + I2); atol=1e-5)
        end
    end

    ### test rule for "θ" message 
    @testset "Rules for θ" begin
    Kuu_inverse_θ = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))
        @testset "q_out, q_in:: Normal" begin
            Ψ0_θ = (θ) -> approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_x)[]
            Ψ1_θ = (θ) -> approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_x)
            Ψ2_θ = (θ) -> approximate_kernel_expectation(ghcubature(21),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_x)
            gt_logbackwardmess_θ = (θ) -> -0.5 * mean(q_w) * (Ψ0_θ(θ) + tr(Ψ2_θ(θ) * (R_v - Kuu_inverse_θ(θ))) ) + mean(q_w) * mean(q_out) * getindex(Ψ1_θ(θ) * μ_v,1)
            ν_θ_1 = @call_rule UniSGP(:θ, Marginalisation) (q_out = q_out, q_in = q_x, q_v = q_v, q_w = q_w, meta = Unimeta_new)
            @test typeof(ν_θ_1) <: ContinuousMultivariateLogPdf
            @test isapprox(logpdf(ν_θ_1,[1,2]), gt_logbackwardmess_θ([1,2]);atol=1e-7)
            @test isapprox(logpdf(ν_θ_1,[0.5,1.4]), gt_logbackwardmess_θ([0.5,1.4]); atol=1e-7)
        end

        @testset "q_out::Normal, q_in::PointMass" begin 
            Ψ0_θ_2 = (θ) -> kernelmatrix(kernel(θ), [1.0], [1.0])[1]
            Ψ1_θ_2 = (θ) -> kernelmatrix(kernel(θ), [1.0], Xu)
            Ψ2_θ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [1.0]) * kernelmatrix(kernel(θ), [1.0], Xu) 
            gt_logbackwardmess_θ_2 = (θ) -> -0.5 * mean(q_w) * (Ψ0_θ_2(θ) + tr(Ψ2_θ_2(θ) * (R_v - Kuu_inverse_θ(θ))) ) + mean(q_w) * mean(q_out) * getindex(Ψ1_θ_2(θ) * μ_v,1)
            ν_θ_2 = @call_rule UniSGP(:θ, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_v = q_v, q_w = q_w, meta = Unimeta_new)
            @test typeof(ν_θ_2) <: ContinuousMultivariateLogPdf
            @test isapprox(logpdf(ν_θ_2,[1,2]), gt_logbackwardmess_θ_2([1,2]);atol=1e-9)
            @test isapprox(logpdf(ν_θ_2,[0.5,1.4]), gt_logbackwardmess_θ_2([0.5,1.4]);atol = 1e-9)
        end

        @testset "q_out, q_in::PointMass" begin
            Ψ0_θ_2 = (θ) -> kernelmatrix(kernel(θ), [1.0], [1.0])[1]
            Ψ1_θ_2 = (θ) -> kernelmatrix(kernel(θ), [1.0], Xu)
            Ψ2_θ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [1.0]) * kernelmatrix(kernel(θ), [1.0], Xu)
            gt_logbackwardmess_θ_3 = (θ) -> -0.5 * mean(q_w) * (Ψ0_θ_2(θ) + tr(Ψ2_θ_2(θ) * (R_v - Kuu_inverse_θ(θ))) ) + mean(q_w) * 2.0 * getindex(Ψ1_θ_2(θ) * μ_v,1)
            ν_θ_3 = @call_rule UniSGP(:θ, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_v = q_v, q_w = q_w, meta = Unimeta_new)
            @test typeof(ν_θ_3) <: ContinuousMultivariateLogPdf
            @test isapprox(logpdf(ν_θ_3,[1,2]), gt_logbackwardmess_θ_3([1,2]);atol=1e-9)
            @test isapprox(logpdf(ν_θ_3,[0.5,1.4]), gt_logbackwardmess_θ_3([0.5,1.4]);atol=1e-9)
        end
    end

    ### Test Average Free energy
    @testset "Test average energy" begin
        @testset "q_out = PointMass(2.0), q_in = PointMass(1.0), q_w::Gamma" begin
            Ψ0_1 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
            Ψ1_1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2_1 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-7*I 
            I1_1 = Ψ0_1 - tr(Kuu_inverse * Ψ2_1)
            I2_1 = 2.0^2 - 2*2.0*getindex(Ψ1_1 * mean(q_v),1) + tr(R_v * Ψ2_1)
            U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1_1 + I2_1)
            
            marginals = (Marginal(PointMass(2.0), false, false, nothing), Marginal(PointMass(1.0), false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta_new)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol = 1e-5)
        end

        @testset "q_out::Normal, q_in = PointMass(1.0), q_w::Gamma" begin 
            Ψ0_2 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
            Ψ1_2 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2_2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-7*I 
            I1_2 = Ψ0_2 - tr(Kuu_inverse * Ψ2_2)
            I2_2 = mean(q_out)^2 + var(q_out)- 2*mean(q_out)*getindex(Ψ1_2 * mean(q_v),1) + tr(R_v * Ψ2_2)
            U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1_2 + I2_2)
            
            marginals = (Marginal(q_out, false, false, nothing), Marginal(PointMass(1.0), false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta_new)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol = 1e-5)
        end

        @testset "q_out, q_in::Normal, q_w::Gamma" begin
            I1_3 = Ψ0_approx - tr(Kuu_inverse * Ψ2_approx)
            I2_3 = mean(q_out)^2 + var(q_out)- 2*mean(q_out)*getindex(Ψ1_approx * mean(q_v),1) + tr(R_v * Ψ2_approx)
            U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1_3 + I2_3)
            
            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_x, false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta_new)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol=1e-5)
        end

        @testset "q_out, q_in::Normal, q_w::PointMass" begin
            w = 5.0
            I1_4 = Ψ0_approx - tr(Kuu_inverse * Ψ2_approx)
            I2_4 = mean(q_out)^2 + var(q_out)- 2*mean(q_out)*getindex(Ψ1_approx * mean(q_v),1) + tr(R_v * Ψ2_approx)
            U_gt = 0.5 * log(2π) - 0.5 * log(w) + 0.5 * w * (I1_4 + I2_4)
            
            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_x, false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(PointMass(w), false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta_new)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt;atol=1e-6);
        end
    end
end;

@testset "Test Multivariate SGP" begin
    q_out = MvNormal([0.5, 1.4], diageye(2))
    q_in =  MvNormal([1.0, 2.7], diageye(2))
    q_v = MvNormalMeanCovariance(rand(2*Nu_2d) |> (x) -> sin.(x), diageye(2*Nu_2d))
    q_w = Wishart(10, 50*diageye(2))
    q_θ = PointMass(θ_val)

    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    R_v = μ_v * μ_v' + Σ_v
    W_mean = mean(q_w)
    E_logdet_W = mean(logdet, q_w)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I)

    sample_x = [rand(q_in) for i=1:10000]
    Ψ2_func = (x) -> kernelmatrix(kernel(θ_val), Xu_2d, [x]) * kernelmatrix(kernel(θ_val), [x], Xu_2d)
    Ψ0_func = (x) -> getindex(kernelmatrix(kernel(θ_val),[x]),1)
    A_x = (x) -> A_2d_xθ(x,θ_val)
    B_x = (x) -> B_2d_xθ(x,θ_val)

    Ψ0_gt = mean(Ψ0_func.(sample_x))
    Ψ1_gt = mean(B_x.(sample_x))
    Ψ2_gt = mean(Ψ2_func.(sample_x)) + 1e-7*I

    Ψ0_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_in)[]
    Ψ1_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], Xu_2d),q_in)
    Ψ2_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), Xu_2d, [x]) * kernelmatrix(kernel(θ_val), [x], Xu_2d), q_in)

    @test Ψ0_approx == Ψ0_gt
    @test isapprox(Ψ1_approx,Ψ1_gt;atol = 0.08)
    @test isapprox(Ψ2_approx, Ψ2_gt; atol = 0.3)

    ### Test rules for out message 
    @testset "Rules for out" begin
        @testset "q_in::MvNormal, q_W::PointMass" begin
            mean_out_gt = kron(C,Ψ1_approx) * μ_v
            cov_out_gt = inv(W_mean)
            ν_out = @call_rule MultiSGP(:out, Marginalisation) (q_in = q_in, q_v = q_v, q_w = PointMass(W_mean), q_θ = q_θ, meta = Multimeta)
            @test typeof(ν_out) <: MultivariateGaussianDistributionsFamily
            @test mean(ν_out) ≈ mean_out_gt
            @test cov(ν_out) == cov_out_gt
        end

        @testset "q_in::MvNormal, q_W::Wishart" begin
            mean_out_gt = kron(C,Ψ1_approx) * μ_v
            cov_out_gt = inv(W_mean)
            ν_out = @call_rule MultiSGP(:out, Marginalisation) (q_in = q_in, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Multimeta)
            @test typeof(ν_out) <: MultivariateGaussianDistributionsFamily
            @test mean(ν_out) ≈ mean_out_gt
            @test cov(ν_out) == cov_out_gt
        end
    end

    ### Test rules for in message 
    @testset "Rules for in" begin
        @testset "q_out::MvNormal, q_w::Wishart" begin
            gt_logbackwardmess_in = (x) -> -0.5 * tr(W_mean * kron(C,A_x(x))) + μ_y' * W_mean * kron(C,B_x(x)) * μ_v - 0.5*tr(R_v * kron(C,B_x(x))' * W_mean * kron(C,B_x(x)))
            ν_in = @call_rule MultiSGP(:in, Marginalisation) (q_out = q_out, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Multimeta)
            @test typeof(ν_in) <: ContinuousMultivariateLogPdf
            @test logpdf(ν_in, [1.0, 1.5]) ≈ gt_logbackwardmess_in([1.0, 1.5])
            @test logpdf(ν_in, [-1.5, 2.0]) ≈ gt_logbackwardmess_in([-1.5, 2.0])
        end

        @testset "q_out::PointMass,q_in::MvNormal, q_w::PointMass" begin
            q_out_pm = PointMass([1.5,2.0])
            μ_y_pm = [1.5,2.0]
            q_w_pm = PointMass(W_mean)
            gt_logbackwardmess_in = (x) -> -0.5 * tr(W_mean * kron(C,A_x(x))) + μ_y_pm' * W_mean * kron(C,B_x(x)) * μ_v - 0.5*tr(R_v * kron(C,B_x(x))' * W_mean * kron(C,B_x(x)))
            gt_neg_logbackwardmess_in = (x) -> - gt_logbackwardmess_in(x)
            res = optimize(gt_neg_logbackwardmess_in,mean(q_in),LBFGS(),Optim.Options(iterations=20))
            m_z = res.minimizer
            W_z = Zygote.hessian(gt_neg_logbackwardmess_in, m_z) 
            ν_in = @call_rule MultiSGP(:in, Marginalisation) (q_out = q_out_pm, q_in = q_in, q_v = q_v, q_w = q_w_pm, q_θ = q_θ, meta = Multimeta)
            @test typeof(ν_in) <: MultivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_in), m_z;atol=0.01) 
            @test isapprox(cov(ν_in), inv(W_z);atol=0.01)
        end
    end

    ### Test rules for v message
    @testset "Rules for v" begin
        @testset "q_out,q_in::MvNormal, q_w::Wishart" begin
            Ψ3 = kron(W_mean, Ψ2_approx)
            Ψ1_tilde = kron(C, Ψ1_approx)
            gt_mean_v = cholinv(Ψ3) * Ψ1_tilde' * W_mean * μ_y 
            gt_cov_v = cholinv(Ψ3)
            ν_v = @call_rule MultiSGP(:v, Marginalisation) (q_out = q_out, q_in = q_in, q_w = q_w, q_θ = q_θ, meta = Multimeta)
            @test typeof(ν_v) <: MultivariateGaussianDistributionsFamily
            @test mean(ν_v) ≈ gt_mean_v 
            @test cov(ν_v) ≈ gt_cov_v
        end

        @testset "q_out::PointMass, q_in::MvNormal, q_w::PointMass" begin
          q_out_pm = PointMass([1.5,2.0])
          q_w_pm = PointMass(W_mean)
          Ψ3 = kron(W_mean, Ψ2_approx)
          Ψ1_tilde = kron(C, Ψ1_approx)
          gt_mean_v = cholinv(Ψ3) * Ψ1_tilde' * W_mean * mean(q_out_pm) 
          gt_cov_v = cholinv(Ψ3)
          ν_v = @call_rule MultiSGP(:v, Marginalisation) (q_out = q_out_pm, q_in = q_in, q_w = q_w_pm, q_θ = q_θ, meta = Multimeta)
          @test typeof(ν_v) <: MultivariateGaussianDistributionsFamily 
          @test mean(ν_v) ≈ gt_mean_v 
          @test cov(ν_v) ≈ gt_cov_v
        end
    end

    ### Test rules for W message 
    @testset "Rules for W" begin
        Ψ1_tilde = kron(C, Ψ1_approx)
        Ψ4_approx = approximate_kernel_expectation(srcubature(),(x) -> kron(C,kernelmatrix(kernel(θ_val), [x], Xu_2d)) * R_v * kron(C,kernelmatrix(kernel(θ_val), Xu_2d, [x])), q_in) #+ 1e-7*I
        I1 = kron(C, Ψ0_approx - tr(Kuu_inverse * Ψ2_approx))
        I2 = μ_y * μ_y' + Σ_y - μ_y * μ_v' * Ψ1_tilde' - Ψ1_tilde * μ_v * μ_y' + Ψ4_approx 
        gt_n_w = length(mean(q_out)) + 2
        gt_V_w = inv(I1 + I2)
        ν_w = @call_rule MultiSGP(:w, Marginalisation) (q_out = q_out, q_in = q_in, q_v = q_v, q_θ = q_θ, meta = Multimeta)
        @test typeof(ν_w) <: WishartDistributionsFamily
        n_w, V_w = params(ν_w)
        @test n_w == gt_n_w 
        @test isapprox(gt_V_w ,V_w; atol=1e-5)
    end

    ### Test rules for θ message 
    @testset "Rules for θ" begin
        Kuu_inverse_θ = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu_2d))
        @testset "q_out, q_in::MvNormal, q_w::Wishart" begin
            Ψ0_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[]
            Ψ1_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], Xu_2d),q_in)
            Ψ2_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), Xu_2d, [x]) * kernelmatrix(kernel(θ), [x], Xu_2d), q_in) + 1e-7*I
            I1_θ = (θ) -> kron(C, Ψ0_θ(θ) - tr(Kuu_inverse_θ(θ) * Ψ2_θ(θ)))
            Ψ1_tilde = (θ) -> kron(C, Ψ1_θ(θ))
            Ψ3_θ = (θ) -> kron(W_mean, Ψ2_θ(θ))
            gt_logbackwardmess_θ = (θ) -> -0.5 * tr(W_mean * I1_θ(θ)) + μ_y' * W_mean * Ψ1_tilde(θ) * μ_v - 0.5 * tr(Ψ3_θ(θ) * R_v)
            ν_θ = @call_rule MultiSGP(:θ, Marginalisation) (q_out = q_out, q_in = q_in, q_v = q_v, q_w = q_w, meta = Multimeta)
            @test typeof(ν_θ) <: ContinuousMultivariateLogPdf
            @test logpdf(ν_θ, [1.2, 2.3]) ≈ gt_logbackwardmess_θ([1.2, 2.3])
            @test logpdf(ν_θ, [0.5, 1.4]) ≈ gt_logbackwardmess_θ([0.5, 1.4])
        end

        @testset "q_out::PointMass, q_in::MvNormal, q_w::PointMass" begin
            q_out_pm = PointMass([1.5,2.0])
            μ_y_pm = mean(q_out_pm)
            q_w_pm = PointMass(W_mean)
            Ψ0_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[]
            Ψ1_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], Xu_2d),q_in)
            Ψ2_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), Xu_2d, [x]) * kernelmatrix(kernel(θ), [x], Xu_2d), q_in) + 1e-7*I
            I1_θ = (θ) -> kron(C, Ψ0_θ(θ) - tr(Kuu_inverse_θ(θ) * Ψ2_θ(θ)))
            Ψ1_tilde = (θ) -> kron(C, Ψ1_θ(θ))
            Ψ3_θ = (θ) -> kron(W_mean, Ψ2_θ(θ))
            gt_logbackwardmess_θ = (θ) -> -0.5 * tr(W_mean * I1_θ(θ)) + μ_y_pm' * W_mean * Ψ1_tilde(θ) * μ_v - 0.5 * tr(Ψ3_θ(θ) * R_v)
            ν_θ = @call_rule MultiSGP(:θ, Marginalisation) (q_out = q_out_pm, q_in = q_in, q_v = q_v, q_w = q_w_pm, meta = Multimeta)
            @test typeof(ν_θ) <: ContinuousMultivariateLogPdf
            @test logpdf(ν_θ, [1.2, 2.3]) ≈ gt_logbackwardmess_θ([1.2, 2.3])
            @test logpdf(ν_θ, [0.5, 1.4]) ≈ gt_logbackwardmess_θ([0.5, 1.4])
        end
    end

    ### Test average energy 
    @testset "Average energy" begin
        @testset "q_out,q_in::MvNormal, q_w::Wishart" begin
            Ψ1_tilde = kron(C, Ψ1_approx)
            Ψ4_approx = approximate_kernel_expectation(srcubature(),(x) -> kron(C,kernelmatrix(kernel(θ_val), [x], Xu_2d)) * R_v * kron(C,kernelmatrix(kernel(θ_val), Xu_2d, [x])), q_in) #+ 1e-7*I
            I1 = kron(C, Ψ0_approx - tr(Kuu_inverse * Ψ2_approx))
            I2 = μ_y * μ_y' + Σ_y - μ_y * μ_v' * Ψ1_tilde' - Ψ1_tilde * μ_v * μ_y' + Ψ4_approx 
            U_gt = 0.5 * tr(W_mean * (I1 + I2)) + length(mean(q_out))/2 * log(2π) - 0.5 * E_logdet_W
            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_in, false, false, nothing), 
            Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), MultiSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Multimeta)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol = 1e-2)
        end

        @testset "q_out::PointMass, q_in::MvNormal, q_w::PointMass" begin
            q_out_pm = PointMass([1.5,2.0])
            q_w_pm = PointMass(W_mean)
            μ_y_pm = mean(q_out_pm)
            Ψ1_tilde = kron(C, Ψ1_approx)
            Ψ4_approx = approximate_kernel_expectation(srcubature(),(x) -> kron(C,kernelmatrix(kernel(θ_val), [x], Xu_2d)) * R_v * kron(C,kernelmatrix(kernel(θ_val), Xu_2d, [x])), q_in) #+ 1e-7*I
            I1 = kron(C, Ψ0_approx - tr(Kuu_inverse * Ψ2_approx))
            I2 = μ_y_pm * μ_y_pm' - μ_y_pm * μ_v' * Ψ1_tilde' - Ψ1_tilde * μ_v * μ_y_pm' + Ψ4_approx 
            U_gt = 0.5 * tr(W_mean * (I1 + I2)) + length(μ_y_pm)/2 * log(2π) - 0.5 * log(det(W_mean))
            marginals = (Marginal(q_out_pm, false, false, nothing), Marginal(q_in, false, false, nothing), 
            Marginal(q_v, false, false, nothing),Marginal(q_w_pm, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), MultiSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Multimeta)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol = 1e-2)
        end
    end
end;
