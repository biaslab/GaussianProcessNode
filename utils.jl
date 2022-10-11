using ReactiveMP, GraphPPL,LinearAlgebra, Random, KernelFunctions
import ReactiveMP: cholinv
import KernelFunctions: Kernel
function predMVN(kernelfunc,meanfunc,xtrain,xtest,y,Σy)
    Kxx                = kernelmatrix(kernelfunc,xtrain,xtrain)
    Kfx                = kernelmatrix(kernelfunc,xtest,xtrain)
    Kff                = kernelmatrix(kernelfunc,xtest,xtest)
    K                  = Kff - Kfx*ReactiveMP.cholinv(Kxx+Σy)*Kfx'
    m                  = meanfunc.(xtest) + Kfx*ReactiveMP.cholinv(Kxx+Σy)*(y-meanfunc.(xtrain))
    
    return m,K
end
struct GaussianProcess
    meanfunction   
    kernelfunction
    finitemarginal
    testinput
    traininput
    invKff
end

@node GaussianProcess Stochastic [out, meanfunc, kernelfunc]


############ FITc 
# function fitc(kernelfunc, meanfunc, xtrain, xinduc,y,Σy)
#     Kxx = kernelmatrix(kernelfunc,xtrain,xtrain) + Σy
#     Kuu = kernelmatrix(kernelfunc,xinduc,xinduc)
#     Kux = kernelmatrix(kernelfunc,xinduc,xtrain)
#     Λ_xx = Diagonal(Kxx - Kux' * cholinv(Kuu) * Kux)

#     Σu = Kuu * cholinv(Kuu + Kux*cholinv(Λ_xx)*Kux') * Kuu
#     μu = meanfunc.(xinduc) + Σu * cholinv(Kuu)*Kux*cholinv(Λ_xx)*(y - meanfunc.(xtrain))

#     return μu, Σu 
# end
# function predMVN(kernelfunc,meanfunc,xtrain,xtest,y,Σy)
#     xinduc             = xtrain[sort(randperm(length(xtrain))[1:Int(round(length(xtrain)/2))])] #inducing points
#     μu, Σu             = fitc(kernelfunc, meanfunc, xtrain, xinduc, y, Σy)
#     Kuu                = kernelmatrix(kernelfunc,xinduc,xinduc)
#     Kfu                = kernelmatrix(kernelfunc,xtest,xinduc)
#     Kff                = kernelmatrix(kernelfunc,xtest,xtest)

#     K                  = Kff - Kfu*cholinv(Kuu)*(Kuu - Σu) * cholinv(Kuu) * Kfu'
#     m                  = meanfunc.(xtest) + Kfu*cholinv(Kuu)*(μu-meanfunc.(xinduc))
    
#     return m,K
# end
###############################################
##### This function uses the computed inverse of Kff for prediction
function predMVN_fast(gp::GaussianProcess,xtest,xtrain,y)      #function for computing FE and message on edge τ
    kernelfunc         = gp.kernelfunction 
    meanfunc           = gp.meanfunction
    Kxx                = kernelmatrix(kernelfunc,xtrain,xtrain)
    Kxf                = kernelmatrix(kernelfunc,xtrain,xtest)
    invKff             = gp.invKff 
    K                  = Kxx - Kxf*invKff*Kxf'
    m                  = meanfunc.(xtrain) + Kxf*invKff*(y-meanfunc.(xtest))
    
    return m,K
end
####################


function ReactiveMP.constvar(name::Symbol, fn::Function) 
  return ReactiveMP.ConstVariable(name, ReactiveMP.VariableIndividual(), PointMass(fn), of(Message(PointMass(fn), true, false)), 0)
end
function ReactiveMP.constvar(name::Symbol, kernel::Kernel ) 
  return ReactiveMP.ConstVariable(name, ReactiveMP.VariableIndividual(), PointMass(kernel), of(Message(PointMass(kernel), true, false)), 0)
end

function make_multivariate_message(messages)
    m = mean.(messages) 
    v = Diagonal(var.(messages))
    return m,v
end

struct ProcessMeta
    index
    Kxx
    Kff
    Kfx
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::PointMass) = begin 
    return GaussianProcess(q_meanfunc.point,q_kernelfunc.point,nothing,nothing,nothing,nothing)
end

# @rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::Kernel) = begin
#     @show fieldnames(typeof(q_kernelfunc))
#     return GaussianProcess(q_meanfunc.point, q_kernelfunc, nothing, nothing,nothing, nothing)
# end
# @rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::Kernel) = begin
#     @show fieldnames(typeof(q_kernelfunc))
#     return missing 
# end

@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Any, q_μ::GaussianProcess,meta::ProcessMeta) = begin
    m_right, cov_right = mean_cov(q_μ.finitemarginal)
    kernelf = q_μ.kernelfunction
    meanf   = q_μ.meanfunction
    test    = q_μ.testinput
    train   = q_μ.traininput
    mμ, vμ = predMVN_fast(q_μ,test,[train[meta.index]],m_right) #changed here
    vμ = clamp(vμ[1],1e-8,huge)
    θ = 2 / (var(q_out) + vμ[1] + abs2(mean(q_out) - mμ[1]))
    α = convert(typeof(θ), 1.5)
    return Gamma(α, θ)
end


###### build KernelFunc node
# struct KernelFunc{K}
#     kernel::K
# end

# @node KernelFunc Deterministic [out, fn, θ]

# @rule KernelFunc(:out, Marginalisation) (q_fn::PointMass, q_θ::UnivariateNormalDistributionsFamily) = begin
#     m_θ = mean_var(q_θ)
#     func = q_fn.point
#     kerfunc = with_lengthscale(func, m_θ)
#     return KernelFunc(kerfunc)
# end

# @rule KernelFunc(:out, Marginalisation) (q_fn::PointMass, q_θ::MultivariateNormalDistributionsFamily) = begin
#     m_θ = mean_cov(q_θ)
#     func = q_fn.point
#     kerfunc = with_lengthscale(func, m_θ)
#     return KernelFunc(kerfunc)
# end

# @rule KernelFunc(:θ, Marginalisation) (q_out::KernelFunc, q_fn::PointMass) = begin
#     kern = q_out.kernel 
#     length_scale = inv.(kern.transform.v)
#     if length(length_scale) > 1
#         return MvNormalMeanCovariance(length_scale, diageye(length(length_scale)))
#     else
#         return NormalMeanVariance(length_scale[],1.)
#     end
# end

###########################

###### Add rule for backward message on edge "kernfunc" of GP node 
# @rule GaussianProcess[:kernfunc, Marginalisation] (q_out::GaussianProcess, q_meanfunc::PointMass,) = begin
#     return missing 
# end

# function Distributions.entropy(pm::PointMass{F}) where {F <: Function}
#     return ReactiveMP.InfCountingReal(Float64,-1)
# end

# function Distributions.entropy(pm::PointMass{F}) where {F <: Kernel}
#     return ReactiveMP.InfCountingReal(Float64,-1)
# end

# @average_energy GaussianProcess (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any) = begin
#     q_out.finitemarginal
#     return -entropy(q_out.finitemarginal)
# end

# @average_energy NormalMeanPrecision (q_out::Any, q_μ::GaussianProcess, q_τ::Any,meta::ProcessMeta) = begin
#     m_right, cov_right = mean_cov(q_μ.finitemarginal)
#     kernelf = q_μ.kernelfunction
#     meanf   = q_μ.meanfunction
#     test    = q_μ.testinput
#     train   = q_μ.traininput
#     μ_mean, μ_var = predMVN(kernelf,meanf,test,[train[meta.index]],m_right,cov_right)
#     μ_var = clamp(μ_var[1],1e-8,huge)
#     μ_mean = μ_mean[1]
#     out_mean, out_var = mean_var(q_out)
#     return (log2π - mean(log, q_τ) + mean(q_τ) * (μ_var + out_var + abs2(μ_mean - out_mean))) / 2
# end

# function ReactiveMP.entropy(p::GaussianProcess)
#     return ReactiveMP.entropy(p.finitemarginal)
# end


#  @rule GaussianProcess(:kernelfunc, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, ) = begin 
#     K = kernelmatrix(q_out.kernelfunction,xtest,xtest)
#     meanfunc = q_meanfunc.point
    
# #     @show meanfunc(q_out.testinput)
#     distp = MvNormalMeanCovariance(meanfunc(xtest),K)
    
    
    
#     return ContinuousUnivariateLogPdf(DomainSets.FullSpace(), θ -> Distances.kl_divergence(q_out.finitemarginal,))
# end

# @marginalrule DeltaFn{f}(:ins) (m_out::ContinuousUnivariateLogPdf, m_ins::NTuple{N,Any}, meta::F) where {f,N}  = begin 
    
#     mproxy, vproxy = ReactiveMP.approximate_meancov(ReactiveMP.ghcubature(11),x -> exp(m_out.logpdf(x)),0.0,1.0)
    
    
    
#     m_out = NormalMeanVariance(mproxy,vproxy)
    
#     return marginalrule(DeltaFn{f}, Val{ :ins }, Val{ (:out, :ins) }, (Message(m_out, false, false), map(m -> Message(m, false, false), m_ins)), nothing, nothing, meta, __node)
# end

# @marginalrule DeltaFn{f}(:ins) (m_out::NormalMeanVariance, m_ins::NTuple{N,Any}, meta::F) where {f,N} = begin 
#     @show kernelmatrix(f(3.0),(meta.xtest,meta.xtest))
#     error(1)
#     return DeltaMarginal(NormalMeanVariance(0.0,1.0),1)
# end
