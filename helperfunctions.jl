import ReactiveMP: diageye 

#-------- Expected Improvement acquisition function -----#
function EI_acquisition(μ,Σ, y_data, λ)
    """
    Expectation Improvement acquisition function 
    """
    y_opt = maximum(y_data)
    σ_x = sqrt.(diag(Σ) .+ 1e-6)
    y_new = []
    for i=1:length(μ)
        temp = (μ[i] - y_opt - λ) * cdf(Normal(),(μ[i] - y_opt - λ)/σ_x[i]) + σ_x[i] * pdf(Normal(),(μ[i] - y_opt - λ)/σ_x[i])
        append!(y_new, temp)
    end
    return y_new
end

#------ Upper Confidence Bound ------#
function UCB(μ, Σ, λ)
    """
    Upper Confidence Bound acquisition function 
    """
    σ_x = sqrt.(diag(Σ) .+ 1e-6)

    return μ .+ λ*σ_x
end

#-------- Probability Improvement acquisition function --------#
function PI_acquisition(μ, Σ, y_data, λ)
    """
    Probability of Improvement acquisition function 
    """
    y_opt = maximum(y_data)
    σ_x = sqrt.(diag(Σ) .+ 1e-6)
    y_new = []
    for i=1:length(μ)
        temp = cdf(Normal(),(μ[i] - y_opt - λ)/σ_x[i]) 
        append!(y_new, temp)
    end
    return y_new
end
## GP prediction 
function gppredict(xtrain, ytrain, xtest, meanfunc, kernel, Kff)
    """
    Compute the predictive distribution of GP 
    xtrain, ytrain : training data 
    xtest          : test input 
    meanfunc       : mean function of GP 
    kernel         : kernel function of GP 
    Kff            : covariance matrix of test input  
    """
    #compute cross-covariance 
    Kfy = kernelmatrix(kernel, xtest, xtrain)
    Kyy = kernelmatrix(kernel,xtrain,xtrain) 

    μ = meanfunc.(xtest) + Kfy * inv(Kyy + 1e-6*diageye(length(ytrain))) * (ytrain - meanfunc.(xtrain))
    Σ = Kff - Kfy * inv(Kyy + 1e-6*diageye(length(ytrain))) * Kfy' 
    return μ, Σ
end
