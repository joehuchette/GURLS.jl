##############################################################################
# Holds the results of our parameter selection process
type ParamselResults <: AbstractResults
	model::AbstractModel
	guesses::Array{Any} # Can contain tuples for sig/lam selection
	performance::Array{Real}
end

##############################################################################
function process{R<:RLS}(train::TrainingProcess{Linear,LOOCV,R})
XX = train.X' * train.X
(n,d) = size(train.X)


# Figure out the lambdas we need to search -- based off of paramsel_lambdaguess.m
(eigs,) = eig(XX)
sort!(eigs,rev = true) 
r = rank(train.X)
lmax = eigs[1]
lmin = max(min(lmax * 1e-8, eigs[r]),200*sqrt(eps()))

powers = linspace(0,1,train.options.nLambda)
guesses = lmin.*(lmax/lmin).^(powers)
guesses = guesses/n


# pre-allocate memory
performance = zeros(train.options.nLambda)

# Test all values for lambda
i = 1
for lambda in guesses
	performance[i] = validate(train,lambda)
	println(performance[i])
	i += 1
end

# Find the best value for lambda
(notused,best) = findmin(performance)
lambdaBest = guesses[best]

# Build the final model-- might as well use all of the training set.
model = buildModel(train,lambdaBest)

results = ParamselResults(model,guesses,performance)
return results
end