##############################################################################
# Holds the results of our parameter selection process
type ParamselResults <: AbstractResults
	model::AbstractModel
	guesses::Array{Any} # Can contain tuples for sig/lam selection
	performance::Array{Real}
end

##############################################################################
function process(train::TrainingProcess{Linear,LOOCV,Primal})
XX = train.X' * train.X
(n,d) = size(train.X)


# Figure out the lambdas we need to search -- based off of paramsel_lambdaguess.m
(eigs,) = eig(XX)
sort!(eigs,rev = true) 
r = rank(train.X)
lmax = eigs[1]
lmin = max(min(lmax * 1e-8, eigs[r]),200*sqrt(eps()))

powers = linspace(0,1,train.nLambda)
guesses = lmin.*(lmax/lmin).^(powers)
guesses = guesses/n


# pre-allocate memory
performance = zeros(train.nLambda)
ws = zeros(train.nLambda,d) # list of fitted values for w's
model = LinearModel(zeros(d,1)) 

# Test all values for lambda
i = 1
for lambda in guesses
	model.w = inv(XX + lambda * eye(d)) * train.X' * train.y
	ws[i,:] = model.w
	performance[i] = LOOCV(model,train.X,train.y)
	ws[i,:] = model.w 
	i += 1
end

# Find the best one and return results
(notused,best) = findmin(performance)
model.w = ws[best,:]'

results = ParamselResults(model,guesses,performance)
return results
end