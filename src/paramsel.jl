##############################################################################
# Holds the results of our parameter selection process
type ParamselResults <: AbstractResults
	model::AbstractModel
	guesses::Array{Any} # Can contain tuples for sig/lam selection
	performance::Array{Real}
end

##############################################################################
function process{K<:Kernel}(train::TrainingProcess{K,LOOCV,Dual})
# XX = train.X' * train.X

(n,d) = size(train.X)

K = buildKernel(train)

# Compute the eigenfactorization of K
(L,Q) = eig(K)
r = rank(train.X)
Qy = Q' * train.y

guesses = getLambdaGuesses(L,r,n,train.options.nLambda)

# pre-allocate memory
performance = zeros(train.options.nLambda)

# Test all values for lambda
i = 1
for lambda in guesses
	# performance[i] = validate(train,lambda)
	performance[i] = validateDual(Q,L,Qy,lambda,train.y)
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

validateDual(x,y,z,w,v) = 0

function getLambdaGuesses(eig,rank,n,nLambda)
# Figure out the lambdas we need to search -- based off of paramsel_lambdaguess.m
	eigs = sort(eig,rev = true) # pass by reference, and order matters for later use. 
	lmax = eigs[1]
	lmin = max(min(lmax * 1e-8, eigs[r]),200*sqrt(eps()))

	powers = linspace(0,1,nLambda)
	guesses = lmin.*(lmax/lmin).^(powers)
	guesses = guesses/n

	return guesses
end