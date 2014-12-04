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
	Xy = train.X' * train.y
	(n,d) = size(train.X)

	(L,Q) = eig(XX)

	guesses = getLambdaGuesses(L,min(n,d),n,train.options.nLambda)

	LEFT = train.X * Q
	RIGHT = Q' * Xy

	# pre-allocate memory
	performance = zeros(train.options.nLambda)

	# Test all values for lambda
	i = 1
	for lambda in guesses
		# performance[i] = validate(train,lambda)
		performance[i] = validatePrimal(LEFT,RIGHT,L,lambda,train.y)[1]
		# println(performance[i])
		i += 1
	end

end


function process{Kern<:Kernel}(train::TrainingProcess{Kern,LOOCV,Dual})

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
		performance[i] = validateDual(Q,L,Qy,lambda,train.y)[1]
		# println(performance[i])
		i += 1
	end

	# Find the best value for lambda
	(notused,best) = findmin(performance)
	lambdaBest = guesses[best]

	# Build the final model-- might as well use all of the training set.
	println("Lambdabest = $lambdaBest")
	model = buildModel(train,lambdaBest,K)

	results = ParamselResults(model,guesses,performance)
	return results
end

function getLambdaGuesses(eig,rank,n,nLambda)
# Figure out the lambdas we need to search -- based off of paramsel_lambdaguess.m

	eigs = sort(eig,rev = true) # pass by reference, and order matters for later use. 
	lmax = eigs[1]
	lmin = max(min(lmax * 1e-8, eigs[rank]),200*sqrt(eps()))

	powers = linspace(0,1,nLambda)
	guesses = lmin.*(lmax/lmin).^(powers)
	guesses = guesses/n

	return guesses
end
