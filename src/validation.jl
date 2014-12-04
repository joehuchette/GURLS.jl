import MLBase


function validate{T,S}(train::TrainingProcess{T,LOOCV,S},lambda::Real,args...)
	# args is an array of arguments we'll need to build the models. For linear, this will
	# contain one argument, the value for lambda. 
	n=size(train.X,1)

	function linearEst(train_inds)
		# Use the types S and T as defined by the method call
		return buildModel(TrainingProcess(train.X[[train_inds],:], train.y[train_inds]; kernel=T(), paramsel=LOOCV(), rls=S()),lambda)
	end

	function evalSSE(model, test_inds)
		error_vec = predict(model,train.X[[test_inds],:]) - train.y[test_inds]
		return sum(error_vec.*error_vec)
	end

	scores = MLBase.cross_validate(linearEst,evalSSE,n,MLBase.LOOCV(n))

	return sum(scores)/n

end



function validateDual(Q,L,Qy,lambda,y)
	# Computes sum of square LOOE given the singular value decomposition of the
	# kernel matrix
	# 
	# INPUTS
	# -Q: eigenvectors of the kernel matrix
	# -L: eigenvalues of the kernel matrix
	# -Qy: result of the matrix multiplication of the transpose of Q times the
	#       labels vector Y (Q*Y)
	# -lambda: regularization parameter
	# 
	# OUTPUT:
	# -perfT: 1xT 2d-array of sum of square LOOE per class

	n, T = size(y)

	C = rls_eigen(Q,L,lambda,n)
	Z = GInverseDiagonal(Q,L,lambda)

	perf = zeros(n,t)
	for t = 1:T
		perf[:,t] = C[:,t]./Z
	end

	perfT = sum(perf.^2,1)

	return perfT
end
