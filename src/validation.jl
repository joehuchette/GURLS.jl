import MLBase


function validate{T,S}(train::Training{T,LOOCV,S},lambda::Real,args...)
	# args is an array of arguments we'll need to build the models. For linear, this will
	# contain one argument, the value for lambda. 
	n=size(train.X,1)

	function linearEst(train_inds)
		# Use the types S and T as defined by the method call
		return buildModel(Training(train.X[[train_inds],:], train.y[train_inds]; kernel=T(), paramsel=LOOCV(), rls=S()),lambda)
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

	n, T = size(y,1), size(y,2)

	C = rls_eigen(Q,L,Qy,lambda,n)
	Z = GInverseDiagonal(Q,L,lambda)

	perf = zeros(n,T)
	for j in 1:T
		for i in 1:n
			perf[i,j] = (C[i,j] / Z[i])^2
		end
	end

	perfT = sum(perf,1)

	return perfT
end


function validatePrimal(LEFT,RIGHT,L,lambda,y)
	# Computes sum of square LOOE given the singular value decomposition of the
	# kernel matrix
	# 
	# INPUTS
	# -LEFT: X*Q  (data*eigenvectors of the kernel matrix)
	# -RIGHT: Q^-1 * X' * y 
	# -L: eigenvalues of the kernel matrix
	# -lambda: regularization parameter
	# 
	# OUTPUT:
	# -perfT: 1xT 2d-array of sum of square LOOE per class

	n, T = size(y,1), size(y,2)

	LL = diagm((L + (n*lambda)).^(-1))
	num = y - LEFT*LL*RIGHT
	den = zeros(n,1)

	for j = 1:n
		den[j] = 1.0 - 	(LEFT[j,:]*LL*LEFT[j,:]')[1]
	end

	perf = zeros(n,T)
	for t = 1:T
		perf[:,t] = num[:,t]./den
	end

	perfT = sum(perf.^2,1)

	return perfT
end



