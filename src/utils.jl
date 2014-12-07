function GInverseDiagonal(Q::Matrix{Float64}, L, lambda)
	n = size(Q,1)
	t = size(lambda,2)
	Z = zeros(Float64, n, t)

	D = Q .^ 2
	for i = 1:t
	    d = L + (n*lambda[i])
	    d = 1 ./ d
	    Z[:,i] = D*d
	end
	
	return Z	
end

function rls_eigen(Q::Matrix{Float64}, L::Vector{Float64}, Qy::Vector{Float64}, lambda, n)
	# Computes RLS estimator given the singular value decomposition of the
	# kernel matrix
	# 
	# INPUTS
	# -Q: eigenvectors of the kernel matrix
	# -L: eigenvalues of the kernel matrix
	# -Qy: result of the matrix multiplication of the transpose of Q times the
	#       labels vector Y (Q*Y)
	# -lambda: regularization parameter
	# -n: number of training samples
	# 
	# OUTPUT:
	# -C: rls coefficient vector

	L = L + n*lambda
	diagL = diagm(1 ./ L)
	C = (Q * diagL) * Qy
	return C
end
