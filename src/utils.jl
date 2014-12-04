

function GInverseDiagonal(Q,L,lambda)
	n = size(Q,1)
	t = size(lambda,2)
	Z = zeros(n,t)

	D = Q.^(2)
	for i = 1:t
	    d = L + (n*lambda[i])
	    d = 1 ./ d
	    Z[:,i] = D*d
	end
	
	return Z	
end




function rls_eigen(Q,L,Qy,lambda,n)
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

	L = L + n*lambda;
	L = diagm(1 ./ L);
	C = (Q*L)*Qy;
	return C
end
