

function GInverseDiagonal(Q, L, lambda)
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

function square_distance(a, b)
	# SQUARE_DISTANCE - computes Euclidean SQUARED distance matrix
	# E = distance(A,B)
	#
	#    A - (DxM) matrix 
	#    B - (DxN) matrix
	#
	# Returns:
	#    d - (MxN) Euclidean SQUARED distances between vectors in A and B

	if (size(a,1) != size(b,1))
		error("A and B should be of same dimensionality")
	end

	aa=sum(a.*a, 1)
	bb=sum(b.*b, 1)
	ab=a'*b;
	d = (abs(repmat(aa',1,size(bb,2)) + repmat(bb,size(aa,2),1) - 2*ab))

end


function rls_eigen(Q, L, Qy, lambda, n)
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
