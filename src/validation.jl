# validateDual(Q,L,Qy,lambda,y::Vector) =
# 	validateDual(Q,L,Qy,lambda,y,Array(Float64,size(y, 1)),Array(Float64,size(Q, 1)))

# function validateDual(Q,L,Qy,lambda,y::Vector, C, Z)
# 	# Computes sum of square LOOE given the singular value decomposition of the
# 	# kernel matrix
# 	# 
# 	# INPUTS
# 	# -Q: eigenvectors of the kernel matrix
# 	# -L: eigenvalues of the kernel matrix
# 	# -Qy: result of the matrix multiplication of the transpose of Q times the
# 	#       labels vector Y (Q*Y)
# 	# -lambda: regularization parameter
# 	# 
# 	# OUTPUT:
# 	# -perfT: 1xT 2d-array of sum of square LOOE per class

# 	n = size(y,1)

# 	println(Qy)

# 	# @time C = rls_eigen(Q,L,Qy,lambda,n)
# 	# @time Z = GInverseDiagonal(Q,L,lambda)
# 	rls_eigen(Q,L,Qy,lambda,n,C)
# 	GInverseDiagonal(Q,L,lambda,Z)
	
# 	@assert (size(C,1),size(C,2)) == (n,1) "size(C) = $(size(C))"
# 	@assert size(Z) == (n,)

# 	acc = 0.0
# 	for i in 1:n
# 		println(C[i] ./ Z[1])
# 		#@inbounds acc += (y[i] - (C[i] ./ Z[1])[1])
# 	end

# 	return 1 - 1/n * acc
# end

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

	@assert (size(C,1),size(C,2)) == (n,T) "size(C) = $(size(C))"
	@assert size(Z) == (n,)

	pred = zeros(size(y))

	for i = 1:T
		pred[:,i] = (C[:,i]./Z)
	end

	perf = zeros(n,T)
	for j in 1:T
		for i in 1:n
			@inbounds perf[i,j] = (C[i,j] / Z[i])^2
		end
	end

	return sum(perf,1)[1]
end


function validatePrimal(LEFT,RIGHT,L,lambda,y::Vector)
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
	d1, d2 = size(LEFT,1), size(LEFT,2)
	@assert d1 == n

	# LL = diagm((L + (n*lambda)).^(-1))
	# num = y - LEFT*LL*RIGHT
	num = copy(y)
	@simd for i in 1:d1
		for j in 1:d2
			@inbounds num[i] -= LEFT[i,j] * RIGHT[j] / (L[j] + (n*lambda))
		end
	end

	den = fill(1.0, n)

	@simd for j in 1:n
		for k in 1:d2
			@inbounds den[j] -= L[k] * LEFT[j,k]^2 
		end
	end

	perfT = 0.0
	@simd for i in 1:n
		@inbounds perfT += (num[i] / den[i])^2
	end

	return perfT
end



