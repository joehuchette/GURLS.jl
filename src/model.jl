# Abstract type to hold sufficient statistics to describe our model. 
abstract AbstractModel <: AbstractResults

# Catch-all to generate errors if we get ahead of ourselves
predict{T<:AbstractModel}(model::T) =
 error("Predict not implemented for models of type $(typeof(model)).")

##############################################################################
# Linear model definition and building
type LinearModel{T<:Real} <: AbstractModel
	w::Array{T,1} # should be 2D to avoid recasting in eval method below
end

function predict(model::LinearModel,X)
	return X * model.w
end

function buildModel{P<:Paramsel}(train::Training{Linear,P,Primal},lambda::Real)
	# w = inv(train.X' * train.X + lambda * eye(size(train.X,2))) * train.X' * train.y

	(n,d) = size(train.X)
	XtX = train.X' * train.X + n * lambda * eye(d)
	Xty = train.X' * train.y
	k = chol(XtX)
	w = k\(k'\Xty)

	return LinearModel(vec(w))
end

function buildModel{P<:Paramsel,R<:Real}(train::Training{Linear,P,Dual},lambda::Real,K::Array{R,2})

	n = size(train.X,1)

	K += n * lambda * eye(n)

	kFact = chol(K)

	w = train.X' * (kFact\(kFact'\train.y))

	return LinearModel(vec(w))

end