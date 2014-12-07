# Abstract type to hold sufficient statistics to describe our model. 
abstract AbstractModel <: AbstractResults

##############################################################################
# Linear model definition and building
type LinearModel{T<:Real} <: AbstractModel
	w::Vector{T}
end

predict(model::LinearModel, X) = X * model.w

function buildModel{P<:Paramsel}(train::Training{Linear,P,Primal}, lambda::Real)

	n, d = size(train.X)
	X, y = train.X, train.y
	XtX = X' * X + n * lambda * eye(d)
	Xty = X' * y
	k = chol(XtX)
	w = k \ (k' \ Xty)

	return LinearModel(vec(w))
end


function buildModel{P<:Paramsel,R<:Real}(train::Training{Linear,P,Dual},lambda::Real,K::Array{R,2})
	w = train.X' * getC(train,lambda,K)
	return LinearModel(vec(w))
end

function getC{R<:Real,Kern<:Kernel,P<:Paramsel}(train::Training{Kern,P,Dual},lambda::Real,K::Array{R,2})
	n = size(train.X,1)

	K += n * lambda * eye(n)
	kFact = chol(K)
	c = kFact \ (kFact' \ train.y)

	return vec(c)
end   

process(p::Prediction, results) = predict(results[p.training].model, p.X)

##############################################################################
# Gaussian model definition and building

type GaussianModel{T<:Real} <: AbstractModel
	c::Array{T,1}
	X::Array{T,2}
	sigma::Real
end

function predict(model::AbstractModel,X)
	k = buildCrossXKernel(model, X)
	return k * model.c''
end

buildModel{P<:Paramsel,R<:Real}(train::Training{Gaussian,P,Dual}, lambda::Real, K::Array{R,2}, sigma) = 
	GaussianModel(getC(train,lambda,K),train.X,sigma)

