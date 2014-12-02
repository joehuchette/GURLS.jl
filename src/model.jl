# Abstract type to hold sufficient statistics to describe our model. 
abstract AbstractModel <: AbstractResults

# Catch-all to generate errors if we get ahead of ourselves
predict{T<:AbstractModel}(model::T) =
 error("Predict not implemented for models of type $(typeof(model)).")

##############################################################################
# Linear model definition and building
type LinearModel <: AbstractModel
	w::Array{Real,2} # should be 2D to avoid recasting in eval method below
end

function predict(model::LinearModel,X)
	return X' * model.w
end

function buildModel{P<:Paramsel}(train::TrainingProcess{Linear,P,Primal},lambda::Real)
	# w = inv(train.X' * train.X + lambda * eye(size(train.X,2))) * train.X' * train.y

	(n,d) = size(train.X)
	XtX = train.X' * train.X + n * lambda * eye(d)
	Xty = train.X' * train.y
	cholfact!(XtX)
	w = XtX\(XtX'\Xty)

	return LinearModel(w)
end