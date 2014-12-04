module GURLS

importall Base

export AbstractProcess, Experiment, AbstractTask, Kernel, Linear, RLS, Primal,
       Dual, Paramsel, LOOCV, Pred, Perf, Conf, TrainingProcess, 
       PredictionProcess, PerformanceProcess, ConfidenceProcess,
       process,predict

###############################################################################
# AbstractProcess: Abstract type for a process in the experiment 
#                  (e.g. training, testing)
abstract AbstractProcess

function Base.print(io::IO,res::AbstractProcess)
	print(io,"$(typeof(res)) with:\n")
	fields = names(res)
	for field in fields
		print(io,"\t$(field): $(typeof(res.(field)))\n")
	end
end
Base.show(io::IO,res::AbstractProcess) = print(io,res)

###############################################################################
# Experiment: Pipeline for a series of processes (training, testing, etc.)
type Experiment
    pipeline::Vector{AbstractProcess}
    options
end
Experiment() = Experiment(AbstractProcess[],nothing)

Base.push!{P<:AbstractProcess}(x::Experiment,y::P) = push!(x.pipeline,y)

###############################################################################
# AbstractTask: What GURLS defines as a "task"
abstract AbstractTask

###############################################################################
# Kernel: Kernel type used in prediction
abstract Kernel <: AbstractTask
type Linear <: Kernel end

###############################################################################
# RLS: Formulation type used in prediction
abstract RLS <: AbstractTask
type Primal <: RLS end
type Dual <: RLS end

###############################################################################
# Paramsel: Parameter selection procedure used
abstract Paramsel <: AbstractTask
type LOOCV <: Paramsel end

abstract Pred <: AbstractTask
abstract Perf <: AbstractTask
abstract Conf <: AbstractTask

###############################################################################
# Classes to hold options for model building

abstract AbstractOptions

function Base.print(io::IO,res::AbstractOptions)
	print(io,"$(typeof(res)) with:\n")
	fields = names(res)
	for field in fields
		print(io,"\t$(field): $(res.(field))\n")
	end
end
Base.show(io::IO,res::AbstractOptions) = print(io,res)

type LinearOptions <: AbstractOptions
	nLambda::Int
end

###############################################################################
# TrainingProcess: Procedure to train data (X,y) using a given kernel, 
#                  parameter selection procedure, and formulation type
type TrainingProcess{K<:Kernel,P<:Paramsel,T<:RLS} <: AbstractProcess
    X
    y
    options::AbstractOptions # hold parameters for model building-- ie nLambda
end

function TrainingProcess{K<:Kernel,P<:Paramsel,T<:RLS}(X, y; kernel::K   = Linear(),
                                                             paramsel::P = LOOCV(),
                                                             rls::T      = Primal())
    options = get_options(kernel,paramsel,rls) # need to actually call constructors,
    												 # otherwise it passes the datatypes 
    												 # themselves, which can't be used for 
    												 # comparison, type hierarchy, etc
    return TrainingProcess{K,P,T}(X,y,options)
end



###############################################################################
# PredictionProcess: Procedure to predict on test data, given a training run
type PredictionProcess <: AbstractProcess
    training::TrainingProcess
    X
    y
end

###############################################################################
# PerformanceProcess: Procedure to assess performance of a given prediction
type PerformanceProcess <: AbstractProcess
    pred::PredictionProcess
    perf::Vector{Perf}
end

###############################################################################
# ConfidenceProcess: Procedure to quantify confidence in a given prediction
type ConfidenceProcess <: AbstractProcess
    pred::PredictionProcess
    conf::Vector{Conf}
end


###############################################################################
# Returns the desired options structure based on types given--- also serves to 
# validate inputs

# catch-all, runs if less-specific case is available.
get_options{K<:Kernel, P <: Paramsel, R <: RLS}(kernal::K,paramsel::P,rls::R) = 
    error("Given training routine is not supported")

get_options(kernel::Linear,paramsel::LOOCV,rls::Primal) =
    LinearOptions(100) # can pick nLambda intelligently later

get_options(kernel::Linear,paramsel::LOOCV,rls::Dual) =
	LinearOptions(100)


##############################################################################
# Type to hold the results of an abstract process
abstract AbstractResults

function Base.print(io::IO,res::AbstractResults)
	print(io,"$(typeof(res)) with:\n")
	fields = names(res)
	for field in fields
		print(io,"\t$(field): $(typeof(res.(field)))\n")
	end
end
Base.show(io::IO,res::AbstractResults) = print(io,res)

##############################################################################
# Main routine that processes an experiment.
function process(e::Experiment)
	results = Array(AbstractResults,length(e.pipeline))
	i = 1
	for task in e.pipeline
		results[i] = process(task)
		i += 1
	end
	return results
end

include("kernel.jl")
include("model.jl")
include("validation.jl")
include("paramsel.jl")
include("legacy.jl")

# Catch-all for undefined processes
process(task) = error("Operation not defined for type $(typeof(task)).")

##############################################################################

end # Module
