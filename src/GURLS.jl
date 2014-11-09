module GURLS

importall Base

export AbstractProcess, Experiment, AbstractTask, Kernel, Linear, RLS, Primal,
       Dual, Paramsel, LOOCV, Pred, Perf, Conf, TrainingProcess, 
       PredictionProcess, PerformanceProcess, ConfidenceProcess,
       process,predict,push!

###############################################################################
# AbstractProcess: Abstract type for a process in the experiment 
#                  (e.g. training, testing)
abstract AbstractProcess

###############################################################################
# Experiment: Pipeline for a series of processes (training, testing, etc.)
type Experiment
    pipeline::Vector{AbstractProcess}
    options
end
Experiment() = Experiment(AbstractProcess[],nothing)

push!{P<:AbstractProcess}(x::Experiment,y::P) = push!(x.pipeline,y)

###############################################################################
# AbstractTask: What GURLS defines as a "task"
abstract AbstractTask

###############################################################################
# Kernel: Kernel type used in prediction
abstract Kernel <: AbstractTask
type Linear <: Kernel end

###############################################################################
# RLS: Formulation type used in prediction
abstract RLS
type Primal <: RLS end
type Dual <: RLS end

###############################################################################
# Paramsel: Parameter selection procedure used
abstract Paramsel
type LOOCV <: Paramsel end

abstract Pred
abstract Perf
abstract Conf

###############################################################################
# Classes to hold options for model building

abstract AbstractOptions

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

function TrainingProcess(X, y; kernel   = Linear,
                               paramsel = LOOCV,
                               rls      = Primal)
    options = get_options(kernel(),paramsel(),rls()) # need to actually call constructors,
    												 # otherwise it passes the datatypes 
    												 # themselves, which can't be used for 
    												 # comparison, type hierarchy, etc
    return TrainingProcess{kernel,paramsel,rls}(X,y,options)
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


##############################################################################
# Type to hold the results of an abstract process
abstract AbstractResults

##############################################################################
# Main routine that processes an experiment.
function process(e::Experiment)
	results = Array(AbstractResults,length(e.pipeline))
	i = 1
	for task in e.pipeline
		results[i] = process(task)
	end
	return results
end

# Catch-all for undefined processes
process{T<:AbstractProcess}(task::T) = error("Operation not defined for type $(typeof(task)).")

##############################################################################

include("model.jl")
include("validation.jl")
include("paramsel.jl")






end # Module