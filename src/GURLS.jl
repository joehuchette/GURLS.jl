module GURLS

using MLBase

export AbstractProcess, Experiment, AbstractTask, Kernel, Linear, RLS, Primal,
       Dual, Paramsel, LOOCV, Pred, Perf, Conf, TrainingProcess, 
       PredictionProcess, PerformanceProcess, ConfidenceProcess

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
# SplittingProcess: Internal task used in hold-out validation (may be better 
#                   idea to roll into that)
type SplittingProcess <: AbstractProcess
    nholdouts::Int
    holdoutprop::Float64
end

###############################################################################
# TrainingProcess: Procedure to train data (X,y) using a given kernel, 
#                  parameter selection procedure, and formulation type
type TrainingProcess{K<:Kernel,P<:Paramsel,T<:RLS} <: AbstractProcess
    X
    y
end

function TrainingProcess(X, y; kernel   = Linear,
                               paramsel = LOOCV,
                               rls      = Primal)
    verify_parameters(kernel,paramsel,rls)
    TrainingProcess{kernel,paramsel,rls}(X,y)
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
# Internal routine that verifies that a given procedure makes sense: that is,
# that the specified configuration is supported
const supported_classes = [
    (Linear,LOOCV,Primal)
]
function verify_parameters(kernel,paramsel,rls)
    (kernel,paramsel,rls) in supported_classes || error("Given training routine is not supported")
end

end
