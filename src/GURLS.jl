module GURLS

importall Base

export AbstractProcess, Experiment, AbstractTask, Kernel, Linear, RLS, Primal,
       Dual, Paramsel, LOOCV, Pred, Perf, Conf, Training, 
       Prediction, MacroAvg, Performance, Confidence,
       process, predict

###############################################################################
# AbstractProcess: Abstract type for a process in the experiment 
#                  (e.g. training, testing)
abstract AbstractProcess

function Base.print(io::IO,res::AbstractProcess)
	print(io,"$(typeof(res)) with:")
	fields = names(res)
	for field in fields
        println(io)
		print(io,"\t$(field): $(typeof(res.(field)))")
	end
end
Base.show(io::IO,res::AbstractProcess) = print(io,res)

###############################################################################
# AbstractTask: What GURLS defines as a "task"
abstract AbstractTask

###############################################################################
# Kernel: Kernel type used in prediction
abstract Kernel <: AbstractTask
type Linear <: Kernel 
    nLambda::Int
end
Linear() = Linear(100)
type Gaussian <: Kernel
    nLambda::Int
    nSigma::Int
end
Gaussian() = Gaussian(100,100)

num_lambda(a::Linear) = a.nLambda
num_lambda(a::Gaussian) = a.nLambda

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
type MacroAvg <: Perf end
abstract Conf <: AbstractTask

###############################################################################
# Experiment: Pipeline for a series of processes (training, testing, etc.)
type Experiment
    pipeline::Vector{AbstractProcess}
    options

    Experiment(args...) = new(collect(args), nothing)
end

Base.push!(x::Experiment,y::AbstractProcess) = push!(x.pipeline,y)


###############################################################################
# Training: Procedure to train data (X,y) using a given kernel, 
#                  parameter selection procedure, and formulation type
type Training{K<:Kernel,P<:Paramsel,T<:RLS} <: AbstractProcess
    X
    y
    kernel::K
    paramsel::P
    rls::T
end
function Training(X, y; kernel   = Linear(),
                        paramsel = LOOCV(),
                        rls      = Primal())
    return Training(X,y,kernel,paramsel,rls)
end

###############################################################################
# Prediction: Procedure to predict on test data, given a model built through
# a training run
type Prediction <: AbstractProcess
    training::Training
    X
end

###############################################################################
# Performance: Procedure to assess performance of a given prediction
type Performance{Perf} <: AbstractProcess
    pred::Prediction
    ytrue
    # perf::Perf
end
Performance{P<:Perf}(pred::Prediction, ytrue, perf::P) = Performance{P}(pred,ytrue) 

###############################################################################
# Confidence: Procedure to quantify confidence in a given prediction
type Confidence{Conf} <: AbstractProcess
    pred::Prediction
    # conf::Conf
end
Confidence(pred::Prediction, conf::Conf) = Confidence{Conf}(pred)

##############################################################################
# Type to hold the results of an abstract process
abstract AbstractResults

function Base.print(io::IO,res::AbstractResults)
	print(io,"$(typeof(res)) with:")
	fields = names(res)
	for field in fields
        println(io)
		print(io,"\t$(field): $(typeof(res.(field)))")
	end
end
Base.show(io::IO,res::AbstractResults) = print(io,res)

##############################################################################
# Main routine that processes an experiment.
process(e::Experiment) = map(process, e.pipeline)
function process(e::Experiment)
    res = Dict{AbstractProcess,Any}()
    for proc in e.pipeline
        res[proc] = process(proc, res)
    end
    return res
end

# Catch-all for undefined processes
process(a,b) = process(a)
process(task) = error("Operation not defined for type $(typeof(task)).")

include("utils.jl")
include("kernel.jl")
include("model.jl")
include("validation.jl")
include("paramsel.jl")
include("performance.jl")
include("legacy.jl")

##############################################################################

end # Module
