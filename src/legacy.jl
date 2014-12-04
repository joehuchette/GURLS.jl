# Support for the "legacy" MATLAB/C++ interface

export defopt, gurls

type TaskDescriptor
    kernel::Set{Kernel}
    rls::Set{RLS}
    paramsel::Set{Paramsel}
    pred::Set{RLS}
    perf::Set{Perf}
    conf::Set{Conf}
end
function TaskDescriptor(;kernel   = Kernel[],
                         rls      = RLS[],
                         paramsel = Paramsel[],
                         pred     = RLS[],
                         perf     = Perf[],
                         conf     = Conf[])
    TaskDescriptor(Set{Kernel}([kernel]),Set{RLS}([rls]),Set{Paramsel}([paramsel]),Set{RLS}([pred]),Set{Perf}([perf]),Set{Conf}([conf]))
end
TaskDescriptor{K<:Kernel,P<:Paramsel,T<:RLS}(::TrainingProcess{K,P,T}) = 
    TaskDescriptor(kernel=K(),rls=P(),paramsel=T())

function merge!(t1::TaskDescriptor,t2::TaskDescriptor)
    union!(t1.kernel,t2.kernel)
    union!(t1.rls,t2.rls)
    union!(t1.paramsel,t2.paramsel)
    union!(t1.pred,t2.pred)
    union!(t1.perf,t2.perf)
    union!(t1.conf,t2.conf)
    return t1
end

type ResultTracker
    train::Set{AbstractModel}
    pred::Set{ParamselResults}
end

ResultTracker(res::AbstractModel)   = 
    ResultTracker(Set{AbstractModel}(res), Set{ParamselResults}())
ResultTracker(res::ParamselResults) = 
    ResultTracker(Set{AbstractModel}(),    Set{ParamselResults}(res))
ResultTracker()                     = 
    ResultTracker(Set{AbstractModel}(),    Set{ParamselResults}())

type LegacyExperiment
    seq::Vector{ASCIIString}
    process::Vector{Vector{Int}}
    tasks::Vector{TaskDescriptor}
    results::Vector
    exper::Experiment
    name::String
end
# If your process requires more than 128 tasks, visit a doctor
LegacyExperiment(name::String) = LegacyExperiment(Vector{ASCIIString}[], fill(Int[],128), Any[], TaskDescriptor[], Experiment(), name)

defopt(name::String) = LegacyExperiment(name)

setoption!(opt::LegacyExperiment,option,value) = 
    setfield!(opt.expr.options,symbol(option),value)

function gurls(X, y, opt::LegacyExperiment, id)
    resize!(opt.results, length(opt.seq))
    process = opt.process[id]
    tdesc = TaskDescriptor()
    res = ResultTracker()
    for (it,task) in enumerate(opt.seq)
        if process[it] == 0 # ignore
            continue
        elseif process[it] in [1,2] # don't yet support writing to disk
            typ, name = split(task, ':')
            process_task!(tdesc, typ, name)
        elseif process[it] == 3 # load from disk...but we already have it in memory!
            merge!(res, ResultTracker(opt.results[it]))
            for k in (id-1):-1:1
                s = opt.process[k]
                if s[it] == 2
                    merge!(tdesc, opt.tasks[k])
                    break
                end
                k == 1 && error("Trying to load a task that has not been executed yet")
            end
        elseif process[it] == 4 # you really want to delete?
            opt.results = nothing
        end
    end
    # validate that task description is valid
    length(tdesc.kernel)   > 1 && error("Too many kernels specified")
    length(tdesc.paramsel) > 1 && error("Too many parameter selection routines specified")
    length(tdesc.rls)      > 1 && error("Too many problem types specified")
    length(tdesc.pred)     > 1 && error("Too many prediction types specified")
    println("tdesc = $tdesc")
    isempty(tdesc.pred) || isempty(tdesc.rls) || (tdesc.pred == tdesc.rls) || error("Prediction type ($(first(tdesc.pred))) and training type ($(first(tdesc.rls))) do not match")
    (!isempty(tdesc.paramsel) && isempty(tdesc.pred) && isempty(tdesc.perf) && isempty(tdesc.conf) ) || 
        error("Cannot train and predict with the same task")

    # validate that results are valid
    isempty(res.train) || isempty(res.pred) || error("Cannot do training and prediction in one task")
    length(res.train) > 1 && error("Cannot attach more than one training")
    length(res.pred)  > 1 && error("Cannot attach more than one prediction")

    # add defaults
    if isempty(tdesc.kernel)
        push!(tdesc.kernel, Linear())
    end
    if isempty(tdesc.rls)
        push!(tdesc.rls, Primal())
    end

    if !isempty(tdesc.paramsel) # training process!
        kernel   = first(tdesc.kernel)
        paramsel = first(tdesc.paramsel)
        rls      = first(tdesc.rls)
        training = TrainingProcess(X, y; kernel=kernel, paramsel=paramsel, rls=rls)
        push!(opt.exper, training)
        process(training)
        return nothing
    else
        try
            training = first(res.train)
        catch
            error("Appropriate training not available")
        end
    end
    if !isempty!(tdesc.pred)
        prediction = PredictionProcess(training, X, y)
        push!(opt.exper, prediction)
    else
        try
            prediction = first(res.pred)
        catch
            error("Appropriate prediction not available")
        end
    end

    if !isempty(tdesc.perf)
        perf = PerformancProcess(prediction, collect(tdesc.perf))
        push!(opt.exper, perf)
        process(perf)
    end
    if !isempty(tdesc.conf)
        conf = ConfidenceProcess(prediction, collect(tdesc.conf))
        push!(opt.exper, conf)
        process(conf)
    end
    return nothing
end

const gurls_funcs = [
    #("split","ho")                   => error("Task not yet implemented"),
    #("paramsel","fixlambda")         => error("Task not yet implemented"),
    ("paramsel","loocvprimal")      => TaskDescriptor(paramsel=LOOCV(), rls=Primal()),
    #("paramsel","loocvdual")         => error("Task not yet implemented"),
    #("paramsel","hoprimal")          => error("Task not yet implemented"),
    #("paramsel","hodual")            => error("Task not yet implemented"),
    #("paramsel","siglam")            => error("Task not yet implemented"),
    #("paramsel","siglamho")          => error("Task not yet implemented"),
    #("paramsel","bfprimal")          => error("Task not yet implemented"),
    #("paramsel","bfdual")            => error("Task not yet implemented"),
    #("paramsel","calibratesgd")      => error("Task not yet implemented"),
    #("paramsel","hoprimalr")         => error("Task not yet implemented"),
    #("paramsel","hodualr")           => error("Task not yet implemented"),
    #("paramsel","horandfeats")       => error("Task not yet implemented"),
    #("paramsel","gpregrLambdaGrid")  => error("Task not yet implemented"),
    #("paramsel","gpregrSigLambGrid") => error("Task not yet implemented"),
    #("paramsel","loogpregr")         => error("Task not yet implemented"),
    #("paramsel","hogpregr")          => error("Task not yet implemented"),
    #("paramsel","siglamhogpregr")    => error("Task not yet implemented"),
    #("paramsel","siglamloogpregr")   => error("Task not yet implemented"),
    #("kernel","chisquared")          => error("Task not yet implemented"),
    ("kernel","linear")              => TaskDescriptor(kernel=Linear()),
    #("kernel","load")                => error("Task not yet implemented"),
    #("kernel","randfeats")           => error("Task not yet implemented"),
    #("kernel","rbf")                 => error("Task not yet implemented"),
    ("rls","primal")                 => TaskDescriptor(rls=Primal()),
    ("rls","dual")                   => TaskDescriptor(rls=Dual())
    #("rls","auto")                   => error("Task not yet implemented"),
    #("rls","pegasos")                => error("Task not yet implemented"),
    #("rls","primalr")                => error("Task not yet implemented"),
    #("rls","dualr")                  => error("Task not yet implemented"),
    #("rls","randfeats")              => error("Task not yet implemented"),
    #("rls","gpregr")                 => error("Task not yet implemented"),
    #("predkernel","traintest")       => error("Task not yet implemented"),
    #("pred","primal")                => error("Task not yet implemented"),
    #("pred","dual")                  => error("Task not yet implemented"),
    #("pred","randfeats")             => error("Task not yet implemented"),
    #("pred","gpregr")                => error("Task not yet implemented"),
    #("perf","macroavg")              => error("Task not yet implemented"),
    #("perf","precrec")               => error("Task not yet implemented"),
    #("perf","rmse")                  => error("Task not yet implemented"),
    #("perf","abserr")                => error("Task not yet implemented"),
    #("conf","maxscore")              => error("Task not yet implemented"),
    #("conf","gap")                   => error("Task not yet implemented"),
    #("conf","boltzmangap")           => error("Task not yet implemented"),
    #("conf","botzman")               => error("Task not yet implemented")
]

function process_task!(tdesc,typ::String,name::String)
    try
        desc = gurls_funcs[(typ,name)]
        merge!(tdesc,desc)
    catch
        error("Unrecognized task '$(typ):$(name)' passed")
    end
end
