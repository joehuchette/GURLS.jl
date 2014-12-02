# Support for the "legacy" MATLAB/C++ interface
type LegacyExperiment
    seq::Vector{ASCIIString}
    process::Vector{Vector{Int}}
    exper::Experiment
    name::String
end
# If your process requires more than 128 tasks, visit a doctor
LegacyExperiment(name::String) = LegacyExperiment(Vector{ASCIIString}[], fill(Int[],128), Experiment(), name)

defopt(name::String) = LegacyExperiment(name)

setoption!(opt::LegacyExperiment,option,value) = 
    setfield!(opt.expr.options,symbol(option),value)

type TaskDescriptor
    kernel::Set{Kernel}
    rls::Set{RLS}
    paramsel::Set{Paramsel}
    pred::Set{Pred}
    perf::Set{Perf}
    conf::Set{Conf}
end
TaskDescriptor() = TaskDescriptor(Set{Kernel}(),Set{RLS}(),Set{Paramsel}(),Set{Pred}(),Set{Perf}(),Set{Conf}())
function TaskDescriptor(;kernel   = Set{Kernel}(),
                         rls      = Set{RLS}(),
                         paramsel = Set{Paramsel}(),
                         pred     = Set{Pred}(),
                         perf     = Set{Perf}(),
                         conf     = Set{Conf}())
    TaskDescriptor(kernel,rls,paramsel,pred,perf,conf)
end

function merge!(t1::TaskDescriptor,t2::TaskDescriptor)
    append!(t1.kernel,t2.kernel)
    append!(t1.rls,t2.rls)
    append!(t1.paramsel,t2.paramsel)
    append!(t1.pred,t2.pred)
    append!(t1.perf,t2.perf)
    append!(t1.conf,t2.conf)
    return t1
end


function validate(tdesc::TaskDescriptor)
    length(kernel) > 1 && error("Too many kernels specified")
    length(rls)    > 1 && error("Too many problem types specified")
    nothing
end

function gurls(X, y, opt::LegacyExperiment, id)
    #id <= length(opt.exper.pipeline) || error("Cannot reuse processes")
    process = opt.process[id]
    tdesc = TaskDescriptor()
    for (it,task) in enumerate(opt.seq)
        if process[it] == 0 # ignore
            continue
        elseif process[it] in [1,2] # don't yet support writing to disk
            typ, name = split(task, ':')
            process_task!(tdesc, typ, name)
        elseif process[it] == 3 # load from disk...but we already have it in memory!
            # do nothing
        elseif process[it] == 4 # don't delete!
            # do nothing
        end
    end
    validate(tdesc)
    
end

const gurls_funcs = [
    ("split","ho")                   => error("Task not yet implemented"),
    ("paramsel","fixlambda")         => error("Task not yet implemented"),
    ("paramsel","loocvprimal")       => error("Task not yet implemented"),
    ("paramsel","loocvdual")         => error("Task not yet implemented"),
    ("paramsel","hoprimal")          => error("Task not yet implemented"),
    ("paramsel","hodual")            => error("Task not yet implemented"),
    ("paramsel","siglam")            => error("Task not yet implemented"),
    ("paramsel","siglamho")          => error("Task not yet implemented"),
    ("paramsel","bfprimal")          => error("Task not yet implemented"),
    ("paramsel","bfdual")            => error("Task not yet implemented"),
    ("paramsel","calibratesgd")      => error("Task not yet implemented"),
    ("paramsel","hoprimalr")         => error("Task not yet implemented"),
    ("paramsel","hodualr")           => error("Task not yet implemented"),
    ("paramsel","horandfeats")       => error("Task not yet implemented"),
    ("paramsel","gpregrLambdaGrid")  => error("Task not yet implemented"),
    ("paramsel","gpregrSigLambGrid") => error("Task not yet implemented"),
    ("paramsel","loogpregr")         => error("Task not yet implemented"),
    ("paramsel","hogpregr")          => error("Task not yet implemented"),
    ("paramsel","siglamhogpregr")    => error("Task not yet implemented"),
    ("paramsel","siglamloogpregr")   => error("Task not yet implemented"),
    ("kernel","chisquared")          => error("Task not yet implemented"),
    ("kernel","linear")              => TaskDescriptor(kernel=Linear),
    ("kernel","load")                => error("Task not yet implemented"),
    ("kernel","randfeats")           => error("Task not yet implemented"),
    ("kernel","rbf")                 => error("Task not yet implemented"),
    ("rls","primal")                 => TaskDescriptor(rls=Primal),
    ("rls","dual")                   => TaskDescriptor(rls=Dual),
    ("rls","auto")                   => error("Task not yet implemented"),
    ("rls","pegasos")                => error("Task not yet implemented"),
    ("rls","primalr")                => error("Task not yet implemented"),
    ("rls","dualr")                  => error("Task not yet implemented"),
    ("rls","randfeats")              => error("Task not yet implemented"),
    ("rls","gpregr")                 => error("Task not yet implemented"),
    ("predkernel","traintest")       => error("Task not yet implemented"),
    ("pred","primal")                => error("Task not yet implemented"),
    ("pred","dual")                  => error("Task not yet implemented"),
    ("pred","randfeats")             => error("Task not yet implemented"),
    ("pred","gpregr")                => error("Task not yet implemented"),
    ("perf","macroavg")              => error("Task not yet implemented"),
    ("perf","precrec")               => error("Task not yet implemented"),
    ("perf","rmse")                  => error("Task not yet implemented"),
    ("perf","abserr")                => error("Task not yet implemented"),
    ("conf","maxscore")              => error("Task not yet implemented"),
    ("conf","gap")                   => error("Task not yet implemented"),
    ("conf","boltzmangap")           => error("Task not yet implemented"),
    ("conf","botzman")               => error("Task not yet implemented")
]

function process_task!(tdesc,typ::String,name::String)
    try
        desc = gurls_funcs[(typ,name)]
        merge!(tdesc,desc)
    catch
        error("Unrecognized task '$(typ):$(name)' passed")
    end
end
