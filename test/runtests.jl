using RDatasets # has to go here, as it needs to be included before GURLS.jl
include(Pkg.dir("GURLS") * "/demo/linearKernel.jl")
include(Pkg.dir("GURLS") * "/demo/gaussian.jl")
include(Pkg.dir("GURLS") * "/test/simple_legacy.jl")
include(Pkg.dir("GURLS") * "/demo/iris.jl")
