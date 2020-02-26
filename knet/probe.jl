using Knet, LinearAlgebra

struct TwoWordPSDProbe
    probe
end

function TwoWordPSDProbe(model_dim::Int, probe_rank::Int)
    probe = param(probe_rank, model_dim)
    TwoWordPSDProbe(probe)
end

function (p::TwoWordPSDProbe)(x)
    transformed = p.probe * x 
    #println("size(transformed): ", size(transformed))

    diffs = []
    for i in 1:size(transformed,2)
        df = hcat(transformed[:,i] .- transformed)
        push!(diffs, df)
    end
    diffs = hcat(diffs...)
    #println("size(diffs): ", size(diffs))

    squared_diffs = abs2.(diffs)
    squared_distances = sum(squared_diffs, dims=1)
    return squared_distances
end


Knet.seed!(1)
println("Two word PSD Probe...")
myprobe = TwoWordPSDProbe(1024, 1024)
x = randn(Float32, 1024, 8)
x = convert(KnetArray{Float32,2}, x)
myprobe(x)
println("End ?")
