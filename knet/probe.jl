using Knet, LinearAlgebra

struct TwoWordPSDProbe
    probe
end

struct OneWordPSDProbe
    probe
end

function TwoWordPSDProbe(model_dim::Int, probe_rank::Int)
    probe = param(probe_rank, model_dim)
    TwoWordPSDProbe(probe)
end


function OneWordPSDProbe(model_dim::Int, probe_rank::Int)
    probe = param(probe_rank, model_dim)
    OneWordPSDProbe(probe)
end

function loadTwoWordPSDProbe(mparams)
    probe = param(mparams)
    TwoWordPSDProbe(probe)
end

function loadOneWordPSDProbe(mparams)
    probe = param(mparams)
    OneWordPSDProbe(probe)
end

mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))


""" 
    twowordprobe(x)
Computes squared L2 distance after projection by a matrix.
For a batch of sentences, computes all n^2 pairs of distances
for each sentence in the batch.
    
"""
function (p::TwoWordPSDProbe)(x)
    squared_distances = []
    transformed = mmul(p.probe, x)  # 1024 x 8 x 1
    for i in 1:size(transformed,2)
        df = hcat(transformed[:,i] .- transformed)
        squared_df = abs2.(df) # 1024 x 8 x 1
        squared_distance = sum(squared_df, dims=1) # 1 x 8 x 1
        push!(squared_distances, squared_distance)
    end
    return vcat(squared_distances...)
end


""" 
    onewordprobe(x)
Computes L1 norm of words after projection by a matrix.
"""
function (p::OneWordPSDProbe)(x)
    norms = []
    transformed = mmul(p.probe, x) 
    for i in 1:size(transformed,2)
        #push!(norms, norm(transformed[:,i]))
        push!(norms, transformed[:,i]' * transformed[:,i])
    end
    return norms
end

