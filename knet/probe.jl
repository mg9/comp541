using Knet, LinearAlgebra
import Distributions: Uniform 
include("utils.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

struct TwoWordPSDProbe
    probe
end

struct OneWordPSDProbe
    probe
end

function TwoWordPSDProbe(model_dim::Int, probe_rank::Int)
    probe = param(rand(Uniform(-0.5,0.5), (probe_rank, model_dim)), atype=_atype)
    TwoWordPSDProbe(probe)
end


function OneWordPSDProbe(model_dim::Int, probe_rank::Int)
    probe = param(probe_rank, model_dim)
    OneWordPSDProbe(probe)
end

function loadTwoWordPSDProbe(mparams)
    probe = param(KnetArray(mparams))
    TwoWordPSDProbe(probe)
end

function loadOneWordPSDProbe(mparams)
    probe = param(mparams)
    OneWordPSDProbe(probe)
end

function mmul(w, x)

    if w == 1 
        return x
    elseif w == 0 
        return 0 
    else 
        return reshape( w * reshape(x, size(x,1), :), 
                        (:, size(x)[2:end]...))
    end
end

""" 
    twowordprobe(x)
Computes squared L2 distance after projection by a matrix.
For a batch of sentences, computes all n^2 pairs of distances
for each sentence in the batch.
    
""" 
# probesize -> P x E 
# x -> E x maxlength x B 
# transformed -> P x maxlength x B
# pred -> maxlength x maxlength x B
function (p::TwoWordPSDProbe)(x, y, masks, sentlengths)
    transformed = mmul(p.probe, convert(_atype, x))  
    maxlength = size(transformed,2)
    B = size(transformed,3)
    preds = []
    tmp = []
    for b in 1:B
        sent = transformed[:,:,b]
        squared_distances = []
        for i in 1:size(sent,2)
            df = hcat(sent[:,i] .- sent)
            squared_df = abs2.(df) #
            squared_distance = sum(squared_df, dims=1) 
            push!(squared_distances, squared_distance)
        end
        pred = vcat(squared_distances...)
        pred = pred .* convert(_atype, masks[:,:,b])
        push!(preds, pred)
    end
    preds = cat(preds..., dims=3)
    loss  = sum(abs.(preds - convert(_atype, y)))
    squared_length = sum(abs2.(sentlengths))
    loss /= squared_length
    println("loss: $loss")
    return loss
end


function (p::TwoWordPSDProbe)(x)
    squared_distances = []
    transformed = mmul(p.probe, x)  # x-> E x T
    for i in 1:size(transformed,2)
        df = hcat(transformed[:,i] .- transformed)
        squared_df = abs2.(df) 
        squared_distance = sum(squared_df, dims=1) 
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



