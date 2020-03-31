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
# batch -> E x T x B 
function probetransform(probe, batch, golds, masks, sentlengths)
    _, lossm = predict(probe, batch, golds, masks, sentlengths)
    return lossm
end

function predict(probe, batch, golds, masks, sentlengths)
    maxlength = sentlengths[1]
    B = length(sentlengths)
    transformed = mmul(probe, convert(_atype,batch))  
    transformed = permutedims(transformed, (3,2,1))
    transformed = reshape(transformed, (1024,maxlength,1,B)) # P x T x 1 x B
    dummy = convert(_atype, zeros(1,1,maxlength,1))
    transformed = transformed .+ dummy
    transposed = permutedims(transformed, (1,3,2,4))
    diffs = transformed - transposed
    squareddists = abs2.(diffs)
    squareddists = sum(diffs, dims=1)
    squareddists = reshape(squareddists, (B,maxlength, maxlength)) # B x T x T
    squareddists = permutedims(squareddists, (3,2,1)) # T x T x B
    squareddists = convert(_atype,masks) .* squareddists
    lossm = sum(abs.(squareddists - convert(_atype,golds)))
    lossm /= sum(sentlengths)
    lossm /= B
    return squareddists, lossm
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



