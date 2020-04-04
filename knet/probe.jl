using Knet, LinearAlgebra
import Distributions: Uniform 
include("utils.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

struct TwoWordPSDProbe
    w
end

struct OneWordPSDProbe
    w
end

function TwoWordPSDProbe(model_dim::Int, probe_rank::Int)
    w = param(rand(Uniform(-0.05,0.05), (probe_rank, model_dim)), atype=_atype)
    TwoWordPSDProbe(w)
end


function OneWordPSDProbe(model_dim::Int, probe_rank::Int)
    w = param(probe_rank, model_dim)
    OneWordPSDProbe(w)
end

function loadTwoWordPSDProbe(mparams)
    w = param(KnetArray(mparams))
    TwoWordPSDProbe(w)
end

function loadOneWordPSDProbe(mparams)
    w = param(mparams)
    OneWordPSDProbe(w)
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
    _, lossm = pred(probe, batch, golds, masks, sentlengths)
    return lossm
end


function pred(probe, batch, golds, masks, sentlengths)
    maxlength = sentlengths[1]
    B = length(sentlengths)
    transformed = mmul(probe.w, convert(_atype,batch))    # P x T x B
    transformed = reshape(transformed, (size(transformed,1),maxlength,1,B)) # P x T x 1 x B
    dummy = convert(_atype, zeros(1,1,maxlength,1))
    transformed = transformed .+ dummy   # P x T x T x B
    transposed = permutedims(transformed, (1,3,2,4))
    diffs = transformed - transposed
    squareddists = abs2.(diffs)
    squareddists = sum(squareddists, dims=1)  # 1 x T x T x B
    squareddists = reshape(squareddists, (maxlength, maxlength,B)) #  T x T x B
    squareddists = convert(_atype,masks) .* squareddists
    
    a = abs.(squareddists - convert(_atype,golds))
    b = reshape(a, (size(a,1)*size(a,2),B))
    b = sum(b,dims=1)
    normalized_sent_losses = vec(b)./ convert(_atype, abs2.(sentlengths))
    batchloss = sum(normalized_sent_losses) /  B
    return squareddists, batchloss
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



