using StatsBase

function report_spearmanr(preds, dataset)
    spsents = Dict()
    for (id, pred_distances) in preds
        sent = dataset[id]
        gold_distances = sent.distances
        sentlength = length(sent.observations) 
        if !(sentlength in keys(spsents))
            spsents[sentlength] = []
        end
        for i in 1:sentlength
            push!(spsents[sentlength],corspearman(Array(pred_distances)[:,i], gold_distances[:,i]))
        end
    end
    mean_spsents = mean.(values(spsents))
    return collect(zip(keys(spsents), mean_spsents))
end