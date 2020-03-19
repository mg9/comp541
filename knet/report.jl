using StatsBase

function report_spearmanr(preds, dataset)
    five_to_fifty = []
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
    for (length, sp) in collect(zip(keys(spsents), mean_spsents))
      if 51>length>4
        push!(five_to_fifty, sp) 
      end
    end
    five_to_fifty_sprmean = mean(five_to_fifty) 
    return five_to_fifty_sprmean
end