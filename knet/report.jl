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


function report_uuas(preds, dataset)
    uuas_total = 0
    uspan_total = 0
    for (id, pred_distances) in preds
        sent = dataset[id]
        gold_edges = union_find(length(sent.observations), pairs_to_distances(sent, sent.distances))
        pred_edges = union_find(length(sent.observations), pairs_to_distances(sent, pred_distances))
        uuas_sent = length(findall(in(pred_edges), gold_edges)) / length(gold_edges)
        if isnan(uuas_sent) continue; end
        uuas_total += uuas_sent
        uspan_total += length(gold_edges)
    end
    return uuas_total/ uspan_total
end


function pairs_to_distances(sent, distances)
    prs_to_distances =  Dict()
    for i in 1:size(distances,1)
        pose_i = sent.observations[i].xpos_sentence
        for j in 1:size(distances, 2)
            pose_j = sent.observations[j].xpos_sentence
            if pose_i in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"] || pose_j in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
                continue # Dont let punctuation tagged words be the part of the tree
            else
                prs_to_distances[(i, j)] = distances[i,j]
            end
        end
    end 
    return prs_to_distances
end


function union_find(n, pairs_to_distances)
    function union(i, j)
        if findparent(i) != findparent(j)
            i_parent = findparent(i)
            parents[i_parent] = j
        end
    end
    function findparent(i)
        i_parent = i
        while true
            if i_parent != parents[i_parent]
                i_parent = parents[i_parent]
            else
                break
            end
        end
        return i_parent
    end
    edges = []
    local parents = collect(1:1:n)
    for ((i_index, j_index), distance) in sort(collect(pairs_to_distances),by=x->x[2])
        i_parent = findparent(i_index)
        j_parent = findparent(j_index)
        if i_parent != j_parent
            union(i_index, j_index)
            push!(edges, (i_index, j_index)) 
        end
    end
    return edges
end