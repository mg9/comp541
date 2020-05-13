using YAML, HDF5

struct Observation
    index
    sentence
    lemma_sentence
    upos_sentence
    xpos_sentence
    morph
    head_indices
    governance_relations
    secondary_relations
    extra_info
end

mutable struct SentenceObservations
    id
    observations
    embeddings
    distances
    depths
    sentencelength
end

mutable struct Dataset
    sents::Array{Any,1}
    batchsize
    ids
end

function Dataset(sents, batchsize)
    dsents =  sort(collect(sents),by=x->x.sentencelength, rev=true)
    i(x) = return x.id
    ids = i.(dsents)
    Dataset(dsents, batchsize, ids)
end

function Observation(lineparts)
    ## TODO refactor here 
    if length(lineparts) == 10
        Observation(lineparts[1],lineparts[2],lineparts[3],lineparts[4],lineparts[5],lineparts[6],lineparts[7],lineparts[8],lineparts[9],lineparts[10])
    end
end

function SentenceObservations(id, observations, embeddings, sentencelength)
   SentenceObservations(id, observations, embeddings,Any, Any, sentencelength)
end


"""
    Returns dictionaries for dataset sentence observations.
"""
function load_conll_dataset(model_layer, corpus_path, embeddings_path)
    
    observations = []
    numsentences = 0 
    embeddings = h5open(embeddings_path, "r") do file
        read(file)
    end
    sent_observations = []

    @info "Embeddings loaded from $embeddings_path"
    for line in readlines(corpus_path)
        obs = Observation(split(line))
        if !isnothing(obs)
            push!(observations, obs)
        else
            numsentences += 1; 
            push!(sent_observations, SentenceObservations(numsentences, observations,embeddings[string(numsentences-1)][:,:, model_layer+1] ,length(observations)))  
            observations = []
        end
    end
    println("Sent observations loaded from corpus: ", size(sent_observations))
    withdistances = add_sentence_distances(sent_observations)
    withdepths = add_sentence_depths(sent_observations)
    return withdepths
end



"""
    Returns dictionaries for dataset sentence observations.
"""
function load_bert_layer(model_layer, corpus_path, dataset_type)
    
    numsentences = 0 
    observations = []
    sent_observations = []

    for line in readlines(corpus_path)
        obs = Observation(split(line))
        if !isnothing(obs)
            push!(observations, obs)
        else
            numsentences += 1; 
            train_embeddings_path= string("resources/", dataset_type,"/bertbase_layer7_embeddings_",numsentences-1,".h5")
            embeddings = h5open(train_embeddings_path, "r") do file
                read(file)
            end
            @info "Embeddings loaded from $train_embeddings_path"
            push!(sent_observations, SentenceObservations(numsentences, observations,embeddings["bertbase_layer7"] ,length(observations)))  
            observations = []
        end
    end
    println("Sent observations loaded from corpus: ", size(sent_observations))
    withdistances = add_sentence_distances(sent_observations)
    withdepths = add_sentence_depths(sent_observations)
    return withdepths
end


function calculate_sentence_distances(sent_observations)
    function helper_between_pairs(i,j, head_indices)
        if i==j; return 0; end
        i_path = [i+1]
        j_path = [j+1]
        i_head = i+1
        j_head = j+1
        i_path_length=0; j_path_length = 0;
        while true
            if !(i_head == 1 && (i_path == [i+1] || i_path[end] == 1))
                i_head = head_indices[i_head - 1]
                append!(i_path,i_head)
            end 
            if !(j_head == 1 && (j_path == [j+1] || j_path[end] == 1))
                j_head = head_indices[j_head - 1]
                append!(j_path,j_head)
            end

            if i_head in j_path
                j_path_length = findfirst(isequal(i_head), j_path)
                i_path_length = length(i_path) - 1
                break
            elseif j_head in i_path
                i_path_length = findfirst(isequal(j_head), i_path)
                j_path_length = length(j_path) - 1
                break
            elseif i_head == j_head
                i_path_length = length(i_path) - 1
                j_path_length = length(j_path) - 1
                break
            end
        end
        total_length = j_path_length + i_path_length
        return total_length - 1
    end
    distances = zeros(length(sent_observations), length(sent_observations))
    head_indices = []
    number_of_underscores = 0
    for obs in sent_observations
        if obs.head_indices == "_"
            push!(head_indices, 1)
            number_of_underscores += 1
        else
            push!(head_indices, parse(Int64, obs.head_indices)+ number_of_underscores +1)
        end
     end

    for i in 1:length(sent_observations)
        for j in 1:length(sent_observations)
            distances[i, j] = helper_between_pairs(i, j, head_indices)
            distances[j, i] = distances[i, j]
        end
    end
    return distances
end

function add_sentence_distances(sent_observations)
    for id in 1:length(sent_observations)
        @info "Calculated sentence $id pairwise distances"
        sent_observations[id].distances = calculate_sentence_distances(sent_observations[id].observations); 
    end
   return sent_observations
end


function calculate_sentence_depths(sent_observations)
    function helper_depth(i, head_indices)
        length = 0
        i_head = i+1
        while true
            i_head = head_indices[i_head - 1]
            if i_head != 1
                length += 1
            else return length; end;
        end
    end

    depths = zeros(length(sent_observations))
    head_indices = []
    number_of_underscores = 0
    
    for obs in sent_observations
        if obs.head_indices == "_"
            push!(head_indices, 1)
            number_of_underscores += 1
        else
            push!(head_indices, parse(Int64, obs.head_indices)+ number_of_underscores +1)
        end
     end
    for i in 1:length(sent_observations)
        depths[i] = helper_depth(i, head_indices)
    end
    return depths
end


function add_sentence_depths(sent_observations)
    for id in 1:length(sent_observations)
        @info "Calculated sentence $id depths"
        sent_observations[id].depths = calculate_sentence_depths(sent_observations[id].observations); 
    end
   return sent_observations
end



"""
    read_from_disk(args)

Reads observations from conllx-formatted files
as specified by the yaml arguments dictionary and 
optionally adds pre-constructed embeddings for them.

Returns:
  A 3-tuple: (train, dev, test) where each element in the
  tuple is a list of Observations for that split of the dataset. 
"""

function read_from_disk(args)
    model_layer = args["model"]["model_layer"]
    type = args["model"]["type"]
    corpus_root = args["dataset"]["corpus"]["root"]
    embeddings_root = args["dataset"]["embeddings"]["root"]

    train_corpus_path = join([corpus_root, args["dataset"]["corpus"]["train_path"]])
    dev_corpus_path = join([corpus_root, args["dataset"]["corpus"]["dev_path"]])
    test_corpus_path = join([corpus_root, args["dataset"]["corpus"]["test_path"]])

    train_embeddings_path = join([embeddings_root,args["dataset"]["embeddings"]["train_path"]])
    dev_embeddings_path = join([embeddings_root, args["dataset"]["embeddings"]["dev_path"]])
    test_embeddings_path = join([embeddings_root,args["dataset"]["embeddings"]["test_path"]])

    train_sents_observations = Any
    dev_sents_observations = Any

    if type == "bert"
        train_sents_observations = load_bert_layer(model_layer, train_corpus_path, "train")
        dev_sents_observations   = load_bert_layer(model_layer, dev_corpus_path, "dev")
    else
        train_sents_observations = load_conll_dataset(model_layer, train_corpus_path, train_embeddings_path)
        dev_sents_observations   = load_conll_dataset(model_layer, dev_corpus_path, dev_embeddings_path)
    end

    return train_sents_observations, dev_sents_observations, Any # test_observations
end