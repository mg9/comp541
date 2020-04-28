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
function load_conll_dataset(model_layer, corpus_path, embeddings_path, distances_path, depths_path, distances_batch_size)
    
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
    withdistances = add_sentence_distances(sent_observations, distances_path, distances_batch_size)
    withdepths = add_sentence_depths(sent_observations, depths_path, distances_batch_size)
    return withdepths
end



"""
    Returns dictionaries for dataset sentence observations.
"""
function load_bert_layer(model_layer, corpus_path, distances_path, depths_path, distances_batch_size, dataset_type)
    
    numsentences = 0 
    observations = []
    sent_observations = []

    for line in readlines(corpus_path)
        obs = Observation(split(line))
        if !isnothing(obs)
            push!(observations, obs)
        else
            numsentences += 1; 

            train_embeddings_path= string("../", dataset_type,"/bertbase_layer7_embeddings_",numsentences-1,".h5")
            embeddings = h5open(train_embeddings_path, "r") do file
                read(file)
            end
            @info "Embeddings loaded from $train_embeddings_path"
            push!(sent_observations, SentenceObservations(numsentences, observations,embeddings["bertbase_layer7"] ,length(observations)))  
            observations = []
        end
    end
    println("Sent observations loaded from corpus: ", size(sent_observations))
    withdistances = add_sentence_distances(sent_observations, distances_path, distances_batch_size)
    return withdistances
end



function add_sentence_distances(sent_observations, pathbase, batchsize)
    for id in 1:length(sent_observations)
        if id% batchsize >0 ? k=floor(id/batchsize) + 1 : k=floor(id/batchsize); end
        distances_file = string(pathbase,string(Integer(k)),".h5")
        distances = h5open(distances_file, "r") do file 
            read(file)
        end
        sentencelength = length(sent_observations[id].observations)
        idmod = (id%batchsize)
        if idmod == 0
            idmod = batchsize
        end
        sent_observations[id].distances = distances["labels"][:,:,idmod][1:sentencelength,1:sentencelength]
    end
    return sent_observations
end


function add_sentence_depths(sent_observations, pathbase, batchsize)
    for id in 1:length(sent_observations) 
        if id% batchsize >0 ? k=floor(id/batchsize) + 1 : k=floor(id/batchsize); end
        depth_file = string(pathbase,string(Integer(k)-1),".h5")
        depths = h5open(depth_file, "r") do file 
            read(file)
        end
        @info "Sentence depth loaded from $depth_file"

        sentencelength = length(sent_observations[id].observations)
        idmod = (id%batchsize)
        if idmod == 0
            idmod = batchsize
        end
        sent_observations[id].depths = depths["labels"][:,idmod][1:sentencelength]
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

    distances_batchsize = args["dataset"]["distances"]["distances_batch_size"]
    corpus_root = args["dataset"]["corpus"]["root"]
    embeddings_root = args["dataset"]["embeddings"]["root"]
    distances_root = args["dataset"]["distances"]["root"]
    depths_root = args["dataset"]["depths"]["root"]

    train_corpus_path = join([corpus_root, args["dataset"]["corpus"]["train_path"]])
    dev_corpus_path = join([corpus_root, args["dataset"]["corpus"]["dev_path"]])
    test_corpus_path = join([corpus_root, args["dataset"]["corpus"]["test_path"]])

    train_embeddings_path = join([embeddings_root,args["dataset"]["embeddings"]["train_path"]])
    dev_embeddings_path = join([embeddings_root, args["dataset"]["embeddings"]["dev_path"]])
    test_embeddings_path = join([embeddings_root,args["dataset"]["embeddings"]["test_path"]])
 
    train_distances_path = join([distances_root,args["dataset"]["distances"]["train_path"]])
    dev_distances_path = join([distances_root,args["dataset"]["distances"]["dev_path"]])


    train_depths_path = join([depths_root,args["dataset"]["depths"]["train_path"]])
    dev_depths_path = join([depths_root,args["dataset"]["depths"]["dev_path"]])

    train_sents_observations = Any
    dev_sents_observations = Any

    if type == "bert"
        train_sents_observations = load_bert_layer(model_layer, train_corpus_path,  train_distances_path, train_depths_path, distances_batchsize, "train")
        dev_sents_observations   = load_bert_layer(model_layer, dev_corpus_path,  dev_distances_path, dev_depths_path, distances_batchsize, "dev")
    else
        train_sents_observations = load_conll_dataset(model_layer, train_corpus_path, train_embeddings_path, train_distances_path, train_depths_path, distances_batchsize)
        dev_sents_observations   = load_conll_dataset(model_layer, dev_corpus_path, dev_embeddings_path, dev_distances_path, dev_depths_path, distances_batchsize)
        #test_observations  = load_conll_dataset(model_layer, test_corpus_path, test_embeddings_path)
    end

    return train_sents_observations, dev_sents_observations, Any # test_observations
end




