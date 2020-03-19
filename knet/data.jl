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
end

mutable struct Dataset
    trn
    dev 
    test 
end

function Dataset(args)
    trn, dev, test = read_from_disk(args)
    Dataset(trn, dev, test)
end

function Observation(lineparts)
    ## TODO refactor here 
    if length(lineparts) == 10
        Observation(lineparts[1],lineparts[2],lineparts[3],lineparts[4],lineparts[5],lineparts[6],lineparts[7],lineparts[8],lineparts[9],lineparts[10])
    end
end

function SentenceObservations(id, observations)
   SentenceObservations(id, observations, Any, Any)
end


"""
    Returns dictionaries for dataset sentence observations.
"""
function load_conll_dataset(model_layer, corpus_path, embeddings_path)
    sent_observations = Dict()
    observations = []
    numsentences = 0
    for line in eachline(corpus_path)
        if startswith(line, "# newdoc") || startswith(line, "# text")  && continue 
        elseif startswith(line, "# sent_id") numsentences += 1; observations = []; continue
        else
            obs = Observation(split(line))
            if !isnothing(obs)
                push!(observations, obs)
            else
                sent_observations[numsentences] =  SentenceObservations(numsentences, observations)  
            end
        end
    end
    sent_observations[numsentences] =  SentenceObservations(numsentences, observations)  ## add last sentence TODO change here!
    withembeddings = add_embeddings(model_layer, sent_observations, embeddings_path)

    ## TODO change here !!!
    if occursin("train",embeddings_path)
        local distances_path = "data/en_ewt-ud-sample/distances/sentencedistances-en_ewt-ud-train.h5"
    elseif occursin("dev",embeddings_path)
        local distances_path = "data/en_ewt-ud-sample/distances/sentencedistances-en_ewt-ud-dev.h5"
    end
    distances = h5open(distances_path, "r") do file 
        read(file)
    end
    withdistances = add_sentence_distances(withembeddings, distances)
    return withdistances
end


function add_embeddings(model_layer, sent_observations, embeddings_path)
    println("Loading Pretrained Embeddings from $embeddings_path , using layer $model_layer")
    embeddings = h5open(embeddings_path, "r") do file
        read(file)
    end
    for id in 1:length(sent_observations)
        sentence_embeddings = embeddings[string(id-1)]  # e.g. 1024 x 18 x 3
        sent_observations[id].embeddings =  sentence_embeddings[:,:, model_layer+1]
    end
    return sent_observations
end


function add_sentence_distances(sent_observations, distances)
    for id in 1:length(sent_observations)
        sentencelength = length(sent_observations[id].observations)
        sent_observations[id].distances = distances["labels"][:,:,id][1:sentencelength,1:sentencelength]
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
    corpus_root = args["dataset"]["corpus"]["root"]
    embeddings_root = args["dataset"]["embeddings"]["root"]

    train_corpus_path = join([corpus_root, args["dataset"]["corpus"]["train_path"]])
    dev_corpus_path = join([corpus_root, args["dataset"]["corpus"]["dev_path"]])
    test_corpus_path = join([corpus_root, args["dataset"]["corpus"]["test_path"]])

    train_embeddings_path = join([embeddings_root,args["dataset"]["embeddings"]["train_path"]])
    dev_embeddings_path = join([embeddings_root, args["dataset"]["embeddings"]["dev_path"]])
    test_embeddings_path = join([embeddings_root,args["dataset"]["embeddings"]["test_path"]])
 
    train_observations = load_conll_dataset(model_layer, train_corpus_path, train_embeddings_path)
    dev_observations   = load_conll_dataset(model_layer, dev_corpus_path, dev_embeddings_path)
    #test_observations  = load_conll_dataset(model_layer, test_corpus_path, test_embeddings_path)

    return train_observations, dev_observations, Any # test_observations
end




