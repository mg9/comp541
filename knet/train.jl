using YAML, Knet, IterTools, Random
import Base: length, iterate
include("data.jl")
include("probe.jl")
include("report.jl")
include("utils.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

function choose_probe(args)
  maximum_rank = args["probe"]["maximum_rank"] 
  embed_dim = args["model"]["hidden_dim"]
  if args["probe"]["task_signature"] == "word"
    return OneWordPSDProbe
  elseif args["probe"]["task_signature"] == "word_pair"
    return TwoWordPSDProbe(embed_dim, maximum_rank)
  end
end

function iterate(d::Dataset, state=randperm(length(d.sents)))
    new_state = state
    new_state_len = length(new_state) 
    max_ind = 0
    if new_state_len == 0 
        return nothing 
    end
    max_ind = min(new_state_len, d.batchsize)  
    sents =  collect(values(d.sents))[new_state[1:max_ind]]
    sents = sort(collect(sents),by=x->x.sentencelength, rev=true)
    f(x) = return x.sentencelength
    sentlengths = f.(sents)
    embeddim = size(sents[1].embeddings,1)
    maxlength = sents[1].sentencelength
    minim = min(d.batchsize, length(sents))
    batch = []
    golds = []
    masks = []

    for b in 1:minim
      push!(batch, wrapmatrixh(sents[b].embeddings, maxlength))
      gold, mask = wrapmatrix(sents[b].distances, maxlength)
      push!(golds, gold)
      push!(masks, mask)
    end

    batch = cat(batch..., dims=3)
    golds = cat(golds..., dims=3)
    masks = cat(masks..., dims=3)
    deleteat!(new_state, 1:max_ind)
    return ((batch, golds, masks, sentlengths), new_state)
end

function length(d::Dataset)
    d, r = divrem(length(d.sents), d.batchsize)
    return r == 0 ? d : d+1
end


function train(probe, trn, dev)
    epoch = adam(probe, ((embeds, golds, masks, sentlengths) for (embeds, golds,masks, sentlengths) in collect(trn)))
    progress!(ncycle(epoch, 10), seconds=5) do x; end
    devpreds = Dict()
    for (id, sent) in collect(dev.sents)
      gold  = convert(_atype, sent.distances)
      embed = convert(_atype, sent.embeddings)
      pred  = probe(embed)
      devpreds[id] = pred
    end
    five_to_fifty_sprmean = report_spearmanr(devpreds, dev.sents)
    uuas = report_uuas(devpreds, dev.sents)
    println("5-50 spearman mean: $five_to_fifty_sprmean, uuas: $uuas")
end



CONFIG_PATH = "config/naacl19/elmo/ptb-prd-ELMo2.yaml"
args = YAML.load(open(CONFIG_PATH))
batchsize = args["dataset"]["batch_size"]
disk = read_from_disk(args)
trn = Dataset(disk[1], batchsize)
dev = Dataset(disk[2], batchsize)
probe = choose_probe(args)
train(probe, trn, dev)

