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

function sl(x) 
  return x.sentencelength
end

function iterate(d::Dataset, state=collect(1:length(d.sents)))
   
    println("hey! ", length(state))
    new_state = copy(state)
    new_state_len = length(new_state) 
    if new_state_len == 0 
        return nothing 
    end
    max_ind = min(new_state_len, d.batchsize)  
    sents = d.sents[new_state[1:max_ind]]
    sentlengths = sl.(sents)
    embeddim = size(sents[1].embeddings,1)
    maxlength = sentlengths[1]
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
    trnbatches = collect(trn)
    devbatches = collect(dev)
    epoch = adam(probetransform, ((probe, batch, golds, masks, sentlengths) for (batch, golds,masks, sentlengths) in trnbatches))
    #i = 3
    progress!(ncycle(epoch, 5), seconds=5) do x; 
      trnloss = 0
      for (batch, golds, masks, sentlengths) in trnbatches[1:50]
        tpreds, tloss = predict(probe, batch, golds, masks, sentlengths)
        trnloss += tloss
      end

      devloss = 0
      for (batch, golds, masks, sentlengths) in devbatches[1:50]
        dpreds, dloss = predict(probe, batch, golds, masks, sentlengths)
        devloss += dloss
      end
      println("trnloss: $trnloss, devloss: $devloss")
      #five_to_fifty_sprmean = report_spearmanr(devpreds, dev.sents)
      #uuas = report_uuas(devpreds, dev.sents)
      #println("5-50 spearman mean: $five_to_fifty_sprmean, uuas: $uuas")
    end
      ## Saving the probe
      #probename = "probe_rank1024_v$i.jld2"
      #@info "Saving the probe $probename" 
      #Knet.save(probename,"probe",probe)
      #i+=1

end



CONFIG_PATH = "config/naacl19/elmo/ptb-prd-ELMo2.yaml"
args = YAML.load(open(CONFIG_PATH))
batchsize = args["dataset"]["batch_size"]
disk = read_from_disk(args)
trn = Dataset(disk[1], batchsize)
dev = Dataset(disk[2], batchsize)
probe = choose_probe(args)
train(probe.probe, trn, dev)

