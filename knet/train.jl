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
   
    #println("hey! ", length(state))
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


function loss(probe, data)
    loss = 0
    for (batch, golds,masks, sentlengths) in data
        loss  += probetransform(probe, batch, golds, masks, sentlengths)
    end
    return loss
end


function train(probe, trn, dev)
    trnbatches = collect(trn)
    devbatches = collect(dev)
    epoch = adam(probetransform, ((probe, batch, golds, masks, sentlengths) for (batch, golds,masks, sentlengths) in trnbatches), lr=0.001)

    for e in 1:10
      progress!(epoch) 
      trnloss = loss(probe, trnbatches)
      devloss = loss(probe, devbatches)
      println("epoch $e, trnloss: $trnloss, devloss: $devloss")
      # Reducing lr 
      lrr = Any
      for p in params(probe)
        p.opt.lr =  p.opt.lr/2
        lrr = p.opt.lr
      end
      println("lr reduced to $lrr")
    end

    # TODO refactor here
    devpreds = Dict()
    for (k, (batch, golds, masks, sentlengths)) in enumerate(devbatches)
      dpreds, _ = pred(probe, batch, golds, masks, sentlengths)
      id = 4*k -3
      devpreds[id] = dpreds[:,:,1][1:sentlengths[1],1:sentlengths[1]]
      devpreds[id+1] = dpreds[:,:,2][1:sentlengths[2],1:sentlengths[2]]
      devpreds[id+2] = dpreds[:,:,3][1:sentlengths[3],1:sentlengths[3]]
      devpreds[id+3] = dpreds[:,:,4][1:sentlengths[4],1:sentlengths[4]]
      k += 1
    end

    five_to_fifty_sprmean = report_spearmanr(devpreds, dev.sents)
    uuas = report_uuas(devpreds, dev.sents)
    println("5-50 spearman mean: $five_to_fifty_sprmean, uuas: $uuas")
    ## Saving the probe
    #probename = "probe_rank1024_bestv1.jld2"
    #@info "Saving the probe $probename" 
    #Knet.save(probename,"probe",probe)
 end




CONFIG_PATH = "config/naacl19/elmo/ptb-prd-ELMo2.yaml"
args = YAML.load(open(CONFIG_PATH))
batchsize = args["dataset"]["batch_size"]
disk = read_from_disk(args)
trn = Dataset(disk[1], batchsize)
dev = Dataset(disk[2], batchsize)
probe = choose_probe(args)
train(probe, trn, dev)

