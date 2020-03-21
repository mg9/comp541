using Dates, YAML, Knet
include("data.jl")
include("probe.jl")
include("report.jl")

function choose_probe(args)
  maximum_rank = args["probe"]["maximum_rank"] 
  embed_dim = args["model"]["hidden_dim"]
  if args["probe"]["task_signature"] == "word"
    return OneWordPSDProbe
  elseif args["probe"]["task_signature"] == "word_pair"
    return TwoWordPSDProbe(embed_dim, maximum_rank)
  end
end


function loss(dataset, batchsize, type)
  batchloss = 0.0
  for (id, sent) in collect(Iterators.take(dataset, batchsize))
    gold  = convert(_atype, sent.distances)
    embed = convert(_atype, sent.embeddings)
    pred  = probe(embed)
    if type == "dev"
      devpreds[id] = pred
    end  
    sent_numwords= size(gold,1)
    sent_loss  = sum(abs.(pred - gold))
    squared_length = abs2.(sent_numwords)
    normalized_sent_loss = sent_loss / squared_length
    batchloss += normalized_sent_loss
  end
  batchloss = batchloss / batchsize
  return batchloss
end



function train(probe, trn, dev)
  global devpreds = Dict()
  batchsize = 100
  trn = sort(collect(trn), by=x->x[1])
  dev = sort(collect(dev), by=x->x[1])
  iter = 0
  while iter < 10
    strn = Iterators.Stateful(trn)
    sdev = Iterators.Stateful(dev)
    J = @diff loss(strn, batchsize, "train")
    trainloss = value(J)
    devloss = loss(sdev, batchsize, "dev")
    for par in params(probe)
      g = grad(J, par)
      update!(value(par), g, eval(Meta.parse("Adam()")))
    end
    five_to_fifty_sprmean = report_spearmanr(devpreds, dataset.dev)
    uuas = report_uuas(devpreds, dataset.dev)
    println("iteration: $iter, trainloss: $trainloss, devloss: $devloss, 5-50 spearman mean: $five_to_fifty_sprmean, uuas: $uuas")
    iter += 1
  end
end


CONFIG_PATH = "config/prd_en_ewt-ud-sample.yaml"
args = YAML.load(open(CONFIG_PATH))
dataset = Dataset(args)
probe = choose_probe(args)
train(probe, dataset.trn, dataset.dev)
